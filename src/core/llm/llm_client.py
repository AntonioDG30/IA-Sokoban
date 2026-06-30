# Client HTTP per dialogare con Ollama in locale.
#
# Usa http.client con connessione persistente (keep-alive) per abbattere la latenza da
# ~2.3 s (nuova connessione TCP a ogni chiamata) a ~0.14 s a regime. Il modello viene
# caricato una volta sola in VRAM nel costruttore e tenuto attivo per tutto il training.

import http.client
import json
import time
import urllib.request
from typing import Optional

from sistema_10x10.config import CONFIG_LLM, MAX_RETRY_LLM, TIMEOUT_LLM_SEC


class ClienteLLM:
    """
    Client per Ollama con connessione HTTP persistente.

    Apre una sola connessione TCP nel costruttore e la riusa per tutte le chiamate, poi
    esegue un warm-up che carica il modello in VRAM con i parametri ottimali (num_gpu=999,
    num_ctx=512). provider al momento può essere solo 'ollama'.
    """

    def __init__(self, provider: str = "ollama") -> None:
        if provider not in CONFIG_LLM:
            raise ValueError(
                f"Provider '{provider}' non riconosciuto. "
                f"Disponibili: {list(CONFIG_LLM)}"
            )

        cfg = CONFIG_LLM[provider]
        self.provider = provider
        self.model = cfg["model"]

        # qwen3 accetta think=False per disattivare il ragionamento interno e rispondere
        # subito, senza l'overhead della chain-of-thought
        self._no_think = "qwen3" in self.model.lower()

        self._ollama_conn: Optional[http.client.HTTPConnection] = None
        self._ollama_connetti()
        self._ollama_warm_up()

        print(
            "[ClienteLLM] model=" + self.model
            + (" [no_think=True]" if self._no_think else "")
        )

    # GESTIONE DELLA CONNESSIONE HTTP

    def _ollama_connetti(self) -> None:
        """
        Apre (o riapre) la connessione HTTP persistente verso Ollama su localhost:11434.

        Chiude prima l'eventuale connessione precedente. Il timeout di 120 s copre anche le
        chiamate più lente durante il training.
        """
        try:
            if self._ollama_conn is not None:
                self._ollama_conn.close()
        except Exception:
            pass
        self._ollama_conn = http.client.HTTPConnection("localhost", 11434, timeout=120)

    def _ollama_get(self, endpoint: str, timeout: float = 5.0) -> dict:
        """
        Esegue una GET sull'API di Ollama tramite urllib.

        Usata solo per /api/ps (modelli attivi) durante il warm-up: una GET non ha body e
        non trae vantaggio dalla connessione persistente come invece le POST.
        """
        req = urllib.request.Request("http://localhost:11434" + endpoint)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _ollama_post(self, endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
        """
        Invia una POST sulla connessione persistente applicando davvero il timeout richiesto.

        Imposta il timeout sia sull'oggetto connessione (per un'eventuale riconnessione) sia
        sul socket già aperto: così il keep-alive resta, ma il timeout per-chiamata è effettivo.
        Se la connessione è caduta (es. dopo un lungo idle) la riapre e riprova una sola volta
        prima di propagare l'eccezione.
        """
        data = json.dumps(payload).encode("utf-8")
        for tentativo in range(2):
            try:
                # Timeout per-chiamata effettivo: sull'oggetto (per una futura connect)
                # e sul socket già aperto, senza sacrificare la connessione persistente.
                self._ollama_conn.timeout = timeout
                if self._ollama_conn.sock is not None:
                    self._ollama_conn.sock.settimeout(timeout)
                self._ollama_conn.request(
                    "POST", endpoint, body=data,
                    headers={"Content-Type": "application/json"},
                )
                resp = self._ollama_conn.getresponse()
                body = resp.read()
                return json.loads(body)
            except Exception:
                if tentativo == 0:
                    # Connessione caduta: riapri e riprova una volta
                    self._ollama_connetti()
                else:
                    raise

    # WARM-UP: CARICA IL MODELLO IN VRAM CON I PARAMETRI OTTIMALI

    def _ollama_warm_up(self) -> None:
        """
        Forza il caricamento del modello in VRAM con num_gpu=999 e num_ctx=512.

        Se il modello è già in RAM ma in split GPU/CPU (size_vram < size totale) lo scarica e
        lo ricarica con i parametri giusti. num_ctx=512 riduce il KV cache da ~4 GB (default
        4096 token) a ~0.1 GB, lasciando spazio in VRAM al modello RL che gira in parallelo;
        keep_alive=-1 impedisce a Ollama di scaricare il modello tra una chiamata e l'altra.
        """
        try:
            d = self._ollama_get("/api/ps", timeout=5.0)
            modelli = d.get("models", [])
            nome_base = self.model.split(":")[0]

            # Rileva se il modello è caricato in split GPU/CPU (in VRAM solo in parte)
            in_split = any(
                m.get("name", "").startswith(nome_base)
                and m.get("size_vram", 0) < m.get("size", 1)
                for m in modelli
            )

            if in_split:
                print("[ClienteLLM] Modello in split GPU/CPU: scarico e ricarico...")
                try:
                    # keep_alive=0 forza lo scaricamento del modello dalla VRAM
                    self._ollama_post(
                        "/api/chat",
                        {
                            "model": self.model,
                            "messages": [],
                            "keep_alive": 0,
                            "stream": False,
                        },
                        timeout=10.0,
                    )
                except Exception:
                    pass
                time.sleep(1.0)

            print("[ClienteLLM] Caricamento modello con num_ctx=512, num_gpu=999...")
            self._ollama_post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_gpu": 999, "num_ctx": 512},
                    "keep_alive": -1,   # tieni il modello in VRAM a tempo indeterminato
                    "stream": False,
                },
                timeout=120.0,
            )
            print("[ClienteLLM] Modello pronto in VRAM.")

        except Exception as e:
            print("[ClienteLLM] Warm-up fallito (Ollama non attivo?): " + str(e))

    # INTERFACCIA PUBBLICA

    def chiedi(
        self,
        prompt: str,
        max_tokens: int = 20,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Invia il prompt al modello e restituisce la risposta testuale (stringa).

        In caso di errore di rete riprova fino a MAX_RETRY_LLM volte, aspettando 0.5 s, 1 s,
        1.5 s tra un tentativo e l'altro; se falliscono tutti restituisce la stringa vuota.
        """
        timeout = timeout or TIMEOUT_LLM_SEC

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "num_gpu":     999,         # tutti i layer del modello sulla GPU
                "num_ctx":     512,         # KV cache ridotto: il modello sta tutto in VRAM
                "num_predict": max_tokens,  # limite ai token generati in risposta
                "temperature": 0,           # risposta deterministica (greedy)
            },
            "keep_alive": -1,   # mantiene il modello in VRAM tra una chiamata e l'altra
            "stream": False,
        }

        if self._no_think:
            # qwen3: disattiva il ragionamento interno per risposte più veloci
            payload["think"] = False

        for tentativo in range(1, MAX_RETRY_LLM + 1):
            try:
                d = self._ollama_post("/api/chat", payload, timeout=timeout)
                testo = d.get("message", {}).get("content", "")
                return testo.strip()
            except Exception as e:
                if tentativo == MAX_RETRY_LLM:
                    print(
                        "[ClienteLLM] Fallimento dopo "
                        + str(MAX_RETRY_LLM) + " tentativi: " + type(e).__name__
                    )
                    return ""
                # Aspetta (backoff crescente) e riapri la connessione prima di riprovare
                time.sleep(0.5 * tentativo)
                self._ollama_connetti()

        return ""
