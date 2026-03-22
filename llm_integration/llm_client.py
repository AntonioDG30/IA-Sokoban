"""Client HTTP per comunicare con Ollama in locale.

Usa http.client con connessione persistente (keep-alive) per ridurre
la latenza da ~2.3s (nuova connessione TCP per ogni chiamata) a ~0.14s
a regime. Il modello viene caricato una volta sola in VRAM al costruttore
e mantenuto attivo per tutta la durata del training.
"""

import http.client
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

from experiments.config import CONFIG_LLM, MAX_RETRY_LLM, TIMEOUT_LLM_SEC


class ClienteLLM:
    """Client per Ollama con connessione HTTP persistente.

    Apre una sola connessione TCP al costruttore e la riusa per tutte
    le chiamate successive. Esegue un warm-up iniziale per caricare
    il modello in VRAM con i parametri ottimali (num_gpu=999, num_ctx=512).

    Parametri:
        provider: attualmente supporta solo 'ollama'.
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

        # qwen3 supporta think=False per disabilitare il ragionamento interno
        # e ottenere risposte dirette senza overhead di chain-of-thought
        self._no_think = "qwen3" in self.model.lower()

        self._ollama_conn: Optional[http.client.HTTPConnection] = None
        self._ollama_connetti()
        self._ollama_warm_up()

        print(
            "[ClienteLLM] model=" + self.model
            + (" [no_think=True]" if self._no_think else "")
        )

    # ------------------------------------------------------------------
    # Gestione connessione HTTP
    # ------------------------------------------------------------------

    def _ollama_connetti(self) -> None:
        """Apre la connessione HTTP persistente verso Ollama su localhost:11434.

        Chiude l'eventuale connessione precedente prima di aprirne una nuova.
        Timeout di 120s per coprire le chiamate piu' lente durante il training.
        """
        try:
            if self._ollama_conn is not None:
                self._ollama_conn.close()
        except Exception:
            pass
        self._ollama_conn = http.client.HTTPConnection("localhost", 11434, timeout=120)

    def _ollama_get(self, endpoint: str, timeout: float = 5.0) -> dict:
        """Esegue una GET sull'API di Ollama tramite urllib.

        Usato solo per /api/ps (lista modelli attivi) durante il warm-up.
        urllib e' usato qui perche' GET non richiede body e non beneficia
        della connessione persistente come le POST.
        """
        req = urllib.request.Request("http://localhost:11434" + endpoint)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _ollama_post(self, endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
        """Invia una POST tramite la connessione persistente.

        Se la connessione e' caduta (es. dopo un lungo idle), la riapre
        automaticamente e riprova una volta sola prima di sollevare eccezione.
        """
        data = json.dumps(payload).encode("utf-8")
        for tentativo in range(2):
            try:
                self._ollama_conn.request(
                    "POST", endpoint, body=data,
                    headers={"Content-Type": "application/json"},
                )
                resp = self._ollama_conn.getresponse()
                body = resp.read()
                return json.loads(body)
            except Exception:
                if tentativo == 0:
                    # Connessione caduta: riapri e riprova
                    self._ollama_connetti()
                else:
                    raise

    # ------------------------------------------------------------------
    # Warm-up: carica il modello in VRAM con i parametri ottimali
    # ------------------------------------------------------------------

    def _ollama_warm_up(self) -> None:
        """Forza il caricamento del modello con num_gpu=999 e num_ctx=512.

        Controlla prima se il modello e' gia' in RAM ma in split GPU/CPU
        (size_vram < size totale). In quel caso lo scarica e lo ricarica
        con i parametri corretti. num_ctx=512 riduce il KV cache da ~4 GB
        (default 4096 token) a ~0.1 GB, lasciando spazio in VRAM per il
        modello RL che gira in parallelo. keep_alive=-1 impedisce a Ollama
        di scaricare il modello tra una chiamata e l'altra.
        """
        try:
            d = self._ollama_get("/api/ps", timeout=5.0)
            modelli = d.get("models", [])
            nome_base = self.model.split(":")[0]

            # Verifica se il modello e' in split GPU/CPU
            in_split = any(
                m.get("name", "").startswith(nome_base)
                and m.get("size_vram", 0) < m.get("size", 1)
                for m in modelli
            )

            if in_split:
                print("[ClienteLLM] Modello in split GPU/CPU: scarico e ricarico...")
                try:
                    # keep_alive=0 scarica il modello dalla VRAM
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
                    "keep_alive": -1,   # tieni il modello in VRAM per sempre
                    "stream": False,
                },
                timeout=120.0,
            )
            print("[ClienteLLM] Modello pronto in VRAM.")

        except Exception as e:
            print("[ClienteLLM] Warm-up fallito (Ollama non attivo?): " + str(e))

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def chiedi(
        self,
        prompt: str,
        max_tokens: int = 20,
        timeout: Optional[float] = None,
    ) -> str:
        """Invia il prompt al modello e restituisce la risposta testuale.

        Riprova fino a MAX_RETRY_LLM volte in caso di errore di rete,
        aspettando 0.5s, 1s, 1.5s tra i tentativi successivi. Restituisce
        stringa vuota se tutti i tentativi falliscono.

        Parametri:
            prompt:     testo del prompt da inviare al modello.
            max_tokens: numero massimo di token da generare nella risposta.
            timeout:    timeout in secondi per la chiamata HTTP.
        """
        timeout = timeout or TIMEOUT_LLM_SEC

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "num_gpu":     999,         # tutti i layer su GPU
                "num_ctx":     512,         # KV cache ridotto: modello tutto in VRAM
                "num_predict": max_tokens,  # limite token risposta
                "temperature": 0,           # risposta deterministica
            },
            "keep_alive": -1,   # mantieni modello in VRAM tra le chiamate
            "stream": False,
        }

        if self._no_think:
            # qwen3: disabilita il ragionamento interno per risposte piu' veloci
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
                # Aspetta prima di riprovare e riapri la connessione
                time.sleep(0.5 * tentativo)
                self._ollama_connetti()

        return ""
