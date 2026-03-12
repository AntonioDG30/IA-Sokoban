"""Client LLM unificato: Groq, Mistral, Ollama.

Per Groq e Mistral usa OpenAI SDK (API-compatible).
Per Ollama usa l'API HTTP nativa (/api/chat) via http.client (connessione
persistente keep-alive): elimina il TCP handshake per ogni chiamata,
riducendo la latenza da ~2.3s a ~0.14s su chiamate warm.
num_gpu=999 e num_ctx=512 vengono rispettati tramite warm_up al costruttore
che forza il ricaricamento del modello se e' in split GPU/CPU.

Gestione automatica qwen3: aggiunge /no_think al prompt per disabilitare
il chain-of-thought e mantenere risposte brevi e veloci.
"""

import http.client
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

import openai

from experiments.config import CONFIG_LLM, MAX_RETRY_LLM, TIMEOUT_LLM_SEC


class ClienteLLM:
    """Wrapper LLM unificato per Groq, Mistral, Ollama.

    Parametri:
        provider: 'groq' | 'ollama' | 'mistral' (default: 'ollama')

    Note:
        - Ollama: usa http.client con connessione persistente (keep-alive) per
          ridurre la latenza da ~2.3s a ~0.14s per chiamata warm. Esegue warm_up
          al costruttore per forzare il ricaricamento con num_gpu=999, num_ctx=512
          se il modello risulta in split GPU/CPU.
        - Groq/Mistral: variabile di ambiente con API key deve essere impostata.
        - qwen3: /no_think aggiunto automaticamente per disabilitare il CoT.
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

        # qwen3: /no_think salta il CoT -> risposta diretta e veloce
        self._no_think = "qwen3" in self.model.lower()

        if provider == "ollama":
            self._ollama_base = "http://localhost:11434"
            self._ollama_conn: Optional[http.client.HTTPConnection] = None
            self._ollama_connetti()   # connessione persistente HTTP keep-alive
            self._ollama_warm_up()   # forza ricaricamento con parametri ottimali
        else:
            api_key_env = cfg.get("api_key_env")
            api_key = os.environ.get(api_key_env, "") if api_key_env else "ollama"
            if api_key_env and not api_key:
                raise EnvironmentError(
                    f"Variabile di ambiente '{api_key_env}' non impostata. "
                    f"Impostala con: set {api_key_env}=<tua_chiave>"
                )
            self._openai_client = openai.OpenAI(
                base_url=cfg["base_url"],
                api_key=api_key,
            )

        print(
            "[ClienteLLM] Provider=" + provider + ", model=" + self.model
            + (" [no_think=True]" if self._no_think else "")
        )

    # ------------------------------------------------------------------
    # Ollama: API nativa HTTP (http.client, connessione persistente)
    # ------------------------------------------------------------------

    def _ollama_connetti(self) -> None:
        """Apre/riapre la connessione HTTP persistente verso Ollama (keep-alive).

        La connessione persistente elimina il TCP handshake per ogni chiamata,
        riducendo la latenza da ~2.3s (urllib.request, nuova conn ogni call)
        a ~0.14s (http.client, riuso della stessa connessione TCP).
        """
        try:
            if self._ollama_conn is not None:
                self._ollama_conn.close()
        except Exception:
            pass
        self._ollama_conn = http.client.HTTPConnection("localhost", 11434, timeout=120)

    def _ollama_get(self, endpoint: str, timeout: float = 5.0) -> dict:
        """Invia richiesta GET all'API di Ollama via urllib (per endpoint /api/ps)."""
        req = urllib.request.Request(self._ollama_base + endpoint)
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _ollama_post(self, endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
        """Invia richiesta POST via connessione HTTP persistente (keep-alive).

        Riprova con riconnessione automatica se la connessione e' caduta.
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
                    # Connessione caduta: riconnetti e riprova una volta
                    self._ollama_connetti()
                else:
                    raise

    def _ollama_warm_up(self) -> None:
        """Forza ricaricamento del modello con num_gpu=999, num_ctx=512.

        Verifica via /api/ps se il modello e' in split GPU/CPU (size_vram < size).
        In caso positivo lo scarica (keep_alive=0) e lo ricarica con i parametri
        corretti. Con num_ctx=512 il KV cache e' ridotto (~0.1 GB invece di ~4 GB
        con ctx=4096) e il modello entra interamente in VRAM.
        """
        try:
            d = self._ollama_get("/api/ps", timeout=5.0)
            modelli = d.get("models", [])
            nome_base = self.model.split(":")[0]
            in_split = any(
                m.get("name", "").startswith(nome_base)
                and m.get("size_vram", 0) < m.get("size", 1)
                for m in modelli
            )

            if in_split:
                print(
                    "[ClienteLLM] Modello in split GPU/CPU rilevato: "
                    "scarico e ricarico con num_ctx=512, num_gpu=999..."
                )
                try:
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

            # Carica/mantiene con parametri ottimali (keep_alive=-1 = mai scaricare)
            print("[ClienteLLM] Caricamento modello con num_ctx=512, num_gpu=999...")
            self._ollama_post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "options": {"num_gpu": 999, "num_ctx": 512},
                    "keep_alive": -1,
                    "stream": False,
                },
                timeout=120.0,
            )
            print("[ClienteLLM] Modello pronto in VRAM.")

        except Exception as e:
            print("[ClienteLLM] Warm-up fallito (Ollama non attivo?): " + str(e))

    def _chiedi_ollama(self, prompt: str, max_tokens: int, timeout: float) -> str:
        """Chiama Ollama via connessione HTTP persistente. Stringa vuota su errore."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "num_gpu":     999,        # tutti i layer su GPU
                "num_ctx":     512,        # KV cache ridotto -> modello tutto in VRAM
                "num_predict": max_tokens, # max token da generare
                "temperature": 0,
            },
            "keep_alive": -1,  # mantiene modello in VRAM tra le chiamate
            "stream": False,
        }
        if self._no_think:
            payload["think"] = False  # qwen3: disabilita CoT interno

        for tentativo in range(1, MAX_RETRY_LLM + 1):
            try:
                d = self._ollama_post("/api/chat", payload, timeout=timeout)
                testo = d.get("message", {}).get("content", "")
                return testo.strip()
            except Exception as e:
                if tentativo == MAX_RETRY_LLM:
                    print(
                        "[ClienteLLM] Ollama: fallimento dopo "
                        + str(MAX_RETRY_LLM) + " tentativi: " + type(e).__name__
                    )
                    return ""
                time.sleep(0.5 * tentativo)
                self._ollama_connetti()  # riconnetti prima del prossimo tentativo

        return ""

    # ------------------------------------------------------------------
    # Groq / Mistral: OpenAI SDK
    # ------------------------------------------------------------------

    def _chiedi_openai(self, prompt: str, max_tokens: int, timeout: float) -> str:
        """Chiama Groq/Mistral via OpenAI SDK. Restituisce stringa vuota su errore."""
        for tentativo in range(1, MAX_RETRY_LLM + 1):
            try:
                risposta = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    timeout=timeout,
                )
                testo = risposta.choices[0].message.content or ""
                return testo.strip()
            except Exception as e:
                if tentativo == MAX_RETRY_LLM:
                    print(
                        "[ClienteLLM] Fallimento dopo " + str(MAX_RETRY_LLM)
                        + " tentativi: " + type(e).__name__
                    )
                    return ""
                time.sleep(0.5 * tentativo)

        return ""

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def chiedi(
        self,
        prompt: str,
        max_tokens: int = 20,
        timeout: Optional[float] = None,
    ) -> str:
        """Invia un prompt e restituisce la risposta testuale.

        Restituisce stringa vuota in caso di fallimento definitivo.
        """
        timeout = timeout or TIMEOUT_LLM_SEC
        if self.provider == "ollama":
            return self._chiedi_ollama(prompt, max_tokens, timeout)
        return self._chiedi_openai(prompt, max_tokens, timeout)
