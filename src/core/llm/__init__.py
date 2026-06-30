# Pacchetto core.llm: tutta l'integrazione del LLM nell'ambiente Sokoban.
#
# Ogni comunicazione con il modello passa da qui. Il modello è qwen3:14b-q4_K_M servito da
# Ollama in locale (localhost:11434), interrogato via http.client con keep-alive per tenere
# bassa la latenza (~0.14-0.21 s a chiamata, a regime).
#
# Moduli:
#   llm_client      client HTTP per Ollama (keep-alive, think=False, gestione dei timeout)
#   sokoban_prompt  griglia->testo, costruzione dei prompt, parsing risposta->azione/reward

from core.llm.llm_client import ClienteLLM
from core.llm.sokoban_prompt import (
    griglia_a_testo,
    crea_prompt,
    parsifica_azione,
    conta_casse,
    crea_prompt_reward,
    parsifica_reward,
)

__all__ = [
    "ClienteLLM",
    "griglia_a_testo",
    "crea_prompt",
    "parsifica_azione",
    "conta_casse",
    "crea_prompt_reward",
    "parsifica_reward",
]
