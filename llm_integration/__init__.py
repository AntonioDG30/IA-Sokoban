"""Pacchetto llm_integration -- integrazione LLM nell'ambiente Sokoban.

Tutta la comunicazione con il LLM passa attraverso questo pacchetto.
Il modello usato e' qwen3:14b-q4_K_M via Ollama locale (localhost:11434).
Il client usa http.client con keep-alive per minimizzare la latenza (~0.14-0.21s/call).

Moduli:
    llm_client       client HTTP per Ollama (keep-alive, think=False, gestione timeout)
    sokoban_prompt   conversione griglia->testo, costruzione prompt, parsing risposta->azione
"""

from llm_integration.llm_client import ClienteLLM
from llm_integration.sokoban_prompt import (
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
