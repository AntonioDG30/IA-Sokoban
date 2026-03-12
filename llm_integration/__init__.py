"""Pacchetto llm_integration -- integrazione LLM nell'ambiente Sokoban.

Moduli:
    llm_client       client unificato per Groq, Mistral, Ollama
    sokoban_prompt   conversione griglia->testo, parsing risposta->azione
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
