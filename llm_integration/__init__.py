"""Pacchetto llm_integration — integrazione LLM nell'ambiente Sokoban.

Moduli:
    llm_client         client unificato per Groq, Mistral, Ollama
    state_formatter    conversione griglia NumPy → testo per LLM
    action_parser      parsing risposta LLM → azione (int)
    reward_parser      parsing risposta LLM → reward (float)
    prompt_templates   template prompt per AG-LLM-ACT e AG-LLM-REW
"""
