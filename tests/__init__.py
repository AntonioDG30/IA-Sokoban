"""Suite di test per il progetto IA-Sokoban.

Struttura: 64 test totali suddivisi in tre moduli.
    test_game_logic.py    18 test  -- logica di gioco pura (applica_mossa, reward, level_generator)
    test_env.py           18 test  -- ambiente Gymnasium (reset, step, spazi, ciclo episodico)
    test_llm_integration.py 28 test -- funzioni LLM (griglia_a_testo, prompt, parsifica_azione,
                                        conta_casse, ClienteLLM con skip automatico se Ollama offline)

Esegui con: pytest tests/
"""
