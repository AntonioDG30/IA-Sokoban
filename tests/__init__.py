# Suite di test del progetto IA-Sokoban.
#
# 64 test totali in tre moduli:
#   test_game_logic.py       18 test — logica di gioco pura (applica_mossa, vittoria, conteggi)
#   test_env.py              18 test — ambiente Gymnasium (reset, step, spazi, ciclo episodico)
#   test_llm_integration.py  28 test — funzioni LLM (griglia_a_testo, prompt, parsifica_azione,
#                            conta_casse) e ClienteLLM, con skip automatico se Ollama è offline
#
# Esegui con: pytest tests/
