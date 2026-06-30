# Curriculum semplificato 7x7: modulo autonomo, separato dal progetto principale 10x10.
#
# Script in src/sistema_7x7/:
#   config_7x7.py           iperparametri, percorsi e factory dell'ambiente 7x7
#   sokoban_gym_7x7.py      ambiente Sokoban 7x7 con observation space nativo (7,7)
#   train_ppo_7x7.py        training AG-PPO sul curriculum C0->C2
#   train_dqn_7x7.py        training AG-DQN sul curriculum C0->C2
#   train_llm_act_7x7.py    valutazione AG-LLM-ACT (nessun training, solo inference)
#   train_llm_guide_7x7.py  training AG-LLM-GUIDE via LfD
#   train_llm_rew_7x7.py    training AG-LLM-REW (PPO con reward aumentata dal LLM)
#   evaluate_7x7.py         valutazione comparativa di tutti e 5 gli agenti
#
# Modelli in artifacts/models/7x7/, log in artifacts/logs/7x7/ (cartelle separate dal progetto 10x10).
