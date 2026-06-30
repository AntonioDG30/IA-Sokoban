# Pacchetto experiments: gli script di training e valutazione degli agenti.
#
# Script nella cartella src/sistema_10x10/:
#   config.py             configurazione centralizzata (iperparametri, percorsi, seed)
#   train_ppo.py          training AG-PPO con curriculum v9 (10x10)
#   train_dqn.py          training AG-DQN con curriculum v9 (10x10)
#   train_llm_act.py      valutazione AG-LLM-ACT a inference time (nessun training)
#   train_llm_guide.py    training AG-LLM-GUIDE via LfD (10x10)
#   train_ppo_llm_rew.py  training AG-LLM-REW (PPO con reward aumentata dal LLM)
#   evaluate_all.py       valutazione comparativa di tutti gli agenti
#
# Sottomodulo:
#   simplified/           sistema 7x7 autonomo, solo livelli generati (nessun Boxoban)
