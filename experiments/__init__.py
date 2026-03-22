"""Pacchetto experiments -- script per training e valutazione degli agenti.

Script presenti nella directory experiments/:
    config.py                       configurazione centralizzata (iperparametri, percorsi, seed)
    train_ppo.py                    training AG-PPO con curriculum v8 (10x10)
    train_dqn.py                    training AG-DQN con curriculum (10x10)
    train_llm_act.py                valutazione AG-LLM-ACT a inference time
    train_llm_guide.py              training AG-LLM-GUIDE via LfD (10x10)
    train_ppo_llm_rew.py            training AG-LLM-REW (PPO + reward LLM)
    evaluate_all.py                 valutazione comparativa tutti gli agenti

Sottomoduli:
    simplified/                     curriculum 7x7 generato (nessun Boxoban)
    curriculum_6x6/                 curriculum 6x6 generato (nessun Boxoban)
"""
