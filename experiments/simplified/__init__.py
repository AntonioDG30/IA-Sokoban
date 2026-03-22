"""Curriculum semplificato 7x7 -- modulo autonomo, separato dal progetto principale 10x10.

Script presenti in experiments/simplified/:
    config_7x7.py         iperparametri, percorsi e factory ambiente 7x7
    sokoban_gym_7x7.py    ambiente Sokoban 7x7 con observation space nativo (7,7)
    train_ppo_7x7.py      training AG-PPO su curriculum C0->C2
    train_dqn_7x7.py      training AG-DQN su curriculum C0->C2
    train_llm_act_7x7.py  valutazione AG-LLM-ACT (nessun training, solo inference)
    train_llm_guide_7x7.py training AG-LLM-GUIDE via LfD
    train_llm_rew_7x7.py  training AG-LLM-REW (PPO + reward LLM)
    evaluate_7x7.py       valutazione comparativa tutti e 5 gli agenti

Modelli salvati in models_7x7/, log in logs_7x7/ (separati dal progetto 10x10).
"""
