"""Pacchetto agents -- implementazioni degli agenti RL e LLM.

Moduli:
    ppo_agent           wrapper training/valutazione AG-PPO (RecurrentPPO via SB3)
    dqn_agent           wrapper training/valutazione AG-DQN (DQN via SB3)
    llm_act_agent       agente AG-LLM-ACT: il LLM sceglie ogni azione a inference time
    llm_guide_agent     agente AG-LLM-GUIDE: LLM raccoglie demo, DQN impara via LfD
    llm_reward_agent    agente AG-LLM-REW: PPO con reward aumentata da LLM ad ogni push
"""
