# Pacchetto agents: i tre agenti basati su LLM del progetto.
#
# Gli agenti RL puri non hanno un wrapper qui: AG-PPO (RecurrentPPO) e AG-DQN sono
# configurati e addestrati direttamente negli script src/sistema_10x10/train_ppo.py e
# src/sistema_10x10/train_dqn.py.
#
# Moduli:
#   llm_act_agent     AG-LLM-ACT: il LLM sceglie ogni azione a inference time, senza RL
#   llm_guide_agent   AG-LLM-GUIDE: il LLM raccoglie demo, il DQN impara da esse (LfD)
#   llm_reward_agent  AG-LLM-REW: PPO con reward aumentata dal LLM a ogni spinta di cassa
