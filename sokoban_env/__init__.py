"""Pacchetto sokoban_env -- ambiente Sokoban custom per Gymnasium.

Observation space: Box(0.0, 7.0, shape=(10,10), dtype=float32)
Action space:      Discrete(4) -- 0=su, 1=giu, 2=sinistra, 3=destra
Valore 7 riservato al padding quando la griglia e' piu' piccola di 10x10.

Esporta:
    SokobanEnv        classe principale dell'ambiente (sokoban_gym.py)
    CaricatoreLivelli caricamento livelli Boxoban da file .txt
    calcola_reward    reward shaping con Manhattan distance e player-box distance
    game_logic        costanti (MURO, PAVIMENTO, ...) e applica_mossa/controlla_vittoria
    AggiuntaCanale    wrapper (H,W) -> (1,H,W) richiesto da SokobanCNN (channels-first)
    SokobanCNN        estrattore CNN custom per CnnPolicy/CnnLstmPolicy di SB3
"""

from sokoban_env.sokoban_gym import SokobanEnv, MAX_STEP_PER_EPISODIO
from sokoban_env.level_loader import CaricatoreLivelli
from sokoban_env.reward import calcola_reward
from sokoban_env import game_logic
from sokoban_env.cnn_wrapper import AggiuntaCanale
from sokoban_env.sokoban_cnn import SokobanCNN

__all__ = [
    "SokobanEnv",
    "MAX_STEP_PER_EPISODIO",
    "CaricatoreLivelli",
    "calcola_reward",
    "game_logic",
    "AggiuntaCanale",
    "SokobanCNN",
]
