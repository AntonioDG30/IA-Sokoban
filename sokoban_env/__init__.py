"""Pacchetto sokoban_env — ambiente Sokoban custom per Gymnasium.

Esporta:
    SokobanEnv        classe principale dell'ambiente
    CaricatoreLivelli caricamento livelli Boxoban
    calcola_reward    funzione reward di default
    game_logic        costanti e funzioni di logica di gioco
    AggiuntaCanale    wrapper che aggiunge dim canale: (H,W) → (1,H,W)
    SokobanCNN        estrattore CNN per CnnPolicy SB3
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
