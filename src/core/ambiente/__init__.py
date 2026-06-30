# Pacchetto core.ambiente: l'ambiente Sokoban custom compatibile con Gymnasium.
#
# Observation space: Box(0.0, 7.0, shape=(10,10), dtype=float32)
# Action space:      Discrete(4) -- 0=su, 1=giù, 2=sinistra, 3=destra
# Il valore 7 è riservato al padding quando la griglia è più piccola di 10x10.
#
# Cosa esporta il pacchetto:
#   SokobanEnv         classe principale dell'ambiente (sokoban_gym.py)
#   CaricatoreLivelli  caricamento dei livelli Boxoban da file .txt
#   calcola_reward     reward shaping con distanza Manhattan e distanza giocatore-cassa
#   game_logic         costanti (MURO, PAVIMENTO, ...) e applica_mossa / controlla_vittoria
#   AggiuntaCanale     wrapper (H,W) -> (1,H,W) richiesto da SokobanCNN (channels-first)
#   SokobanCNN         estrattore CNN custom per le policy CnnPolicy / CnnLstmPolicy di SB3

from core.ambiente.sokoban_gym import SokobanEnv, MAX_STEP_PER_EPISODIO
from core.ambiente.level_loader import CaricatoreLivelli
from core.ambiente.reward import calcola_reward
from core.ambiente import game_logic
from core.ambiente.cnn_wrapper import AggiuntaCanale
from core.ambiente.sokoban_cnn import SokobanCNN

__all__ = [
    "SokobanEnv",
    "MAX_STEP_PER_EPISODIO",
    "CaricatoreLivelli",
    "calcola_reward",
    "game_logic",
    "AggiuntaCanale",
    "SokobanCNN",
]
