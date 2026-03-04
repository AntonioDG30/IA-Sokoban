"""Funzione di reward di default per l'ambiente Sokoban.

Schema reward:
    +10.0  completamento livello (tutte le casse su target)
    +1.0   per ogni cassa appena posizionata su target
    -1.0   per ogni cassa appena rimossa da target (es. spinta fuori)
    -0.1   penalità per ogni step (incentiva soluzioni brevi)
"""

import numpy as np
from sokoban_env.game_logic import conta_casse_su_target


def calcola_reward(
    griglia_precedente: np.ndarray,
    griglia_nuova: np.ndarray,
    terminato: bool,
) -> float:
    """Calcola la reward per una transizione di stato.

    Parametri:
        griglia_precedente: stato prima della mossa.
        griglia_nuova:      stato dopo la mossa.
        terminato:          True se il livello è stato completato.

    Restituisce:
        Valore float della reward per questo step.
    """
    casse_prima = conta_casse_su_target(griglia_precedente)
    casse_dopo = conta_casse_su_target(griglia_nuova)
    delta_casse = casse_dopo - casse_prima

    reward = -0.1  # penalità step

    if terminato:
        reward += 10.0  # bonus completamento livello

    # +1 per ogni cassa messa su target, -1 per ogni cassa tolta
    reward += float(delta_casse)

    return reward
