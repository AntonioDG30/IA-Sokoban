"""Funzione di reward per l'ambiente Sokoban.

Schema reward di default:
    +10.0  completamento livello (tutte le casse su target)
    +1.0   per ogni cassa appena posizionata su target
    -1.0   per ogni cassa appena rimossa da target (es. spinta fuori)
    -0.005 penalita' per ogni step (v9: ridotta da -0.01 per favorire esplorazione)

Reward shaping aggiuntivo (v10):
    +/- scala_manhattan * (dist_prec - dist_att)
        dove dist = somma distanze Manhattan ottimali casse->target
        calcolata tramite algoritmo Ungherese (assegnamento ottimale O(k^3)).
        Positivo quando ci si avvicina al goal, negativo quando ci si allontana.

    +/- scala_player_box * (dist_pb_prec - dist_pb_att)    [v10 Option A]
        dove dist_pb = distanza Manhattan dal giocatore alla cassa piu' vicina
        non ancora su target. Guida la navigazione verso i box (fase 1 del task).
        Delta-based: sicuro, non hackabile (oscillare = net zero).
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from sokoban_env.game_logic import (
    conta_casse_su_target, CASSA, CASSA_SU_TARGET, TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
)


PENALITA_STEP      = -0.005   # v10: mantenuto da v9 (C4/C5 hanno 300 step, -0.01 penalizzerebbe troppo)
BONUS_COMPLETAMENTO =  10.0   # tutte le casse su target (dominante: max shaping=0 ora)
BONUS_CASSA_SU_TGT  =   1.0   # cassa appena messa su target
MALUS_CASSA_FUORI   =  -1.0   # cassa appena tolta da target
BONUS_PROXIMITY     =   0.0   # v10: disabilitato (reward hacking con shaping per-step)


def calcola_reward(
    griglia_precedente: np.ndarray,
    griglia_nuova: np.ndarray,
    terminato: bool,
    scala_manhattan: float = 0.0,
    adiacente_cassa: bool = False,
    scala_player_box: float = 0.0,
) -> float:
    """Calcola la reward per una transizione di stato.

    Parametri:
        griglia_precedente: stato prima della mossa.
        griglia_nuova:      stato dopo la mossa.
        terminato:          True se il livello e' stato completato.
        scala_manhattan:    fattore di scala per il reward shaping Manhattan casse->target
                            (0.0 = disabilitato, default v10: 0.3).
        adiacente_cassa:    non piu' usato (BONUS_PROXIMITY=0.0). Mantenuto per API compat.
        scala_player_box:   fattore di scala per il reward shaping giocatore->cassa
                            (0.0 = disabilitato, default v10: 0.1). [v10 Option A]

    Restituisce:
        Valore float della reward per questo step.
    """
    casse_prima = conta_casse_su_target(griglia_precedente)
    casse_dopo  = conta_casse_su_target(griglia_nuova)
    delta_casse = casse_dopo - casse_prima

    # Penalita' step base
    reward = PENALITA_STEP

    if terminato:
        reward += BONUS_COMPLETAMENTO

    # +1/-1 per ogni cassa spostata su/da target
    reward += float(delta_casse)

    # Shaping Manhattan casse->target (v9+): incentiva avvicinamento casse ai target.
    if scala_manhattan > 0.0:
        target_pos = _trova_target(griglia_precedente)
        if target_pos:
            dist_prec = _distanza_totale_ottimale(griglia_precedente, target_pos)
            dist_att  = _distanza_totale_ottimale(griglia_nuova, target_pos)
            reward += (dist_prec - dist_att) * scala_manhattan

    # Shaping giocatore->cassa (v10 Option A): incentiva avvicinamento ai box
    # prima che vengano spinti (fase 1 del credit assignment a 2 stadi).
    # Delta-based: sicuro (oscillare = net zero, completare = reward gratis).
    if scala_player_box > 0.0:
        dist_pb_prec = _distanza_giocatore_cassa(griglia_precedente)
        dist_pb_att  = _distanza_giocatore_cassa(griglia_nuova)
        reward += (dist_pb_prec - dist_pb_att) * scala_player_box

    return reward


# ---------------------------------------------------------------------------
# Funzioni di supporto per il reward shaping
# ---------------------------------------------------------------------------

def _trova_target(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """Restituisce le posizioni (riga, col) di tutti i target nella griglia."""
    posizioni = np.argwhere(
        (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _trova_casse(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """Restituisce le posizioni (riga, col) di tutte le casse nella griglia."""
    posizioni = np.argwhere(
        (griglia == CASSA) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _distanza_giocatore_cassa(griglia: np.ndarray) -> float:
    """Distanza Manhattan minima dal giocatore alla cassa non-su-target piu' vicina.

    Restituisce 0.0 se non ci sono casse libere (tutte su target) o se il
    giocatore non e' trovato nella griglia.
    """
    # Trova il giocatore
    pos_g = np.argwhere((griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET))
    if len(pos_g) == 0:
        return 0.0
    riga_g, col_g = int(pos_g[0, 0]), int(pos_g[0, 1])

    # Trova casse NON ancora su target
    casse = np.argwhere(griglia == CASSA)
    if len(casse) == 0:
        return 0.0

    return float(min(abs(int(r) - riga_g) + abs(int(c) - col_g) for r, c in casse))


def _distanza_totale_ottimale(
    griglia: np.ndarray,
    target_positions: List[Tuple[int, int]],
) -> float:
    """Somma delle distanze Manhattan ottimali casse->target via algoritmo Ungherese O(k^3).

    A differenza dell'assegnamento greedy, garantisce l'assegnamento globalmente
    ottimale: nessuna coppia (cassa, target) puo' essere scambiata per ridurre
    la distanza totale. Con k=3 casse il costo computazionale e' trascurabile.
    """
    casse = _trova_casse(griglia)
    if not casse or not target_positions:
        return 0.0
    costo = np.array([
        [abs(cr - tr) + abs(cc - tc) for tr, tc in target_positions]
        for cr, cc in casse
    ], dtype=np.float64)
    righe_opt, col_opt = linear_sum_assignment(costo)
    return float(costo[righe_opt, col_opt].sum())
