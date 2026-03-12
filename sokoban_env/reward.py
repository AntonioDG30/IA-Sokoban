"""Funzione di reward per l'ambiente Sokoban.

Schema reward di default:
    +10.0  completamento livello (tutte le casse su target)
    +1.0   per ogni cassa appena posizionata su target
    -1.0   per ogni cassa appena rimossa da target (es. spinta fuori)
    -0.005 penalita' per ogni step (v9: ridotta da -0.01 per favorire esplorazione)

Reward shaping aggiuntivo (v9):
    +/- scala_manhattan * (dist_prec - dist_att)
        dove dist = somma distanze Manhattan ottimali casse->target
        calcolata tramite algoritmo Ungherese (assegnamento ottimale O(k^3)).
        Positivo quando ci si avvicina al goal, negativo quando ci si allontana.

    +0.05  proximity bonus: giocatore adiacente a una cassa prima della mossa.
        Guida il navigazione verso le casse prima che vengano spinte.
        Basato su info['giocatore_adiacente_cassa'] calcolato in SokobanEnv.step().
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from sokoban_env.game_logic import conta_casse_su_target, CASSA, CASSA_SU_TARGET, TARGET


PENALITA_STEP      = -0.005   # v9: ridotta da -0.01 per favorire esplorazione
BONUS_COMPLETAMENTO =  10.0   # tutte le casse su target
BONUS_CASSA_SU_TGT  =   1.0   # cassa appena messa su target
MALUS_CASSA_FUORI   =  -1.0   # cassa appena tolta da target
BONUS_PROXIMITY     =   0.05  # giocatore adiacente a una cassa (v9)


def calcola_reward(
    griglia_precedente: np.ndarray,
    griglia_nuova: np.ndarray,
    terminato: bool,
    scala_manhattan: float = 0.0,
    adiacente_cassa: bool = False,
) -> float:
    """Calcola la reward per una transizione di stato.

    Parametri:
        griglia_precedente: stato prima della mossa.
        griglia_nuova:      stato dopo la mossa.
        terminato:          True se il livello e' stato completato.
        scala_manhattan:    fattore di scala per il reward shaping Manhattan
                            (0.0 = disabilitato, valore tipico: 2.0 in v9).
        adiacente_cassa:    True se il giocatore era adiacente a una cassa
                            prima della mossa (da SokobanEnv._giocatore_adiacente_cassa).
                            Abilita il proximity bonus +0.05.

    Restituisce:
        Valore float della reward per questo step.
    """
    casse_prima = conta_casse_su_target(griglia_precedente)
    casse_dopo  = conta_casse_su_target(griglia_nuova)
    delta_casse = casse_dopo - casse_prima

    # Penalita' step base (v9: -0.005, ridotta da -0.01 per non scoraggiare
    # soluzioni che richiedono molte mosse di navigazione verso le casse)
    reward = PENALITA_STEP

    if terminato:
        reward += BONUS_COMPLETAMENTO

    # +1/-1 per ogni cassa spostata su/da target
    reward += float(delta_casse)

    # Proximity bonus (v9): guida il navigazione verso le casse.
    # Si applica quando il giocatore e' adiacente a una cassa prima della mossa,
    # creando un gradiente che incentiva l'avvicinamento ai box prima di spingerli.
    if adiacente_cassa:
        reward += BONUS_PROXIMITY

    # Reward shaping basata su distanza Manhattan ottimale (Fase 2.2 + v9 scala=2.0)
    # Penalizza quando le casse si allontanano dai target, premia quando si avvicinano.
    if scala_manhattan > 0.0:
        target_pos = _trova_target(griglia_precedente)
        if target_pos:
            dist_prec = _distanza_totale_ottimale(griglia_precedente, target_pos)
            dist_att  = _distanza_totale_ottimale(griglia_nuova, target_pos)
            reward += (dist_prec - dist_att) * scala_manhattan

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
