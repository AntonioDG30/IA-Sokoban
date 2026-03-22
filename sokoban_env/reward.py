"""Funzione di reward per l'ambiente Sokoban (versione definitiva v10).

Schema reward:
    +10.0  bonus di completamento (tutte le casse su target)
    +1.0   per ogni cassa appena posizionata su target
    -1.0   per ogni cassa appena tolta da target (spinta fuori)
    -0.005 penalita' per ogni step eseguito

Reward shaping aggiuntivo:
    Manhattan casse->target (scala_manhattan=0.3):
        +/- scala_manhattan * (dist_prec - dist_att)
        dist = somma distanze Manhattan ottimali, calcolata via algoritmo
        Ungherese (assegnamento globalmente ottimale, non greedy).
        Positivo quando le casse si avvicinano ai target.

    Giocatore->cassa (scala_player_box=0.1):
        +/- scala_player_box * (dist_pb_prec - dist_pb_att)
        dist_pb = distanza Manhattan dal giocatore alla cassa libera piu' vicina.
        Guida il giocatore verso le casse prima di spingerle.
        Delta-based: oscillare da net=0, non hackabile.
"""

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from sokoban_env.game_logic import (
    conta_casse_su_target, CASSA, CASSA_SU_TARGET, TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
)


# Costanti della reward: valori definitivi dopo i test sul curriculum
PENALITA_STEP      = -0.005   # penalita' per step: ridotta da -0.01 per non punire
                               # le soluzioni lunghe nelle fasi C4/C5 (max 300 step)
BONUS_COMPLETAMENTO =  10.0   # segnale dominante: garantisce che completare sia
                               # sempre preferibile a raccogliere shaping senza finire
BONUS_CASSA_SU_TGT  =   1.0   # rinforzo immediato per ogni cassa posizionata
MALUS_CASSA_FUORI   =  -1.0   # penalita' per aver spostato una cassa fuori dal target
BONUS_PROXIMITY     =   0.0   # disabilitato in v10: causava reward hacking


def calcola_reward(
    griglia_precedente: np.ndarray,
    griglia_nuova: np.ndarray,
    terminato: bool,
    scala_manhattan: float = 0.0,
    adiacente_cassa: bool = False,
    scala_player_box: float = 0.0,
) -> float:
    """Calcola la reward per una singola transizione di stato.

    Combina la reward base (step, completamento, casse) con il reward shaping
    Manhattan e giocatore->cassa se i rispettivi fattori di scala sono > 0.

    Parametri:
        griglia_precedente: stato della griglia prima della mossa.
        griglia_nuova:      stato della griglia dopo la mossa.
        terminato:          True se il livello e' stato completato (tutte le casse su target).
        scala_manhattan:    fattore di scala per lo shaping Manhattan (0.0 = disabilitato).
        adiacente_cassa:    parametro non piu' usato (BONUS_PROXIMITY=0); mantenuto
                            per compatibilita' con le chiamate esistenti.
        scala_player_box:   fattore di scala per lo shaping giocatore->cassa (0.0 = disabilitato).

    Restituisce:
        Valore float della reward per questo step.
    """
    casse_prima = conta_casse_su_target(griglia_precedente)
    casse_dopo  = conta_casse_su_target(griglia_nuova)
    delta_casse = casse_dopo - casse_prima

    # Penalita' per ogni step eseguito (scoraggia soluzioni inutilmente lunghe)
    reward = PENALITA_STEP

    if terminato:
        reward += BONUS_COMPLETAMENTO

    # +1 per ogni cassa appena messa su target, -1 per ogni cassa spostata via
    reward += float(delta_casse)

    # Shaping Manhattan: incentiva ad avvicinare le casse ai target
    if scala_manhattan > 0.0:
        target_pos = _trova_target(griglia_precedente)
        if target_pos:
            dist_prec = _distanza_totale_ottimale(griglia_precedente, target_pos)
            dist_att  = _distanza_totale_ottimale(griglia_nuova, target_pos)
            reward += (dist_prec - dist_att) * scala_manhattan

    # Shaping giocatore->cassa: incentiva il giocatore ad avvicinarsi ai box
    if scala_player_box > 0.0:
        dist_pb_prec = _distanza_giocatore_cassa(griglia_precedente)
        dist_pb_att  = _distanza_giocatore_cassa(griglia_nuova)
        reward += (dist_pb_prec - dist_pb_att) * scala_player_box

    return reward


# ---------------------------------------------------------------------------
# Funzioni di supporto per il reward shaping
# ---------------------------------------------------------------------------

def _trova_target(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """Restituisce le posizioni (riga, col) di tutti i target nella griglia.

    Include sia i target liberi (TARGET) sia quelli occupati da casse
    (CASSA_SU_TARGET), perche' quelli sono comunque target del livello.
    """
    posizioni = np.argwhere(
        (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _trova_casse(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """Restituisce le posizioni (riga, col) di tutte le casse nella griglia.

    Include sia le casse libere (CASSA) sia quelle gia' su target (CASSA_SU_TARGET).
    """
    posizioni = np.argwhere(
        (griglia == CASSA) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _distanza_giocatore_cassa(griglia: np.ndarray) -> float:
    """Calcola la distanza Manhattan minima dal giocatore alla cassa libera piu' vicina.

    Considera solo le casse NON ancora su target: quando tutte le casse sono
    a posto il livello e' gia' risolto e questa distanza non serve piu'.
    Restituisce 0.0 se non ci sono casse libere o se il giocatore non e' trovato.

    Parametri:
        griglia: matrice NumPy di forma (H, W).
    """
    pos_g = np.argwhere((griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET))
    if len(pos_g) == 0:
        return 0.0
    riga_g, col_g = int(pos_g[0, 0]), int(pos_g[0, 1])

    # Solo le casse non ancora posizionate sui target
    casse = np.argwhere(griglia == CASSA)
    if len(casse) == 0:
        return 0.0

    # Distanza Manhattan dalla posizione del giocatore a ciascuna cassa
    return float(min(abs(int(r) - riga_g) + abs(int(c) - col_g) for r, c in casse))


def _distanza_totale_ottimale(
    griglia: np.ndarray,
    target_positions: List[Tuple[int, int]],
) -> float:
    """Somma delle distanze Manhattan ottimali casse->target tramite algoritmo Ungherese.

    A differenza di un assegnamento greedy (abbina la cassa al target piu' vicino),
    l'algoritmo Ungherese O(k^3) trova l'assegnamento globalmente ottimale: nessuno
    scambio tra due coppie (cassa, target) puo' ridurre la distanza totale. Con k=4
    casse il costo computazionale e' trascurabile (~microsecondo).

    Parametri:
        griglia:          matrice NumPy di forma (H, W).
        target_positions: lista di coordinate (riga, col) dei target nel livello.
    """
    casse = _trova_casse(griglia)
    if not casse or not target_positions:
        return 0.0

    # Matrice dei costi: costo[i][j] = distanza Manhattan tra cassa i e target j
    costo = np.array([
        [abs(cr - tr) + abs(cc - tc) for tr, tc in target_positions]
        for cr, cc in casse
    ], dtype=np.float64)

    # Algoritmo Ungherese: trova l'assegnamento a costo minimo
    righe_opt, col_opt = linear_sum_assignment(costo)
    return float(costo[righe_opt, col_opt].sum())
