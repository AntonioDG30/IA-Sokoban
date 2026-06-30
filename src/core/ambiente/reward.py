# Funzione di reward dell'ambiente Sokoban (versione definitiva v10).
#
# Reward di base, sommata a ogni step:
#   +10.0   bonus di completamento (tutte le casse sui bersagli)
#   +1.0    per ogni cassa appena messa a posto
#   -1.0    per ogni cassa appena tolta da un bersaglio (spinta fuori)
#   -0.005  penalità fissa di step (scoraggia le soluzioni inutilmente lunghe)
#
# Reward shaping aggiuntivo (gradiente denso che guida l'agente verso la soluzione):
#   - Manhattan casse->target (scala 0.3): premia l'avvicinamento delle casse ai
#     bersagli. La distanza è la somma degli accoppiamenti ottimali cassa-target,
#     calcolata con l'algoritmo Ungherese (assegnamento globale, non greedy).
#   - Giocatore->cassa (scala 0.1): premia l'avvicinamento del giocatore alla cassa
#     libera più vicina, così l'agente impara prima a raggiungere le casse e poi a spingerle.
# Entrambe le componenti sono delta-based (differenza tra due step consecutivi):
# oscillando avanti e indietro il guadagno netto è zero, quindi non sono "hackabili".

import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment   # algoritmo Ungherese per l'assegnamento ottimale
from core.ambiente.game_logic import (
    conta_casse_su_target, CASSA, CASSA_SU_TARGET, TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
)


# COSTANTI DELLA REWARD (valori definitivi dopo i test sul curriculum)
PENALITA_STEP       = -0.005   # penalità di step: ridotta da -0.01 per non punire le
                               #   soluzioni lunghe delle fasi C4/C5 (fino a 300 step)
BONUS_COMPLETAMENTO =  10.0    # segnale dominante: completare deve valere sempre più
                               #   che accumulare shaping senza chiudere il livello
BONUS_CASSA_SU_TGT  =   1.0    # rinforzo immediato per ogni cassa appena posizionata
MALUS_CASSA_FUORI   =  -1.0    # penalità per aver spinto una cassa fuori da un bersaglio
BONUS_PROXIMITY     =   0.0    # disabilitato in v10: causava reward hacking (giri a vuoto)


def calcola_reward(
    griglia_precedente: np.ndarray,
    griglia_nuova: np.ndarray,
    terminato: bool,
    scala_manhattan: float = 0.0,
    adiacente_cassa: bool = False,
    scala_player_box: float = 0.0,
) -> float:
    """
    Calcola la reward di una singola transizione (griglia_precedente -> griglia_nuova).

    Somma in ordine cinque componenti:
      1. La penalità fissa di step.
      2. Il bonus di completamento, se il livello è terminato.
      3. +1/-1 per ogni cassa entrata/uscita da un bersaglio in questo step.
      4. Lo shaping Manhattan casse->target, se scala_manhattan > 0.
      5. Lo shaping giocatore->cassa, se scala_player_box > 0.

    Il parametro adiacente_cassa non è più usato (BONUS_PROXIMITY=0): resta nella
    firma solo per compatibilità con i chiamanti esistenti. Restituisce la reward (float).
    """
    casse_prima = conta_casse_su_target(griglia_precedente)
    casse_dopo  = conta_casse_su_target(griglia_nuova)
    delta_casse = casse_dopo - casse_prima      # +1 cassa appena messa, -1 cassa appena tolta

    # 1. Penalità fissa: ogni step "costa", così l'agente preferisce soluzioni brevi
    reward = PENALITA_STEP

    # 2. Bonus dominante di fine livello
    if terminato:
        reward += BONUS_COMPLETAMENTO

    # 3. Rinforzo immediato sulle casse: BONUS_CASSA_SU_TGT per ognuna appena messa,
    #    MALUS_CASSA_FUORI per ognuna appena tolta da un bersaglio
    casse_messe = max(delta_casse, 0)        # casse entrate su un target in questo step
    casse_tolte = max(-delta_casse, 0)       # casse uscite da un target in questo step
    reward += casse_messe * BONUS_CASSA_SU_TGT + casse_tolte * MALUS_CASSA_FUORI

    # 4. Shaping Manhattan: premia (valore positivo) l'avvicinamento globale casse->target
    if scala_manhattan > 0.0:
        target_pos = _trova_target(griglia_precedente)
        if target_pos:
            dist_prec = _distanza_totale_ottimale(griglia_precedente, target_pos)  # prima della mossa
            dist_att  = _distanza_totale_ottimale(griglia_nuova, target_pos)       # dopo la mossa
            reward += (dist_prec - dist_att) * scala_manhattan   # >0 se la distanza è diminuita

    # 5. Shaping giocatore->cassa: premia l'avvicinamento alla cassa libera più vicina
    if scala_player_box > 0.0:
        dist_pb_prec = _distanza_giocatore_cassa(griglia_precedente)
        dist_pb_att  = _distanza_giocatore_cassa(griglia_nuova)
        reward += (dist_pb_prec - dist_pb_att) * scala_player_box

    return reward


# FUNZIONI DI SUPPORTO PER IL REWARD SHAPING

def _trova_target(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """
    Restituisce le coordinate (riga, col) di tutti i bersagli del livello.

    Include sia i target liberi (TARGET) sia quelli già coperti da una cassa
    (CASSA_SU_TARGET): anche questi ultimi restano bersagli a tutti gli effetti.
    """
    posizioni = np.argwhere(
        (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _trova_casse(griglia: np.ndarray) -> List[Tuple[int, int]]:
    """
    Restituisce le coordinate (riga, col) di tutte le casse del livello, sia quelle
    libere (CASSA) sia quelle già posizionate su un bersaglio (CASSA_SU_TARGET).
    """
    posizioni = np.argwhere(
        (griglia == CASSA) | (griglia == CASSA_SU_TARGET)
    )
    return [(int(r), int(c)) for r, c in posizioni]


def _distanza_giocatore_cassa(griglia: np.ndarray) -> float:
    """
    Distanza Manhattan dal giocatore alla cassa libera più vicina.

    Considera solo le casse NON ancora a posto: quando tutte le casse sono sui
    bersagli il livello è già risolto e questa distanza non serve più. Restituisce
    0.0 se non ci sono casse libere oppure se il giocatore non viene trovato.
    """
    pos_g = np.argwhere((griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET))
    if len(pos_g) == 0:
        return 0.0
    riga_g, col_g = int(pos_g[0, 0]), int(pos_g[0, 1])

    # Solo le casse ancora libere: quelle già a posto non vanno più raggiunte
    casse = np.argwhere(griglia == CASSA)
    if len(casse) == 0:
        return 0.0

    # Minima distanza Manhattan dal giocatore tra tutte le casse libere
    return float(min(abs(int(r) - riga_g) + abs(int(c) - col_g) for r, c in casse))


def _distanza_totale_ottimale(
    griglia: np.ndarray,
    target_positions: List[Tuple[int, int]],
) -> float:
    """
    Somma minima delle distanze Manhattan casse->target via algoritmo Ungherese.

    A differenza di un abbinamento greedy (ogni cassa al target più vicino), l'algoritmo
    Ungherese O(k^3) trova l'assegnamento globalmente ottimale: nessuno scambio tra due
    coppie (cassa, target) può ridurre ulteriormente la distanza totale. Con k=4 casse il
    costo è trascurabile (~microsecondi). Restituisce 0.0 se mancano le casse o i target.
    """
    casse = _trova_casse(griglia)
    if not casse or not target_positions:
        return 0.0

    # Matrice dei costi: costo[i][j] = distanza Manhattan tra la cassa i e il target j
    costo = np.array([
        [abs(cr - tr) + abs(cc - tc) for tr, tc in target_positions]
        for cr, cc in casse
    ], dtype=np.float64)

    # linear_sum_assignment risolve l'assegnamento a costo minimo (righe=casse, colonne=target)
    righe_opt, col_opt = linear_sum_assignment(costo)
    return float(costo[righe_opt, col_opt].sum())   # somma dei costi delle coppie scelte
