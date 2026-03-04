"""Logica di gioco pura per Sokoban.

Funzioni stateless per la manipolazione della griglia. Non dipende da Gymnasium
né da Pygame — può essere importato e testato in isolamento.

Codifica celle:
    0 = MURO        (#)
    1 = PAVIMENTO   ( )
    2 = TARGET      (.)
    3 = CASSA       ($)
    4 = CASSA_SU_TARGET  (*)
    5 = GIOCATORE   (@)
    6 = GIOCATORE_SU_TARGET  (+)
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Costanti — codifica celle
# ---------------------------------------------------------------------------

MURO = 0
PAVIMENTO = 1
TARGET = 2
CASSA = 3
CASSA_SU_TARGET = 4
GIOCATORE = 5
GIOCATORE_SU_TARGET = 6

# Mapping azione (intero) → delta (riga, colonna)
DELTA_AZIONE: dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # su
    1: ( 1,  0),   # giù
    2: ( 0, -1),   # sinistra
    3: ( 0,  1),   # destra
}

NOMI_AZIONI: dict[int, str] = {
    0: "su",
    1: "giù",
    2: "sinistra",
    3: "destra",
}

# Celle che rappresentano superfici "calpestabili" (senza oggetti)
CELLE_LIBERE = (PAVIMENTO, TARGET)

# Celle che rappresentano casse
CELLE_CASSA = (CASSA, CASSA_SU_TARGET)


# ---------------------------------------------------------------------------
# Funzioni di ispezione
# ---------------------------------------------------------------------------

def trova_giocatore(griglia: np.ndarray) -> Tuple[int, int]:
    """Restituisce la posizione (riga, colonna) del giocatore nella griglia.

    Solleva ValueError se il giocatore non è presente.
    """
    posizioni = np.argwhere(
        (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
    )
    if len(posizioni) == 0:
        raise ValueError("Giocatore non trovato nella griglia.")
    return int(posizioni[0, 0]), int(posizioni[0, 1])


def conta_casse_su_target(griglia: np.ndarray) -> int:
    """Restituisce il numero di casse attualmente posizionate sui target."""
    return int(np.sum(griglia == CASSA_SU_TARGET))


def conta_casse_totali(griglia: np.ndarray) -> int:
    """Restituisce il numero totale di casse (su target e non)."""
    return int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))


def controlla_vittoria(griglia: np.ndarray) -> bool:
    """Restituisce True se tutte le casse sono posizionate sui target.

    Condizione: nessuna cassa libera (CASSA) rimane nella griglia
    e almeno una cassa è su target.
    """
    return (
        int(np.sum(griglia == CASSA)) == 0
        and conta_casse_su_target(griglia) > 0
    )


# ---------------------------------------------------------------------------
# Funzione principale — applica mossa
# ---------------------------------------------------------------------------

def applica_mossa(
    griglia: np.ndarray, azione: int
) -> Tuple[np.ndarray, bool, bool]:
    """Applica una mossa alla griglia e restituisce il nuovo stato.

    Parametri:
        griglia: matrice NumPy di forma (10, 10), dtype int8.
        azione: intero in {0,1,2,3} — su, giù, sinistra, destra.

    Restituisce una tupla (nuova_griglia, mossa_eseguita, cassa_spostata):
        nuova_griglia:   stato aggiornato (copia della griglia originale se
                         la mossa non è valida).
        mossa_eseguita:  True se il giocatore si è effettivamente spostato.
        cassa_spostata:  True se durante la mossa è stata spinta una cassa.

    Solleva ValueError se l'azione non è in {0,1,2,3}.
    """
    if azione not in DELTA_AZIONE:
        raise ValueError(
            f"Azione non valida: {azione}. Deve essere uno tra {{0,1,2,3}}."
        )

    nuova_griglia = griglia.copy()
    dr, dc = DELTA_AZIONE[azione]

    riga_g, col_g = trova_giocatore(griglia)
    riga_dest = riga_g + dr
    col_dest = col_g + dc

    n_righe, n_col = griglia.shape

    # Fuori dai bordi → mossa impossibile
    if not (0 <= riga_dest < n_righe and 0 <= col_dest < n_col):
        return nuova_griglia, False, False

    cella_dest = int(griglia[riga_dest, col_dest])

    # Destinazione è un muro → mossa impossibile
    if cella_dest == MURO:
        return nuova_griglia, False, False

    cassa_spostata = False

    # Destinazione contiene una cassa → prova a spingerla
    if cella_dest in CELLE_CASSA:
        riga_cassa_dest = riga_dest + dr
        col_cassa_dest = col_dest + dc

        # La cassa uscirebbe dai bordi → mossa impossibile
        if not (0 <= riga_cassa_dest < n_righe and 0 <= col_cassa_dest < n_col):
            return nuova_griglia, False, False

        cella_dopo_cassa = int(griglia[riga_cassa_dest, col_cassa_dest])

        # La cassa può essere spostata solo su pavimento o target liberi
        if cella_dopo_cassa not in CELLE_LIBERE:
            return nuova_griglia, False, False

        # Sposta la cassa nella sua nuova posizione
        nuova_griglia[riga_cassa_dest, col_cassa_dest] = (
            CASSA_SU_TARGET if cella_dopo_cassa == TARGET else CASSA
        )
        cassa_spostata = True

    # Aggiorna la cella originale del giocatore (poteva essere su un target)
    nuova_griglia[riga_g, col_g] = (
        TARGET if griglia[riga_g, col_g] == GIOCATORE_SU_TARGET else PAVIMENTO
    )

    # Posiziona il giocatore nella cella di destinazione
    if cassa_spostata:
        # Se c'era CASSA_SU_TARGET, sotto c'era un target → giocatore è su target
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == CASSA_SU_TARGET else GIOCATORE
        )
    else:
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == TARGET else GIOCATORE
        )

    return nuova_griglia, True, cassa_spostata
