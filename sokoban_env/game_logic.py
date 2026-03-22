"""Logica di gioco Sokoban: costanti, funzioni di movimento e verifica vittoria.

Nessuna dipendenza esterna (no Gymnasium, no Pygame): si puo' testare
direttamente con un array NumPy senza dover istanziare l'ambiente.

Codifica numerica delle celle:
    0 = MURO            (#)
    1 = PAVIMENTO       ( )
    2 = TARGET          (.)
    3 = CASSA           ($)
    4 = CASSA_SU_TARGET (*)
    5 = GIOCATORE       (@)
    6 = GIOCATORE_SU_TARGET (+)
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Costanti: codifica numerica delle celle
# ---------------------------------------------------------------------------

MURO                 = 0
PAVIMENTO            = 1
TARGET               = 2
CASSA                = 3
CASSA_SU_TARGET      = 4
GIOCATORE            = 5
GIOCATORE_SU_TARGET  = 6

# Delta (riga, colonna) per ciascuna delle 4 azioni discrete
DELTA_AZIONE: dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # su
    1: ( 1,  0),   # giu
    2: ( 0, -1),   # sinistra
    3: ( 0,  1),   # destra
}

# Nome italiano di ciascuna azione, usato nei prompt LLM e nel logging
NOMI_AZIONI: dict[int, str] = {
    0: "su",
    1: "giù",
    2: "sinistra",
    3: "destra",
}

# Celle su cui il giocatore puo' camminare senza spostare nulla
CELLE_LIBERE = (PAVIMENTO, TARGET)

# Celle che rappresentano una cassa (con o senza target sotto)
CELLE_CASSA = (CASSA, CASSA_SU_TARGET)


# ---------------------------------------------------------------------------
# Funzioni di ispezione della griglia
# ---------------------------------------------------------------------------

def trova_giocatore(griglia: np.ndarray) -> Tuple[int, int]:
    """Restituisce la posizione (riga, colonna) del giocatore nella griglia.

    Il giocatore puo' trovarsi su un pavimento (GIOCATORE) o su un target
    (GIOCATORE_SU_TARGET): entrambi i casi vengono cercati.

    Solleva ValueError se il giocatore non e' presente nella griglia.

    Parametri:
        griglia: matrice NumPy di forma (H, W), dtype int8.
    """
    posizioni = np.argwhere(
        (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
    )
    if len(posizioni) == 0:
        raise ValueError("Giocatore non trovato nella griglia.")
    return int(posizioni[0, 0]), int(posizioni[0, 1])


def conta_casse_su_target(griglia: np.ndarray) -> int:
    """Restituisce il numero di casse attualmente posizionate sui target.

    Parametri:
        griglia: matrice NumPy di forma (H, W).
    """
    return int(np.sum(griglia == CASSA_SU_TARGET))


def conta_casse_totali(griglia: np.ndarray) -> int:
    """Restituisce il numero totale di casse nel livello (su target e non).

    Parametri:
        griglia: matrice NumPy di forma (H, W).
    """
    return int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))


def controlla_vittoria(griglia: np.ndarray) -> bool:
    """Restituisce True se tutte le casse sono posizionate sui target.

    La vittoria si verifica quando non rimane nessuna cassa libera (CASSA)
    nella griglia e almeno una cassa e' su target (il livello non e' vuoto).

    Parametri:
        griglia: matrice NumPy di forma (H, W).
    """
    return (
        int(np.sum(griglia == CASSA)) == 0
        and conta_casse_su_target(griglia) > 0
    )


# ---------------------------------------------------------------------------
# Funzione principale: applica una mossa alla griglia
# ---------------------------------------------------------------------------

def applica_mossa(
    griglia: np.ndarray, azione: int
) -> Tuple[np.ndarray, bool, bool]:
    """Applica una mossa alla griglia e restituisce il nuovo stato.

    Gestisce tutti i casi: movimento su pavimento, movimento su target,
    spinta di una cassa su pavimento, spinta su target, mossa bloccata
    da muro, mossa bloccata da cassa contro muro o altro ostacolo.

    Parametri:
        griglia: matrice NumPy di forma (H, W), dtype int8.
        azione:  intero in {0, 1, 2, 3} — su, giu, sinistra, destra.

    Restituisce una tupla (nuova_griglia, mossa_eseguita, cassa_spostata):
        nuova_griglia:  griglia aggiornata (copia della originale se la mossa
                        non e' valida).
        mossa_eseguita: True se il giocatore si e' spostato di almeno una cella.
        cassa_spostata: True se durante la mossa e' stata spinta una cassa.

    Solleva ValueError se l'azione non e' in {0, 1, 2, 3}.
    """
    if azione not in DELTA_AZIONE:
        raise ValueError(
            f"Azione non valida: {azione}. Deve essere uno tra {{0,1,2,3}}."
        )

    nuova_griglia = griglia.copy()
    dr, dc = DELTA_AZIONE[azione]

    riga_g, col_g = trova_giocatore(griglia)
    riga_dest = riga_g + dr
    col_dest  = col_g + dc

    n_righe, n_col = griglia.shape

    # La cella di destinazione e' fuori dalla griglia: mossa impossibile
    if not (0 <= riga_dest < n_righe and 0 <= col_dest < n_col):
        return nuova_griglia, False, False

    cella_dest = int(griglia[riga_dest, col_dest])

    # Muro davanti: il giocatore non puo' passare
    if cella_dest == MURO:
        return nuova_griglia, False, False

    cassa_spostata = False

    # C'e' una cassa davanti: prova a spingerla nella cella successiva
    if cella_dest in CELLE_CASSA:
        riga_cassa_dest = riga_dest + dr
        col_cassa_dest  = col_dest + dc

        # La cassa uscirebbe dalla griglia: mossa impossibile
        if not (0 <= riga_cassa_dest < n_righe and 0 <= col_cassa_dest < n_col):
            return nuova_griglia, False, False

        cella_dopo_cassa = int(griglia[riga_cassa_dest, col_cassa_dest])

        # La cassa puo' essere spinta solo su pavimento libero o su un target libero
        if cella_dopo_cassa not in CELLE_LIBERE:
            return nuova_griglia, False, False

        # Sposta la cassa: se atterro su target diventa CASSA_SU_TARGET
        nuova_griglia[riga_cassa_dest, col_cassa_dest] = (
            CASSA_SU_TARGET if cella_dopo_cassa == TARGET else CASSA
        )
        cassa_spostata = True

    # Libera la cella originale del giocatore: se era su un target, lascia il target
    nuova_griglia[riga_g, col_g] = (
        TARGET if griglia[riga_g, col_g] == GIOCATORE_SU_TARGET else PAVIMENTO
    )

    # Posiziona il giocatore nella nuova cella
    if cassa_spostata:
        # Se sotto la cassa c'era un target (CASSA_SU_TARGET), il giocatore e' ora su target
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == CASSA_SU_TARGET else GIOCATORE
        )
    else:
        # Movimento semplice: se la cella di destinazione era un target, segnalo
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == TARGET else GIOCATORE
        )

    return nuova_griglia, True, cassa_spostata
