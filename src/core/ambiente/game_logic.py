# Logica di gioco Sokoban: costanti delle celle, movimento e verifica della vittoria.
# Nessuna dipendenza esterna (niente Gymnasium, niente Pygame): ogni funzione riceve
# un array NumPy e ne restituisce uno nuovo, quindi si testa passando una matrice a
# mano senza dover istanziare l'ambiente completo.
#
# Codifica numerica delle celle (un intero per cella):
#   0 = MURO                 (#)   muro invalicabile
#   1 = PAVIMENTO            ( )   cella vuota calpestabile
#   2 = TARGET               (.)   bersaglio su cui va posizionata una cassa
#   3 = CASSA                ($)   cassa libera, da spingere
#   4 = CASSA_SU_TARGET      (*)   cassa già a posto su un bersaglio
#   5 = GIOCATORE            (@)   giocatore su pavimento
#   6 = GIOCATORE_SU_TARGET  (+)   giocatore fermo su un bersaglio

import numpy as np
from typing import Tuple


# COSTANTI: CODIFICA NUMERICA DELLE CELLE
MURO                 = 0   # cella invalicabile
PAVIMENTO            = 1   # cella vuota calpestabile
TARGET               = 2   # bersaglio da coprire con una cassa
CASSA                = 3   # cassa libera, ancora da posizionare
CASSA_SU_TARGET      = 4   # cassa già su un bersaglio (conta per la vittoria)
GIOCATORE            = 5   # giocatore su pavimento
GIOCATORE_SU_TARGET  = 6   # giocatore su un bersaglio

# Delta (riga, colonna) di spostamento per ciascuna delle 4 azioni discrete:
# le righe crescono verso il basso, le colonne verso destra (convenzione matrice).
DELTA_AZIONE: dict[int, Tuple[int, int]] = {
    0: (-1,  0),   # su:       una riga in alto
    1: ( 1,  0),   # giù:      una riga in basso
    2: ( 0, -1),   # sinistra: una colonna a sinistra
    3: ( 0,  1),   # destra:   una colonna a destra
}

# Celle su cui il giocatore può camminare senza spingere nulla.
CELLE_LIBERE = (PAVIMENTO, TARGET)

# Celle che contengono una cassa (libera oppure già su bersaglio).
CELLE_CASSA = (CASSA, CASSA_SU_TARGET)


# FUNZIONI DI ISPEZIONE DELLA GRIGLIA

def trova_giocatore(griglia: np.ndarray) -> Tuple[int, int]:
    """
    Trova la posizione (riga, colonna) del giocatore nella griglia.

    Cerca sia GIOCATORE (su pavimento) sia GIOCATORE_SU_TARGET (su un bersaglio):
    il giocatore è uno solo, ma può trovarsi in entrambe le condizioni.
    Solleva ValueError se nessun giocatore è presente (griglia malformata).
    """
    # argwhere restituisce tutte le posizioni che soddisfano la condizione booleana
    posizioni = np.argwhere(
        (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
    )
    if len(posizioni) == 0:
        raise ValueError("Giocatore non trovato nella griglia.")
    return int(posizioni[0, 0]), int(posizioni[0, 1])   # prima (e unica) posizione


def conta_casse_su_target(griglia: np.ndarray) -> int:
    """Conta le casse attualmente a posto: il numero di celle CASSA_SU_TARGET."""
    return int(np.sum(griglia == CASSA_SU_TARGET))


def conta_casse_totali(griglia: np.ndarray) -> int:
    """Conta tutte le casse del livello, sia quelle libere sia quelle già a posto."""
    return int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))


def controlla_vittoria(griglia: np.ndarray) -> bool:
    """
    Verifica se il livello è risolto.

    Il livello è vinto quando non resta nessuna cassa libera e c'è almeno una cassa
    a posto: la seconda condizione esclude le griglie vuote o malformate, che senza
    casse risulterebbero "vinte" per assenza di casse da posizionare.
    """
    return (
        int(np.sum(griglia == CASSA)) == 0          # nessuna cassa ancora libera
        and conta_casse_su_target(griglia) > 0      # ...e almeno una è a posto
    )


# FUNZIONE PRINCIPALE: APPLICA UNA MOSSA ALLA GRIGLIA

def applica_mossa(
    griglia: np.ndarray, azione: int
) -> Tuple[np.ndarray, bool, bool]:
    """
    Applica una mossa e restituisce il nuovo stato della griglia.

    Copre tutti i casi della fisica di Sokoban:
      1. Movimento semplice su pavimento o su un bersaglio.
      2. Spinta di una cassa su pavimento o su un bersaglio libero.
      3. Mossa bloccata: muro davanti, bordo della griglia, oppure cassa non
         spingibile perché ha un muro o un'altra cassa subito dietro.

    Restituisce la tupla (nuova_griglia, mossa_eseguita, cassa_spostata):
      - nuova_griglia:  griglia aggiornata, o copia invariata se la mossa è illegale
      - mossa_eseguita: True se il giocatore si è spostato di almeno una cella
      - cassa_spostata: True se la mossa ha spinto una cassa
    Solleva ValueError se l'azione non è in {0, 1, 2, 3}.
    """
    if azione not in DELTA_AZIONE:
        raise ValueError(
            f"Azione non valida: {azione}. Deve essere uno tra {{0,1,2,3}}."
        )

    # Si lavora su una copia: la griglia originale non va mai mutata in-place
    nuova_griglia = griglia.copy()
    dr, dc = DELTA_AZIONE[azione]          # spostamento richiesto dall'azione

    riga_g, col_g = trova_giocatore(griglia)
    riga_dest = riga_g + dr                # cella in cui il giocatore vuole entrare
    col_dest  = col_g + dc

    n_righe, n_col = griglia.shape

    # Destinazione fuori dai bordi: mossa impossibile, stato invariato
    if not (0 <= riga_dest < n_righe and 0 <= col_dest < n_col):
        return nuova_griglia, False, False

    cella_dest = int(griglia[riga_dest, col_dest])

    # Muro davanti: il giocatore non passa
    if cella_dest == MURO:
        return nuova_griglia, False, False

    cassa_spostata = False

    # Cassa davanti: si tenta di spingerla nella cella immediatamente successiva
    if cella_dest in CELLE_CASSA:
        riga_cassa_dest = riga_dest + dr   # dove finirebbe la cassa (un passo oltre)
        col_cassa_dest  = col_dest + dc

        # La cassa uscirebbe dalla griglia: spinta impossibile
        if not (0 <= riga_cassa_dest < n_righe and 0 <= col_cassa_dest < n_col):
            return nuova_griglia, False, False

        cella_dopo_cassa = int(griglia[riga_cassa_dest, col_cassa_dest])

        # Una cassa si spinge solo su pavimento o su un bersaglio libero:
        # mai contro un muro né contro un'altra cassa
        if cella_dopo_cassa not in CELLE_LIBERE:
            return nuova_griglia, False, False

        # Sposta la cassa: se atterra su un bersaglio diventa CASSA_SU_TARGET
        nuova_griglia[riga_cassa_dest, col_cassa_dest] = (
            CASSA_SU_TARGET if cella_dopo_cassa == TARGET else CASSA
        )
        cassa_spostata = True

    # Svuota la cella di partenza: se il giocatore stava su un bersaglio, lo lascia scoperto
    nuova_griglia[riga_g, col_g] = (
        TARGET if griglia[riga_g, col_g] == GIOCATORE_SU_TARGET else PAVIMENTO
    )

    # Scrive il giocatore nella nuova cella, ricordando se sotto c'è un bersaglio
    if cassa_spostata:
        # La destinazione conteneva la cassa: se era CASSA_SU_TARGET sotto c'era un
        # bersaglio, quindi ora il giocatore risulta GIOCATORE_SU_TARGET
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == CASSA_SU_TARGET else GIOCATORE
        )
    else:
        # Movimento semplice: giocatore su target se la destinazione era un bersaglio
        nuova_griglia[riga_dest, col_dest] = (
            GIOCATORE_SU_TARGET if cella_dest == TARGET else GIOCATORE
        )

    return nuova_griglia, True, cassa_spostata
