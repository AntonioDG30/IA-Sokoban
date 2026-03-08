"""Generatore di livelli Sokoban procedurali per il curriculum learning (Fase 2.3).

Genera livelli validi e risolvibili per griglie di dimensione arbitraria
(tipicamente 5x5 con 2 casse e 7x7 con 3 casse) da usare nelle fasi
iniziali del curriculum learning prima di passare a Boxoban 10x10.

Algoritmo:
    1. Genera una griglia con muri perimetrali e interno vuoto.
    2. Posiziona casualmente giocatore, casse e target in celle libere.
    3. Verifica la risolvibilita' tramite BFS nello spazio degli stati.
    4. Se non risolvibile, riprova (max N tentativi).

Nota: la verifica BFS e' esatta ma costosa per griglie grandi.
Per 5x5 e 7x7 con 2-3 casse e' pienamente praticabile.
"""

import numpy as np
import random
from collections import deque
from typing import Optional, Tuple

from sokoban_env.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET, GIOCATORE,
    applica_mossa, controlla_vittoria,
)

# Valore usato per il padding nelle griglie piu' piccole di 10x10.
# Distinto da MURO (0) cosi' la CNN puo' imparare a ignorare il bordo artificiale.
PADDING = 7


class GeneratoreLivelli:
    """Genera livelli Sokoban procedurali validi e risolvibili.

    Parametri costruttore:
        seme: seed per la riproducibilita'. None = casuale.
    """

    MAX_TENTATIVI = 500  # tentativi massimi per trovare un livello risolvibile

    def __init__(self, seme: Optional[int] = None):
        self._rng = random.Random(seme)
        self._np_rng = np.random.default_rng(seme)

    def genera(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> np.ndarray:
        """Genera un livello Sokoban valido e risolvibile.

        Parametri:
            righe:   numero di righe della griglia.
            colonne: numero di colonne della griglia.
            n_casse: numero di casse (e target) da posizionare.

        Restituisce:
            Griglia NumPy (righe, colonne) dtype int8.

        Solleva:
            RuntimeError se non trova un livello risolvibile entro MAX_TENTATIVI.
        """
        for _ in range(self.MAX_TENTATIVI):
            griglia = self._genera_griglia_base(righe, colonne, n_casse)
            if griglia is not None and self._verifica_risolvibile(griglia):
                return griglia

        raise RuntimeError(
            f"Impossibile generare livello risolvibile ({righe}x{colonne}, "
            f"{n_casse} casse) in {self.MAX_TENTATIVI} tentativi."
        )

    # ------------------------------------------------------------------
    # Generazione griglia
    # ------------------------------------------------------------------

    def _genera_griglia_base(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> Optional[np.ndarray]:
        """Genera una griglia con muri perimetrali e oggetti posizionati casualmente.

        Restituisce None se non ci sono abbastanza celle libere.
        """
        griglia = np.zeros((righe, colonne), dtype=np.int8)

        # Muri perimetrali
        griglia[0, :] = MURO
        griglia[-1, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, -1] = MURO

        # Interno: pavimento
        griglia[1:-1, 1:-1] = PAVIMENTO

        # Aggiungi qualche muro interno casuale (max 20% celle interne)
        celle_interne = (righe - 2) * (colonne - 2)
        n_muri_interni = self._rng.randint(0, max(0, celle_interne // 5))
        posizioni_interne = [
            (r, c)
            for r in range(1, righe - 1)
            for c in range(1, colonne - 1)
        ]
        self._rng.shuffle(posizioni_interne)
        for r, c in posizioni_interne[:n_muri_interni]:
            griglia[r, c] = MURO

        # Celle libere disponibili
        libere = [(r, c) for r, c in posizioni_interne if griglia[r, c] == PAVIMENTO]

        # Serve almeno: n_casse target + n_casse casse + 1 giocatore
        if len(libere) < 2 * n_casse + 1:
            return None

        self._rng.shuffle(libere)

        # Posiziona target
        for i in range(n_casse):
            r, c = libere[i]
            griglia[r, c] = TARGET

        # Posiziona casse (su pavimento, non su target)
        idx = n_casse
        for i in range(n_casse):
            r, c = libere[idx + i]
            griglia[r, c] = CASSA

        # Posiziona giocatore
        r, c = libere[2 * n_casse]
        griglia[r, c] = GIOCATORE

        return griglia

    # ------------------------------------------------------------------
    # Verifica risolvibilita' tramite BFS
    # ------------------------------------------------------------------

    def _verifica_risolvibile(self, griglia: np.ndarray) -> bool:
        """Esegue BFS nello spazio degli stati per verificare la risolvibilita'.

        Restituisce True se esiste una sequenza di mosse che porta alla vittoria.
        Limite: MAX_STATI_BFS stati esplorati per evitare OOM su griglie grandi.
        """
        MAX_STATI_BFS = 50_000

        stato_iniziale = griglia.tobytes()
        visitati = {stato_iniziale}
        coda = deque([griglia.copy()])
        n_esplorati = 0

        while coda and n_esplorati < MAX_STATI_BFS:
            stato = coda.popleft()
            n_esplorati += 1

            for azione in range(4):
                nuovo_stato, _, _ = applica_mossa(stato, azione)
                chiave = nuovo_stato.tobytes()

                if controlla_vittoria(nuovo_stato):
                    return True

                if chiave not in visitati:
                    visitati.add(chiave)
                    coda.append(nuovo_stato)

        return False


# ---------------------------------------------------------------------------
# Funzione di utilita' per il padding a dimensione fissa
# ---------------------------------------------------------------------------

def padding_a_10x10(griglia: np.ndarray) -> np.ndarray:
    """Inserisce la griglia in un frame 10x10 con padding = PADDING (7).

    Usato per mantenere l'observation space fisso a (10,10) con SB3
    anche quando la griglia e' piu' piccola di 10x10.

    Il valore di padding (7) e' distinto da MURO (0) e da tutte le celle
    di gioco (1-6), cosi' la CNN puo' imparare a ignorare il bordo artificiale
    e sviluppare features invarianti al cambio di dimensione della griglia.

    La griglia viene centrata nel frame 10x10.

    Parametri:
        griglia: array NumPy (righe, colonne) con righe<=10 e colonne<=10.

    Restituisce:
        Array NumPy (10, 10) dtype int8 con la griglia centrata e bordi a PADDING.
    """
    righe, colonne = griglia.shape
    assert righe <= 10 and colonne <= 10, (
        f"La griglia ({righe}x{colonne}) supera le dimensioni massime 10x10."
    )

    padded = np.full((10, 10), fill_value=PADDING, dtype=np.int8)
    offset_r = (10 - righe) // 2
    offset_c = (10 - colonne) // 2
    padded[offset_r:offset_r + righe, offset_c:offset_c + colonne] = griglia
    return padded
