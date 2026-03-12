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
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
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
    # Verifica risolvibilita': BFS esatto o euristica dead-corner
    # ------------------------------------------------------------------

    # Soglia stati BFS: per spazi piccoli usa BFS esatto, altrimenti euristica.
    # 10x10 con 1 cassa: ~64^2 = 4096 stati -> BFS esatto.
    # 10x10 con 2+ casse: >262144 stati -> euristica dead-corner.
    MAX_STATI_BFS = 50_000

    def _verifica_risolvibile(self, griglia: np.ndarray) -> bool:
        """Sceglie tra BFS esatto ed euristica in base alla complessita' del problema.

        Per spazi stati piccoli (1 cassa su griglia piccola) usa BFS esatto.
        Per spazi grandi (2+ casse su 10x10) usa l'euristica dead-corner:
        piu' veloce, non garantisce risolvibilita' ma filtra i casi ovviamente
        impossibili (casse bloccate in angoli senza target adiacente).
        """
        righe, colonne = griglia.shape
        n_casse = int((griglia == CASSA).sum())
        # Stima conservativa dello spazio stati: celle_interne^(n_casse+1)
        celle_interne = max(1, (righe - 2) * (colonne - 2))
        stima_stati = celle_interne ** (n_casse + 1)

        if stima_stati <= self.MAX_STATI_BFS:
            # BFS esatto: possibile solo per configurazioni piccole
            return self._bfs_esatto(griglia)
        else:
            # Euristica: controllo dead-corner + connettivita' giocatore-casse
            return self._euristica_dead_corner(griglia)

    def _bfs_esatto(self, griglia: np.ndarray) -> bool:
        """BFS nello spazio degli stati completo (esatto ma costoso).

        Restituisce True se esiste una soluzione entro MAX_STATI_BFS stati.
        """
        stato_iniziale = griglia.tobytes()
        visitati = {stato_iniziale}
        coda = deque([griglia.copy()])
        n_esplorati = 0

        while coda and n_esplorati < self.MAX_STATI_BFS:
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

    def _euristica_dead_corner(self, griglia: np.ndarray) -> bool:
        """Verifica euristica per griglie grandi (necessaria, non sufficiente).

        Controlla due condizioni che ESCLUDONO livelli ovviamente irrisolvibili:
        1. Nessuna cassa libera e' in dead-corner: angolo formato da due muri
           perpendicolari senza un target esattamente in quell'angolo.
        2. Il giocatore puo' raggiungere almeno una cassa (connettivita' base).

        Falsi positivi possibili (livello accettato ma non risolvibile) sono
        tollerabili: l'agente RL impara comunque da episodi troncati.
        """
        righe, colonne = griglia.shape

        # --- 1. Dead-corner check ---
        # Posizioni target (la cassa non e' dead-corner SE e' gia' su target)
        target_pos = set(
            map(tuple, np.argwhere(
                (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
            ))
        )

        for r in range(1, righe - 1):
            for c in range(1, colonne - 1):
                if griglia[r, c] != CASSA:
                    continue  # Solo casse libere
                if (r, c) in target_pos:
                    continue  # Gia' su target: non e' dead

                # Angolo morto: bloccata da muro in almeno una direzione verticale
                # E in almeno una direzione orizzontale
                muro_v = (griglia[r - 1, c] == MURO) or (griglia[r + 1, c] == MURO)
                muro_h = (griglia[r, c - 1] == MURO) or (griglia[r, c + 1] == MURO)
                if muro_v and muro_h:
                    return False  # Cassa irrecuperabile

        # --- 2. Connettivita' giocatore-casse (flood fill) ---
        pos_g = np.argwhere(
            (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
        )
        if len(pos_g) == 0:
            return False
        r_g, c_g = int(pos_g[0, 0]), int(pos_g[0, 1])

        # BFS sul solo movimento del giocatore (casse = ostacoli fissi)
        raggiungibili: set = set()
        coda_ff: deque = deque([(r_g, c_g)])
        while coda_ff:
            r, c = coda_ff.popleft()
            if (r, c) in raggiungibili:
                continue
            if not (0 <= r < righe and 0 <= c < colonne):
                continue
            cella = griglia[r, c]
            if cella == MURO or cella == CASSA:
                continue
            raggiungibili.add((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                coda_ff.append((r + dr, c + dc))

        # Almeno una cassa deve essere adiacente a una cella raggiungibile
        casse = np.argwhere((griglia == CASSA) | (griglia == CASSA_SU_TARGET))
        for r_c, c_c in casse:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r_c + dr, c_c + dc) in raggiungibili:
                    return True  # Giocatore puo' raggiungere questa cassa

        return False  # Nessuna cassa accessibile


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
