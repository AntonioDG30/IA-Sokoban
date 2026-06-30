# Generatore procedurale di livelli Sokoban per il curriculum learning.
#
# Genera livelli validi e risolvibili su griglie 10x10 (o di altra dimensione), usati
# nelle fasi C0-C3 del curriculum prima di passare ai livelli reali di Boxoban.
#
# Algoritmo di generazione (per ogni tentativo):
#   1. Crea una griglia con muri perimetrali e interno tutto a pavimento.
#   2. Aggiunge muri interni casuali (fino al 20% delle celle interne).
#   3. Posiziona a caso giocatore, casse e target tra le celle libere.
#   4. Verifica la risolvibilità con BFS esatto oppure euristica dead-corner.
#   5. Se il livello non è valido/risolvibile, riprova fino a MAX_TENTATIVI volte.
#
# Sul metodo di verifica: il BFS è esatto ma costoso. Con 1 cassa su 10x10 lo spazio è
# ~4096 stati (BFS rapido); con 2+ casse supera i 50K stati e si passa all'euristica
# dead-corner, che scarta i casi palesemente impossibili ma non garantisce la soluzione.

import numpy as np
import random
from collections import deque
from typing import Optional

from core.ambiente.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
    applica_mossa, controlla_vittoria,
)

# Valore con cui si riempie il bordo quando una griglia più piccola viene inserita in un
# frame 10x10. È distinto da MURO (0) così la CNN può imparare a ignorare il bordo finto.
PADDING = 7


class GeneratoreLivelli:
    """
    Genera livelli Sokoban procedurali, validi e risolvibili.

    seme rende riproducibile la generazione (None = casuale).
    """

    # Tentativi massimi di generazione+verifica prima di arrendersi con un RuntimeError.
    MAX_TENTATIVI = 500

    def __init__(self, seme: Optional[int] = None):
        # Due RNG separati: uno (random) per le scelte di posizionamento, uno per NumPy
        self._rng    = random.Random(seme)
        self._np_rng = np.random.default_rng(seme)

    def genera(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> np.ndarray:
        """
        Genera un livello valido e risolvibile, riprovando finché non ci riesce.

        Ripete fino a MAX_TENTATIVI cicli di generazione + verifica (BFS esatto sugli spazi
        piccoli, euristica su quelli grandi). righe/colonne sono le dimensioni della griglia
        e n_casse il numero di casse (con altrettanti target). Restituisce la griglia (int8)
        oppure solleva RuntimeError se non trova nulla di risolvibile entro i tentativi.
        """
        for _ in range(self.MAX_TENTATIVI):
            griglia = self._genera_griglia_base(righe, colonne, n_casse)
            if griglia is not None and self._verifica_risolvibile(griglia):
                return griglia

        raise RuntimeError(
            f"Impossibile generare livello risolvibile ({righe}x{colonne}, "
            f"{n_casse} casse) in {self.MAX_TENTATIVI} tentativi."
        )

    # GENERAZIONE DELLA STRUTTURA DELLA GRIGLIA

    def _genera_griglia_base(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> Optional[np.ndarray]:
        """
        Costruisce una griglia con muri perimetrali, muri interni casuali e oggetti.

        Aggiunge muri interni (fino al 20% delle celle interne) per variare i livelli, poi
        piazza target, casse e giocatore nelle celle ancora libere. Restituisce None se,
        dopo i muri, le celle libere non bastano per tutti gli oggetti.
        """
        griglia = np.zeros((righe, colonne), dtype=np.int8)

        # Muri perimetrali: chiudono la griglia sui quattro lati
        griglia[0, :]  = MURO
        griglia[-1, :] = MURO
        griglia[:, 0]  = MURO
        griglia[:, -1] = MURO

        # Interno inizialmente tutto pavimento
        griglia[1:-1, 1:-1] = PAVIMENTO

        # Muri interni casuali (0-20% delle celle interne) per dare varietà ai livelli
        celle_interne = (righe - 2) * (colonne - 2)
        n_muri_interni = self._rng.randint(0, max(0, celle_interne // 5))   # // 5 = 20% al massimo
        posizioni_interne = [
            (r, c)
            for r in range(1, righe - 1)
            for c in range(1, colonne - 1)
        ]
        self._rng.shuffle(posizioni_interne)
        for r, c in posizioni_interne[:n_muri_interni]:
            griglia[r, c] = MURO

        # Celle ancora libere dopo aver piazzato i muri interni
        libere = [(r, c) for r, c in posizioni_interne if griglia[r, c] == PAVIMENTO]

        # Servono almeno: n_casse target + n_casse casse + 1 giocatore
        if len(libere) < 2 * n_casse + 1:
            return None

        self._rng.shuffle(libere)

        # Target nelle prime n_casse celle libere
        for i in range(n_casse):
            r, c = libere[i]
            griglia[r, c] = TARGET

        # Casse nelle n_casse celle successive (mai sopra ai target appena piazzati)
        idx = n_casse
        for i in range(n_casse):
            r, c = libere[idx + i]
            griglia[r, c] = CASSA

        # Giocatore nella prima cella libera rimasta
        r, c = libere[2 * n_casse]
        griglia[r, c] = GIOCATORE

        return griglia

    # VERIFICA DI RISOLVIBILITÀ

    # Soglia per scegliere tra BFS esatto ed euristica:
    #   10x10 con 1 cassa ~ 64^2  = 4096 stati   -> BFS fattibile
    #   10x10 con 2 casse ~ 64^3  = 262144 stati -> troppo lento, si usa l'euristica
    MAX_STATI_BFS = 50_000

    def _verifica_risolvibile(self, griglia: np.ndarray) -> bool:
        """
        Sceglie il metodo di verifica in base alla dimensione stimata dello spazio stati.

        La stima è conservativa: celle_interne^(n_casse+1). Sotto la soglia MAX_STATI_BFS
        usa il BFS esatto (garanzia completa di risolvibilità); sopra la soglia ripiega
        sull'euristica dead-corner (più veloce, ma con possibili falsi positivi).
        """
        righe, colonne = griglia.shape
        n_casse = int((griglia == CASSA).sum())

        # Stima conservativa: (celle interne) ^ (n_casse + posizione del giocatore)
        celle_interne = max(1, (righe - 2) * (colonne - 2))
        stima_stati = celle_interne ** (n_casse + 1)

        if stima_stati <= self.MAX_STATI_BFS:
            return self._bfs_esatto(griglia)
        else:
            return self._euristica_dead_corner(griglia)

    def _bfs_esatto(self, griglia: np.ndarray) -> bool:
        """
        Esplora con BFS l'intero spazio degli stati alla ricerca di una soluzione.

        Restituisce True appena raggiunge una configurazione vinta, False se esaurisce la
        coda o supera MAX_STATI_BFS stati visitati. Usato solo quando lo spazio è abbastanza
        piccolo da rendere il BFS veloce.
        """
        stato_iniziale = griglia.tobytes()
        visitati = {stato_iniziale}           # stati già visti, serializzati a bytes
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
        """
        Verifica euristica usata quando il BFS sarebbe troppo lento.

        Controlla due condizioni necessarie (ma non sufficienti) per la risolvibilità:
          1. Nessuna cassa libera è in un angolo morto (bloccata da due muri perpendicolari
             senza un target proprio in quell'angolo).
          2. Il giocatore può raggiungere almeno una cassa (flood fill sul solo movimento
             del giocatore, trattando le casse come ostacoli fissi).
        I falsi positivi (livello accettato ma in realtà irrisolvibile) sono tollerati:
        l'agente RL impara comunque dagli episodi troncati.
        """
        righe, colonne = griglia.shape

        # Posizioni di tutti i target: una cassa su un target non è un dead-corner
        target_pos = set(
            map(tuple, np.argwhere(
                (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
            ))
        )

        # Controllo 1: cerca casse libere bloccate in un angolo
        for r in range(1, righe - 1):
            for c in range(1, colonne - 1):
                if griglia[r, c] != CASSA:
                    continue
                if (r, c) in target_pos:
                    continue   # già su un target: recuperata, non è un dead-corner

                # Angolo morto: muro su un lato verticale E su un lato orizzontale
                muro_v = (griglia[r - 1, c] == MURO) or (griglia[r + 1, c] == MURO)
                muro_h = (griglia[r, c - 1] == MURO) or (griglia[r, c + 1] == MURO)
                if muro_v and muro_h:
                    return False   # cassa irrecuperabile -> livello scartato

        # Controllo 2: connettività giocatore-casse tramite flood fill
        pos_g = np.argwhere(
            (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
        )
        if len(pos_g) == 0:
            return False
        r_g, c_g = int(pos_g[0, 0]), int(pos_g[0, 1])

        # Flood fill sul solo spostamento del giocatore (le casse sono ostacoli fissi)
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

        # Basta che il giocatore arrivi accanto ad almeno una cassa (da un lato)
        casse = np.argwhere((griglia == CASSA) | (griglia == CASSA_SU_TARGET))
        for r_c, c_c in casse:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r_c + dr, c_c + dc) in raggiungibili:
                    return True   # almeno una cassa è raggiungibile

        return False   # nessuna cassa accessibile: livello inutile


# FUNZIONE DI UTILITÀ: INSERISCE UNA GRIGLIA IN UN FRAME 10x10

def padding_a_10x10(griglia: np.ndarray) -> np.ndarray:
    """
    Centra una griglia più piccola dentro un frame 10x10, riempiendo il bordo con PADDING (7).

    Serve a tenere l'observation space fisso a (10,10) quando la griglia di gioco è più
    piccola. Il valore 7 è distinto da MURO (0) e da tutte le celle di gioco (1-6), così la
    CNN può imparare a ignorare il bordo artificiale. Restituisce la griglia (10, 10) int8.
    """
    righe, colonne = griglia.shape
    assert righe <= 10 and colonne <= 10, (
        f"La griglia ({righe}x{colonne}) supera le dimensioni massime 10x10."
    )

    # Frame di partenza interamente a PADDING
    padded = np.full((10, 10), fill_value=PADDING, dtype=np.int8)

    # Offset per centrare la griglia nel frame
    offset_r = (10 - righe) // 2
    offset_c = (10 - colonne) // 2
    padded[offset_r:offset_r + righe, offset_c:offset_c + colonne] = griglia

    return padded
