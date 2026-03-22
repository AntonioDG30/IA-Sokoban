"""Generatore procedurale di livelli Sokoban per il curriculum learning.

Genera livelli validi e risolvibili su griglie 10x10 (o di altra dimensione),
usati nelle fasi C0-C3 del curriculum prima di passare ai livelli Boxoban reali.

Algoritmo di generazione:
    1. Crea una griglia con muri perimetrali e interno a pavimento.
    2. Aggiunge muri interni casuali (max 20% delle celle interne).
    3. Posiziona casualmente giocatore, casse e target in celle libere.
    4. Verifica la risolvibilita' con BFS esatto o euristica dead-corner.
    5. Se non risolvibile, riprova fino a MAX_TENTATIVI volte.

Nota sul metodo di verifica:
    Il BFS e' esatto ma costoso per spazi grandi. Per 1 cassa su 10x10
    lo spazio e' ~4096 stati (BFS rapido). Per 2+ casse supera 50K stati
    e si usa l'euristica dead-corner che filtra i casi ovviamente impossibili
    ma non garantisce la risolvibilita'.
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

# Valore di padding usato per riempire il bordo quando una griglia piu' piccola
# viene inserita in un frame 10x10. Distinto da MURO (0) per permettere alla CNN
# di imparare a ignorare il bordo artificiale
PADDING = 7


class GeneratoreLivelli:
    """Genera livelli Sokoban procedurali validi e risolvibili.

    Parametri:
        seme: seed per la riproducibilita'. None = seme casuale.
    """

    # Numero massimo di tentativi prima di sollevare RuntimeError
    MAX_TENTATIVI = 500

    def __init__(self, seme: Optional[int] = None):
        # Due RNG separati: uno per le scelte di posizionamento, uno per NumPy
        self._rng    = random.Random(seme)
        self._np_rng = np.random.default_rng(seme)

    def genera(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> np.ndarray:
        """Genera un livello Sokoban valido e risolvibile.

        Esegue fino a MAX_TENTATIVI iterazioni di generazione+verifica.
        La verifica usa BFS esatto per spazi piccoli, euristica per spazi grandi.

        Parametri:
            righe:   numero di righe della griglia da generare.
            colonne: numero di colonne della griglia da generare.
            n_casse: numero di casse (e corrispondenti target) da posizionare.

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
    # Generazione della struttura della griglia
    # ------------------------------------------------------------------

    def _genera_griglia_base(
        self,
        righe: int,
        colonne: int,
        n_casse: int,
    ) -> Optional[np.ndarray]:
        """Genera una griglia con muri perimetrali e oggetti posizionati casualmente.

        Muri interni casuali vengono aggiunti per aumentare la varieta' dei livelli,
        ma solo fino al 20% delle celle interne per non bloccare troppo lo spazio.

        Restituisce None se le celle libere disponibili non sono sufficienti
        per posizionare giocatore, casse e target.
        """
        griglia = np.zeros((righe, colonne), dtype=np.int8)

        # Muri perimetrali: delimitano la griglia su tutti i lati
        griglia[0, :]  = MURO
        griglia[-1, :] = MURO
        griglia[:, 0]  = MURO
        griglia[:, -1] = MURO

        # Interno: tutto pavimento inizialmente
        griglia[1:-1, 1:-1] = PAVIMENTO

        # Muri interni casuali (0-20% delle celle interne) per variare i livelli
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

        # Celle ancora libere dopo aver piazzato i muri interni
        libere = [(r, c) for r, c in posizioni_interne if griglia[r, c] == PAVIMENTO]

        # Serve almeno: n_casse target + n_casse casse + 1 giocatore
        if len(libere) < 2 * n_casse + 1:
            return None

        self._rng.shuffle(libere)

        # Posiziona i target nelle prime n_casse celle libere
        for i in range(n_casse):
            r, c = libere[i]
            griglia[r, c] = TARGET

        # Posiziona le casse nelle n_casse celle successive (non sulle stesse dei target)
        idx = n_casse
        for i in range(n_casse):
            r, c = libere[idx + i]
            griglia[r, c] = CASSA

        # Posiziona il giocatore nella cella successiva disponibile
        r, c = libere[2 * n_casse]
        griglia[r, c] = GIOCATORE

        return griglia

    # ------------------------------------------------------------------
    # Verifica risolvibilita'
    # ------------------------------------------------------------------

    # Soglia per scegliere tra BFS esatto ed euristica:
    # 10x10 con 1 cassa ~ 64^2 = 4096 stati -> BFS fattibile
    # 10x10 con 2 casse ~ 64^3 = 262144 stati -> troppo lento, si usa euristica
    MAX_STATI_BFS = 50_000

    def _verifica_risolvibile(self, griglia: np.ndarray) -> bool:
        """Sceglie tra BFS esatto ed euristica in base alla dimensione dello spazio stati.

        La stima dello spazio stati e' conservativa: celle_interne^(n_casse+1).
        Se sotto la soglia, usa il BFS esatto (garanzia completa).
        Altrimenti usa l'euristica dead-corner (piu' veloce, falsi positivi possibili).
        """
        righe, colonne = griglia.shape
        n_casse = int((griglia == CASSA).sum())

        # Stima conservativa: (celle interne)^(n_casse + posizione giocatore)
        celle_interne = max(1, (righe - 2) * (colonne - 2))
        stima_stati = celle_interne ** (n_casse + 1)

        if stima_stati <= self.MAX_STATI_BFS:
            return self._bfs_esatto(griglia)
        else:
            return self._euristica_dead_corner(griglia)

    def _bfs_esatto(self, griglia: np.ndarray) -> bool:
        """Esplora lo spazio degli stati completo con BFS per trovare una soluzione.

        Restituisce True se trova la vittoria entro MAX_STATI_BFS stati visitati.
        Usato solo quando lo spazio e' abbastanza piccolo da rendere il BFS veloce.
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
        """Verifica euristica per griglie dove il BFS sarebbe troppo lento.

        Controlla due condizioni necessarie (ma non sufficienti) per la risolvibilita':
        1. Nessuna cassa libera e' in un angolo morto (bloccata da due muri perpendicolari
           senza un target esattamente in quell'angolo).
        2. Il giocatore puo' raggiungere almeno una cassa (flood fill sul solo movimento
           del giocatore, trattando le casse come ostacoli fissi).

        I falsi positivi (livello accettato ma non risolvibile) sono tollerabili:
        l'agente RL impara comunque da episodi troncati.
        """
        righe, colonne = griglia.shape

        # Posizioni di tutti i target nel livello (la cassa non e' in dead-corner
        # se e' gia' posizionata su un target)
        target_pos = set(
            map(tuple, np.argwhere(
                (griglia == TARGET) | (griglia == CASSA_SU_TARGET)
            ))
        )

        # Controllo 1: dead-corner per ogni cassa libera
        for r in range(1, righe - 1):
            for c in range(1, colonne - 1):
                if griglia[r, c] != CASSA:
                    continue
                if (r, c) in target_pos:
                    continue   # gia' su target: non e' un dead-corner

                # Angolo morto: bloccata da almeno un muro verticale E uno orizzontale
                muro_v = (griglia[r - 1, c] == MURO) or (griglia[r + 1, c] == MURO)
                muro_h = (griglia[r, c - 1] == MURO) or (griglia[r, c + 1] == MURO)
                if muro_v and muro_h:
                    return False   # cassa irrecuperabile -> livello scartato

        # Controllo 2: connettivita' giocatore-casse via flood fill
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

        # Il giocatore deve poter raggiungere almeno una cassa (di lato)
        casse = np.argwhere((griglia == CASSA) | (griglia == CASSA_SU_TARGET))
        for r_c, c_c in casse:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if (r_c + dr, c_c + dc) in raggiungibili:
                    return True   # almeno una cassa e' raggiungibile

        return False   # nessuna cassa accessibile: livello inutile


# ---------------------------------------------------------------------------
# Funzione di utilita': inserisce una griglia in un frame 10x10
# ---------------------------------------------------------------------------

def padding_a_10x10(griglia: np.ndarray) -> np.ndarray:
    """Inserisce la griglia in un frame 10x10 con padding = PADDING (7).

    Usato per mantenere l'observation space fisso a (10,10) quando la griglia
    di gioco e' piu' piccola. Il valore 7 e' distinto da MURO (0) e da tutte
    le celle di gioco (1-6), cosi' la CNN puo' imparare a ignorare il bordo
    artificiale. La griglia viene centrata nel frame.

    Parametri:
        griglia: array NumPy (righe, colonne) con righe<=10 e colonne<=10.

    Restituisce:
        Array NumPy (10, 10) dtype int8 con la griglia centrata e bordi a PADDING.
    """
    righe, colonne = griglia.shape
    assert righe <= 10 and colonne <= 10, (
        f"La griglia ({righe}x{colonne}) supera le dimensioni massime 10x10."
    )

    # Frame iniziale tutto a PADDING
    padded = np.full((10, 10), fill_value=PADDING, dtype=np.int8)

    # Offset per centrare la griglia nel frame
    offset_r = (10 - righe) // 2
    offset_c = (10 - colonne) // 2
    padded[offset_r:offset_r + righe, offset_c:offset_c + colonne] = griglia

    return padded
