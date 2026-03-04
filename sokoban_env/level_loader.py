"""Caricamento e parsing dei livelli Boxoban.

Supporta il formato testuale del dataset DeepMind Boxoban:
    https://github.com/google-deepmind/boxoban-levels

Formato file:
    ; level 0
    ##########
    #.  $   @#
    ...
    ##########

    ; level 1
    ...

Caratteri:
    '#' → MURO (0)
    ' ' → PAVIMENTO (1)
    '.' → TARGET (2)
    '$' → CASSA (3)
    '*' → CASSA_SU_TARGET (4)
    '@' → GIOCATORE (5)
    '+' → GIOCATORE_SU_TARGET (6)

Se i dati Boxoban non sono disponibili, il loader usa livelli di test integrati.
"""

import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from sokoban_env.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
)

# ---------------------------------------------------------------------------
# Mappatura caratteri → valori numerici
# ---------------------------------------------------------------------------

MAPPA_CARATTERI: dict[str, int] = {
    "#": MURO,
    " ": PAVIMENTO,
    "-": PAVIMENTO,   # variante usata in alcuni file Boxoban
    ".": TARGET,
    "$": CASSA,
    "*": CASSA_SU_TARGET,
    "@": GIOCATORE,
    "+": GIOCATORE_SU_TARGET,
}

# Dimensioni standard Boxoban
DIMENSIONE_GRIGLIA = (10, 10)

# ---------------------------------------------------------------------------
# Livelli di test integrati (usati quando Boxoban non è disponibile)
# ---------------------------------------------------------------------------

# Formato: lista di stringhe, ciascuna rappresenta una riga del livello.
# Livello 0 — risolto spingendo la cassa a destra (1 step)
_LIVELLO_TEST_0 = [
    "##########",
    "#        #",
    "# @$.    #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
]

# Livello 1 — richiede 2 mosse
_LIVELLO_TEST_1 = [
    "##########",
    "#        #",
    "#  .     #",
    "#  $     #",
    "#  @     #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
]

# Livello 2 — angolo, richiede pianificazione
_LIVELLO_TEST_2 = [
    "##########",
    "#  .     #",
    "#        #",
    "#  $     #",
    "#  @     #",
    "#        #",
    "#        #",
    "#        #",
    "#        #",
    "##########",
]

LIVELLI_BUILTIN: List[List[str]] = [
    _LIVELLO_TEST_0,
    _LIVELLO_TEST_1,
    _LIVELLO_TEST_2,
]


# ---------------------------------------------------------------------------
# Funzioni di parsing
# ---------------------------------------------------------------------------

def _riga_a_valori(riga: str, larghezza: int) -> Optional[List[int]]:
    """Converte una riga testuale in una lista di valori interi.

    Se la riga contiene caratteri non riconosciuti, restituisce None.
    """
    valori: List[int] = []
    # Padding o troncatura per adattarsi alla larghezza attesa
    riga_normalizzata = riga.ljust(larghezza)[:larghezza]
    for carattere in riga_normalizzata:
        if carattere not in MAPPA_CARATTERI:
            return None
        valori.append(MAPPA_CARATTERI[carattere])
    return valori


def _righe_a_griglia(righe: List[str]) -> Optional[np.ndarray]:
    """Converte una lista di righe testuali in una matrice NumPy.

    Restituisce None se le righe non formano una griglia valida.
    """
    n_righe_attese, n_col_attese = DIMENSIONE_GRIGLIA
    if len(righe) != n_righe_attese:
        return None

    matrice: List[List[int]] = []
    for riga in righe:
        valori = _riga_a_valori(riga, n_col_attese)
        if valori is None:
            return None
        matrice.append(valori)

    griglia = np.array(matrice, dtype=np.int8)

    # Validazione: deve esserci esattamente un giocatore
    n_giocatori = int(np.sum(griglia == GIOCATORE)) + int(
        np.sum(griglia == GIOCATORE_SU_TARGET)
    )
    if n_giocatori != 1:
        return None

    # Deve esserci almeno una cassa e un target
    n_casse = int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))
    n_target = int(np.sum(griglia == TARGET)) + int(np.sum(griglia == CASSA_SU_TARGET))
    if n_casse == 0 or n_target == 0:
        return None

    return griglia


def _analizza_testo_livelli(contenuto: str) -> List[np.ndarray]:
    """Analizza il contenuto testuale di un file Boxoban.

    Restituisce una lista di griglie NumPy valide.
    """
    livelli: List[np.ndarray] = []
    righe_correnti: List[str] = []

    for riga in contenuto.splitlines():
        riga_strip = riga.rstrip()

        # Riga di intestazione/commento → salva il livello accumulato (se valido)
        if riga_strip.startswith(";"):
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        # Riga vuota → fine del blocco livello corrente
        if not riga_strip:
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        righe_correnti.append(riga_strip)

    # Gestione ultimo livello senza riga vuota finale
    if righe_correnti:
        griglia = _righe_a_griglia(righe_correnti)
        if griglia is not None:
            livelli.append(griglia)

    return livelli


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------

class CaricatoreLivelli:
    """Carica livelli Sokoban da file Boxoban o dai livelli integrati.

    Parametri:
        directory_base: percorso alla directory contenente i dati Boxoban
                        (es. 'data/boxoban/'). Se None o assente, usa i
                        livelli integrati.
        difficolta:     'unfiltered', 'medium' o 'hard'.
        split:          'train', 'valid' o 'test'.
        seme:           seme per la randomizzazione.
    """

    def __init__(
        self,
        directory_base: Optional[str] = None,
        difficolta: str = "unfiltered",
        split: str = "train",
        seme: Optional[int] = None,
    ):
        self.difficolta = difficolta
        self.split = split
        self._rng = random.Random(seme)
        self._livelli: List[np.ndarray] = []
        self._caricato = False

        if directory_base is not None:
            self._directory = Path(directory_base) / difficolta / split
        else:
            self._directory = None

    def _carica_se_necessario(self) -> None:
        """Carica i livelli la prima volta che vengono richiesti (lazy load)."""
        if self._caricato:
            return

        if self._directory is not None and self._directory.exists():
            self._livelli = self._carica_da_directory(self._directory)

        if not self._livelli:
            print(
                f"[CaricatoreLivelli] Dati Boxoban non trovati in "
                f"'{self._directory}'. Uso livelli builtin."
            )
            self._livelli = self._carica_livelli_builtin()

        print(f"[CaricatoreLivelli] Caricati {len(self._livelli)} livelli "
              f"({self.difficolta}/{self.split}).")
        self._caricato = True

    @staticmethod
    def _carica_da_directory(directory: Path) -> List[np.ndarray]:
        """Carica tutti i livelli da file .txt in una directory."""
        livelli: List[np.ndarray] = []
        file_txt = sorted(directory.glob("*.txt"))
        for percorso in file_txt:
            try:
                contenuto = percorso.read_text(encoding="utf-8")
                nuovi = _analizza_testo_livelli(contenuto)
                livelli.extend(nuovi)
            except (OSError, UnicodeDecodeError) as e:
                print(f"[CaricatoreLivelli] Errore lettura {percorso}: {e}")
        return livelli

    @staticmethod
    def _carica_livelli_builtin() -> List[np.ndarray]:
        """Converte i livelli testuali integrati in griglie NumPy."""
        livelli: List[np.ndarray] = []
        for righe in LIVELLI_BUILTIN:
            griglia = _righe_a_griglia(righe)
            if griglia is not None:
                livelli.append(griglia)
        return livelli

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        self._carica_se_necessario()
        return len(self._livelli)

    def ottieni(self, indice: int) -> np.ndarray:
        """Restituisce il livello all'indice specificato (copia)."""
        self._carica_se_necessario()
        return self._livelli[indice % len(self._livelli)].copy()

    def casuale(self) -> np.ndarray:
        """Restituisce un livello casuale (copia)."""
        self._carica_se_necessario()
        return self._rng.choice(self._livelli).copy()

    def tutti(self) -> List[np.ndarray]:
        """Restituisce la lista completa delle griglie (copie)."""
        self._carica_se_necessario()
        return [g.copy() for g in self._livelli]
