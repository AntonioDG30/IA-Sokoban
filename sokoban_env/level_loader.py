"""Caricamento e parsing dei livelli dal dataset DeepMind Boxoban.

Il dataset Boxoban usa un formato testuale con livelli separati da righe
di commento ('; level N') e righe vuote. Ogni livello e' una griglia 10x10
di caratteri ASCII. Questo modulo converte quei caratteri nei valori numerici
usati da game_logic.py.

Formato file Boxoban:
    ; level 0
    ##########
    #.  $   @#
    ...
    ##########

    ; level 1
    ...

Mappatura caratteri -> valori numerici:
    '#' -> MURO (0)
    ' ' -> PAVIMENTO (1)
    '.' -> TARGET (2)
    '$' -> CASSA (3)
    '*' -> CASSA_SU_TARGET (4)
    '@' -> GIOCATORE (5)
    '+' -> GIOCATORE_SU_TARGET (6)

Se i dati Boxoban non sono disponibili, il caricatore usa tre livelli
di test semplici integrati nel codice come fallback.
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
# Mappatura caratteri -> valori numerici
# ---------------------------------------------------------------------------

MAPPA_CARATTERI: dict[str, int] = {
    "#": MURO,
    " ": PAVIMENTO,
    "-": PAVIMENTO,   # variante usata in alcuni file Boxoban al posto dello spazio
    ".": TARGET,
    "$": CASSA,
    "*": CASSA_SU_TARGET,
    "@": GIOCATORE,
    "+": GIOCATORE_SU_TARGET,
}

# Dimensioni standard di tutti i livelli Boxoban: 10 righe x 10 colonne
DIMENSIONE_GRIGLIA = (10, 10)

# ---------------------------------------------------------------------------
# Livelli builtin: usati come fallback se Boxoban non e' disponibile
# ---------------------------------------------------------------------------

# Livello 0: una sola cassa, risolvibile in un singolo step (spingi a destra)
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

# Livello 1: una cassa sopra al giocatore, richiede due mosse
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

# Livello 2: cassa e target distanziati, richiede pianificazione minima
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
    """Converte una riga di testo Boxoban in una lista di valori interi.

    Adatta la riga alla larghezza attesa tramite padding o troncatura.
    Restituisce None se la riga contiene caratteri non riconosciuti.

    Parametri:
        riga:     stringa di una riga del livello Boxoban.
        larghezza: numero di colonne atteso (10 per Boxoban standard).
    """
    valori: List[int] = []
    # Garantisce la larghezza corretta: padding con spazio o troncatura
    riga_normalizzata = riga.ljust(larghezza)[:larghezza]
    for carattere in riga_normalizzata:
        if carattere not in MAPPA_CARATTERI:
            return None
        valori.append(MAPPA_CARATTERI[carattere])
    return valori


def _righe_a_griglia(righe: List[str]) -> Optional[np.ndarray]:
    """Converte una lista di righe testuali in una matrice NumPy validata.

    Verifica che il livello abbia esattamente un giocatore e almeno una cassa
    e un target, altrimenti scarta il livello.

    Parametri:
        righe: lista di stringhe, una per riga del livello.

    Restituisce:
        Matrice NumPy (10, 10) dtype int8, oppure None se il livello non e' valido.
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

    # Deve esserci esattamente un giocatore nel livello
    n_giocatori = int(np.sum(griglia == GIOCATORE)) + int(
        np.sum(griglia == GIOCATORE_SU_TARGET)
    )
    if n_giocatori != 1:
        return None

    # Deve esserci almeno una cassa e almeno un target
    n_casse  = int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))
    n_target = int(np.sum(griglia == TARGET)) + int(np.sum(griglia == CASSA_SU_TARGET))
    if n_casse == 0 or n_target == 0:
        return None

    return griglia


def _analizza_testo_livelli(contenuto: str) -> List[np.ndarray]:
    """Analizza il contenuto testuale di un file Boxoban e restituisce i livelli.

    Scorre il file riga per riga accumulando le righe di ciascun livello.
    Un livello e' terminato da una riga vuota o da una riga di intestazione ';'.

    Parametri:
        contenuto: stringa con il contenuto completo del file.

    Restituisce:
        Lista di griglie NumPy (10, 10) valide.
    """
    livelli: List[np.ndarray] = []
    righe_correnti: List[str] = []

    for riga in contenuto.splitlines():
        riga_strip = riga.rstrip()

        # Riga di intestazione '; level N': salva il livello accumulato e ricomincia
        if riga_strip.startswith(";"):
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        # Riga vuota: fine del blocco corrente
        if not riga_strip:
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        righe_correnti.append(riga_strip)

    # Gestisce l'ultimo livello se il file non termina con riga vuota
    if righe_correnti:
        griglia = _righe_a_griglia(righe_correnti)
        if griglia is not None:
            livelli.append(griglia)

    return livelli


# ---------------------------------------------------------------------------
# Classe principale per il caricamento
# ---------------------------------------------------------------------------

class CaricatoreLivelli:
    """Carica livelli Sokoban da file Boxoban o dai livelli integrati.

    Il caricamento e' lazy: i file vengono letti solo alla prima richiesta,
    non al costruttore. Questo evita di caricare centinaia di migliaia di
    livelli se l'ambiente viene istanziato ma non usato subito.

    Parametri:
        directory_base: percorso a data/boxoban/ (es. 'data/boxoban/'). Se None
                        o non esistente, usa i livelli builtin come fallback.
        difficolta:     'unfiltered', 'medium' o 'hard'.
        split:          'train', 'valid' o 'test'.
        seme:           seed per la selezione casuale dei livelli.
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
        self._caricato = False   # flag lazy loading

        if directory_base is not None:
            # Costruisce il percorso: data/boxoban/<difficolta>/<split>/
            self._directory = Path(directory_base) / difficolta / split
        else:
            self._directory = None

    def _carica_se_necessario(self) -> None:
        """Carica i livelli la prima volta che vengono richiesti.

        Tenta prima di caricare da directory Boxoban; se non disponibile
        o vuota, usa i livelli builtin come fallback.
        """
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
        """Legge tutti i file .txt nella directory e ne estrae i livelli.

        I file vengono ordinati per nome per garantire un ordine deterministico
        indipendentemente dal filesystem.
        """
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
        """Converte i livelli testuali hardcoded in griglie NumPy.

        Usato solo come fallback quando Boxoban non e' disponibile.
        """
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
        """Restituisce il numero totale di livelli disponibili."""
        self._carica_se_necessario()
        return len(self._livelli)

    def ottieni(self, indice: int) -> np.ndarray:
        """Restituisce il livello all'indice specificato (copia, non reference).

        L'indice viene calcolato in modulo per evitare IndexError.
        """
        self._carica_se_necessario()
        return self._livelli[indice % len(self._livelli)].copy()

    def casuale(self) -> np.ndarray:
        """Restituisce un livello scelto casualmente (copia)."""
        self._carica_se_necessario()
        return self._rng.choice(self._livelli).copy()

    def tutti(self) -> List[np.ndarray]:
        """Restituisce tutte le griglie come lista di copie."""
        self._carica_se_necessario()
        return [g.copy() for g in self._livelli]
