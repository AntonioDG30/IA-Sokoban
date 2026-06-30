# Caricamento e parsing dei livelli dal dataset DeepMind Boxoban.
#
# Il dataset Boxoban usa un formato testuale: i livelli sono separati da righe di
# intestazione ('; level N') e da righe vuote, e ogni livello è una griglia 10x10 di
# caratteri ASCII. Questo modulo traduce quei caratteri nei valori numerici di game_logic.
#
# Formato di un file Boxoban:
#   ; level 0
#   ##########
#   #.  $   @#
#   ...
#   ##########
#
#   ; level 1
#   ...
#
# Mappatura caratteri -> valori numerici:
#   '#' -> MURO (0)          '$' -> CASSA (3)             '@' -> GIOCATORE (5)
#   ' ' -> PAVIMENTO (1)     '*' -> CASSA_SU_TARGET (4)   '+' -> GIOCATORE_SU_TARGET (6)
#   '.' -> TARGET (2)
#
# Se i dati Boxoban non sono disponibili, il caricatore ripiega su tre livelli di test
# semplici, definiti direttamente nel codice.

import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.ambiente.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
)

# MAPPATURA CARATTERI -> VALORI NUMERICI
MAPPA_CARATTERI: dict[str, int] = {
    "#": MURO,
    " ": PAVIMENTO,
    "-": PAVIMENTO,   # alcuni file Boxoban usano '-' al posto dello spazio per il pavimento
    ".": TARGET,
    "$": CASSA,
    "*": CASSA_SU_TARGET,
    "@": GIOCATORE,
    "+": GIOCATORE_SU_TARGET,
}

# Dimensione standard di tutti i livelli Boxoban: 10 righe x 10 colonne.
DIMENSIONE_GRIGLIA = (10, 10)


# LIVELLI BUILTIN: FALLBACK QUANDO BOXOBAN NON È DISPONIBILE

# Livello 0: una sola cassa, risolvibile in un singolo step (basta spingere a destra).
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

# Livello 1: cassa sopra al giocatore, richiede due mosse.
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

# Livello 2: cassa e target più distanziati, richiede una minima pianificazione.
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


# FUNZIONI DI PARSING

def _riga_a_valori(riga: str, larghezza: int) -> Optional[List[int]]:
    """
    Converte una riga di testo Boxoban nella lista dei suoi valori interi.

    Normalizza la riga alla larghezza attesa (riempie con spazi o tronca) e traduce ogni
    carattere tramite MAPPA_CARATTERI. Restituisce None appena incontra un carattere non
    riconosciuto, così l'intero livello verrà scartato a monte.
    """
    valori: List[int] = []
    # Forza la larghezza esatta: riempie di spazi a destra e poi tronca
    riga_normalizzata = riga.ljust(larghezza)[:larghezza]
    for carattere in riga_normalizzata:
        if carattere not in MAPPA_CARATTERI:
            return None
        valori.append(MAPPA_CARATTERI[carattere])
    return valori


def _righe_a_griglia(righe: List[str]) -> Optional[np.ndarray]:
    """
    Converte una lista di righe testuali in una matrice NumPy (10, 10) int8, validandola.

    Restituisce None (scartando il livello) se il numero di righe non è quello atteso, o
    se non sono rispettati i vincoli minimi: esattamente un giocatore, almeno una cassa e
    almeno un target.
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

    # Vincolo 1: deve esserci esattamente un giocatore (su pavimento o su target)
    n_giocatori = int(np.sum(griglia == GIOCATORE)) + int(
        np.sum(griglia == GIOCATORE_SU_TARGET)
    )
    if n_giocatori != 1:
        return None

    # Vincolo 2: almeno una cassa e almeno un target, altrimenti il livello è degenere
    n_casse  = int(np.sum(griglia == CASSA)) + int(np.sum(griglia == CASSA_SU_TARGET))
    n_target = int(np.sum(griglia == TARGET)) + int(np.sum(griglia == CASSA_SU_TARGET))
    if n_casse == 0 or n_target == 0:
        return None

    return griglia


def _analizza_testo_livelli(contenuto: str) -> List[np.ndarray]:
    """
    Estrae tutte le griglie valide dal contenuto testuale di un file Boxoban.

    Scorre il file riga per riga accumulando le righe del livello corrente; un livello si
    chiude su una riga vuota o su una riga di intestazione ';'. Le griglie che non superano
    la validazione di _righe_a_griglia vengono semplicemente saltate.
    """
    livelli: List[np.ndarray] = []
    righe_correnti: List[str] = []

    for riga in contenuto.splitlines():
        riga_strip = riga.rstrip()

        # Intestazione '; level N': chiude il livello accumulato e ne apre uno nuovo
        if riga_strip.startswith(";"):
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        # Riga vuota: anch'essa chiude il blocco corrente
        if not riga_strip:
            if righe_correnti:
                griglia = _righe_a_griglia(righe_correnti)
                if griglia is not None:
                    livelli.append(griglia)
                righe_correnti = []
            continue

        righe_correnti.append(riga_strip)

    # Ultimo livello del file, se non è seguito da una riga vuota di chiusura
    if righe_correnti:
        griglia = _righe_a_griglia(righe_correnti)
        if griglia is not None:
            livelli.append(griglia)

    return livelli


# CLASSE PRINCIPALE PER IL CARICAMENTO

class CaricatoreLivelli:
    """
    Carica livelli Sokoban dai file Boxoban oppure dai livelli builtin di fallback.

    Il caricamento è lazy: i file vengono letti solo alla prima richiesta, non nel
    costruttore, per non leggere centinaia di migliaia di livelli se l'ambiente viene
    creato ma non subito usato. directory_base punta a dataset/boxoban/ (None o percorso
    inesistente -> fallback builtin); difficolta e split scelgono la sottocartella; seme
    rende deterministica la scelta casuale dei livelli.
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
        self._caricato = False   # flag del lazy loading: True dopo il primo caricamento

        if directory_base is not None:
            # Percorso completo della cartella: dataset/boxoban/<difficolta>/<split>/
            self._directory = Path(directory_base) / difficolta / split
        else:
            self._directory = None

    def _carica_se_necessario(self) -> None:
        """
        Carica i livelli alla prima richiesta (alle chiamate successive non fa nulla).

        Prova prima a leggere dalla directory Boxoban; se manca o è vuota ripiega sui
        livelli builtin, così l'ambiente resta utilizzabile anche senza dataset.
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
        """
        Legge tutti i file .txt della directory e ne estrae i livelli.

        I file vengono ordinati per nome così l'ordine di caricamento è deterministico e
        indipendente dal filesystem. Un errore di lettura su un file viene segnalato e
        ignorato, senza interrompere il caricamento degli altri.
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
        """Converte i tre livelli testuali hardcoded in griglie NumPy (solo come fallback)."""
        livelli: List[np.ndarray] = []
        for righe in LIVELLI_BUILTIN:
            griglia = _righe_a_griglia(righe)
            if griglia is not None:
                livelli.append(griglia)
        return livelli

    # API PUBBLICA

    def __len__(self) -> int:
        """Numero totale di livelli disponibili (forza il caricamento se non ancora fatto)."""
        self._carica_se_necessario()
        return len(self._livelli)

    def ottieni(self, indice: int) -> np.ndarray:
        """
        Restituisce una copia del livello all'indice dato.
        L'indice è preso in modulo sul numero di livelli, quindi non solleva mai IndexError.
        """
        self._carica_se_necessario()
        return self._livelli[indice % len(self._livelli)].copy()

    def casuale(self) -> np.ndarray:
        """Restituisce una copia di un livello scelto a caso (con l'RNG seminato)."""
        self._carica_se_necessario()
        return self._rng.choice(self._livelli).copy()
