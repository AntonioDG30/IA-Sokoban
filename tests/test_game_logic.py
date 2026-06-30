# Test unitari per core/ambiente/game_logic.py.
#
# Verifica la logica di gioco stateless: movimento del giocatore, spinta delle casse,
# condizione di vittoria e gestione dei valori di cella (target, giocatore su target). Non
# usa l'ambiente Gymnasium: chiama direttamente le funzioni di game_logic su griglie NumPy
# costruite a mano.
#
# Esegui con: pytest tests/test_game_logic.py -v

import numpy as np
import pytest

from core.ambiente.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE,
    trova_giocatore,
    conta_casse_su_target,
    conta_casse_totali,
    controlla_vittoria,
    applica_mossa,
)


# FIXTURE — GRIGLIE DI TEST

def _griglia_da_testo(righe: list[str]) -> np.ndarray:
    """
    Costruisce una griglia NumPy int8 da una lista di righe ASCII.

    Traduce i simboli Boxoban nei valori numerici tramite MAPPA_CARATTERI del caricatore;
    ogni riga è riempita con spazi (o troncata) fino a DIMENSIONE_GRIGLIA.
    """
    from core.ambiente.level_loader import MAPPA_CARATTERI, DIMENSIONE_GRIGLIA
    n_r, n_c = DIMENSIONE_GRIGLIA
    matrice = []
    for riga in righe:
        # ljust allunga le righe corte, [:n_c] tronca quelle troppo lunghe
        riga_pad = riga.ljust(n_c)[:n_c]
        matrice.append([MAPPA_CARATTERI[c] for c in riga_pad])
    return np.array(matrice, dtype=np.int8)


@pytest.fixture
def griglia_semplice():
    """
    Griglia 10x10 con giocatore in (2,2), cassa in (2,3), target in (2,4).
    Basta una sola azione "destra" (3) per vincere.
    """
    righe = [
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
    return _griglia_da_testo(righe)


@pytest.fixture
def griglia_cassa_su_target():
    """Griglia con la cassa già posizionata sul target (simbolo '*')."""
    righe = [
        "##########",
        "#        #",
        "# @*     #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "##########",
    ]
    return _griglia_da_testo(righe)


@pytest.fixture
def griglia_giocatore_su_target():
    """
    Griglia con il giocatore su un target (simbolo '+').
    Serve a verificare che, quando il giocatore se ne va, la cella torni a TARGET e non resti
    GIOCATORE.
    """
    righe = [
        "##########",
        "#        #",
        "#  +$    #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "#        #",
        "##########",
    ]
    return _griglia_da_testo(righe)


# TEST trova_giocatore

class TestTrovaGiocatore:
    def test_trova_giocatore_normale(self, griglia_semplice):
        """Trova il giocatore su una cella GIOCATORE (valore 5)."""
        riga, col = trova_giocatore(griglia_semplice)
        assert riga == 2
        assert col == 2

    def test_trova_giocatore_su_target(self, griglia_giocatore_su_target):
        """Riconosce anche GIOCATORE_SU_TARGET (valore 6) come posizione valida."""
        riga, col = trova_giocatore(griglia_giocatore_su_target)
        assert riga == 2
        assert col == 3

    def test_giocatore_assente_solleva_errore(self):
        """Una griglia senza giocatore deve sollevare ValueError con un messaggio chiaro."""
        griglia_vuota = np.zeros((10, 10), dtype=np.int8)
        with pytest.raises(ValueError, match="Giocatore non trovato"):
            trova_giocatore(griglia_vuota)


# TEST conta_casse_su_target

class TestContaCasse:
    def test_nessuna_cassa_su_target(self, griglia_semplice):
        """Con la cassa ancora libera il contatore deve essere zero."""
        assert conta_casse_su_target(griglia_semplice) == 0

    def test_una_cassa_su_target(self, griglia_cassa_su_target):
        """Il valore CASSA_SU_TARGET va contato come cassa piazzata."""
        assert conta_casse_su_target(griglia_cassa_su_target) == 1

    def test_conta_totali(self, griglia_semplice):
        """conta_casse_totali conta le casse libere (non su target)."""
        assert conta_casse_totali(griglia_semplice) == 1

    def test_conta_totali_con_cassa_su_target(self, griglia_cassa_su_target):
        # CASSA_SU_TARGET è comunque una cassa: deve essere contata
        assert conta_casse_totali(griglia_cassa_su_target) == 1


# TEST controlla_vittoria

class TestControlllaVittoria:
    def test_non_vinto_cassa_libera(self, griglia_semplice):
        """Con almeno una cassa non sul target la vittoria non è raggiunta."""
        assert controlla_vittoria(griglia_semplice) is False

    def test_vinto_una_cassa_su_target(self):
        """Griglia con solo CASSA_SU_TARGET e nessuna CASSA libera: è vittoria."""
        griglia = np.zeros((10, 10), dtype=np.int8)
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        # Riempie l'interno di pavimento
        griglia[1:9, 1:9] = PAVIMENTO
        griglia[1, 1] = CASSA_SU_TARGET
        griglia[1, 2] = GIOCATORE
        assert controlla_vittoria(griglia) is True

    def test_non_vinto_griglia_vuota(self):
        """Griglia senza casse: nessuna vittoria (non c'è alcun target da soddisfare)."""
        griglia = np.zeros((10, 10), dtype=np.int8)
        assert controlla_vittoria(griglia) is False


# TEST applica_mossa

class TestApplicaMossa:
    def test_movimento_su_pavimento(self, griglia_semplice):
        """Il giocatore si sposta su una cella di pavimento libera."""
        # Giocatore in (2,2), azione "giù" (1): la cella (3,2) è pavimento
        nuova, mossa, cassa = applica_mossa(griglia_semplice, 1)
        assert mossa is True
        assert cassa is False
        assert nuova[3, 2] == GIOCATORE
        assert nuova[2, 2] == PAVIMENTO

    def test_movimento_bloccato_da_muro(self, griglia_semplice):
        """Il giocatore non deve attraversare le celle MURO."""
        griglia = griglia_semplice.copy()
        griglia[2, 2] = PAVIMENTO
        griglia[1, 2] = GIOCATORE   # giocatore adiacente al bordo superiore
        # Azione "su" (0): la cella (0,2) è MURO
        nuova, mossa, cassa = applica_mossa(griglia, 0)
        assert mossa is False
        assert cassa is False
        assert nuova[1, 2] == GIOCATORE   # il giocatore non si è mosso

    def test_spinta_cassa_su_pavimento(self, griglia_semplice):
        """Spingendo la cassa verso una cella libera, giocatore e cassa si spostano insieme."""
        # Configurazione: giocatore (2,2), cassa (2,3), target (2,4).
        # "destra" (3): la cassa va su (2,4)=target -> CASSA_SU_TARGET
        nuova, mossa, cassa = applica_mossa(griglia_semplice, 3)
        assert mossa is True
        assert cassa is True
        assert nuova[2, 3] == GIOCATORE
        assert nuova[2, 4] == CASSA_SU_TARGET
        assert nuova[2, 2] == PAVIMENTO

    def test_spinta_cassa_su_target_porta_a_vittoria(self, griglia_semplice):
        """Dopo aver spinto l'unica cassa sul target, controlla_vittoria deve dare True."""
        nuova, _, _ = applica_mossa(griglia_semplice, 3)
        assert controlla_vittoria(nuova) is True

    def test_cassa_bloccata_da_muro(self):
        """Non si può spingere una cassa contro un muro."""
        griglia = np.ones((10, 10), dtype=np.int8) * PAVIMENTO
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        # Cassa in (1,8): la cella (1,9) è MURO, quindi non è spingibile
        griglia[1, 7] = GIOCATORE
        griglia[1, 8] = CASSA
        nuova, mossa, cassa = applica_mossa(griglia, 3)
        assert mossa is False
        assert cassa is False
        assert nuova[1, 7] == GIOCATORE

    def test_cassa_bloccata_da_altra_cassa(self):
        """Una cassa non è spingibile se un'altra cassa la blocca."""
        griglia = np.ones((10, 10), dtype=np.int8) * PAVIMENTO
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        griglia[1, 1] = GIOCATORE
        griglia[1, 2] = CASSA
        griglia[1, 3] = CASSA   # la seconda cassa blocca la prima
        nuova, mossa, cassa = applica_mossa(griglia, 3)
        assert mossa is False

    def test_azione_non_valida_solleva_errore(self, griglia_semplice):
        """Un indice di azione fuori da {0,1,2,3} deve sollevare ValueError."""
        with pytest.raises(ValueError):
            applica_mossa(griglia_semplice, 99)

    def test_giocatore_su_target_lascia_target(self, griglia_giocatore_su_target):
        """Quando il giocatore lascia un target, la cella deve tornare a TARGET."""
        # Giocatore_su_target in (2,3), azione "su" (0): il giocatore va in (1,3)
        nuova, mossa, _ = applica_mossa(griglia_giocatore_su_target, 0)
        assert mossa is True
        assert nuova[2, 3] == TARGET      # il target non scompare
        assert nuova[1, 3] == GIOCATORE   # il giocatore si è spostato
