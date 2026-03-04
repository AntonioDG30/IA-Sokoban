"""Test unitari per sokoban_env/game_logic.py.

Esegui con: pytest tests/test_game_logic.py -v
"""

import numpy as np
import pytest

from sokoban_env.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA, CASSA_SU_TARGET,
    GIOCATORE, GIOCATORE_SU_TARGET,
    trova_giocatore,
    conta_casse_su_target,
    conta_casse_totali,
    controlla_vittoria,
    applica_mossa,
)


# ---------------------------------------------------------------------------
# Fixture — griglie di test
# ---------------------------------------------------------------------------

def _griglia_da_testo(righe: list[str]) -> np.ndarray:
    """Converte una lista di stringhe in griglia NumPy (solo per test)."""
    from sokoban_env.level_loader import MAPPA_CARATTERI, DIMENSIONE_GRIGLIA
    n_r, n_c = DIMENSIONE_GRIGLIA
    matrice = []
    for riga in righe:
        riga_pad = riga.ljust(n_c)[:n_c]
        matrice.append([MAPPA_CARATTERI[c] for c in riga_pad])
    return np.array(matrice, dtype=np.int8)


@pytest.fixture
def griglia_semplice():
    """
    Griglia 10×10 con 1 cassa, 1 target, giocatore a sinistra della cassa.
    Per vincere: azione destra (3).

    ##########
    #        #
    # @$.    #
    #        #
    ...
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
    """Griglia con una cassa già posizionata sul target."""
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
    """Griglia con il giocatore su un target."""
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


# ---------------------------------------------------------------------------
# Test trova_giocatore
# ---------------------------------------------------------------------------

class TestTrovaGiocatore:
    def test_trova_giocatore_normale(self, griglia_semplice):
        riga, col = trova_giocatore(griglia_semplice)
        assert riga == 2
        assert col == 2

    def test_trova_giocatore_su_target(self, griglia_giocatore_su_target):
        riga, col = trova_giocatore(griglia_giocatore_su_target)
        assert riga == 2
        assert col == 3

    def test_giocatore_assente_solleva_errore(self):
        griglia_vuota = np.zeros((10, 10), dtype=np.int8)
        with pytest.raises(ValueError, match="Giocatore non trovato"):
            trova_giocatore(griglia_vuota)


# ---------------------------------------------------------------------------
# Test conta_casse_su_target
# ---------------------------------------------------------------------------

class TestContaCasse:
    def test_nessuna_cassa_su_target(self, griglia_semplice):
        assert conta_casse_su_target(griglia_semplice) == 0

    def test_una_cassa_su_target(self, griglia_cassa_su_target):
        assert conta_casse_su_target(griglia_cassa_su_target) == 1

    def test_conta_totali(self, griglia_semplice):
        assert conta_casse_totali(griglia_semplice) == 1

    def test_conta_totali_con_cassa_su_target(self, griglia_cassa_su_target):
        # CASSA_SU_TARGET conta come cassa totale
        assert conta_casse_totali(griglia_cassa_su_target) == 1


# ---------------------------------------------------------------------------
# Test controlla_vittoria
# ---------------------------------------------------------------------------

class TestControlllaVittoria:
    def test_non_vinto_cassa_libera(self, griglia_semplice):
        assert controlla_vittoria(griglia_semplice) is False

    def test_vinto_una_cassa_su_target(self):
        """Griglia con solo CASSA_SU_TARGET (nessuna CASSA libera)."""
        griglia = np.zeros((10, 10), dtype=np.int8)
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        # Riempi l'interno di pavimento
        griglia[1:9, 1:9] = PAVIMENTO
        griglia[1, 1] = CASSA_SU_TARGET
        griglia[1, 2] = GIOCATORE
        assert controlla_vittoria(griglia) is True

    def test_non_vinto_griglia_vuota(self):
        griglia = np.zeros((10, 10), dtype=np.int8)
        assert controlla_vittoria(griglia) is False


# ---------------------------------------------------------------------------
# Test applica_mossa
# ---------------------------------------------------------------------------

class TestApplicaMossa:
    def test_movimento_su_pavimento(self, griglia_semplice):
        """Il giocatore si sposta su una cella di pavimento libera."""
        # Giocatore in (2,2), azione giù → (3,2) è pavimento
        nuova, mossa, cassa = applica_mossa(griglia_semplice, 1)  # giù
        assert mossa is True
        assert cassa is False
        assert nuova[3, 2] == GIOCATORE
        assert nuova[2, 2] == PAVIMENTO

    def test_movimento_bloccato_da_muro(self, griglia_semplice):
        """Il giocatore non può attraversare un muro."""
        # Giocatore in (2,2), azione sinistra → (2,1) è pavimento, ok
        # Ma azione su → (1,2) è pavimento, ok
        # Per testare il muro: dobbiamo spostare il giocatore vicino al bordo
        griglia = griglia_semplice.copy()
        griglia[2, 2] = PAVIMENTO
        griglia[1, 2] = GIOCATORE  # giocatore in (1,2)
        # azione su → (0,2) è MURO
        nuova, mossa, cassa = applica_mossa(griglia, 0)  # su
        assert mossa is False
        assert cassa is False
        assert nuova[1, 2] == GIOCATORE  # non si è mosso

    def test_spinta_cassa_su_pavimento(self, griglia_semplice):
        """Il giocatore spinge la cassa su pavimento."""
        # Griglia: @(2,2) $(2,3) .(2,4)
        # Azione destra: cassa va in (2,4)=target → CASSA_SU_TARGET
        nuova, mossa, cassa = applica_mossa(griglia_semplice, 3)  # destra
        assert mossa is True
        assert cassa is True
        assert nuova[2, 3] == GIOCATORE
        assert nuova[2, 4] == CASSA_SU_TARGET
        assert nuova[2, 2] == PAVIMENTO

    def test_spinta_cassa_su_target_porta_a_vittoria(self, griglia_semplice):
        """Spingere la cassa sul target soddisfa la condizione di vittoria."""
        nuova, _, _ = applica_mossa(griglia_semplice, 3)  # destra
        assert controlla_vittoria(nuova) is True

    def test_cassa_bloccata_da_muro(self, griglia_semplice):
        """La cassa non può essere spinta contro un muro."""
        # Giocatore in (2,2), cassa in (2,3).
        # Se il giocatore si sposta a sinistra, poi su, poi 2 volte destra...
        # Costruiamo una situazione dove la cassa è contro il muro destro.
        griglia = np.ones((10, 10), dtype=np.int8) * PAVIMENTO
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        griglia[1, 7] = GIOCATORE
        griglia[1, 8] = CASSA   # cassa contro muro destro in (1,9)=MURO
        nuova, mossa, cassa = applica_mossa(griglia, 3)  # destra
        assert mossa is False
        assert cassa is False
        assert nuova[1, 7] == GIOCATORE

    def test_cassa_bloccata_da_altra_cassa(self):
        """La cassa non può essere spinta se un'altra cassa la blocca."""
        griglia = np.ones((10, 10), dtype=np.int8) * PAVIMENTO
        griglia[0, :] = MURO
        griglia[9, :] = MURO
        griglia[:, 0] = MURO
        griglia[:, 9] = MURO
        griglia[1, 1] = GIOCATORE
        griglia[1, 2] = CASSA
        griglia[1, 3] = CASSA   # seconda cassa blocca la prima
        nuova, mossa, cassa = applica_mossa(griglia, 3)  # destra
        assert mossa is False

    def test_azione_non_valida_solleva_errore(self, griglia_semplice):
        with pytest.raises(ValueError):
            applica_mossa(griglia_semplice, 99)

    def test_giocatore_su_target_lascia_target(self, griglia_giocatore_su_target):
        """Quando il giocatore lascia un target, la cella torna a TARGET."""
        # Giocatore_su_target in (2,3), cassa in (2,4)
        # Azione su → giocatore va in (1,3)
        nuova, mossa, _ = applica_mossa(griglia_giocatore_su_target, 0)  # su
        assert mossa is True
        assert nuova[2, 3] == TARGET       # target rimane
        assert nuova[1, 3] == GIOCATORE   # giocatore si sposta
