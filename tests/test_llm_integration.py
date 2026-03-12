"""Test unitari per llm_integration.

Testa le funzioni pure (griglia_a_testo, crea_prompt, parsifica_azione, conta_casse)
senza richiedere connettivita' LLM.

I test di ClienteLLM.chiedi() vengono saltati se Ollama non e' attivo (skip automatico).
"""

import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

import numpy as np
import pytest

from llm_integration.sokoban_prompt import (
    conta_casse,
    crea_prompt,
    griglia_a_testo,
    parsifica_azione,
    SIMBOLI,
)


# ---------------------------------------------------------------------------
# Fixture: griglie di test
# ---------------------------------------------------------------------------

@pytest.fixture
def griglia_vuota() -> np.ndarray:
    """Griglia 10x10 tutta muri (valore 0)."""
    return np.zeros((10, 10), dtype=np.float32)


@pytest.fixture
def griglia_semplice() -> np.ndarray:
    """Griglia con un layout minimo: giocatore, cassa, target."""
    g = np.zeros((10, 10), dtype=np.float32)
    g[1:-1, 1:-1] = 1.0   # pavimento interno
    g[5, 5] = 5.0          # giocatore
    g[3, 3] = 3.0          # cassa
    g[2, 2] = 2.0          # target
    return g


@pytest.fixture
def griglia_con_casse_su_target() -> np.ndarray:
    """Griglia con 2 casse su target (CASSA_SU_TARGET=4) e 1 cassa libera."""
    g = np.ones((10, 10), dtype=np.float32)
    g[5, 5] = 5.0   # giocatore
    g[3, 3] = 4.0   # cassa su target
    g[4, 4] = 4.0   # cassa su target
    g[6, 6] = 3.0   # cassa libera
    return g


# ---------------------------------------------------------------------------
# Test griglia_a_testo
# ---------------------------------------------------------------------------

class TestGrigliaATesto:
    def test_restituisce_10_righe(self, griglia_semplice):
        testo = griglia_a_testo(griglia_semplice)
        righe = testo.split("\n")
        assert len(righe) == 10, f"Attese 10 righe, ottenute {len(righe)}"

    def test_ogni_riga_ha_10_caratteri(self, griglia_semplice):
        testo = griglia_a_testo(griglia_semplice)
        for i, riga in enumerate(testo.split("\n")):
            assert len(riga) == 10, f"Riga {i}: attesi 10 char, ottenuti {len(riga)}"

    def test_simboli_corretti(self, griglia_semplice):
        testo = griglia_a_testo(griglia_semplice)
        righe = testo.split("\n")
        # Giocatore in (5,5)
        assert righe[5][5] == SIMBOLI[5], "Giocatore '@' non trovato in (5,5)"
        # Cassa in (3,3)
        assert righe[3][3] == SIMBOLI[3], "Cassa '$' non trovata in (3,3)"
        # Target in (2,2)
        assert righe[2][2] == SIMBOLI[2], "Target '.' non trovato in (2,2)"
        # Muro in (0,0)
        assert righe[0][0] == SIMBOLI[0], "Muro '#' non trovato in (0,0)"

    def test_griglia_vuota(self, griglia_vuota):
        testo = griglia_a_testo(griglia_vuota)
        # Tutti i caratteri devono essere '#' (muro=0)
        for riga in testo.split("\n"):
            assert all(c == SIMBOLI[0] for c in riga)

# ---------------------------------------------------------------------------
# Test crea_prompt
# ---------------------------------------------------------------------------

class TestCreaPrompt:
    def test_ritorna_stringa_non_vuota(self, griglia_semplice):
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=0, n_casse=1)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contiene_griglia(self, griglia_semplice):
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=0, n_casse=1)
        # Il testo della griglia deve essere nel prompt
        for riga in txt.split("\n"):
            assert riga in prompt, f"Riga griglia non trovata nel prompt: {riga!r}"

    def test_contiene_info_casse(self, griglia_semplice):
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=2, n_casse=4)
        assert "2" in prompt, "Info casse su target non nel prompt"
        assert "4" in prompt, "Info n_casse non nel prompt"

    def test_contiene_azioni_valide(self, griglia_semplice):
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=0, n_casse=1)
        for azione in ("su", "giu", "sinistra", "destra"):
            assert azione in prompt.lower(), f"Azione '{azione}' non nel prompt"


# ---------------------------------------------------------------------------
# Test parsifica_azione
# ---------------------------------------------------------------------------

class TestParsificaAzione:
    @pytest.mark.parametrize("testo,atteso", [
        ("su", 0),
        ("SU", 0),
        ("  su  ", 0),
        ("giu", 1),
        ("GIU", 1),
        ("giu'", 1),
        ("sinistra", 2),
        ("SINISTRA", 2),
        ("destra", 3),
        ("DESTRA", 3),
    ])
    def test_parole_valide(self, testo, atteso):
        assert parsifica_azione(testo) == atteso

    def test_con_punteggiatura(self):
        # "su!" -> "su " dopo re.sub -> match "su" -> 0
        assert parsifica_azione("su!") == 0
        assert parsifica_azione("destra.") == 3

    def test_fallback_su_testo_invalido(self, monkeypatch):
        """Testo sconosciuto deve restituire un intero in {0,1,2,3}."""
        risultato = parsifica_azione("blablabla_sconosciuto")
        assert risultato in (0, 1, 2, 3), f"Fallback non in {{0,1,2,3}}: {risultato}"

    def test_stringa_vuota_fallback(self):
        risultato = parsifica_azione("")
        assert risultato in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Test conta_casse
# ---------------------------------------------------------------------------

class TestContaCasse:
    def test_nessuna_cassa(self, griglia_vuota):
        su_t, tot = conta_casse(griglia_vuota)
        assert su_t == 0
        assert tot == 0

    def test_cassa_su_target(self, griglia_con_casse_su_target):
        su_t, tot = conta_casse(griglia_con_casse_su_target)
        assert su_t == 2, f"Attese 2 casse su target, ottenute {su_t}"
        assert tot == 3, f"Attese 3 casse totali, ottenute {tot}"

    def test_solo_casse_libere(self, griglia_semplice):
        su_t, tot = conta_casse(griglia_semplice)
        assert su_t == 0
        assert tot == 1

    def test_griglia_piena_casse(self):
        g = np.full((10, 10), fill_value=3.0, dtype=np.float32)  # tutto casse
        su_t, tot = conta_casse(g)
        assert su_t == 0
        assert tot == 100


# ---------------------------------------------------------------------------
# Test ClienteLLM (skip se Ollama non attivo)
# ---------------------------------------------------------------------------

def _ollama_attivo() -> bool:
    """Verifica se Ollama e' raggiungibile (senza dipendenza da openai)."""
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_attivo(), reason="Ollama non attivo su localhost:11434")
class TestClienteLLM:
    def test_istanziazione_ollama(self):
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        assert c is not None

    def test_chiedi_risposta_stringa(self):
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        risposta = c.chiedi("Rispondi con una sola parola: ciao", max_tokens=5, timeout=15.0)
        assert isinstance(risposta, str)

    def test_chiedi_timeout_non_crasha(self):
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        # Timeout molto breve -> potrebbe fallire -> deve restituire "" non lanciare
        risposta = c.chiedi("Elenca tutti i numeri da 1 a 1000", max_tokens=200, timeout=0.01)
        assert isinstance(risposta, str)  # "" se timeout
