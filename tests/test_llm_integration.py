"""Test unitari per llm_integration.

Copre le funzioni pure di sokoban_prompt (griglia_a_testo, crea_prompt,
parsifica_azione, conta_casse) senza richiedere connettivita' LLM.

I test di ClienteLLM.chiedi() vengono saltati automaticamente se Ollama
non e' attivo su localhost:11434, cosi' la suite rimane verde anche
in ambienti senza GPU o senza il server locale.

Esegui con: pytest tests/test_llm_integration.py -v
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
    """Griglia 10x10 tutta muri (valore 0), usata per testare i casi limite."""
    return np.zeros((10, 10), dtype=np.float32)


@pytest.fixture
def griglia_semplice() -> np.ndarray:
    """Griglia con un layout minimo: pavimento interno, giocatore, cassa e target.

    Il pavimento occupa la zona interna (bordo escluso) mentre le celle
    di interesse vengono piazzate in posizioni note per facilitare le asserzioni.
    """
    g = np.zeros((10, 10), dtype=np.float32)
    g[1:-1, 1:-1] = 1.0   # pavimento interno (bordo resta muro)
    g[5, 5] = 5.0          # giocatore
    g[3, 3] = 3.0          # cassa libera
    g[2, 2] = 2.0          # target vuoto
    return g


@pytest.fixture
def griglia_con_casse_su_target() -> np.ndarray:
    """Griglia con 2 casse gia' su target (valore 4) e 1 cassa ancora libera.

    Utile per verificare che conta_casse distingua correttamente
    le casse posizionate da quelle ancora da spostare.
    """
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
        """La griglia 10x10 deve produrre esattamente 10 righe di testo."""
        testo = griglia_a_testo(griglia_semplice)
        righe = testo.split("\n")
        assert len(righe) == 10, f"Attese 10 righe, ottenute {len(righe)}"

    def test_ogni_riga_ha_10_caratteri(self, griglia_semplice):
        """Ogni riga deve avere esattamente 10 caratteri (uno per cella)."""
        testo = griglia_a_testo(griglia_semplice)
        for i, riga in enumerate(testo.split("\n")):
            assert len(riga) == 10, f"Riga {i}: attesi 10 char, ottenuti {len(riga)}"

    def test_simboli_corretti(self, griglia_semplice):
        """Le celle significative devono corrispondere ai simboli attesi in SIMBOLI."""
        testo = griglia_a_testo(griglia_semplice)
        righe = testo.split("\n")
        # Giocatore in (5,5)
        assert righe[5][5] == SIMBOLI[5], "Giocatore '@' non trovato in (5,5)"
        # Cassa in (3,3)
        assert righe[3][3] == SIMBOLI[3], "Cassa '$' non trovata in (3,3)"
        # Target in (2,2)
        assert righe[2][2] == SIMBOLI[2], "Target '.' non trovato in (2,2)"
        # Muro in (0,0) -- il bordo deve rimanere muro
        assert righe[0][0] == SIMBOLI[0], "Muro '#' non trovato in (0,0)"

    def test_griglia_vuota(self, griglia_vuota):
        """Una griglia tutta-zero deve produrre solo caratteri muro."""
        testo = griglia_a_testo(griglia_vuota)
        for riga in testo.split("\n"):
            assert all(c == SIMBOLI[0] for c in riga)


# ---------------------------------------------------------------------------
# Test crea_prompt
# ---------------------------------------------------------------------------

class TestCreaPrompt:
    def test_ritorna_stringa_non_vuota(self, griglia_semplice):
        """Il prompt deve essere una stringa non vuota."""
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=0, n_casse=1)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_contiene_griglia(self, griglia_semplice):
        """Il testo della griglia deve essere incluso nel prompt per contesto visivo."""
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=0, n_casse=1)
        for riga in txt.split("\n"):
            assert riga in prompt, f"Riga griglia non trovata nel prompt: {riga!r}"

    def test_contiene_info_casse(self, griglia_semplice):
        """I numeri di casse su target e totali devono comparire nel prompt."""
        txt = griglia_a_testo(griglia_semplice)
        prompt = crea_prompt(txt, casse_su_target=2, n_casse=4)
        assert "2" in prompt, "Info casse su target non nel prompt"
        assert "4" in prompt, "Info n_casse non nel prompt"

    def test_contiene_azioni_valide(self, griglia_semplice):
        """Il prompt deve elencare le 4 azioni valide cosi' il LLM sa cosa rispondere."""
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
        ("  su  ", 0),      # spazi iniziali/finali
        ("giu", 1),
        ("GIU", 1),
        ("giu'", 1),        # apostrofo tipico del LLM italiano
        ("sinistra", 2),
        ("SINISTRA", 2),
        ("destra", 3),
        ("DESTRA", 3),
    ])
    def test_parole_valide(self, testo, atteso):
        """Tutte le varianti attese del vocabolario devono mappare sull'azione corretta."""
        assert parsifica_azione(testo) == atteso

    def test_con_punteggiatura(self):
        """La punteggiatura finale non deve impedire il riconoscimento dell'azione."""
        assert parsifica_azione("su!") == 0
        assert parsifica_azione("destra.") == 3

    def test_fallback_su_testo_invalido(self, monkeypatch):
        """Testo sconosciuto deve restituire un intero valido in {0,1,2,3}.

        Il fallback casuale e' accettabile: l'importante e' non crashare
        e non restituire valori fuori dallo spazio delle azioni.
        """
        risultato = parsifica_azione("blablabla_sconosciuto")
        assert risultato in (0, 1, 2, 3), f"Fallback non in {{0,1,2,3}}: {risultato}"

    def test_stringa_vuota_fallback(self):
        """La stringa vuota e' un caso limite che non deve sollevare eccezioni."""
        risultato = parsifica_azione("")
        assert risultato in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# Test conta_casse
# ---------------------------------------------------------------------------

class TestContaCasse:
    def test_nessuna_cassa(self, griglia_vuota):
        """Su una griglia tutta-muri non ci sono casse di alcun tipo."""
        su_t, tot = conta_casse(griglia_vuota)
        assert su_t == 0
        assert tot == 0

    def test_cassa_su_target(self, griglia_con_casse_su_target):
        """Deve contare separatamente le casse su target (4) e le libere (3)."""
        su_t, tot = conta_casse(griglia_con_casse_su_target)
        assert su_t == 2, f"Attese 2 casse su target, ottenute {su_t}"
        assert tot == 3, f"Attese 3 casse totali, ottenute {tot}"

    def test_solo_casse_libere(self, griglia_semplice):
        """Griglia con 1 sola cassa libera: su_target deve essere 0."""
        su_t, tot = conta_casse(griglia_semplice)
        assert su_t == 0
        assert tot == 1

    def test_griglia_piena_casse(self):
        """Caso estremo: 100 casse libere, nessuna su target."""
        g = np.full((10, 10), fill_value=3.0, dtype=np.float32)
        su_t, tot = conta_casse(g)
        assert su_t == 0
        assert tot == 100


# ---------------------------------------------------------------------------
# Test ClienteLLM (skip se Ollama non attivo)
# ---------------------------------------------------------------------------

def _ollama_attivo() -> bool:
    """Controlla se il server Ollama risponde su localhost:11434.

    Usa urllib puro per non aggiungere dipendenze a runtime.
    Il timeout breve (2s) evita attese lunghe in CI dove Ollama non gira.
    """
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_attivo(), reason="Ollama non attivo su localhost:11434")
class TestClienteLLM:
    def test_istanziazione_ollama(self):
        """ClienteLLM('ollama') deve istanziarsi senza errori se Ollama e' attivo."""
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        assert c is not None

    def test_chiedi_risposta_stringa(self):
        """Una richiesta semplice deve restituire una stringa non None."""
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        risposta = c.chiedi("Rispondi con una sola parola: ciao", max_tokens=5, timeout=15.0)
        assert isinstance(risposta, str)

    def test_chiedi_timeout_non_crasha(self):
        """Con timeout quasi zero il client deve restituire stringa vuota, non crashare.

        Il contratto e' che ClienteLLM gestisca internamente i timeout
        restituendo "" anziche' propagare l'eccezione al chiamante.
        """
        from llm_integration import ClienteLLM
        c = ClienteLLM("ollama")
        risposta = c.chiedi("Elenca tutti i numeri da 1 a 1000", max_tokens=200, timeout=0.01)
        assert isinstance(risposta, str)  # "" se timeout, mai None
