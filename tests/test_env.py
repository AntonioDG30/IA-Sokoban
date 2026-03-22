"""Test unitari per sokoban_env/sokoban_gym.py.

Verifica il comportamento dell'ambiente Gymnasium: reset, step, spazi
di azione/osservazione e ciclo episodico completo.
I livelli builtin vengono usati ovunque: non serve il dataset Boxoban.

Esegui con: pytest tests/test_env.py -v
"""

import numpy as np
import pytest

from sokoban_env import SokobanEnv


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Crea un ambiente headless (no Pygame) per i test automatici.

    render_mode=None evita l'inizializzazione di Pygame, che richiederebbe
    un display fisico e rallenterebbe l'esecuzione in CI.
    """
    ambiente = SokobanEnv(render_mode=None, seme=42)
    yield ambiente
    ambiente.close()


@pytest.fixture
def env_con_indice(env):
    """Ambiente pre-resettato sul livello 0 (builtin, 1 cassa, vincibile in 1 mossa)."""
    env.reset(options={"indice_livello": 0})
    return env


# ---------------------------------------------------------------------------
# Test reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restituisce_osservazione_e_info(self, env):
        """reset() deve restituire (obs ndarray float32 10x10, info dict)."""
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10, 10)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_step_azzerato(self, env):
        """Dopo ogni reset il contatore step_corrente torna a 0."""
        env.reset()
        env.step(0)
        env.step(1)
        # Un secondo reset deve azzerare il contatore
        _, info = env.reset()
        assert info["step_corrente"] == 0

    def test_reset_con_indice_specifico(self, env):
        """Due reset sullo stesso indice devono produrre la griglia identica."""
        obs1, _ = env.reset(options={"indice_livello": 0})
        obs2, _ = env.reset(options={"indice_livello": 0})
        np.testing.assert_array_equal(obs1, obs2)

    def test_osservazione_valori_nel_range(self, env):
        """I valori di osservazione devono stare in [0, 6].

        7 e' il valore di PADDING usato internamente per griglie piu' piccole,
        ma non deve mai comparire nell'osservazione esportata all'agente.
        """
        obs, _ = env.reset()
        assert obs.min() >= 0
        assert obs.max() <= 6


# ---------------------------------------------------------------------------
# Test step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_restituisce_struttura_corretta(self, env):
        """step() deve restituire una tupla di 5 elementi con i tipi corretti."""
        env.reset()
        risultato = env.step(0)
        assert len(risultato) == 5
        obs, reward, terminated, truncated, info = risultato
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10, 10)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_penalita_per_mossa(self, env):
        """Un movimento senza push deve dare esattamente -0.005 (penalita' step)."""
        env.reset(options={"indice_livello": 0})
        # Azione su: il giocatore si sposta su pavimento libero, nessuna cassa
        _, reward, terminated, _, _ = env.step(0)
        assert not terminated
        assert reward == pytest.approx(-0.005, abs=1e-6)

    def test_step_vittoria_reward_positiva(self, env):
        """Livello 0: giocatore-cassa-target allineati. Destra porta alla vittoria."""
        env.reset(options={"indice_livello": 0})
        # Nel livello 0: giocatore in (2,2), cassa in (2,3), target in (2,4)
        # Una sola azione destra (3) spinge la cassa sul target
        _, reward, terminated, _, _ = env.step(3)
        assert terminated is True
        # reward = +10 (completamento) + 1 (cassa su target) - 0.005 (step) = 10.995
        assert reward == pytest.approx(10.995, abs=1e-6)

    def test_step_truncation_dopo_max_step(self):
        """Dopo max_step senza vittoria l'episodio finisce con truncated=True."""
        env = SokobanEnv(max_step=3, render_mode=None, seme=42)
        env.reset(options={"indice_livello": 2})
        truncated = False
        # Azione "giu'" ripetuta non porta alla vittoria: garantisce la truncation
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(1)
            if terminated or truncated:
                break
        env.close()
        assert truncated is True

    def test_step_non_terminato_se_casse_non_tutte_su_target(self, env):
        """Un singolo step su livello con piu' casse non basta per vincere."""
        env.reset(options={"indice_livello": 2})
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False

    def test_info_contiene_step_corrente(self, env):
        """info['step_corrente'] deve essere 1 dopo il primo step."""
        env.reset()
        _, _, _, _, info = env.step(1)
        assert "step_corrente" in info
        assert info["step_corrente"] == 1

    def test_info_contiene_casse_su_target(self, env):
        """info deve sempre includere il campo 'casse_su_target' come int."""
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "casse_su_target" in info
        assert isinstance(info["casse_su_target"], int)

    def test_step_senza_reset_solleva_errore(self):
        """Chiamare step() prima di reset() deve sollevare RuntimeError."""
        env = SokobanEnv(render_mode=None)
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(0)
        env.close()


# ---------------------------------------------------------------------------
# Test spazi Gymnasium
# ---------------------------------------------------------------------------

class TestSpazi:
    def test_action_space_discrete4(self, env):
        """Lo spazio azioni deve essere Discrete(4): su, giu, sinistra, destra."""
        assert env.action_space.n == 4

    def test_observation_space_box(self, env):
        """Observation space: Box float32, shape (10,10), valori in [0, 7].

        float32 e' obbligatorio per SB3/PyTorch.
        7 e' PADDING (usato per griglie piu' piccole di 10x10).
        """
        assert env.observation_space.shape == (10, 10)
        assert env.observation_space.dtype == np.float32
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 7.0

    def test_azioni_valide_nel_action_space(self, env):
        """Tutti e quattro gli interi 0-3 devono essere nel action_space."""
        for azione in range(4):
            assert env.action_space.contains(azione)

    def test_osservazione_nel_observation_space(self, env):
        """L'osservazione prodotta da reset() deve stare nell'observation_space."""
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Test agente random (sanity check)
# ---------------------------------------------------------------------------

class TestAgenteRandom:
    def test_episodio_completo_agente_random(self):
        """Un episodio completo con azioni casuali non deve causare eccezioni."""
        env = SokobanEnv(render_mode=None, seme=0)
        obs, info = env.reset()
        totale_reward = 0.0
        n_step = 0
        for _ in range(200):
            azione = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(azione)
            totale_reward += reward
            n_step += 1
            if terminated or truncated:
                break
        env.close()
        assert n_step > 0
        assert isinstance(totale_reward, float)

    def test_piu_episodi_consecutivi(self):
        """Cinque reset consecutivi non devono causare errori o perdite di stato."""
        env = SokobanEnv(render_mode=None, seme=1)
        for _ in range(5):
            obs, _ = env.reset()
            assert obs.shape == (10, 10)
            for _ in range(10):
                obs, _, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                if terminated or truncated:
                    break
        env.close()
