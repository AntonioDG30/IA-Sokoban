"""Test unitari per sokoban_env/sokoban_gym.py.

Esegui con: pytest tests/test_env.py -v

Nota: i test usano i livelli builtin (non richiedono il dataset Boxoban).
"""

import numpy as np
import pytest

from sokoban_env import SokobanEnv


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    """Ambiente con render_mode=None (headless, per CI/CD)."""
    ambiente = SokobanEnv(render_mode=None, seme=42)
    yield ambiente
    ambiente.close()


@pytest.fixture
def env_con_indice(env):
    """Ambiente resettato con il livello 0 (builtin, facilmente vincibile)."""
    env.reset(options={"indice_livello": 0})
    return env


# ---------------------------------------------------------------------------
# Test reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restituisce_osservazione_e_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (10, 10)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_reset_step_azzerato(self, env):
        env.reset()
        env.step(0)
        env.step(1)
        env.reset()
        _, info = env.reset()
        assert info["step_corrente"] == 0

    def test_reset_con_indice_specifico(self, env):
        obs1, _ = env.reset(options={"indice_livello": 0})
        obs2, _ = env.reset(options={"indice_livello": 0})
        np.testing.assert_array_equal(obs1, obs2)

    def test_osservazione_valori_nel_range(self, env):
        obs, _ = env.reset()
        assert obs.min() >= 0
        assert obs.max() <= 6


# ---------------------------------------------------------------------------
# Test step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_restituisce_struttura_corretta(self, env):
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
        env.reset(options={"indice_livello": 0})
        # Prima azione: su — il giocatore si sposta senza spingere la cassa
        _, reward, terminated, _, _ = env.step(0)
        # Reward deve essere -0.005 (nessun cambio casse su target, no vittoria)
        assert not terminated
        assert reward == pytest.approx(-0.005, abs=1e-6)

    def test_step_vittoria_reward_positiva(self, env):
        """Livello 0: @$.  ->  azione destra -> cassa su target -> vittoria."""
        env.reset(options={"indice_livello": 0})
        # Nel livello 0: giocatore in (2,2), cassa in (2,3), target in (2,4)
        # azione destra (3) spinge cassa su target
        _, reward, terminated, _, _ = env.step(3)  # destra
        assert terminated is True
        # reward = +10 (completamento) + 1 (cassa su target) - 0.005 (step) = 10.995
        assert reward == pytest.approx(10.995, abs=1e-6)

    def test_step_truncation_dopo_max_step(self):
        """L'episodio viene troncato dopo max_step senza vittoria."""
        env = SokobanEnv(max_step=3, render_mode=None, seme=42)
        env.reset(options={"indice_livello": 2})  # livello non banale
        truncated = False
        # Azione "giù" (1): allontana il giocatore dalla cassa → nessuna vittoria
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(1)  # sempre giù
            if terminated or truncated:
                break
        env.close()
        assert truncated is True

    def test_step_non_terminato_se_casse_non_tutte_su_target(self, env):
        env.reset(options={"indice_livello": 2})
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False

    def test_info_contiene_step_corrente(self, env):
        env.reset()
        _, _, _, _, info = env.step(1)
        assert "step_corrente" in info
        assert info["step_corrente"] == 1

    def test_info_contiene_casse_su_target(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "casse_su_target" in info
        assert isinstance(info["casse_su_target"], int)

    def test_step_senza_reset_solleva_errore(self):
        env = SokobanEnv(render_mode=None)
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(0)
        env.close()


# ---------------------------------------------------------------------------
# Test spazi Gymnasium
# ---------------------------------------------------------------------------

class TestSpazi:
    def test_action_space_discrete4(self, env):
        assert env.action_space.n == 4

    def test_observation_space_box(self, env):
        assert env.observation_space.shape == (10, 10)
        assert env.observation_space.dtype == np.float32  # float32 per SB3/PyTorch
        assert env.observation_space.low.min() == 0.0
        assert env.observation_space.high.max() == 7.0  # PADDING=7 e' il valore massimo

    def test_azioni_valide_nel_action_space(self, env):
        for azione in range(4):
            assert env.action_space.contains(azione)

    def test_osservazione_nel_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


# ---------------------------------------------------------------------------
# Test agente random (sanity check)
# ---------------------------------------------------------------------------

class TestAgenteRandom:
    def test_episodio_completo_agente_random(self):
        """Verifica che l'ambiente funzioni per un episodio completo senza crash."""
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
        """Reset multipli non causano errori."""
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
