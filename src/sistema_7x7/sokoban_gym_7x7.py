# Ambiente Sokoban 7x7 con observation space nativo (7,7).
#
# Differenze rispetto a sokoban_gym.py (il 10x10 Boxoban):
#   - observation_space Box(0,6,(7,7),float32) NATIVO, senza padding;
#   - usa GeneratoreLivelli.genera(7,7,n_casse) su griglia (7,7) nativa;
#   - niente padding_a_10x10(), quindi zero offset per la CNN;
#   - n_casse e max_step sono parametri del costruttore;
#   - nessuna dipendenza da dataset/boxoban/.
#
# Non tocca nessun file di core/ambiente/: riusa game_logic (applica_mossa, controlla_vittoria,
# conta_casse_su_target), level_generator (GeneratoreLivelli) e reward (calcola_reward).

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from core.ambiente.game_logic import (
    applica_mossa,
    controlla_vittoria,
    conta_casse_su_target,
)
from core.ambiente.level_generator import GeneratoreLivelli
from core.ambiente.reward import calcola_reward


class SokobanEnv7x7(gymnasium.Env):
    """
    Ambiente Sokoban 7x7 con observation space nativo (7,7).

    Observation space: Box(0.0, 6.0, shape=(7,7), dtype=float32); action space: Discrete(4)
    (0=su, 1=giù, 2=sinistra, 3=destra). A ogni reset() genera un livello valido e risolvibile
    con GeneratoreLivelli.genera(7,7,n_casse), senza usare nessun dataset Boxoban.
    """

    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        n_casse: int = 1,
        max_step: int = 100,
        seme: Optional[int] = None,
    ) -> None:
        """
        Inizializza l'ambiente 7x7 con il generatore procedurale.
        n_casse è il numero di casse del livello, max_step il limite di passi prima del
        truncated, seme il seed del generatore (None = casuale).
        """
        super().__init__()

        self.n_casse = n_casse
        self.max_step = max_step

        # Observation space nativo 7x7, valori 0-6 (nessun padding, quindi high=6.0)
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=6.0, shape=(7, 7), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(4)

        self._generatore = GeneratoreLivelli(seme=seme)
        self._griglia: Optional[np.ndarray] = None
        self._step_corrente: int = 0
        self.np_random: np.random.Generator = np.random.default_rng(seme)

    # API GYMNASIUM

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Genera un nuovo livello 7x7, azzera il contatore di step e restituisce
        (osservazione float32 (7,7), info dict).
        """
        super().reset(seed=seed)
        # Genera la griglia 7x7 nativa (nessun padding)
        self._griglia = self._generatore.genera(7, 7, self.n_casse)
        self._step_corrente = 0
        info = self._info(mossa_eseguita=False, cassa_spostata=False)
        return self._griglia.astype(np.float32), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Applica un'azione (0..3) e restituisce la transizione standard Gymnasium
        (obs, reward, terminated, truncated, info). Il reward shaping è attivo con gli stessi
        pesi del 10x10 (scala_manhattan=0.3, scala_player_box=0.1).
        """
        if self._griglia is None:
            raise RuntimeError("Chiamare reset() prima di step().")

        griglia_pre = self._griglia.copy()
        nuova_griglia, mossa_eseguita, cassa_spostata = applica_mossa(
            self._griglia, int(action)
        )
        self._griglia = nuova_griglia
        self._step_corrente += 1

        terminated = controlla_vittoria(self._griglia)
        truncated = (self._step_corrente >= self.max_step) and not terminated
        reward = calcola_reward(griglia_pre, self._griglia, terminated,
                               scala_manhattan=0.3, scala_player_box=0.1)

        info = self._info(
            mossa_eseguita=mossa_eseguita, cassa_spostata=cassa_spostata
        )
        return self._griglia.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        """Rendering non implementato: questo ambiente è headless."""
        return None

    def close(self) -> None:
        """Chiude l'ambiente; non ci sono risorse da rilasciare."""
        pass

    # HELPER

    def _info(self, mossa_eseguita: bool, cassa_spostata: bool) -> Dict[str, Any]:
        """
        Costruisce il dizionario info restituito da reset() e step(): step_corrente,
        casse_su_target, mossa_eseguita e cassa_spostata.
        """
        if self._griglia is None:
            return {}
        return {
            "step_corrente":   self._step_corrente,
            "casse_su_target": conta_casse_su_target(self._griglia),
            "mossa_eseguita":  mossa_eseguita,
            "cassa_spostata":  cassa_spostata,
        }
