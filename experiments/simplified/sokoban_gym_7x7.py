"""Ambiente Sokoban 7x7 con observation space nativo (7,7).

DIFFERENZE rispetto a sokoban_gym.py (10x10 Boxoban):
    - observation_space: Box(0,6,(7,7),float32) -- NATIVO, nessun padding
    - Usa GeneratoreLivelli.genera(7,7,n_casse) -- griglia (7,7) nativa
    - Nessun padding_a_10x10() -- zero offset CNN
    - n_casse e max_step come parametri costruttore
    - Nessuna dipendenza da data/boxoban/

ZERO modifiche a sokoban_env/ o a qualunque altro file esistente.

Importa da sokoban_env:
    - game_logic: applica_mossa, controlla_vittoria, conta_casse_su_target
    - level_generator: GeneratoreLivelli
    - reward: calcola_reward
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sokoban_env.game_logic import (
    applica_mossa,
    controlla_vittoria,
    conta_casse_su_target,
)
from sokoban_env.level_generator import GeneratoreLivelli
from sokoban_env.reward import calcola_reward


class SokobanEnv7x7(gymnasium.Env):
    """Ambiente Sokoban 7x7 con observation space nativo (7,7).

    Observation space: Box(0.0, 6.0, shape=(7,7), dtype=float32)
    Action space:      Discrete(4) -- 0=su, 1=giu, 2=sinistra, 3=destra

    Generatore procedurale: crea livelli validi e risolvibili ogni reset().
    Usa GeneratoreLivelli.genera(7,7,n_casse) -- nessun Boxoban dataset.
    """

    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        n_casse: int = 1,
        max_step: int = 100,
        seme: Optional[int] = None,
    ) -> None:
        """Inizializza l'ambiente 7x7 con generatore procedurale.

        Parametri:
            n_casse:  numero di casse da piazzare nel livello generato
            max_step: passi massimi per episodio prima del truncated
            seme:     seed per il generatore di livelli (None = casuale)
        """
        super().__init__()

        self.n_casse = n_casse
        self.max_step = max_step

        # Observation space: nativo 7x7, valori 0-6 (nessun padding)
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=6.0, shape=(7, 7), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(4)

        self._generatore = GeneratoreLivelli(seme=seme)
        self._griglia: Optional[np.ndarray] = None
        self._step_corrente: int = 0
        self.np_random: np.random.Generator = np.random.default_rng(seme)

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Genera un nuovo livello 7x7 e resetta il contatore step.

        Restituisce:
            (obs float32 (7,7), info dict con metriche iniziali)
        """
        super().reset(seed=seed)
        # Genera griglia 7x7 nativa -- nessun padding
        self._griglia = self._generatore.genera(7, 7, self.n_casse)
        self._step_corrente = 0
        info = self._info(mossa_eseguita=False, cassa_spostata=False)
        return self._griglia.astype(np.float32), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applica un'azione e restituisce la transizione (obs, reward, terminated, truncated, info).

        Parametri:
            action: intero in {0,1,2,3} -- su/giu/sinistra/destra
        Restituisce:
            tupla standard Gymnasium (obs, reward, terminated, truncated, info)
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
        """Rendering non implementato per questo ambiente headless."""
        return None

    def close(self) -> None:
        """Chiude l'ambiente; nessuna risorsa da rilasciare."""
        pass

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _info(self, mossa_eseguita: bool, cassa_spostata: bool) -> Dict[str, Any]:
        """Costruisce il dizionario info da restituire con reset() e step().

        Parametri:
            mossa_eseguita: True se il giocatore si e' effettivamente mosso
            cassa_spostata: True se almeno una cassa e' stata spostata in questo step
        Restituisce:
            dizionario con step_corrente, casse_su_target, mossa_eseguita, cassa_spostata
        """
        if self._griglia is None:
            return {}
        return {
            "step_corrente":   self._step_corrente,
            "casse_su_target": conta_casse_su_target(self._griglia),
            "mossa_eseguita":  mossa_eseguita,
            "cassa_spostata":  cassa_spostata,
        }

    @property
    def griglia(self) -> Optional[np.ndarray]:
        """Vista della griglia corrente (7x7 nativa, dtype int8)."""
        if self._griglia is None:
            return None
        return self._griglia.copy()
