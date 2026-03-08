"""Wrapper gymnasium che aggiunge la dimensione canale all'osservazione.

Trasforma (10, 10) float32 → (1, 10, 10) float32 (channels-first)
per compatibilita' con CnnPolicy di Stable Baselines 3.

Con la CNN l'agente impara filtri convoluzionali invarianti alla traslazione
(es. "cassa vicina a target → spingi"), che si generalizzano tra griglie
di dimensione diversa (5x5 → 7x7 → 10x10 nel curriculum).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AggiuntaCanale(gym.ObservationWrapper):
    """Aggiunge dimensione canale: (H, W) float32 → (1, H, W) float32.

    Uso:
        env = SokobanEnv(griglia_size=(5, 5), n_casse=1)
        env = AggiuntaCanale(env)
        # env.observation_space.shape == (1, 10, 10)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space
        if len(old.shape) != 2:
            raise ValueError(
                f"AggiuntaCanale si aspetta obs 2D (H, W), "
                f"ricevuto shape {old.shape}"
            )
        h, w = old.shape
        self.observation_space = spaces.Box(
            low=np.zeros((1, h, w), dtype=np.float32),
            high=np.full((1, h, w), float(old.high.max()), dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """(H, W) → (1, H, W)."""
        return obs[np.newaxis, ...].astype(np.float32)
