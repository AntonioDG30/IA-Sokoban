"""Wrapper Gymnasium che aggiunge la dimensione canale all'osservazione.

SokobanEnv restituisce osservazioni di forma (H, W) float32. Le policy CNN
di Stable Baselines 3 si aspettano tensori channels-first (C, H, W). Questo
wrapper inserisce il canale in posizione 0, trasformando (H, W) in (1, H, W).

Utilizzo tipico nella catena di wrapping per il training:
    env = SokobanEnv(...)
    env = AggiuntaCanale(env)     # (10,10) -> (1,10,10)
    env = Monitor(env)            # raccoglie statistiche episodiche
    env = VecEnv(env)             # parallelizzazione (solo PPO)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AggiuntaCanale(gym.ObservationWrapper):
    """Trasforma l'osservazione da (H, W) a (1, H, W) float32.

    Parametri:
        env: ambiente Gymnasium con observation_space 2D (H, W).

    Solleva ValueError se l'observation space non e' 2D.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space

        # Verifica che l'obs space sia 2D: il wrapper ha senso solo in questo caso
        if len(old.shape) != 2:
            raise ValueError(
                f"AggiuntaCanale si aspetta obs 2D (H, W), "
                f"ricevuto shape {old.shape}"
            )

        h, w = old.shape

        # Aggiorna l'observation space per riflettere la nuova forma (1, H, W)
        self.observation_space = spaces.Box(
            low=np.zeros((1, h, w), dtype=np.float32),
            high=np.full((1, h, w), float(old.high.max()), dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Aggiunge la dimensione canale in posizione 0: (H, W) -> (1, H, W).

        Parametri:
            obs: array float32 di forma (H, W).

        Restituisce:
            Array float32 di forma (1, H, W).
        """
        return obs[np.newaxis, ...].astype(np.float32)
