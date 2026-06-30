# Wrapper Gymnasium che aggiunge la dimensione del canale all'osservazione.
#
# SokobanEnv restituisce osservazioni di forma (H, W) float32, ma le policy CNN di
# Stable Baselines 3 si aspettano tensori channels-first (C, H, W). Questo wrapper
# inserisce il canale in posizione 0, trasformando (H, W) in (1, H, W).
#
# Posizione tipica nella catena di wrapping del training:
#   env = SokobanEnv(...)
#   env = AggiuntaCanale(env)     # (10,10) -> (1,10,10)
#   env = Monitor(env)            # raccoglie le statistiche episodiche
#   env = VecEnv(env)             # parallelizzazione (solo PPO)

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AggiuntaCanale(gym.ObservationWrapper):
    """
    Trasforma l'osservazione da (H, W) a (1, H, W) float32.

    Avvolge un ambiente con observation_space 2D e ne aggiorna lo spazio in (1, H, W),
    così la CNN lo riceve nel formato channels-first atteso. Solleva ValueError se
    l'observation space avvolto non è 2D.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old = env.observation_space

        # Il wrapper ha senso solo su osservazioni 2D (H, W): ogni altra forma è un errore
        if len(old.shape) != 2:
            raise ValueError(
                f"AggiuntaCanale si aspetta obs 2D (H, W), "
                f"ricevuto shape {old.shape}"
            )

        h, w = old.shape

        # Nuovo observation space con il canale in testa: (1, H, W).
        # high.max() conserva il valore massimo del vecchio spazio (7 con padding, 6 senza)
        self.observation_space = spaces.Box(
            low=np.zeros((1, h, w), dtype=np.float32),
            high=np.full((1, h, w), float(old.high.max()), dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Aggiunge la dimensione di canale in testa: (H, W) -> (1, H, W).
        Gymnasium la chiama in automatico a ogni reset() e step().
        """
        return obs[np.newaxis, ...].astype(np.float32)   # np.newaxis inserisce l'asse del canale
