"""Estrattore di features CNN personalizzato per griglie Sokoban 10x10.

Architettura compatta a 3 layer convoluzionali progettata per input (1, 10, 10).
I valori di cella sono in [0, 7]: celle di gioco 0-6, padding artificiale = 7.
Normalizzazione: /7.0 → range [0, 1].

Il padding artificiale (7) e' distinto dai muri reali (0), permettendo alla CNN
di imparare filtri che ignorano il bordo e generalizzano tra griglie di diverse
dimensioni durante il curriculum learning (Opzione B).

Flusso dati:
    (1, 10, 10) → /7.0 (normalizzazione)
               → Conv(1→32, 3x3) → ReLU
               → Conv(32→64, 3x3, stride=2) → ReLU   [output: (64, 5, 5)]
               → Conv(64→64, 3x3) → ReLU              [output: (64, 5, 5)]
               → Flatten (1600)
               → Linear(1600 → features_dim) → ReLU
               → (features_dim,)                       [default: 256]
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SokobanCNN(BaseFeaturesExtractor):
    """CNN per osservazioni Sokoban (1, H, W) float32.

    Args:
        observation_space: spazio osservazioni con shape (1, H, W), valori in [0, 6].
        features_dim:      dimensione del vettore di output (default 256).
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        n_canali = observation_space.shape[0]  # 1

        self.cnn = nn.Sequential(
            # Layer 1: (1, 10, 10) → (32, 10, 10)
            nn.Conv2d(n_canali, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: (32, 10, 10) → (64, 5, 5)  [stride=2 dimezza le dimensioni]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: (64, 5, 5) → (64, 5, 5)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcola dimensione flatten automaticamente
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]  # 64 * 5 * 5 = 1600 per input 10x10

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Normalizza [0,7]->]0,1] poi estrae features CNN.

        Il valore 7 corrisponde al padding artificiale (bordo non di gioco).
        Dividendo per 7.0 tutti i valori cadono in [0, 1].
        """
        normalized = observations / 7.0
        return self.linear(self.cnn(normalized))
