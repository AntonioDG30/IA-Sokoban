"""Estrattore di feature CNN personalizzato per griglie Sokoban.

Estende BaseFeaturesExtractor di SB3 per essere iniettato nelle policy CNN
tramite policy_kwargs. Funziona su qualsiasi dimensione di griglia: la
dimensione del layer flatten viene calcolata automaticamente con una forward
pass su un tensore dummy al momento della costruzione.

Architettura per input (1, 10, 10):
    Conv(1->32, 3x3, stride=1, pad=1)  ->  (32, 10, 10)  -> ReLU
    Conv(32->64, 3x3, stride=2, pad=1) ->  (64,  5,  5)  -> ReLU
    Conv(64->64, 3x3, stride=1, pad=1) ->  (64,  5,  5)  -> ReLU
    Flatten                             ->  1600
    Linear(1600 -> features_dim)        ->  256            -> ReLU

Per input (1, 7, 7) il flatten produce 1024 invece di 1600; il linear
layer si adatta automaticamente grazie al calcolo dinamico.

I valori di cella [0, 7] vengono normalizzati dividendo per 7.0 per
portare l'input nell CNN nel range [0, 1].
"""

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SokobanCNN(BaseFeaturesExtractor):
    """CNN compatta per osservazioni Sokoban (1, H, W) float32.

    Parametri:
        observation_space: spazio osservazioni con shape (1, H, W).
        features_dim:      dimensione del vettore di output. Default 256.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Numero di canali in ingresso (1 per le griglie Sokoban: un solo canale)
        n_canali = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Layer 1: mantiene le dimensioni spaziali (stride=1, padding=1)
            nn.Conv2d(n_canali, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: dimezza le dimensioni spaziali (stride=2) aumentando i canali
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: raffina le feature senza cambiare le dimensioni
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcola la dimensione del flatten automaticamente con un tensore dummy.
        # Questo evita di hardcodare 1600 (10x10) o 1024 (7x7): la stessa CNN
        # funziona su qualsiasi dimensione di griglia senza modifiche al codice
        with th.no_grad():
            sample   = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        # Linear finale: porta le feature CNN alla dimensione di output desiderata
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Normalizza l'input e calcola il vettore di feature.

        I valori di cella sono in [0, 7]: dividendo per 7.0 cadono in [0, 1],
        range piu' adatto per l'addestramento della rete neurale.
        Il valore 7 corrisponde al padding artificiale (bordo non di gioco).

        Parametri:
            observations: tensore (batch, 1, H, W) con valori in [0, 7].

        Restituisce:
            Tensore (batch, features_dim) con le feature estratte.
        """
        # Normalizzazione [0,7] -> [0,1]
        normalized = observations / 7.0
        return self.linear(self.cnn(normalized))
