# Estrattore di feature CNN personalizzato per le griglie Sokoban.
#
# Estende BaseFeaturesExtractor di SB3 per essere iniettato nelle policy CNN tramite
# policy_kwargs. Funziona su qualsiasi dimensione di griglia: la dimensione del layer
# flatten viene calcolata automaticamente con una forward pass su un tensore fittizio
# alla costruzione, quindi non c'è nessun valore hardcoded da aggiornare a mano.
#
# Architettura per input (1, 10, 10):
#   Conv(1->32, 3x3, stride=1, pad=1)   ->  (32, 10, 10)  -> ReLU
#   Conv(32->64, 3x3, stride=2, pad=1)  ->  (64,  5,  5)  -> ReLU
#   Conv(64->64, 3x3, stride=1, pad=1)  ->  (64,  5,  5)  -> ReLU
#   Flatten                              ->  1600
#   Linear(1600 -> features_dim)         ->  256           -> ReLU
#
# Per input (1, 7, 7) il flatten produce 1024 invece di 1600: il layer Linear si adatta
# da solo grazie al calcolo dinamico. I valori di cella [0, 7] vengono normalizzati
# dividendo per 7.0, così l'input della CNN cade nel range [0, 1].

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SokobanCNN(BaseFeaturesExtractor):
    """
    CNN compatta che estrae un vettore di feature da un'osservazione (1, H, W) float32.

    features_dim è la dimensione del vettore restituito (default 256), cioè l'input che
    le head di policy e value di SB3 ricevono a valle dell'estrattore.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Canali in ingresso: 1 per le griglie Sokoban (un solo "colore", non immagini RGB)
        n_canali = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Layer 1: estrae pattern locali mantenendo la risoluzione (stride=1, padding=1)
            nn.Conv2d(n_canali, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Layer 2: dimezza la risoluzione spaziale (stride=2) e raddoppia i canali
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: raffina le feature senza cambiare le dimensioni
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcola la dimensione del flatten passando un tensore fittizio di zeri nella CNN:
        # evita di hardcodare 1600 (10x10) o 1024 (7x7) e rende la rete indipendente dalla griglia
        with th.no_grad():
            sample   = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        # Layer Linear finale: comprime le feature convolutive nei features_dim di output
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Normalizza l'input e calcola il vettore di feature.

        I valori di cella stanno in [0, 7]: dividendo per 7.0 cadono in [0, 1], range più
        adatto all'addestramento (il 7 è il padding artificiale, il bordo non di gioco).
        Riceve un tensore (batch, 1, H, W) e restituisce (batch, features_dim).
        """
        # Normalizzazione [0, 7] -> [0, 1]
        normalized = observations / 7.0
        return self.linear(self.cnn(normalized))
