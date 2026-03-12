"""AG-LLM-REW: PPO addestrato con reward aumentata da LLM.

Il LLM valuta l'azione appena eseguita confrontando la griglia prima e dopo
la mossa. Viene chiamato quando il giocatore era adiacente a una cassa prima
della mossa (~20% degli step). A inference time solo il PPO agisce: il LLM
non e' piu' necessario.

Architettura:
    SokobanEnv -> RicompensaLLM -> Monitor -> PPO

Il wrapper RicompensaLLM intercetta step():
  - Salva la griglia pre-mossa (obs corrente prima di chiamare env.step)
  - Chiama env.step(action) per ottenere la griglia post-mossa
  - Se info['giocatore_adiacente_cassa'] e' True (giocatore era vicino a una cassa):
    - Controlla cache (obs_pre.tobytes(), action) -> score
    - Cache miss: chiede al LLM una valutazione 0-3 confrontando pre e post
    - Normalizza score [-0.5, +0.5], scala per LAMBDA_LLM, aggiunge a reward
  - Altrimenti: reward invariata (~80% degli step, risparmio LLM)

Fix rispetto a versione precedente:
  - Trigger cambiato: solo_push (5%) -> adiacente_cassa (20%) per feedback piu' denso
  - Cache key: (obs_pre, action) invece di solo obs_post per correttezza semantica
  - Prompt: mostra griglia pre+post + nome azione (traccia: "analizzando l'azione eseguita")
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from llm_integration import ClienteLLM, griglia_a_testo, conta_casse
from llm_integration.sokoban_prompt import (
    NOMI_AZIONI,
    crea_prompt_reward,
    parsifica_reward,
)
from experiments.config import MAX_TOKENS_REWARD, TIMEOUT_LLM_SEC, LAMBDA_LLM


class RicompensaLLM(gymnasium.Wrapper):
    """Gymnasium Wrapper che aumenta la reward con valutazione LLM.

    Chiama il LLM quando il giocatore era adiacente a una cassa prima della
    mossa (~20% degli step). Mantiene una cache keyed su (obs_pre, action)
    per evitare chiamate duplicate sulle stesse transizioni.

    Parametri:
        env:            SokobanEnv (obs 2D float32, info['giocatore_adiacente_cassa']).
        client:         ClienteLLM istanziato.
        lambda_llm:     fattore moltiplicativo del segnale LLM (default LAMBDA_LLM).
        solo_adiacente: se True chiama LLM solo quando giocatore era vicino a cassa.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        client: ClienteLLM,
        lambda_llm: float = LAMBDA_LLM,
        solo_adiacente: bool = True,
    ) -> None:
        super().__init__(env)
        self._client = client
        self._lambda = lambda_llm
        self._solo_adiacente = solo_adiacente
        # Cache: (obs_pre bytes, action int) -> score float
        self._cache: Dict[Tuple[bytes, int], float] = {}
        self._n_chiamate_llm: int = 0
        self._n_cache_hit: int = 0
        # Obs corrente (pre-step), aggiornato in reset() e step()
        self._obs_corrente: Optional[np.ndarray] = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Reset: salva obs iniziale per uso come pre-step al primo step()."""
        obs, info = self.env.reset(**kwargs)
        self._obs_corrente = obs.copy()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Esegue lo step e aggiunge segnale LLM se giocatore era adiacente a cassa."""
        # Salva griglia pre-mossa
        obs_pre = self._obs_corrente

        # Esegui la mossa
        obs_post, reward, terminated, truncated, info = self.env.step(action)

        # Aggiorna stato corrente per il prossimo step
        self._obs_corrente = obs_post.copy()

        # Decide se chiamare il LLM
        adiacente = info.get("giocatore_adiacente_cassa", False)
        if not self._solo_adiacente or adiacente:
            cache_key = (obs_pre.tobytes(), int(action))
            if cache_key in self._cache:
                score = self._cache[cache_key]
                self._n_cache_hit += 1
            else:
                casse_tgt, n_casse = conta_casse(obs_post)
                grid_pre  = griglia_a_testo(obs_pre)
                grid_post = griglia_a_testo(obs_post)
                azione_nome = NOMI_AZIONI[int(action)]
                prompt = crea_prompt_reward(
                    grid_text_pre=grid_pre,
                    grid_text_post=grid_post,
                    azione_nome=azione_nome,
                    casse_su_target=casse_tgt,
                    n_casse=n_casse,
                )
                risposta = self._client.chiedi(
                    prompt,
                    max_tokens=MAX_TOKENS_REWARD,
                    timeout=TIMEOUT_LLM_SEC,
                )
                score = parsifica_reward(risposta)
                self._cache[cache_key] = score
                self._n_chiamate_llm += 1

            reward += self._lambda * score

        return obs_post, reward, terminated, truncated, info

    @property
    def statistiche_llm(self) -> Dict[str, int]:
        """Statistiche utilizzo LLM: chiamate, cache hit, dimensione cache."""
        return {
            "n_chiamate": self._n_chiamate_llm,
            "n_cache_hit": self._n_cache_hit,
            "cache_size":  len(self._cache),
        }


class AgenteRicompensaLLM:
    """Factory per RicompensaLLM.

    Crea e configura il wrapper LLM reward.
    Espone avvolgi_env() usato in experiments/train_ppo_llm_rew.py.

    Parametri:
        provider:       provider LLM ("ollama", "groq", "mistral").
        lambda_llm:     fattore di scala del segnale LLM.
        solo_adiacente: True = chiama LLM solo quando giocatore era vicino a cassa.
    """

    def __init__(
        self,
        provider: str = "ollama",
        lambda_llm: float = LAMBDA_LLM,
        solo_adiacente: bool = True,
    ) -> None:
        self.provider = provider
        self.lambda_llm = lambda_llm
        self.solo_adiacente = solo_adiacente
        self._client = ClienteLLM(provider)

    def avvolgi_env(self, env: gymnasium.Env) -> RicompensaLLM:
        """Avvolge SokobanEnv con il wrapper LLM reward."""
        return RicompensaLLM(
            env=env,
            client=self._client,
            lambda_llm=self.lambda_llm,
            solo_adiacente=self.solo_adiacente,
        )
