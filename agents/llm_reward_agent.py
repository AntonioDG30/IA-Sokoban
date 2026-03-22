"""Agente AG-LLM-REW: PPO addestrato con reward arricchita da valutazione LLM.

Il LLM valuta la qualita' di ogni spinta di cassa confrontando la griglia
prima e dopo la mossa. Il giudizio viene scalato e aggiunto alla reward
standard dell'ambiente. A inference time solo il PPO agisce: il LLM
non e' piu' necessario.

Il wrapper RicompensaLLM intercetta ogni chiamata a step() e:
    1. Salva la griglia pre-mossa (obs corrente prima della chiamata).
    2. Esegue env.step(action) per ottenere la griglia post-mossa.
    3. Se una cassa e' stata spostata (info['cassa_spostata']=True):
       - Controlla la cache (obs_pre, action) -> score gia' calcolato.
       - Cache miss: chiede al LLM un punteggio 0-3 sul confronto pre/post.
       - Aggiunge lambda_llm * score alla reward originale.
    4. Per tutti gli altri step (~95%): reward invariata, nessuna chiamata LLM.

Usare solo push effettivi come trigger riduce le chiamate LLM di circa 20x
rispetto a chiamare il LLM ad ogni step, rendendo il training fattibile.
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
    """Wrapper Gymnasium che aggiunge la valutazione LLM alla reward.

    Intercetta step() e, quando una cassa viene spostata, chiede al LLM
    di valutare la mossa su scala 0-3. Il punteggio normalizzato in
    [-0.5, +0.5] viene scalato per lambda_llm e aggiunto alla reward base.

    Una cache keyed su (obs_pre.tobytes(), action) evita di chiamare il LLM
    piu' volte per la stessa coppia (stato, azione), cosa comune nelle fasi
    iniziali del training quando l'agente tende a ripetere le stesse mosse.

    Parametri:
        env:        SokobanEnv (obs 2D float32, info['cassa_spostata'] presente).
        client:     ClienteLLM istanziato e pronto.
        lambda_llm: peso del segnale LLM nella reward complessiva.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        client: ClienteLLM,
        lambda_llm: float = LAMBDA_LLM,
    ) -> None:
        super().__init__(env)
        self._client  = client
        self._lambda  = lambda_llm

        # Cache: (hash obs_pre, azione) -> score float in [-0.5, +0.5]
        self._cache: Dict[Tuple[bytes, int], float] = {}
        self._n_chiamate_llm: int = 0
        self._n_cache_hit: int    = 0

        # Obs corrente (aggiornata a ogni reset() e step()), usata come pre-step
        self._obs_corrente: Optional[np.ndarray] = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """Resetta l'ambiente e memorizza l'osservazione iniziale come pre-step.

        L'obs iniziale serve per avere la griglia pre-mossa disponibile
        senza dover chiamare l'ambiente una seconda volta.
        """
        obs, info = self.env.reset(**kwargs)
        self._obs_corrente = obs.copy()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Esegue lo step e aggiunge la valutazione LLM se una cassa e' stata spostata.

        Parametri:
            action: intero in {0, 1, 2, 3}.

        Restituisce:
            (obs_post, reward_aumentata, terminated, truncated, info)
        """
        # Griglia pre-mossa: e' l'obs del passo precedente
        obs_pre = self._obs_corrente

        obs_post, reward, terminated, truncated, info = self.env.step(action)

        # Aggiorna l'obs corrente per il prossimo step
        self._obs_corrente = obs_post.copy()

        # Chiamata LLM solo se una cassa e' stata effettivamente spostata (~5% degli step)
        if info.get("cassa_spostata", False):
            cache_key = (obs_pre.tobytes(), int(action))

            if cache_key in self._cache:
                # Stesso stato e stessa azione gia' valutati: riusa il punteggio
                score = self._cache[cache_key]
                self._n_cache_hit += 1
            else:
                # Nuova valutazione: costruisce il prompt con griglia pre/post
                casse_tgt, n_casse = conta_casse(obs_post)
                grid_pre   = griglia_a_testo(obs_pre)
                grid_post  = griglia_a_testo(obs_post)
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
                self._n_chiamate_llm  += 1

            # Aggiunge il contributo LLM alla reward base
            reward += self._lambda * score

        return obs_post, reward, terminated, truncated, info

    @property
    def statistiche_llm(self) -> Dict[str, int]:
        """Restituisce le statistiche di utilizzo del LLM durante il training."""
        return {
            "n_chiamate":  self._n_chiamate_llm,
            "n_cache_hit": self._n_cache_hit,
            "cache_size":  len(self._cache),
        }


class AgenteRicompensaLLM:
    """Factory che crea e configura il wrapper RicompensaLLM.

    Istanzia il ClienteLLM e lo inietta nel wrapper. Espone avvolgi_env()
    usato dagli script di training in experiments/train_ppo_llm_rew.py.

    Parametri:
        lambda_llm: peso del segnale LLM nella reward (default: LAMBDA_LLM da config).
    """

    def __init__(
        self,
        provider: str = "ollama",
        lambda_llm: float = LAMBDA_LLM,
    ) -> None:
        self.lambda_llm = lambda_llm
        self._client = ClienteLLM(provider)

    def avvolgi_env(self, env: gymnasium.Env) -> RicompensaLLM:
        """Avvolge l'ambiente SokobanEnv con il wrapper LLM reward.

        Parametri:
            env: SokobanEnv non ancora avvolto (senza Monitor).

        Restituisce:
            RicompensaLLM pronto per essere avvolto in Monitor e poi in PPO.
        """
        return RicompensaLLM(
            env=env,
            client=self._client,
            lambda_llm=self.lambda_llm,
        )
