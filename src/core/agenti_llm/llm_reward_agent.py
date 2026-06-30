# Agente AG-LLM-REW: PPO addestrato con una reward arricchita dal giudizio del LLM.
#
# Il LLM valuta la qualità di ogni spinta di cassa confrontando la griglia prima e dopo la
# mossa; il giudizio viene scalato e sommato alla reward standard dell'ambiente. A inference
# time agisce solo il PPO: il LLM non serve più.
#
# Il wrapper RicompensaLLM intercetta ogni step() e:
#   1. Tiene da parte la griglia pre-mossa (l'osservazione corrente prima della chiamata).
#   2. Esegue env.step(action) e ottiene la griglia post-mossa.
#   3. Se una cassa è stata spostata (info['cassa_spostata']=True):
#        - cache hit (obs_pre, action): riusa lo score già calcolato;
#        - cache miss: chiede al LLM un punteggio 0-3 sul confronto pre/post;
#        - somma lambda_llm * score alla reward originale.
#   4. Per tutti gli altri step (~95%): reward invariata, nessuna chiamata al LLM.
#
# Triggerare solo sulle spinte reali riduce le chiamate LLM di circa 20 volte rispetto a
# interrogare il LLM a ogni step, rendendo il training fattibile.

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from core.llm import ClienteLLM, griglia_a_testo, conta_casse
from core.llm.sokoban_prompt import (
    NOMI_AZIONI,
    crea_prompt_reward,
    parsifica_reward,
)
from sistema_10x10.config import MAX_TOKENS_REWARD, TIMEOUT_LLM_SEC, LAMBDA_LLM


class RicompensaLLM(gymnasium.Wrapper):
    """
    Wrapper Gymnasium che somma alla reward base la valutazione del LLM.

    Intercetta step() e, quando una cassa viene spostata, fa valutare la mossa al LLM su
    scala 0-3; il punteggio normalizzato in [-0.5, +0.5] viene moltiplicato per lambda_llm e
    aggiunto alla reward. Una cache su (obs_pre.tobytes(), action) evita di interrogare il
    LLM più volte sulla stessa coppia (stato, azione), frequente nelle prime fasi del
    training quando l'agente ripete le stesse mosse. env deve essere il SokobanEnv (obs 2D,
    con info['cassa_spostata']); client è un ClienteLLM già pronto.
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

        # Cache: (hash della obs_pre, azione) -> score float in [-0.5, +0.5]
        self._cache: Dict[Tuple[bytes, int], float] = {}
        self._n_chiamate_llm: int = 0
        self._n_cache_hit: int    = 0

        # Osservazione corrente (aggiornata a ogni reset() e step()), usata come pre-mossa
        self._obs_corrente: Optional[np.ndarray] = None

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Resetta l'ambiente e memorizza l'osservazione iniziale come stato pre-mossa.
        Averla già da parte evita di interrogare l'ambiente una seconda volta al primo step.
        """
        obs, info = self.env.reset(**kwargs)
        self._obs_corrente = obs.copy()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue lo step e, se una cassa è stata spostata, aggiunge la valutazione del LLM.
        Restituisce (obs_post, reward_eventualmente_aumentata, terminated, truncated, info).
        """
        # Griglia pre-mossa: è l'osservazione del passo precedente
        obs_pre = self._obs_corrente

        obs_post, reward, terminated, truncated, info = self.env.step(action)

        # Aggiorna l'osservazione corrente per lo step successivo
        self._obs_corrente = obs_post.copy()

        # Interroga il LLM solo se una cassa è stata davvero spostata (~5% degli step)
        if info.get("cassa_spostata", False):
            cache_key = (obs_pre.tobytes(), int(action))

            if cache_key in self._cache:
                # Stessa coppia (stato, azione) già valutata: riusa il punteggio in cache
                score = self._cache[cache_key]
                self._n_cache_hit += 1
            else:
                # Nuova valutazione: costruisce il prompt con la griglia pre e post mossa
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

            # Somma il contributo del LLM alla reward di base
            reward += self._lambda * score

        return obs_post, reward, terminated, truncated, info

    @property
    def statistiche_llm(self) -> Dict[str, int]:
        """Statistiche d'uso del LLM durante il training (chiamate, cache hit, dimensione cache)."""
        return {
            "n_chiamate":  self._n_chiamate_llm,
            "n_cache_hit": self._n_cache_hit,
            "cache_size":  len(self._cache),
        }


class AgenteRicompensaLLM:
    """
    Factory che crea e configura il wrapper RicompensaLLM.

    Istanzia il ClienteLLM una volta sola e lo inietta nel wrapper. Espone avvolgi_env(),
    usato dallo script src/sistema_10x10/train_ppo_llm_rew.py. lambda_llm è il peso del segnale
    LLM nella reward (default LAMBDA_LLM da config).
    """

    def __init__(
        self,
        provider: str = "ollama",
        lambda_llm: float = LAMBDA_LLM,
    ) -> None:
        self.lambda_llm = lambda_llm
        self._client = ClienteLLM(provider)

    def avvolgi_env(self, env: gymnasium.Env) -> RicompensaLLM:
        """
        Avvolge il SokobanEnv (non ancora avvolto in Monitor) con il wrapper di reward LLM.
        Restituisce il RicompensaLLM, pronto a essere a sua volta avvolto in Monitor e PPO.
        """
        return RicompensaLLM(
            env=env,
            client=self._client,
            lambda_llm=self.lambda_llm,
        )
