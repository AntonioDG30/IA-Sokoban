# Agente AG-LLM-GUIDE: il LLM raccoglie dimostrazioni, il DQN impara da esse (LfD).
#
# Learning from Demonstrations con il LLM come teacher, in tre fasi per ogni livello del
# curriculum:
#   1. RACCOLTA: il LLM gioca N episodi salvando le transizioni (obs, azione, reward,
#      next_obs, done) in formato (1,10,10).
#   2. PRE-FILL: le transizioni vengono caricate nel replay buffer del DQN con
#      riempi_replay_buffer() prima di avviare learn().
#   3. TRAINING DQN: il DQN apprende sia dalle demo del LLM (già nel buffer) sia dalla
#      propria esperienza raccolta online.
# A inference time il LLM non serve più: decide solo il DQN.
#
# Differenza rispetto agli altri agenti LLM del progetto:
#   - AG-LLM-ACT  (llm_act_agent.py):    il LLM agisce a inference time, nessun RL.
#   - AG-LLM-GUIDE (questo file):         il LLM guida il training, il DQN agisce a inference.
#   - AG-LLM-REW  (llm_reward_agent.py):  l'RL agisce, il LLM valuta la qualità della mossa.
#
# Riferimento: Hester et al. (2018), Deep Q-learning from Demonstrations (DQfD),
# https://arxiv.org/abs/1704.03732

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from core.llm.llm_client import ClienteLLM
from core.llm.sokoban_prompt import (
    griglia_a_testo,
    crea_prompt,
    parsifica_azione,
    conta_casse,
    NOMI_AZIONI,
)
from sistema_10x10.config import MAX_TOKENS_AZIONE

# Alias di tipo per una transizione: obs (1,10,10), azione, reward, next_obs (1,10,10), done
Transizione = Tuple[np.ndarray, int, float, np.ndarray, bool]


class AgenteAgLLMGuide:
    """
    Agente AG-LLM-GUIDE: LfD con il LLM come teacher e il DQN come policy finale.

    seme è il seed di riproducibilità passato al reset dell'ambiente; provider seleziona il
    backend LLM usato per raccogliere le dimostrazioni.
    """

    def __init__(self, provider: str = "ollama", seme: int = 42) -> None:
        self.seme   = seme
        self.client = ClienteLLM(provider=provider)

    # ESTRAZIONE COORDINATE ESPLICITE PER ARRICCHIRE IL PROMPT LLM

    @staticmethod
    def _estrai_posizioni(obs_2d: np.ndarray) -> str:
        """
        Converte l'obs (10,10) in una stringa con le coordinate 1-indexed degli oggetti.

        Le coordinate esplicite aiutano il LLM a ragionare spazialmente senza interpretare
        la griglia ASCII pura; le celle di padding (valore 7) vengono ignorate. Formato:
        'Giocatore: riga R, colonna C | Cassa 1: riga R, colonna C | ...'.
        """
        grid = np.round(obs_2d).astype(int)

        player_pos = np.argwhere((grid == 5) | (grid == 6))             # 5=GIOCATORE, 6=GIOCATORE_SU_TARGET
        box_pos    = np.argwhere((grid == 3) | (grid == 4))             # 3=CASSA, 4=CASSA_SU_TARGET
        tgt_pos    = np.argwhere((grid == 2) | (grid == 4) | (grid == 6))  # tutti i target (anche coperti)

        parti = []
        if len(player_pos) > 0:
            r, c = int(player_pos[0, 0]), int(player_pos[0, 1])
            parti.append("Giocatore: riga " + str(r + 1) + ", colonna " + str(c + 1))
        for i, (r, c) in enumerate(box_pos):
            parti.append(
                "Cassa " + str(i + 1) + ": riga " + str(int(r) + 1)
                + ", colonna " + str(int(c) + 1)
            )
        for i, (r, c) in enumerate(tgt_pos):
            parti.append(
                "Target " + str(i + 1) + ": riga " + str(int(r) + 1)
                + ", colonna " + str(int(c) + 1)
            )
        return " | ".join(parti)

    # FASE 1: RACCOLTA DELLE DIMOSTRAZIONI LLM

    def raccoglie_demo_fase(
        self,
        env,
        n_episodi: int,
        max_step: int,
        nome_fase: str = "",
    ) -> List[List[Transizione]]:
        """
        Raccoglie le dimostrazioni del LLM per una fase del curriculum.

        Il LLM gioca da policy per n_episodi (vedendo griglia ASCII, coordinate esplicite e
        ultime 5 mosse). Le transizioni sono salvate in formato (1,10,10) per essere
        compatibili con il replay buffer del DQN CnnPolicy, quindi env deve essere già
        avvolto con AggiuntaCanale. Restituisce la lista degli episodi, ognuno come lista di
        Transizioni (obs_1c, azione, reward, next_obs_1c, done).
        """
        episodi: List[List[Transizione]] = []
        n_risolti = 0
        t0 = time.time()

        for i_ep in range(n_episodi):
            obs_1c, _ = env.reset()         # shape (1,10,10) grazie ad AggiuntaCanale
            episodio: List[Transizione] = []

            # Storico delle ultime 5 mosse: riduce i loop (es. su->giu->su->giu...)
            storico_mosse: Deque[str] = deque(maxlen=5)
            step_ep = 0

            while True:
                # Il LLM lavora sulla griglia 2D: si toglie l'asse del canale
                obs_2d = obs_1c[0]

                grid_text       = griglia_a_testo(obs_2d)
                casse_su_tgt, n_casse = conta_casse(obs_2d)
                posizioni       = self._estrai_posizioni(obs_2d)

                # Aggiunge lo storico delle mosse al campo posizioni del prompt
                if storico_mosse:
                    posizioni += " | Ultime mosse: " + " ".join(storico_mosse)

                prompt = crea_prompt(
                    grid_text=grid_text,
                    casse_su_target=casse_su_tgt,
                    n_casse=n_casse,
                    step_corrente=step_ep,
                    max_step=max_step,
                    posizioni=posizioni,
                )

                risposta = self.client.chiedi(prompt, max_tokens=MAX_TOKENS_AZIONE)
                azione   = parsifica_azione(risposta)
                storico_mosse.append(NOMI_AZIONI[azione])

                next_obs_1c, reward, terminated, truncated, _ = env.step(azione)
                done = terminated or truncated

                # Salva la transizione con obs (1,10,10), pronta per il replay buffer DQN
                episodio.append(
                    (obs_1c, azione, float(reward), next_obs_1c, done)
                )
                obs_1c   = next_obs_1c
                step_ep += 1

                if done:
                    if terminated:
                        n_risolti += 1
                    break

            episodi.append(episodio)

            if (i_ep + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(
                    "[AG-LLM-GUIDE] " + (nome_fase + " " if nome_fase else "")
                    + "demo " + str(i_ep + 1) + "/" + str(n_episodi)
                    + " | risolti=" + str(n_risolti)
                    + " | elapsed=" + str(round(elapsed, 1)) + "s"
                )

        elapsed_tot = time.time() - t0
        n_trans     = sum(len(ep) for ep in episodi)
        solve_pct   = round(n_risolti / n_episodi * 100, 1)
        print(
            "[AG-LLM-GUIDE] Demo " + (nome_fase + " " if nome_fase else "")
            + "completate: " + str(n_episodi) + " episodi | "
            + str(n_trans) + " transizioni | "
            + "risolti=" + str(n_risolti) + " (" + str(solve_pct) + "%) | "
            + str(round(elapsed_tot / 60, 1)) + " min"
        )
        return episodi

    # FASE 2: PRE-CARICAMENTO DEL REPLAY BUFFER DQN CON LE DEMO LLM

    @staticmethod
    def riempi_replay_buffer(modello_dqn, episodi: List[List[Transizione]]) -> int:
        """
        Carica le transizioni del LLM nel replay buffer del modello DQN.

        Va chiamata dopo aver creato il modello DQN e prima di learn(): così il DQN si
        aggiorna anche sulle demo del LLM fin dal primo mini-batch, non solo sulla propria
        esperienza. Il replay buffer SB3 con n_envs=1 si aspetta obs (1,1,10,10), mentre le
        transizioni raccolte hanno obs (1,10,10): si aggiunge la dimensione batch con
        np.expand_dims prima di buf.add(). Restituisce il numero di transizioni caricate.
        """
        n_caricate = 0
        buf        = modello_dqn.replay_buffer

        for episodio in episodi:
            for obs_1c, azione, reward, next_obs_1c, done in episodio:
                # SB3 vuole shape (n_envs, *obs_shape) = (1, 1, 10, 10)
                obs_buf      = np.expand_dims(obs_1c, axis=0)
                next_obs_buf = np.expand_dims(next_obs_1c, axis=0)

                buf.add(
                    obs=obs_buf,
                    next_obs=next_obs_buf,
                    action=np.array([[azione]]),   # shape (1, 1)
                    reward=np.array([[reward]]),   # shape (1, 1)
                    done=np.array([[done]]),        # shape (1, 1)
                    infos=[{}],
                )
                n_caricate += 1

        occupazione_pct = round(n_caricate / buf.buffer_size * 100, 1)
        print(
            "[AG-LLM-GUIDE] Replay buffer pre-caricato: "
            + str(n_caricate) + " transizioni LLM ("
            + str(occupazione_pct) + "% del buffer da "
            + str(buf.buffer_size) + ")"
        )
        return n_caricate
