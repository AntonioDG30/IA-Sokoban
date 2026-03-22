"""Agente AG-LLM-GUIDE: il LLM raccoglie dimostrazioni, il DQN impara da esse (LfD).

Implementazione di Learning from Demonstrations (LfD) con LLM come teacher:
    Fase 1 — RACCOLTA: il LLM gioca N episodi per fase, salvando le transizioni
              (obs, azione, reward, next_obs, done) in formato (1,10,10).
    Fase 2 — PRE-FILL: le transizioni vengono caricate nel replay buffer del DQN
              tramite riempi_replay_buffer() prima di avviare learn().
    Fase 3 — TRAINING DQN: il DQN apprende sia dalle demo LLM (nel buffer)
              sia dalla propria esperienza generata durante il training.

A inference time il LLM non serve: solo il DQN decide le azioni.

Il funzionamento e' diverso dagli altri agenti LLM del progetto:
    - AG-LLM (llm_act_agent.py):  LLM agisce a inference time, nessun RL.
    - AG-LLM-GUIDE (questo file): LLM guida il training, DQN agisce a inference.
    - AG-LLM-REW (llm_reward_agent.py): RL agisce, LLM valuta la qualita' della mossa.

Riferimento:
    Hester et al. (2018), Deep Q-learning from Demonstrations (DQfD).
    https://arxiv.org/abs/1704.03732
"""

import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from llm_integration.llm_client import ClienteLLM
from llm_integration.sokoban_prompt import (
    griglia_a_testo,
    crea_prompt,
    parsifica_azione,
    conta_casse,
    NOMI_AZIONI,
)

# Tipo alias per una singola transizione: obs (1,10,10), azione, reward, next_obs, done
Transizione = Tuple[np.ndarray, int, float, np.ndarray, bool]


class AgenteAgLLMGuide:
    """Agente AG-LLM-GUIDE: LfD con LLM come teacher e DQN come policy finale.

    Parametri:
        seme: seed per la riproducibilita' (passato al reset dell'ambiente).
    """

    def __init__(self, provider: str = "ollama", seme: int = 42) -> None:
        self.seme   = seme
        self.client = ClienteLLM(provider=provider)

    # ------------------------------------------------------------------
    # Estrazione coordinate esplicite per arricchire il prompt LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _estrai_posizioni(obs_2d: np.ndarray) -> str:
        """Converte l'obs (10,10) in una stringa con coordinate 1-indexed.

        Le coordinate esplicite aiutano il LLM a ragionare spazialmente
        senza dover interpretare la griglia ASCII pura. Le celle di
        padding (valore 7) vengono ignorate.

        Formato output: 'Giocatore: riga R, colonna C | Cassa 1: riga R, colonna C | ...'

        Parametri:
            obs_2d: array float32 di forma (10, 10).
        """
        grid = np.round(obs_2d).astype(int)

        # Valori corrispondenti: 5=GIOCATORE, 6=GIOCATORE_SU_TARGET
        player_pos = np.argwhere((grid == 5) | (grid == 6))
        # 3=CASSA, 4=CASSA_SU_TARGET
        box_pos    = np.argwhere((grid == 3) | (grid == 4))
        # 2=TARGET, 4=CASSA_SU_TARGET, 6=GIOCATORE_SU_TARGET
        tgt_pos    = np.argwhere((grid == 2) | (grid == 4) | (grid == 6))

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

    # ------------------------------------------------------------------
    # Fase 1: raccolta dimostrazioni LLM
    # ------------------------------------------------------------------

    def raccoglie_demo_fase(
        self,
        env,
        n_episodi: int,
        max_step: int,
        nome_fase: str = "",
    ) -> List[List[Transizione]]:
        """Raccoglie dimostrazioni LLM per una fase del curriculum.

        Il LLM agisce come policy per n_episodi, osservando la griglia ASCII,
        le coordinate esplicite e le ultime 5 mosse (per ridurre i loop). Le
        transizioni vengono salvate in formato (1,10,10) per compatibilita' con
        il replay buffer di DQN CnnPolicy. L'env deve essere gia' avvolto con
        AggiuntaCanale.

        Parametri:
            env:       SokobanEnv avvolto con AggiuntaCanale (obs shape: 1,10,10).
            n_episodi: numero di episodi da raccogliere.
            max_step:  limite di step per episodio (per il contatore nel prompt).
            nome_fase: nome della fase per il logging.

        Restituisce:
            Lista di episodi; ogni episodio e' una lista di Transizioni
            (obs_1c, azione, reward, next_obs_1c, done) con obs_1c (1,10,10).
        """
        episodi: List[List[Transizione]] = []
        n_risolti = 0
        t0 = time.time()

        for i_ep in range(n_episodi):
            obs_1c, _ = env.reset()         # shape (1,10,10)
            episodio: List[Transizione] = []

            # Storico delle ultime 5 mosse: riduce i loop (es. su->giu->su->giu...)
            storico_mosse: Deque[str] = deque(maxlen=5)
            step_ep = 0

            while True:
                # Il LLM lavora sulla griglia 2D (squeeze del canale)
                obs_2d = obs_1c[0]

                grid_text       = griglia_a_testo(obs_2d)
                casse_su_tgt, n_casse = conta_casse(obs_2d)
                posizioni       = self._estrai_posizioni(obs_2d)

                # Aggiunge lo storico mosse al campo posizioni del prompt
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

                risposta = self.client.chiedi(prompt, max_tokens=10)
                azione   = parsifica_azione(risposta)
                storico_mosse.append(NOMI_AZIONI[azione])

                next_obs_1c, reward, terminated, truncated, _ = env.step(azione)
                done = terminated or truncated

                # Salva la transizione con obs in formato (1,10,10) per il replay buffer DQN
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

    # ------------------------------------------------------------------
    # Fase 2: pre-caricamento del replay buffer DQN con le demo LLM
    # ------------------------------------------------------------------

    @staticmethod
    def riempi_replay_buffer(modello_dqn, episodi: List[List[Transizione]]) -> int:
        """Carica le transizioni LLM nel replay buffer del modello DQN.

        Deve essere chiamato dopo la creazione del modello DQN e prima di
        learn(). In questo modo il DQN inizia ad aggiornarsi anche sulle demo
        LLM fin dal primo mini-batch, non solo sulla propria esperienza.

        Il replay buffer SB3 con n_envs=1 si aspetta obs di shape (1,1,10,10):
        le transizioni raccolte hanno obs shape (1,10,10), quindi si aggiunge
        la dimensione batch con np.expand_dims prima di chiamare buf.add().

        Parametri:
            modello_dqn: modello DQN SB3 gia' creato (replay_buffer deve esistere).
            episodi:     lista di episodi restituita da raccoglie_demo_fase().

        Restituisce:
            Numero totale di transizioni caricate nel buffer.
        """
        n_caricate = 0
        buf        = modello_dqn.replay_buffer

        for episodio in episodi:
            for obs_1c, azione, reward, next_obs_1c, done in episodio:
                # SB3 si aspetta shape (n_envs, *obs_shape) = (1, 1, 10, 10)
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
