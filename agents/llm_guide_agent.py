"""Agente AG-LLM-GUIDE: LLM raccoglie dimostrazioni -> DQN impara da esse (LfD).

Implementazione fedele alla risposta del professore (2026-03-06):
    "durante il training dell'agente di RL si ha che l'LLM decide l'azione
     che sara' poi eseguita dal RL, una sorta di 'aiutante' per tutto
     l'addestramento"

Flusso per ogni fase del curriculum C0->C5:
    1. RACCOLTA DEMO: il LLM agisce come policy su N episodi, salvando
       le transizioni (obs, action, reward, next_obs, done).
       L'osservazione e' in formato (1,10,10) — compatibile con DQN CnnPolicy.
    2. PRE-FILL BUFFER: le transizioni LLM vengono caricate nel replay buffer
       del DQN prima dell'addestramento RL (tramite riempi_replay_buffer()).
    3. TRAINING DQN: l'agente DQN apprende sia dalle demo LLM (gia' nel buffer)
       sia dalla propria esperienza generata durante il training.

A inference time il LLM non serve piu': solo il DQN decide le azioni.

Distinzione con gli altri agenti LLM del progetto:
    - AG-LLM (llm_act_agent.py):  LLM agisce a inference time, nessun RL.
                                   Corrisponde alla traccia PDF ("genera l'azione
                                   ogni turno").
    - AG-LLM-GUIDE (questo file): LLM guida il training RL via LfD, DQN agisce
                                   a inference time. Corrisponde alla mail prof.
    - AG-LLM-REW (llm_reward_agent.py): RL agisce, LLM valuta la reward.

Riferimento tecnico:
    DQfD (Deep Q-learning from Demonstrations), Hester et al. (2018).
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

# Tipo alias: una transizione (obs_1c, azione, reward, next_obs_1c, done)
# dove obs_1c ha shape (1,10,10) — gia' formato DQN CnnPolicy channels-first.
Transizione = Tuple[np.ndarray, int, float, np.ndarray, bool]


class AgenteAgLLMGuide:
    """Agente AG-LLM-GUIDE: LfD con LLM come guida e DQN come policy finale.

    Parametri:
        provider: provider LLM ('ollama' | 'groq' | 'mistral').
        seme:     seed per riproducibilita' (usato per reset env).
    """

    def __init__(self, provider: str = "ollama", seme: int = 42) -> None:
        self.provider = provider
        self.seme = seme
        self.client = ClienteLLM(provider=provider)

    # ------------------------------------------------------------------
    # Helper: estrazione coordinate esplicite (identica ad AgenteAgLLM)
    # Aiuta il LLM a ragionare spazialmente sulla griglia.
    # ------------------------------------------------------------------

    @staticmethod
    def _estrai_posizioni(obs_2d: np.ndarray) -> str:
        """Converte obs (10,10) in stringa con coordinate esplicite 1-indexed.

        Formato: 'Giocatore: riga R, colonna C | Cassa 1: riga R, colonna C | ...'
        Ignora celle di padding (valore 7).
        """
        grid = np.round(obs_2d).astype(int)
        # 5=GIOCATORE, 6=GIOCATORE_SU_TARGET
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
    # Fase 1: raccolta demo LLM
    # ------------------------------------------------------------------

    def raccoglie_demo_fase(
        self,
        env,
        n_episodi: int,
        max_step: int,
        nome_fase: str = "",
    ) -> List[List[Transizione]]:
        """Raccoglie demo LLM su n_episodi per una fase del curriculum.

        Il LLM agisce come policy osservando griglia ASCII + coordinate esplicite
        + storico ultime 5 mosse (anti-loop).

        L'env deve essere gia' avvolto con AggiuntaCanale in modo che le
        osservazioni abbiano shape (1,10,10) — compatibili con il replay buffer
        DQN CnnPolicy. Il LLM utilizza obs[0] (shape 10,10) per i prompt.

        Parametri:
            env:        SokobanEnv gia' wrappato con AggiuntaCanale.
            n_episodi:  numero di episodi da raccogliere.
            max_step:   limite di step per episodio (per il prompt "step rimasti").
            nome_fase:  nome della fase per il logging.

        Restituisce lista di episodi, ciascuno lista di Transizioni.
        Ogni Transizione = (obs_1c, azione, reward, next_obs_1c, done)
        con obs_1c shape (1,10,10).
        """
        episodi: List[List[Transizione]] = []
        n_risolti = 0
        t0 = time.time()

        for i_ep in range(n_episodi):
            obs_1c, _ = env.reset()          # shape (1,10,10)
            episodio: List[Transizione] = []

            # Storico ultime N mosse (anti-loop: il LLM lo riceve nel prompt)
            storico_mosse: Deque[str] = deque(maxlen=5)
            step_ep = 0

            while True:
                # Il LLM usa la griglia 2D (10,10) — squeezia il canale
                obs_2d = obs_1c[0]

                grid_text  = griglia_a_testo(obs_2d)
                casse_su_tgt, n_casse = conta_casse(obs_2d)
                posizioni  = self._estrai_posizioni(obs_2d)

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

                # Salva transizione con obs in formato (1,10,10)
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
        n_trans = sum(len(ep) for ep in episodi)
        solve_pct = round(n_risolti / n_episodi * 100, 1)
        print(
            "[AG-LLM-GUIDE] Demo " + (nome_fase + " " if nome_fase else "")
            + "completate: " + str(n_episodi) + " episodi | "
            + str(n_trans) + " transizioni | "
            + "risolti=" + str(n_risolti) + " (" + str(solve_pct) + "%) | "
            + str(round(elapsed_tot / 60, 1)) + " min"
        )
        return episodi

    # ------------------------------------------------------------------
    # Fase 2: pre-fill replay buffer DQN con demo LLM
    # ------------------------------------------------------------------

    @staticmethod
    def riempi_replay_buffer(modello_dqn, episodi: List[List[Transizione]]) -> int:
        """Carica le transizioni LLM nel replay buffer del modello DQN.

        Deve essere chiamato DOPO la creazione del modello DQN (replay_buffer
        deve esistere) e PRIMA di learn() per far si' che il DQN inizi
        ad apprendere dalle demo LLM fin dal primo update.

        Il replay buffer SB3 per n_envs=1 attende obs di shape
        (n_envs, *obs_shape) = (1, 1, 10, 10). Le transizioni raccolte
        hanno obs_1c di shape (1,10,10), quindi si aggiunge la dimensione
        batch con np.expand_dims(obs_1c, axis=0).

        Parametri:
            modello_dqn: modello DQN SB3 gia' creato.
            episodi:     lista di episodi da raccoglie_demo_fase().

        Restituisce il numero di transizioni caricate nel buffer.
        """
        n_caricate = 0
        buf = modello_dqn.replay_buffer

        for episodio in episodi:
            for obs_1c, azione, reward, next_obs_1c, done in episodio:
                # SB3 replay buffer add() si aspetta shape (n_envs, *obs_shape)
                # n_envs=1, obs_shape=(1,10,10) -> shape richiesta: (1,1,10,10)
                obs_buf      = np.expand_dims(obs_1c, axis=0)
                next_obs_buf = np.expand_dims(next_obs_1c, axis=0)

                buf.add(
                    obs=obs_buf,
                    next_obs=next_obs_buf,
                    action=np.array([[azione]]),          # shape (1,1)
                    reward=np.array([[reward]]),           # shape (1,1)
                    done=np.array([[done]]),               # shape (1,1)
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
