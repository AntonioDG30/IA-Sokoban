"""Agente AG-DQN: Deep Q-Network su Sokoban.

Wrapper attorno a DQN di Stable Baselines 3. A differenza di AG-PPO,
DQN non supporta ambienti vettorizzati: usa un singolo ambiente sia
per il training sia per la valutazione. Il replay buffer non viene
azzerato tra le fasi del curriculum (buffer carry-over), permettendo
al modello di mantenere le competenze acquisite nelle fasi precedenti.

Riferimento:
    Mnih et al. (2015), Human-level control through deep reinforcement learning.
    Nature 518, 529-533.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from sokoban_env import SokobanEnv


def _crea_env(
    directory_livelli: Optional[str],
    difficolta: str,
    split: str,
    max_step: int,
    render_mode: Optional[str] = None,
) -> Monitor:
    """Crea un SokobanEnv avvolto in Monitor per la raccolta delle metriche.

    Parametri:
        directory_livelli: percorso a data/boxoban/ (None per livelli builtin).
        difficolta:        'unfiltered' | 'medium' | 'hard'.
        split:             'train' | 'valid' | 'test'.
        max_step:          limite di step per episodio.
        render_mode:       None | 'human' | 'rgb_array'.
    """
    env = SokobanEnv(
        directory_livelli=directory_livelli,
        difficolta=difficolta,
        split=split,
        max_step=max_step,
        render_mode=render_mode,
    )
    return Monitor(env)


class AgenteDQN:
    """Wrapper SB3-DQN per il training e la valutazione su Sokoban.

    DQN usa un replay buffer off-policy: le esperienze raccolte vengono
    accumulate e riusate piu' volte per l'aggiornamento. Il buffer non
    viene azzerato tra le fasi del curriculum, permettendo al modello
    di continuare ad apprendere dalle competenze delle fasi precedenti.

    Parametri:
        config_dqn:        dizionario con gli iperparametri DQN (da experiments/config.py).
        directory_livelli: percorso a data/boxoban/.
        difficolta:        difficolta' del dataset Boxoban da usare.
        max_step:          step massimi per episodio.
        seme:              seed per la riproducibilita'.
    """

    def __init__(
        self,
        config_dqn: Dict[str, Any],
        directory_livelli: Optional[str] = None,
        difficolta: str = "unfiltered",
        max_step: int = 120,
        seme: int = 42,
    ):
        self.config_dqn        = config_dqn
        self.directory_livelli = directory_livelli
        self.difficolta        = difficolta
        self.max_step          = max_step
        self.seme              = seme
        self.modello: Optional[DQN] = None

    # ------------------------------------------------------------------
    # Costruzione ambienti
    # ------------------------------------------------------------------

    def _costruisci_env_train(self) -> Monitor:
        """Crea l'ambiente di training (singolo, avvolto in Monitor)."""
        return _crea_env(
            self.directory_livelli, self.difficolta, "train", self.max_step
        )

    def _costruisci_env_val(self) -> Monitor:
        """Crea l'ambiente di validazione (singolo, split=valid)."""
        return _crea_env(
            self.directory_livelli, self.difficolta, "valid", self.max_step
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def addestra(
        self,
        totale_timesteps: int,
        dir_log: Optional[str] = None,
        dir_modello: Optional[str] = None,
        frequenza_eval: int = 10_000,
        frequenza_checkpoint: int = 100_000,
    ) -> None:
        """Addestra l'agente DQN per il numero di timestep specificato.

        Parametri:
            totale_timesteps:     numero totale di timestep di training.
            dir_log:              directory per i log TensorBoard.
            dir_modello:          directory per i checkpoint.
            frequenza_eval:       step tra una valutazione e la successiva.
            frequenza_checkpoint: step tra un checkpoint e l'altro.
        """
        env_train = self._costruisci_env_train()
        env_val   = self._costruisci_env_val()

        # tensorboard_log e' gestito separatamente: non va in **config
        config = {k: v for k, v in self.config_dqn.items()
                  if k != "tensorboard_log"}

        self.modello = DQN(
            env=env_train,
            seed=self.seme,
            tensorboard_log=dir_log,
            **config,
        )

        print(
            f"\n[AgenteDQN] Avvio training — "
            f"seed={self.seme}, timesteps={totale_timesteps:,}, "
            f"difficolta'={self.difficolta}"
        )
        print(f"[AgenteDQN] Policy: {self.modello.policy}")

        callbacks: List[BaseCallback] = []

        if dir_modello is not None:
            Path(dir_modello).mkdir(parents=True, exist_ok=True)

            # Valutazione periodica: salva il checkpoint con la reward media piu' alta
            eval_callback = EvalCallback(
                env_val,
                best_model_save_path=str(Path(dir_modello) / "best"),
                log_path=str(Path(dir_modello) / "eval_logs"),
                eval_freq=frequenza_eval,
                n_eval_episodes=20,
                deterministic=True,
                render=False,
                verbose=1,
            )
            callbacks.append(eval_callback)

            # Checkpoint periodico per analisi post-training
            checkpoint_callback = CheckpointCallback(
                save_freq=frequenza_checkpoint,
                save_path=str(Path(dir_modello) / "checkpoints"),
                name_prefix=f"dqn_seed{self.seme}",
                verbose=1,
            )
            callbacks.append(checkpoint_callback)

        self.modello.learn(
            total_timesteps=totale_timesteps,
            callback=callbacks if callbacks else None,
            tb_log_name=f"DQN_seed{self.seme}",
            reset_num_timesteps=True,
        )

        print(f"[AgenteDQN] Training completato ({totale_timesteps:,} timesteps).")

    # ------------------------------------------------------------------
    # Valutazione
    # ------------------------------------------------------------------

    def valuta(
        self,
        n_episodi: int = 100,
        difficolta: Optional[str] = None,
        split: str = "test",
        deterministico: bool = True,
    ) -> Dict[str, float]:
        """Valuta il modello su n_episodi e restituisce le metriche aggregate.

        Parametri:
            n_episodi:      numero di episodi da eseguire.
            difficolta:     difficolta' del dataset (default: quella del training).
            split:          'train' | 'valid' | 'test'.
            deterministico: True per usare la policy deterministica (argmax Q).

        Restituisce dizionario con:
            solve_rate:        % episodi risolti.
            mosse_medie:       media mosse per episodi risolti.
            reward_cumulativa: media reward totale su tutti gli episodi.
            casse_su_target:   media casse su target a fine episodio.
        """
        if self.modello is None:
            raise RuntimeError("Il modello non e' stato addestrato. Chiamare addestra().")

        diff = difficolta or self.difficolta
        env  = _crea_env(self.directory_livelli, diff, split, self.max_step)

        n_risolti = 0
        mosse_risolti: List[int]   = []
        reward_totali: List[float] = []
        casse_totali:  List[int]   = []

        for _ in range(n_episodi):
            obs, _ = env.reset()
            reward_ep  = 0.0
            step_ep    = 0
            done       = False
            casse_finali = 0

            while not done:
                azione, _ = self.modello.predict(obs, deterministic=deterministico)
                obs, reward, terminated, truncated, info = env.step(int(azione))
                reward_ep   += float(reward)
                step_ep     += 1
                casse_finali = info.get("casse_su_target", 0)
                done = terminated or truncated

            if terminated:   # vittoria effettiva
                n_risolti += 1
                mosse_risolti.append(step_ep)

            reward_totali.append(reward_ep)
            casse_totali.append(casse_finali)

        env.close()

        solve_rate   = n_risolti / n_episodi * 100
        mosse_medie  = float(np.mean(mosse_risolti)) if mosse_risolti else 0.0
        reward_media = float(np.mean(reward_totali))
        casse_medie  = float(np.mean(casse_totali))

        metriche = {
            "solve_rate":        round(solve_rate, 2),
            "mosse_medie":       round(mosse_medie, 2),
            "reward_cumulativa": round(reward_media, 4),
            "casse_su_target":   round(casse_medie, 3),
            "n_episodi":         n_episodi,
            "n_risolti":         n_risolti,
        }

        print(
            f"\n[AgenteDQN] Valutazione ({diff}/{split}, {n_episodi} episodi):\n"
            f"  Solve rate:        {solve_rate:.1f}%\n"
            f"  Mosse medie:       {mosse_medie:.1f} (solo risolti)\n"
            f"  Reward cumulativa: {reward_media:.3f}\n"
            f"  Casse su target:   {casse_medie:.2f}/4"
        )

        return metriche

    # ------------------------------------------------------------------
    # Salvataggio e caricamento
    # ------------------------------------------------------------------

    def salva(self, percorso: str) -> None:
        """Salva il modello nel percorso specificato (SB3 aggiunge .zip)."""
        if self.modello is None:
            raise RuntimeError("Nessun modello da salvare. Chiamare addestra().")
        Path(percorso).parent.mkdir(parents=True, exist_ok=True)
        self.modello.save(percorso)
        print(f"[AgenteDQN] Modello salvato: {percorso}.zip")

    def carica(self, percorso: str) -> None:
        """Carica un modello precedentemente salvato dal percorso specificato."""
        env = self._costruisci_env_train()
        self.modello = DQN.load(percorso, env=env)
        print(f"[AgenteDQN] Modello caricato: {percorso}")
