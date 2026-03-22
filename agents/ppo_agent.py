"""Agente AG-PPO: Proximal Policy Optimization su Sokoban.

Wrapper attorno a RecurrentPPO di Stable Baselines 3 (sb3-contrib) con
SokobanCNN come estrattore di feature. Gestisce il ciclo completo:
creazione ambiente vettorizzato, training, valutazione periodica, checkpoint.

Riferimento:
    Schulman et al. (2017), Proximal Policy Optimization Algorithms.
    https://arxiv.org/abs/1707.06347
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from sokoban_env import SokobanEnv


def _crea_env(
    directory_livelli: Optional[str],
    difficolta: str,
    split: str,
    max_step: int,
    render_mode: Optional[str] = None,
) -> Monitor:
    """Crea un SokobanEnv avvolto in Monitor per la raccolta delle metriche.

    Monitor registra reward e lunghezza di ogni episodio in formato compatibile
    con SB3 e TensorBoard. Usato sia per training che per valutazione.

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


class AgentePPO:
    """Wrapper SB3-PPO per il training e la valutazione su Sokoban.

    Parametri:
        config_ppo:        dizionario con gli iperparametri PPO (da experiments/config.py).
        directory_livelli: percorso a data/boxoban/.
        difficolta:        difficolta' del dataset Boxoban da usare.
        max_step:          step massimi per episodio.
        n_envs:            numero di ambienti paralleli per il training.
        seme:              seed per la riproducibilita'.
    """

    def __init__(
        self,
        config_ppo: Dict[str, Any],
        directory_livelli: Optional[str] = None,
        difficolta: str = "unfiltered",
        max_step: int = 120,
        n_envs: int = 4,
        seme: int = 42,
    ):
        self.config_ppo        = config_ppo
        self.directory_livelli = directory_livelli
        self.difficolta        = difficolta
        self.max_step          = max_step
        self.n_envs            = n_envs
        self.seme              = seme
        self.modello: Optional[PPO] = None
        self._env_train: Optional[VecEnv] = None

    # ------------------------------------------------------------------
    # Costruzione ambienti
    # ------------------------------------------------------------------

    def _costruisci_env_train(self) -> VecEnv:
        """Crea il VecEnv di training con n_envs ambienti paralleli.

        make_vec_env gestisce automaticamente la parallelizzazione tramite
        SubprocVecEnv o DummyVecEnv in base alla piattaforma.
        """
        def _factory():
            return _crea_env(
                self.directory_livelli, self.difficolta, "train", self.max_step
            )
        return make_vec_env(_factory, n_envs=self.n_envs, seed=self.seme)

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
        """Addestra l'agente PPO per il numero di timestep specificato.

        Salva il miglior modello (per reward media su valutazione) nella
        sottocartella best/ e checkpoint regolari nella cartella checkpoints/.
        La valutazione avviene ogni ~20K interazioni ambientali totali
        (frequenza_eval viene divisa per n_envs per normalizzare).

        Parametri:
            totale_timesteps:     numero totale di timestep di training.
            dir_log:              directory per i log TensorBoard.
            dir_modello:          directory per i checkpoint.
            frequenza_eval:       interazioni totali tra una valutazione e l'altra.
            frequenza_checkpoint: interazioni totali tra un checkpoint e l'altro.
        """
        self._env_train = self._costruisci_env_train()
        env_val = self._costruisci_env_val()

        # Rimuove tensorboard_log dalla config perche' viene passato separatamente
        config = {k: v for k, v in self.config_ppo.items()
                  if k != "tensorboard_log"}

        self.modello = PPO(
            env=self._env_train,
            seed=self.seme,
            tensorboard_log=dir_log,
            **config,
        )

        print(
            f"\n[AgentePPO] Avvio training — "
            f"seed={self.seme}, timesteps={totale_timesteps:,}, "
            f"n_envs={self.n_envs}, difficolta'={self.difficolta}"
        )
        print(f"[AgentePPO] Policy: {self.modello.policy}")

        callbacks: List[BaseCallback] = []

        if dir_modello is not None:
            Path(dir_modello).mkdir(parents=True, exist_ok=True)

            # Valutazione periodica: salva il miglior modello nella cartella best/
            eval_callback = EvalCallback(
                env_val,
                best_model_save_path=str(Path(dir_modello) / "best"),
                log_path=str(Path(dir_modello) / "eval_logs"),
                eval_freq=max(frequenza_eval // self.n_envs, 1),
                n_eval_episodes=20,
                deterministic=True,
                render=False,
                verbose=1,
            )
            callbacks.append(eval_callback)

            # Checkpoint periodico: utile per analizzare la curva di apprendimento
            checkpoint_callback = CheckpointCallback(
                save_freq=max(frequenza_checkpoint // self.n_envs, 1),
                save_path=str(Path(dir_modello) / "checkpoints"),
                name_prefix=f"ppo_seed{self.seme}",
                verbose=1,
            )
            callbacks.append(checkpoint_callback)

        self.modello.learn(
            total_timesteps=totale_timesteps,
            callback=callbacks if callbacks else None,
            tb_log_name=f"PPO_seed{self.seme}",
            reset_num_timesteps=True,
        )

        print(f"[AgentePPO] Training completato ({totale_timesteps:,} timesteps).")

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

        Usa la policy deterministica (argmax) per la valutazione finale.
        Il solve rate e' calcolato su terminated=True, non sulla reward.

        Parametri:
            n_episodi:      numero di episodi da eseguire.
            difficolta:     difficolta' del dataset (default: quella del training).
            split:          'train' | 'valid' | 'test'.
            deterministico: True per usare la policy deterministica.

        Restituisce dizionario con:
            solve_rate:        % episodi risolti (terminated=True).
            mosse_medie:       media mosse per episodi risolti (0.0 se nessuno).
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

            if terminated:   # vittoria effettiva: tutte le casse su target
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
            f"\n[AgentePPO] Valutazione ({diff}/{split}, {n_episodi} episodi):\n"
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
        """Salva il modello nel percorso specificato (SB3 aggiunge .zip).

        Crea automaticamente le directory intermedie se non esistono.
        """
        if self.modello is None:
            raise RuntimeError("Nessun modello da salvare. Chiamare addestra().")
        Path(percorso).parent.mkdir(parents=True, exist_ok=True)
        self.modello.save(percorso)
        print(f"[AgentePPO] Modello salvato: {percorso}.zip")

    def carica(self, percorso: str) -> None:
        """Carica un modello precedentemente salvato dal percorso specificato."""
        env = self._costruisci_env_train()
        self.modello = PPO.load(percorso, env=env)
        print(f"[AgentePPO] Modello caricato: {percorso}")
