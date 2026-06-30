# Training di AG-DQN con curriculum learning C0->C5.
#
# Curriculum adattivo identico a quello di AG-PPO ma con budget di step ridotti (DQN è meno
# efficiente in termini di campioni):
#   C0: 1 cassa, generata            (400K step base, soglia 15% solve rate)
#   C1: 2 casse, generata            (700K step base, soglia 10%)
#   C2: 3 casse, generata            (1M step base,   soglia  5%)
#   C3: 4 casse, generata            (1.4M step base, soglia  3%)
#   C4: 4 casse, Boxoban medium      (1.2M step base, soglia  2%)
#   C5: 4 casse, Boxoban unfiltered  (1.3M step base, nessuna soglia)
#   Totale: ~6M step (di più se una fase viene ripetuta perché sotto soglia)
#
# DQN usa un singolo ambiente (niente VecEnv) e la stessa CnnPolicy di AG-PPO. Il replay
# buffer off-policy non viene azzerato tra le fasi: porta l'esperienza da una fase all'altra,
# un vantaggio rispetto a PPO.
#
# Uso: python src/sistema_10x10/train_dqn.py [--seed 42] [--dir-dati dataset/boxoban]
# Il modello finale viene salvato in artifacts/models/10x10/dqn/dqn_seed{seed}.zip

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from sistema_10x10.config import (
    FASI_CURRICULUM_V9,
    CONFIG_DQN,
    DIR_DATI,
    DIR_LOG,
    N_EPISODI_VALUTAZIONE,
    crea_env_da_fase,
    percorso_modello_dqn,
)
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale


# CURRICULUM ADATTIVO — SOGLIE E PARAMETRI DI RIPETIZIONE (IDENTICI A train_ppo.py)

SOGLIE_CURRICULUM = {
    "C0-1box-gen":        15.0,
    "C1-2box-gen":        10.0,
    "C2-3box-gen":         5.0,
    "C3-4box-gen":         3.0,
    "C4-4box-medium":      2.0,
    "C5-4box-unfiltered":  0.0,
}

MAX_RIPETIZIONI_FASE = 2   # => budget massimo 3x per fase

_SOGLIA_RISOLTO = 9.0  # reward minima per dire "episodio risolto" (vedi train_ppo.py)


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """
    Legge dai log di valutazione SB3 il massimo solve rate (%) raggiunto.
    Usa la soglia 9.0 per separare gli episodi completati dai falsi positivi prodotti dal
    reward shaping (SCALA_MANHATTAN + SCALA_PLAYER_BOX). Restituisce 0.0 se il file manca.
    """
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        return float((d["results"] >= _SOGLIA_RISOLTO).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# CLI

def _parse_args():
    """Legge gli argomenti da riga di comando: --seed (int) e --dir-dati (path a dataset/boxoban)."""
    p = argparse.ArgumentParser(description="Training AG-DQN curriculum v9")
    p.add_argument("--seed",     type=int, default=42,            help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI), help="Path dataset/boxoban")
    return p.parse_args()


# TRAINING

def addestra_curriculum(seed: int, dir_dati: str) -> None:
    """
    Addestra AG-DQN con il curriculum adattivo completo C0->C5.

    Stessa struttura di train_ppo.py ma con DQN: un solo ambiente e un replay buffer
    persistente tra le fasi. A ogni fase cambia ambiente con set_env() senza svuotare il
    buffer, così le esperienze delle fasi precedenti restano disponibili.
    """
    dir_dati_path = Path(dir_dati)
    dir_modello   = percorso_modello_dqn(seed).parent
    dir_log_dqn   = DIR_LOG / "dqn" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_dqn.mkdir(parents=True, exist_ok=True)

    print("[AG-DQN] Curriculum v9 - seed=" + str(seed))

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_dqn = fase["timestep_dqn"]
        max_s  = fase["max_step"]

        print("\n[AG-DQN] Fase " + str(i_fase) + ": " + nome)
        print("[AG-DQN] Timestep=" + str(ts_dqn) + " | max_step=" + str(max_s))

        # Ambienti di training e validazione (singoli, avvolti in Monitor).
        # AggiuntaCanale: (10,10) -> (1,10,10), formato channels-first per SokobanCNN
        env_train = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")))
        env_val   = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")))

        # Policy kwargs: stessa CNN custom di AG-PPO (qui senza LSTM)
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )

        config = {**CONFIG_DQN}
        config.pop("verbose", None)

        if modello is None:
            modello = DQN(
                policy="CnnPolicy",
                env=env_train,
                seed=seed,
                tensorboard_log=str(dir_log_dqn),
                policy_kwargs=policy_kwargs,
                verbose=1,
                **config,
            )
        else:
            modello.set_env(env_train)   # buffer conservato: niente reset tra le fasi

        # Callback della fase (riusati tra le ripetizioni così i log si accumulano)
        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=20_000,
                n_eval_episodes=N_EPISODI_VALUTAZIONE,
                deterministic=True,
                render=False,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=100_000,
                save_path=str(dir_fase / "checkpoints"),
                name_prefix="dqn_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # Curriculum adattivo: ripete la fase finché il solve rate non raggiunge la soglia
        soglia = SOGLIE_CURRICULUM.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE + 1):
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_dqn,
                callback=callbacks,
                tb_log_name="DQN_" + nome + "_seed" + str(seed),
                reset_num_timesteps=(i_fase == 0 and rep == 0),
            )
            elapsed_rep = time.time() - t0

            max_sr = _leggi_max_solve_rate(dir_fase / "eval_logs")
            print(
                "[AG-DQN] Fase " + nome + " rep " + str(rep)
                + " completata in " + str(round(elapsed_rep / 60, 1)) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-DQN] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) - avanzamento forzato"
                    )
                break

            print(
                "[AG-DQN] Solve rate " + str(round(max_sr, 1)) + "% < soglia "
                + str(soglia) + "% - ripetizione " + str(rep + 1)
                + "/" + str(MAX_RIPETIZIONI_FASE)
            )

        elapsed_tot_fase = time.time() - t_fase
        print(
            "[AG-DQN] Fase " + nome + " totale: "
            + str(round(elapsed_tot_fase / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salvataggio del modello finale
    percorso_finale = str(percorso_modello_dqn(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("[AG-DQN] Training completato in " + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-DQN] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, dir_dati=args.dir_dati)
