# Training di AG-PPO sul curriculum 7x7 (C0->C2) — gemello di train_ppo.py.
#
# RecurrentPPO (CnnLstmPolicy) con SokobanCNN su griglia nativa 7x7. Identico a
# src/sistema_10x10/train_ppo.py tranne che per: l'ambiente SokobanEnv7x7 (7,7) al posto di
# SokobanEnv (10,10), 3 fasi generate C0/C1/C2 invece di 6, e il budget scalato (~67%).
#
# Curriculum adattivo (MAX_RIPETIZIONI_FASE=2, soglie del 7x7):
#   C0: 1 cassa  (400K step base, soglia 50%)
#   C1: 2 casse  (700K step base, soglia 15%)
#   C2: 3 casse  (1M step base,   nessuna soglia)
#
# Uso: python src/sistema_7x7/train_ppo_7x7.py [--seed 42]
# Modello finale salvato in artifacts/models/7x7/ppo/ppo_7x7_seed{seed}.zip

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sistema_7x7.config_7x7 import (
    FASI_CURRICULUM_7x7,
    CONFIG_PPO_7x7,
    DIR_LOG_7x7,
    N_ENVS_PPO_7x7,
    N_EPISODI_VALUTAZIONE_7x7,
    MAX_RIPETIZIONI_FASE_7x7,
    SOGLIE_7x7,
    SOGLIA_RISOLTO_7x7,
    percorso_ppo_7x7,
    crea_env_7x7,
)
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale


# CURRICULUM ADATTIVO — IDENTICO A train_ppo.py

_SOGLIA_RISOLTO = SOGLIA_RISOLTO_7x7


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """
    Legge dai log SB3 il massimo solve rate (%) raggiunto. Come in train_ppo.py, usa la
    soglia 9.0 per separare gli episodi completati dai falsi positivi del reward shaping.
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
    """Legge il seed dalla riga di comando (default 42 per riproducibilità)."""
    p = argparse.ArgumentParser(description="Training AG-PPO curriculum 7x7")
    p.add_argument("--seed", type=int, default=42, help="Seed fisso")
    return p.parse_args()


# TRAINING

def addestra_curriculum(seed: int) -> None:
    """
    Addestra AG-PPO con il curriculum 7x7 C0->C2.
    Stessa logica di train_ppo.py::addestra_curriculum() ma con SokobanEnv7x7 (7,7), 3 fasi
    generate invece di 6 e percorsi in artifacts/models/7x7/ e artifacts/logs/7x7/.
    """
    dir_modello = percorso_ppo_7x7(seed).parent
    dir_log_ppo = DIR_LOG_7x7 / "ppo" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_ppo.mkdir(parents=True, exist_ok=True)

    print("[AG-PPO-7x7] Curriculum 7x7 -- seed=" + str(seed))
    print("[AG-PPO-7x7] n_envs=" + str(N_ENVS_PPO_7x7) + " | SokobanCNN (1,7,7)")

    modello = None
    t_totale = time.time()

    # Policy kwargs: SokobanCNN + LSTM, identici a train_ppo.py.
    # SokobanCNN calcola n_flatten in modo dinamico, quindi funziona anche per (1,7,7).
    policy_kwargs = dict(
        features_extractor_class=SokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    for i_fase, fase in enumerate(FASI_CURRICULUM_7x7):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        ent    = fase["ent_coef"]

        print("\n[AG-PPO-7x7] Fase " + str(i_fase) + ": " + nome)
        print("[AG-PPO-7x7] Timestep=" + str(ts_ppo) + " | max_step=" + str(fase["max_step"])
              + " | ent_coef=" + str(ent))

        # Ambiente di training (VecEnv parallelo).
        # AggiuntaCanale: (7,7) -> (1,7,7), il formato channels-first per SokobanCNN
        def _factory_train(fase=fase):
            return Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))

        env_train = make_vec_env(_factory_train, n_envs=N_ENVS_PPO_7x7, seed=seed)

        # Ambiente di validazione (singolo)
        env_val = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))

        config = {**CONFIG_PPO_7x7, "ent_coef": ent}
        config.pop("verbose", None)

        if modello is None:
            modello = RecurrentPPO(
                policy="CnnLstmPolicy",
                env=env_train,
                seed=seed,
                tensorboard_log=str(dir_log_ppo),
                policy_kwargs=policy_kwargs,
                verbose=1,
                **config,
            )
        else:
            modello.set_env(env_train)
            modello.ent_coef = ent

        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=max(20_000 // N_ENVS_PPO_7x7, 1),
                n_eval_episodes=N_EPISODI_VALUTAZIONE_7x7,
                deterministic=True,
                render=False,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=max(100_000 // N_ENVS_PPO_7x7, 1),
                save_path=str(dir_fase / "checkpoints"),
                name_prefix="ppo_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # Curriculum adattivo (stessa logica di train_ppo.py)
        soglia = SOGLIE_7x7.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE_7x7 + 1):
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_ppo,
                callback=callbacks,
                tb_log_name="PPO_" + nome + "_seed" + str(seed),
                reset_num_timesteps=(i_fase == 0 and rep == 0),
            )
            elapsed_rep = time.time() - t0

            max_sr = _leggi_max_solve_rate(dir_fase / "eval_logs")
            print(
                "[AG-PPO-7x7] Fase " + nome + " rep " + str(rep)
                + " completata in " + str(round(elapsed_rep / 60, 1)) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE_7x7:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-PPO-7x7] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) -- avanzamento forzato"
                    )
                break

            print(
                "[AG-PPO-7x7] Solve rate " + str(round(max_sr, 1)) + "% < soglia "
                + str(soglia) + "% -- ripetizione " + str(rep + 1)
                + "/" + str(MAX_RIPETIZIONI_FASE_7x7)
            )

        elapsed_tot_fase = time.time() - t_fase
        print(
            "[AG-PPO-7x7] Fase " + nome + " totale: "
            + str(round(elapsed_tot_fase / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salvataggio del modello finale
    percorso_finale = str(percorso_ppo_7x7(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("[AG-PPO-7x7] Training completato in " + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-PPO-7x7] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed)
