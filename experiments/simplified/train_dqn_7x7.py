"""Training AG-DQN su curriculum 7x7 (C0->C2) -- identico a train_dqn.py.

DQN (CnnPolicy) con SokobanCNN su griglia nativa 7x7.
Identico a experiments/train_dqn.py eccetto:
    - Env: SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
    - Fasi: 3 fasi generate C0/C1/C2 invece di 6 fasi C0->C5
    - Budget: scalato (~67% dell'originale per le 3 fasi generative)

Curriculum adattivo identico (MAX_RIPETIZIONI_FASE=2, soglie 7x7):
    C0: 1 cassa  (280K step base, soglia 50%)
    C1: 2 casse  (500K step base, soglia 15%)
    C2: 3 casse  (700K step base, nessuna soglia)

Uso:
    python experiments/simplified/train_dqn_7x7.py [--seed 42]

Modello salvato in models_7x7/dqn/dqn_7x7_seed{seed}.zip
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from experiments.simplified.config_7x7 import (
    FASI_CURRICULUM_7x7,
    CONFIG_DQN_7x7,
    DIR_MODELLI_7x7,
    DIR_LOG_7x7,
    N_EPISODI_VALUTAZIONE_7x7,
    MAX_RIPETIZIONI_FASE_7x7,
    SOGLIE_7x7,
    SOGLIA_RISOLTO_7x7,
    percorso_dqn_7x7,
    crea_env_7x7,
)
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# Curriculum adattivo -- IDENTICO a train_dqn.py
# ---------------------------------------------------------------------------

_SOGLIA_RISOLTO = SOGLIA_RISOLTO_7x7


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """Legge il massimo solve rate (%) dai log di valutazione SB3.

    Identico a train_dqn.py: usa soglia 9.0 per distinguere episodi completati
    da falsi positivi causati dal reward shaping (SCALA_MANHATTAN + SCALA_PLAYER_BOX).
    """
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        return float((d["results"] >= _SOGLIA_RISOLTO).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Training AG-DQN curriculum 7x7")
    p.add_argument("--seed", type=int, default=42, help="Seed fisso")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int) -> None:
    """Addestra AG-DQN con curriculum 7x7 C0->C2.

    Identico a train_dqn.py::addestra_curriculum() eccetto:
        - Usa SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
        - 3 fasi generate invece di 6
        - Percorsi modelli/log in models_7x7/ e logs_7x7/
    """
    dir_modello = percorso_dqn_7x7(seed).parent
    dir_log_dqn = DIR_LOG_7x7 / "dqn" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_dqn.mkdir(parents=True, exist_ok=True)

    print("\n[AG-DQN-7x7] ========================================")
    print("[AG-DQN-7x7] Curriculum 7x7 -- seed=" + str(seed))
    print("[AG-DQN-7x7] SokobanCNN (1,7,7)")
    print("[AG-DQN-7x7] ========================================\n")

    modello = None
    t_totale = time.time()

    # Policy kwargs: CNN custom identica a train_dqn.py
    policy_kwargs = dict(
        features_extractor_class=SokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    for i_fase, fase in enumerate(FASI_CURRICULUM_7x7):
        nome   = fase["nome"]
        ts_dqn = fase["timestep_dqn"]

        print("\n[AG-DQN-7x7] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print("[AG-DQN-7x7] Timestep=" + str(ts_dqn) + " | max_step=" + str(fase["max_step"]))

        # Ambiente di training (singolo, Monitor) -- identico a train_dqn.py
        # AggiuntaCanale: (7,7) -> (1,7,7) richiesto da SokobanCNN (channels-first)
        env_train = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
        env_val   = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))

        config = {**CONFIG_DQN_7x7}
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
            modello.set_env(env_train)

        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=20_000,
                n_eval_episodes=N_EPISODI_VALUTAZIONE_7x7,
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

        # --- Curriculum adattivo -- identico a train_dqn.py ---
        soglia = SOGLIE_7x7.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE_7x7 + 1):
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
                "[AG-DQN-7x7] Fase " + nome + " rep " + str(rep)
                + " completata in " + str(round(elapsed_rep / 60, 1)) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE_7x7:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-DQN-7x7] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) -- avanzamento forzato"
                    )
                break

            print(
                "[AG-DQN-7x7] Solve rate " + str(round(max_sr, 1)) + "% < soglia "
                + str(soglia) + "% -- ripetizione " + str(rep + 1)
                + "/" + str(MAX_RIPETIZIONI_FASE_7x7)
            )

        elapsed_tot_fase = time.time() - t_fase
        print(
            "[AG-DQN-7x7] Fase " + nome + " totale: "
            + str(round(elapsed_tot_fase / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salva modello finale
    percorso_finale = str(percorso_dqn_7x7(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("\n[AG-DQN-7x7] ========================================")
    print("[AG-DQN-7x7] Training completato in " + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-DQN-7x7] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-DQN-7x7] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed)
