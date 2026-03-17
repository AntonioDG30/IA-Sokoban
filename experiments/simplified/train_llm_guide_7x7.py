"""Training AG-LLM-GUIDE su curriculum 7x7 (C0->C2) -- identico a train_llm_guide.py.

Paradigma LfD (Learning from Demonstrations) con LLM come guida.
Per ogni fase del curriculum:
    1. RACCOLTA DEMO: il LLM gioca N_DEMO_LLM_7x7 episodi salvando
       le transizioni (obs, action, reward, next_obs, done).
    2. PRE-FILL BUFFER: le demo vengono caricate nel replay buffer DQN.
    3. TRAINING DQN: apprende sia dalle demo LLM sia dalla propria esperienza.

Identico a experiments/train_llm_guide.py eccetto:
    - Env: SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
    - Fasi: 3 fasi generate C0/C1/C2 invece di 6 fasi C0->C5
    - Nessun dataset Boxoban (solo livelli generati)

Architettura DQN identica ad AG-DQN (train_dqn_7x7.py):
    - CnnPolicy con SokobanCNN (features_dim=256)
    - AggiuntaCanale: (7,7) -> (1,7,7) channels-first
    - Curriculum adattivo con soglie solve_rate identiche ad AG-DQN

Uso:
    python experiments/simplified/train_llm_guide_7x7.py [--seed 42] [--provider ollama]

Modello salvato in models_7x7/llm_guide/llm_guide_7x7_seed{seed}.zip
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
    N_DEMO_LLM_7x7,
    SOGLIE_7x7,
    SOGLIA_RISOLTO_7x7,
    PROVIDER_DEFAULT,
    percorso_llm_guide_7x7,
    crea_env_7x7,
)
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale
from agents.llm_guide_agent import AgenteAgLLMGuide


# ---------------------------------------------------------------------------
# Curriculum adattivo -- IDENTICO a train_llm_guide.py
# ---------------------------------------------------------------------------

_SOGLIA_RISOLTO = SOGLIA_RISOLTO_7x7


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """Legge il massimo solve rate (%) dai log di valutazione SB3."""
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
    p = argparse.ArgumentParser(description="Training AG-LLM-GUIDE curriculum 7x7 LfD")
    p.add_argument("--seed",     type=int, default=42,
                   help="Seed fisso per riproducibilita'")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT,
                   help="Provider LLM per raccolta demo (ollama|groq|mistral)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training curriculum
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int, provider: str) -> None:
    """Addestra AG-LLM-GUIDE con curriculum LfD 7x7 C0->C2.

    Identico a train_llm_guide.py eccetto:
        - Usa SokobanEnv7x7 (7,7)
        - 3 fasi generate invece di 6
        - Nessun split "valid" (solo livelli generati)
    """
    dir_modello   = percorso_llm_guide_7x7(seed).parent
    dir_log_guide = DIR_LOG_7x7 / "llm_guide" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_guide.mkdir(parents=True, exist_ok=True)

    # Agente guida LLM -- identico a train_llm_guide.py
    guida = AgenteAgLLMGuide(provider=provider, seme=seed)

    print("\n[AG-LLM-GUIDE-7x7] ============================================")
    print("[AG-LLM-GUIDE-7x7] Curriculum 7x7 LfD -- seed=" + str(seed))
    print("[AG-LLM-GUIDE-7x7] N_DEMO_LLM_7x7=" + str(N_DEMO_LLM_7x7))
    print("[AG-LLM-GUIDE-7x7] Provider LLM=" + provider)
    print("[AG-LLM-GUIDE-7x7] ============================================\n")

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_7x7):
        nome   = fase["nome"]
        ts_dqn = fase["timestep_dqn"]
        max_s  = fase["max_step"]

        print("\n[AG-LLM-GUIDE-7x7] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print(
            "[AG-LLM-GUIDE-7x7] Raccolta " + str(N_DEMO_LLM_7x7)
            + " demo LLM (max_step=" + str(max_s) + ")..."
        )

        # ---------------------------------------------------------------
        # FASE 1: raccolta demo LLM
        # AggiuntaCanale: (7,7) -> (1,7,7) -- identico a train_llm_guide.py
        # ---------------------------------------------------------------
        env_demo = AggiuntaCanale(crea_env_7x7(fase, seed))
        t_demo = time.time()
        episodi_demo = guida.raccoglie_demo_fase(
            env=env_demo,
            n_episodi=N_DEMO_LLM_7x7,
            max_step=max_s,
            nome_fase=nome,
        )
        env_demo.close()
        print(
            "[AG-LLM-GUIDE-7x7] Demo raccolte in "
            + str(round((time.time() - t_demo) / 60, 1)) + " min"
        )

        # ---------------------------------------------------------------
        # Ambienti DQN training e validazione -- identici a train_llm_guide.py
        # ---------------------------------------------------------------
        env_train = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
        env_val   = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))

        # CnnPolicy identica ad AG-DQN per confronto equo
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        config = {k: v for k, v in CONFIG_DQN_7x7.items() if k != "verbose"}

        # ---------------------------------------------------------------
        # FASE 2: crea o aggiorna modello DQN -- identico a train_llm_guide.py
        # ---------------------------------------------------------------
        if modello is None:
            modello = DQN(
                policy="CnnPolicy",
                env=env_train,
                seed=seed,
                tensorboard_log=str(dir_log_guide),
                policy_kwargs=policy_kwargs,
                verbose=1,
                **config,
            )
        else:
            modello.set_env(env_train)

        # Pre-fill replay buffer con demo LLM di questa fase
        n_trans = guida.riempi_replay_buffer(modello, episodi_demo)
        print(
            "[AG-LLM-GUIDE-7x7] Buffer occupazione: "
            + str(n_trans) + " nuove transizioni LLM caricate"
        )

        # ---------------------------------------------------------------
        # Callbacks -- identici a train_llm_guide.py
        # ---------------------------------------------------------------
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
                name_prefix="llm_guide_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # ---------------------------------------------------------------
        # FASE 3: training DQN con curriculum adattivo -- identico a train_llm_guide.py
        # ---------------------------------------------------------------
        soglia = SOGLIE_7x7.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE_7x7 + 1):
            print(
                "[AG-LLM-GUIDE-7x7] Training DQN " + nome
                + " rep=" + str(rep) + "/" + str(MAX_RIPETIZIONI_FASE_7x7)
                + " (" + str(ts_dqn) + " step)..."
            )
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_dqn,
                callback=callbacks,
                tb_log_name="LLMGuide_" + nome + "_seed" + str(seed),
                reset_num_timesteps=(i_fase == 0 and rep == 0),
            )
            elapsed_rep = round((time.time() - t0) / 60, 1)

            max_sr = _leggi_max_solve_rate(dir_fase / "eval_logs")
            print(
                "[AG-LLM-GUIDE-7x7] Rep " + str(rep)
                + " completata in " + str(elapsed_rep) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE_7x7:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-LLM-GUIDE-7x7] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) -- avanzamento forzato"
                    )
                break

            print(
                "[AG-LLM-GUIDE-7x7] Solve rate " + str(round(max_sr, 1))
                + "% < soglia " + str(soglia) + "% -- ripetizione "
                + str(rep + 1) + "/" + str(MAX_RIPETIZIONI_FASE_7x7)
            )

        print(
            "[AG-LLM-GUIDE-7x7] Fase " + nome + " completata in "
            + str(round((time.time() - t_fase) / 60, 1)) + " min totali"
        )
        env_train.close()
        env_val.close()

    # Salva modello finale (DQN trained with LLM guidance)
    percorso_finale = str(percorso_llm_guide_7x7(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = round((time.time() - t_totale) / 3600, 2)
    print("\n[AG-LLM-GUIDE-7x7] ============================================")
    print("[AG-LLM-GUIDE-7x7] Training completato in " + str(elapsed_tot) + " ore")
    print("[AG-LLM-GUIDE-7x7] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-LLM-GUIDE-7x7] ============================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, provider=args.provider)
