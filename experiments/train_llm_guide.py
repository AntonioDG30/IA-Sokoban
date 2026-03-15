"""Training AG-LLM-GUIDE con curriculum learning C0->C5 (v9).

Paradigma: LfD (Learning from Demonstrations) con LLM come guida.
Per ogni fase del curriculum:
    1. RACCOLTA DEMO: il LLM gioca N_DEMO_LLM_FASE episodi salvando
       le transizioni (obs, action, reward, next_obs, done).
    2. PRE-FILL BUFFER: le demo vengono caricate nel replay buffer DQN.
    3. TRAINING DQN: apprende sia dalle demo LLM sia dalla propria esperienza.

Questo script implementa la versione dell'agente confermata dal professore:
    "durante il training dell'agente di RL si ha che l'LLM decide l'azione
     che sara' poi eseguita dal RL, una sorta di 'aiutante' per tutto
     l'addestramento" [risposta mail professore, 2026-03-06]

Confronto agenti LLM del progetto:
    - AG-LLM       (train_llm_act.py):    LLM agisce a inference time (traccia PDF)
    - AG-LLM-GUIDE (questo script):       LLM guida training via LfD   (mail prof)
    - AG-LLM-REW   (train_ppo_llm_rew.py): LLM valuta la reward         (traccia PDF)

Architettura DQN identica ad AG-DQN (train_dqn.py):
    - CnnPolicy con SokobanCNN (features_dim=256)
    - AggiuntaCanale: (10,10) -> (1,10,10) channels-first
    - Curriculum adattivo con soglie solve_rate (identiche ad AG-DQN)
    - Replay buffer: pre-riempito con demo LLM ad ogni fase

Uso:
    python experiments/train_llm_guide.py [--seed 42] [--dir-dati data/boxoban]

Modello finale salvato in: models/llm_guide/llm_guide_seed{seed}.zip
Log TensorBoard in:         logs/llm_guide/seed{seed}/
"""

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

from experiments.config import (
    FASI_CURRICULUM_V9,
    CONFIG_DQN,
    DIR_DATI,
    DIR_LOG,
    N_EPISODI_VALUTAZIONE,
    crea_env_da_fase,
)
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale
from agents.llm_guide_agent import AgenteAgLLMGuide


# ---------------------------------------------------------------------------
# Parametri LfD — raccolta demo LLM per fase
# ---------------------------------------------------------------------------

# Episodi LLM per fase prima del training DQN.
# Con 2% solve rate su C0 (30 ep -> ~0-1 risolti), le transizioni "fallite"
# sono comunque utili: mostrano al DQN come avvicinarsi alle casse e
# ricevere proximity bonus. Su C1+ (0%) forniscono esplorazione iniziale.
N_DEMO_LLM_FASE = 30


# ---------------------------------------------------------------------------
# Curriculum adattivo — identico ad AG-DQN per confronto equo
# ---------------------------------------------------------------------------

SOGLIE_CURRICULUM = {
    "C0-1box-gen":        15.0,
    "C1-2box-gen":        10.0,
    "C2-3box-gen":         5.0,
    "C3-4box-gen":         3.0,
    "C4-4box-medium":      2.0,
    "C5-4box-unfiltered":  0.0,
}

MAX_RIPETIZIONI_FASE = 2   # max 3x budget per fase se soglia non raggiunta


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """Legge il massimo solve rate (%) dai log di valutazione SB3."""
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        return float((d["results"] > 0).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Training AG-LLM-GUIDE curriculum v9 LfD")
    p.add_argument("--seed",     type=int, default=42,
                   help="Seed fisso per riproducibilita'")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),
                   help="Percorso directory data/boxoban")
    p.add_argument("--provider", type=str, default="ollama",
                   help="Provider LLM per raccolta demo (ollama|groq|mistral)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training curriculum
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int, dir_dati: str, provider: str) -> None:
    """Addestra AG-LLM-GUIDE con curriculum LfD C0->C5.

    Per ogni fase:
        1. Raccoglie N_DEMO_LLM_FASE episodi con LLM come policy.
        2. Pre-carica il replay buffer DQN con le transizioni LLM.
        3. Addestra DQN per timestep_dqn step (curriculum adattivo).
    """
    dir_dati_path = Path(dir_dati)
    dir_modello   = Path("models/llm_guide")
    dir_log_guide = DIR_LOG / "llm_guide" / ("seed" + str(seed))
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_guide.mkdir(parents=True, exist_ok=True)

    # Agente guida LLM — stesso client LLM di AG-LLM
    guida = AgenteAgLLMGuide(provider=provider, seme=seed)

    print("\n[AG-LLM-GUIDE] ============================================")
    print("[AG-LLM-GUIDE] Curriculum v9 LfD — seed=" + str(seed))
    print("[AG-LLM-GUIDE] N_DEMO_LLM_FASE=" + str(N_DEMO_LLM_FASE))
    print("[AG-LLM-GUIDE] Provider LLM=" + provider)
    print("[AG-LLM-GUIDE] ============================================\n")

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_dqn = fase["timestep_dqn"]
        max_s  = fase["max_step"]

        print("\n[AG-LLM-GUIDE] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print(
            "[AG-LLM-GUIDE] Raccolta " + str(N_DEMO_LLM_FASE)
            + " demo LLM (max_step=" + str(max_s) + ")..."
        )

        # ---------------------------------------------------------------
        # FASE 1: raccolta demo LLM
        # L'env usa AggiuntaCanale per avere obs (1,10,10) gia' in
        # formato DQN — il LLM usa obs[0] (shape 10,10) per i prompt.
        # ---------------------------------------------------------------
        env_demo = AggiuntaCanale(
            crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")
        )
        t_demo = time.time()
        episodi_demo = guida.raccoglie_demo_fase(
            env=env_demo,
            n_episodi=N_DEMO_LLM_FASE,
            max_step=max_s,
            nome_fase=nome,
        )
        env_demo.close()
        print(
            "[AG-LLM-GUIDE] Demo raccolte in "
            + str(round((time.time() - t_demo) / 60, 1)) + " min"
        )

        # ---------------------------------------------------------------
        # Ambienti DQN training e validazione
        # Stessa catena di wrap di AG-DQN: SokobanEnv -> AggiuntaCanale -> Monitor
        # ---------------------------------------------------------------
        env_train = Monitor(AggiuntaCanale(
            crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")
        ))
        env_val = Monitor(AggiuntaCanale(
            crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")
        ))

        # CnnPolicy identica ad AG-DQN per confronto equo
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        config = {k: v for k, v in CONFIG_DQN.items() if k != "verbose"}

        # ---------------------------------------------------------------
        # FASE 2: crea o aggiorna modello DQN
        # Prima fase: crea modello da zero con replay buffer vuoto.
        # Fasi successive: aggiorna solo l'env (buffer conserva esperienza
        # accumulata, incluse le demo LLM delle fasi precedenti).
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
            "[AG-LLM-GUIDE] Buffer occupazione: "
            + str(n_trans) + " nuove transizioni LLM caricate"
        )

        # ---------------------------------------------------------------
        # Callbacks: eval periodica + checkpoint
        # Struttura cartelle identica ad AG-DQN per valutazione uniforme
        # ---------------------------------------------------------------
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
                name_prefix="llm_guide_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # ---------------------------------------------------------------
        # FASE 3: training DQN con curriculum adattivo
        # Il DQN impara sia dalle demo LLM (gia' nel buffer) sia dalla
        # propria esperienza generata durante il training (off-policy).
        # ---------------------------------------------------------------
        soglia = SOGLIE_CURRICULUM.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE + 1):
            print(
                "[AG-LLM-GUIDE] Training DQN " + nome
                + " rep=" + str(rep) + "/" + str(MAX_RIPETIZIONI_FASE)
                + " (" + str(ts_dqn) + " step)..."
            )
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_dqn,
                callback=callbacks,
                tb_log_name="LLMGuide_" + nome + "_seed" + str(seed),
                # reset_num_timesteps solo al primissimo apprendimento
                # per mantenere il contatore globale degli step
                reset_num_timesteps=(i_fase == 0 and rep == 0),
            )
            elapsed_rep = round((time.time() - t0) / 60, 1)

            max_sr = _leggi_max_solve_rate(dir_fase / "eval_logs")
            print(
                "[AG-LLM-GUIDE] Rep " + str(rep)
                + " completata in " + str(elapsed_rep) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-LLM-GUIDE] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) — avanzamento forzato"
                    )
                break

            print(
                "[AG-LLM-GUIDE] Solve rate " + str(round(max_sr, 1))
                + "% < soglia " + str(soglia) + "% — ripetizione "
                + str(rep + 1) + "/" + str(MAX_RIPETIZIONI_FASE)
            )

        print(
            "[AG-LLM-GUIDE] Fase " + nome + " completata in "
            + str(round((time.time() - t_fase) / 60, 1)) + " min totali"
        )
        env_train.close()
        env_val.close()

    # Salva modello finale (DQN trained with LLM guidance)
    percorso_finale = str(dir_modello / ("llm_guide_seed" + str(seed)))
    modello.save(percorso_finale)

    elapsed_tot = round((time.time() - t_totale) / 3600, 2)
    print("\n[AG-LLM-GUIDE] ============================================")
    print("[AG-LLM-GUIDE] Training completato in " + str(elapsed_tot) + " ore")
    print("[AG-LLM-GUIDE] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-LLM-GUIDE] ============================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(
        seed=args.seed,
        dir_dati=args.dir_dati,
        provider=args.provider,
    )
