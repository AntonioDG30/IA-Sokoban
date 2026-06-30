# Training di AG-LLM-GUIDE con curriculum learning C0->C5 (paradigma LfD).
#
# Learning from Demonstrations con il LLM come guida. Per ogni fase del curriculum:
#   1. RACCOLTA DEMO: il LLM gioca N_DEMO_LLM_FASE episodi salvando le transizioni
#      (obs, action, reward, next_obs, done).
#   2. PRE-FILL DEL BUFFER: le demo vengono caricate nel replay buffer del DQN.
#   3. TRAINING DQN: il DQN impara sia dalle demo del LLM sia dalla propria esperienza.
#
# È la versione dell'agente confermata dal professore:
#   "durante il training dell'agente di RL si ha che l'LLM decide l'azione che sarà poi
#    eseguita dal RL, una sorta di 'aiutante' per tutto l'addestramento"
#   [risposta mail professore, 2026-03-06]
#
# I tre agenti LLM del progetto a confronto:
#   - AG-LLM-ACT   (train_llm_act.py):     il LLM agisce a inference time (traccia PDF)
#   - AG-LLM-GUIDE (questo script):        il LLM guida il training via LfD (mail prof)
#   - AG-LLM-REW   (train_ppo_llm_rew.py): il LLM valuta la reward (traccia PDF)
#
# Architettura DQN identica ad AG-DQN (train_dqn.py): CnnPolicy con SokobanCNN (features_dim
# =256), AggiuntaCanale (10,10)->(1,10,10), stesse soglie del curriculum adattivo; in più il
# replay buffer viene pre-riempito con le demo del LLM a ogni fase.
#
# Uso: python src/sistema_10x10/train_llm_guide.py [--seed 42] [--dir-dati dataset/boxoban]
# Modello finale in artifacts/models/10x10/llm_guide/llm_guide_seed{seed}.zip, log in artifacts/logs/10x10/llm_guide/seed{seed}/.

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
    percorso_modello_llm_guide,
)
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale
from core.agenti_llm.llm_guide_agent import AgenteAgLLMGuide


# PARAMETRI LfD — RACCOLTA DELLE DEMO LLM PER FASE

# Episodi giocati dal LLM per fase prima del training DQN. Anche con un solve rate del 2% su
# C0 (30 ep -> ~0-1 risolti) le transizioni "fallite" sono utili: mostrano al DQN come
# avvicinarsi alle casse; su C1+ (0%) forniscono almeno un'esplorazione iniziale.
N_DEMO_LLM_FASE = 30


# CURRICULUM ADATTIVO — IDENTICO AD AG-DQN PER UN CONFRONTO EQUO

SOGLIE_CURRICULUM = {
    "C0-1box-gen":        15.0,
    "C1-2box-gen":        10.0,
    "C2-3box-gen":         5.0,
    "C3-4box-gen":         3.0,
    "C4-4box-medium":      2.0,
    "C5-4box-unfiltered":  0.0,
}

MAX_RIPETIZIONI_FASE = 2   # budget massimo 3x per fase se la soglia non viene raggiunta

# Reward minima per dire "episodio risolto", identica ad AG-DQN/AG-PPO. NON usare > 0: con
# lo shaping attivo un episodio non risolto può accumulare reward positiva (~20% falsi
# positivi), mentre un episodio davvero risolto ha sempre reward >= 9.5 (bonus +10).
_SOGLIA_RISOLTO = 9.0


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """
    Legge dai log di valutazione SB3 il massimo solve rate (%) raggiunto, usando la soglia
    9.0 per separare gli episodi completati dai falsi positivi del reward shaping (come negli
    altri script di training). Restituisce 0.0 se il file manca.
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
    """Legge gli argomenti da riga di comando: --seed (int), --dir-dati (path), --provider (str)."""
    p = argparse.ArgumentParser(description="Training AG-LLM-GUIDE curriculum v9 LfD")
    p.add_argument("--seed",     type=int, default=42,
                   help="Seed fisso per riproducibilita'")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),
                   help="Percorso directory dataset/boxoban")
    p.add_argument("--provider", type=str, default="ollama",
                   help="Provider LLM per raccolta demo (ollama)")
    return p.parse_args()


# TRAINING DEL CURRICULUM

def addestra_curriculum(seed: int, dir_dati: str, provider: str) -> None:
    """
    Addestra AG-LLM-GUIDE con il curriculum LfD C0->C5.

    Per ogni fase: (1) raccoglie N_DEMO_LLM_FASE episodi con il LLM come policy, (2)
    pre-riempie il replay buffer del DQN con quelle transizioni, (3) addestra il DQN per
    timestep_dqn step con curriculum adattivo.
    """
    dir_dati_path = Path(dir_dati)
    dir_modello   = percorso_modello_llm_guide(seed).parent
    dir_log_guide = DIR_LOG / "llm_guide" / ("seed" + str(seed))
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_guide.mkdir(parents=True, exist_ok=True)

    # Agente guida LLM — stesso client LLM di AG-LLM-ACT
    guida = AgenteAgLLMGuide(provider=provider, seme=seed)

    print("[AG-LLM-GUIDE] Curriculum v9 LfD - seed=" + str(seed))
    print("[AG-LLM-GUIDE] N_DEMO_LLM_FASE=" + str(N_DEMO_LLM_FASE))
    print("[AG-LLM-GUIDE] Provider LLM=" + provider)

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_dqn = fase["timestep_dqn"]
        max_s  = fase["max_step"]

        print("\n[AG-LLM-GUIDE] Fase " + str(i_fase) + ": " + nome)
        print(
            "[AG-LLM-GUIDE] Raccolta " + str(N_DEMO_LLM_FASE)
            + " demo LLM (max_step=" + str(max_s) + ")..."
        )

        # FASE 1: raccolta delle demo LLM.
        # L'env è avvolto in AggiuntaCanale per avere obs (1,10,10) già in formato DQN;
        # il LLM usa obs[0] (shape 10,10) per costruire i prompt.
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

        # Ambienti DQN di training e validazione.
        # Stessa catena di wrap di AG-DQN: SokobanEnv -> AggiuntaCanale -> Monitor
        env_train = Monitor(AggiuntaCanale(
            crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")
        ))
        env_val = Monitor(AggiuntaCanale(
            crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")
        ))

        # CnnPolicy identica ad AG-DQN per un confronto equo
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        config = {k: v for k, v in CONFIG_DQN.items() if k != "verbose"}

        # FASE 2: crea o aggiorna il modello DQN.
        # Prima fase: modello da zero con replay buffer vuoto. Fasi successive: solo set_env(),
        # così il buffer conserva l'esperienza accumulata (incluse le demo delle fasi precedenti).
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

        # Pre-riempie il replay buffer con le demo LLM di questa fase
        n_trans = guida.riempi_replay_buffer(modello, episodi_demo)
        print(
            "[AG-LLM-GUIDE] Buffer occupazione: "
            + str(n_trans) + " nuove transizioni LLM caricate"
        )

        # Callback: valutazione periodica + checkpoint (stessa struttura cartelle di AG-DQN)
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

        # FASE 3: training DQN con curriculum adattivo.
        # Il DQN impara sia dalle demo del LLM (già nel buffer) sia dalla propria esperienza
        # raccolta online (off-policy).
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
                # reset_num_timesteps solo al primissimo learn(), per tenere il contatore
                # globale degli step
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
                        + " (" + str(round(max_sr, 1)) + "%) - avanzamento forzato"
                    )
                break

            print(
                "[AG-LLM-GUIDE] Solve rate " + str(round(max_sr, 1))
                + "% < soglia " + str(soglia) + "% - ripetizione "
                + str(rep + 1) + "/" + str(MAX_RIPETIZIONI_FASE)
            )

        print(
            "[AG-LLM-GUIDE] Fase " + nome + " completata in "
            + str(round((time.time() - t_fase) / 60, 1)) + " min totali"
        )
        env_train.close()
        env_val.close()

    # Salvataggio del modello finale (il DQN addestrato con la guida del LLM)
    percorso_finale = str(dir_modello / ("llm_guide_seed" + str(seed)))
    modello.save(percorso_finale)

    elapsed_tot = round((time.time() - t_totale) / 3600, 2)
    print("[AG-LLM-GUIDE] Training completato in " + str(elapsed_tot) + " ore")
    print("[AG-LLM-GUIDE] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(
        seed=args.seed,
        dir_dati=args.dir_dati,
        provider=args.provider,
    )
