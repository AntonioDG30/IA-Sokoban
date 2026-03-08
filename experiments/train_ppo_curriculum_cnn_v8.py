"""Script di training AG-PPO v8 — curriculum 10x10 fisso.

Differenza chiave rispetto a v7:
    La griglia e' SEMPRE 10x10 per tutte le fasi.
    Questo elimina la causa radice dei fallimenti v3-v7:
    padding_a_10x10() centrava le griglie piccole con offset variabile,
    facendo imparare alla CNN feature di posizione assoluta non trasferibili.

    Con griglia=(10,10) nativa, non viene applicato alcun padding/offset.
    La fase finale (C3: Boxoban 10x10/4box) e' identica per struttura alle
    fasi precedenti — nessun mismatch di observation.

Fix reward: step penalty -0.1 -> -0.01 (in reward.py).
    Soluzioni di 150 mosse ora danno reward positiva netta.

Struttura curriculum v8:
    C0: 10x10/1box   500k  max_step=150  ent_coef=0.01
    C1: 10x10/2box   800k  max_step=200  ent_coef=0.01  (+cassa)
    C2: 10x10/3box  1200k  max_step=250  ent_coef=0.02  (+cassa)
    C3: 10x10/4box  1500k  max_step=300  ent_coef=0.03  (+cassa, Boxoban)
    Totale: 4.0M step

Uso:
    python experiments/train_ppo_curriculum_cnn_v8.py --seed 42 --n_envs 4
    python experiments/train_ppo_curriculum_cnn_v8.py --seed 42 --solo_valuta

Modelli salvati in: models/ppo_v8/
"""

import argparse
import sys
from pathlib import Path
from typing import List

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sokoban_env import SokobanEnv, AggiuntaCanale
from sokoban_env.sokoban_cnn import SokobanCNN
from experiments.config import (
    CONFIG_PPO, SEEDS, FASI_CURRICULUM_V8, SCALA_MANHATTAN,
    DIR_DATI, DIR_LOG, DIR_MODELLI,
    percorso_modello_ppo_v8,
)

POLICY_KWARGS_CNN = {
    "features_extractor_class":  SokobanCNN,
    "features_extractor_kwargs": {"features_dim": 256},
}


def _crea_env_fase(fase: dict, dir_dati: str, seme: int) -> Monitor:
    """Crea SokobanEnv + AggiuntaCanale + Monitor per la fase specificata."""
    griglia = tuple(fase["griglia"])
    max_step = fase["max_step"]
    if fase["dataset"] == "generato":
        env = SokobanEnv(
            griglia_size=griglia,
            n_casse=fase["n_casse"],
            scala_manhattan=SCALA_MANHATTAN,
            max_step=max_step,
            seme=seme,
        )
    else:
        env = SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split="train",
            griglia_size=griglia,
            n_casse=fase["n_casse"],
            scala_manhattan=SCALA_MANHATTAN,
            max_step=max_step,
            seme=seme,
        )
    return Monitor(AggiuntaCanale(env))


def _crea_vecenv_fase(fase: dict, dir_dati: str, seme: int, n_envs: int):
    """Crea VecEnv con AggiuntaCanale per la fase specificata."""
    griglia = tuple(fase["griglia"])
    max_step = fase["max_step"]
    if fase["dataset"] == "generato":
        def _factory():
            return AggiuntaCanale(SokobanEnv(
                griglia_size=griglia,
                n_casse=fase["n_casse"],
                scala_manhattan=SCALA_MANHATTAN,
                max_step=max_step,
            ))
    else:
        def _factory():
            return AggiuntaCanale(SokobanEnv(
                directory_livelli=dir_dati,
                difficolta="unfiltered",
                split="train",
                griglia_size=griglia,
                n_casse=fase["n_casse"],
                scala_manhattan=SCALA_MANHATTAN,
                max_step=max_step,
            ))
    return make_vec_env(_factory, n_envs=n_envs, seed=seme)


def addestra_curriculum(seme: int, n_envs: int, solo_valuta: bool) -> None:
    """Esegue il training curriculum PPO-CNN v8 per un singolo seed."""
    dir_dati = str(DIR_DATI) if DIR_DATI.exists() else None
    dir_output = str(DIR_MODELLI / "ppo_v8")
    percorso = percorso_modello_ppo_v8(seme)

    if solo_valuta:
        if not percorso.with_suffix(".zip").exists():
            print(f"[train_ppo_v8] Modello non trovato: {percorso}.zip")
            return
        env_test = Monitor(AggiuntaCanale(SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split="test",
        )))
        modello = PPO.load(str(percorso), env=env_test)
        _valuta_finale(modello, dir_dati, seme)
        return

    Path(dir_output).mkdir(parents=True, exist_ok=True)

    config = {k: v for k, v in CONFIG_PPO.items() if k not in ("tensorboard_log", "policy")}
    modello = None

    for i, fase in enumerate(FASI_CURRICULUM_V8):
        nome_fase = fase["nome"]
        timestep_fase = fase["timestep"]
        ent_coef_fase = fase["ent_coef"]
        max_step_fase = fase["max_step"]
        prima_fase = (i == 0)

        print(f"\n{'=' * 60}")
        print(f"[PPO-CNN-v8] Seed {seme} | {nome_fase} | {timestep_fase:,} step")
        print(f"  Griglia: {fase['griglia']} | Casse: {fase['n_casse']}")
        print(f"  max_step={max_step_fase} | ent_coef={ent_coef_fase} | padding=7")
        print(f"  scala_manhattan={SCALA_MANHATTAN} | step_penalty=-0.01")
        print(f"{'=' * 60}")

        env_train = _crea_vecenv_fase(fase, dir_dati, seme, n_envs)
        env_val   = _crea_env_fase(fase, dir_dati, seme)

        if prima_fase:
            modello = PPO(
                policy="CnnPolicy",
                env=env_train,
                seed=seme,
                tensorboard_log=str(DIR_LOG),
                policy_kwargs=POLICY_KWARGS_CNN,
                **config,
            )
        else:
            modello.set_env(env_train)

        modello.ent_coef = ent_coef_fase

        callbacks: List = []

        eval_cb = EvalCallback(
            env_val,
            best_model_save_path=f"{dir_output}/best_{nome_fase}_seed{seme}",
            log_path=f"{dir_output}/eval_{nome_fase}_seed{seme}",
            eval_freq=max(10_000 // n_envs, 1),
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            verbose=0,
        )
        callbacks.append(eval_cb)

        ckpt_cb = CheckpointCallback(
            save_freq=max(100_000 // n_envs, 1),
            save_path=f"{dir_output}/checkpoints",
            name_prefix=f"ppo_cnn_v8_{nome_fase}_seed{seme}",
            verbose=0,
        )
        callbacks.append(ckpt_cb)

        modello.learn(
            total_timesteps=timestep_fase,
            callback=callbacks,
            tb_log_name=f"PPO_CNN_V8_seed{seme}",
            reset_num_timesteps=prima_fase,
            progress_bar=False,
        )

        print(f"[PPO-CNN-v8] {nome_fase} completata. ent_coef={ent_coef_fase}")

        # Ricarica il best model della fase appena completata prima di passare
        # alla successiva. Evita di portare regressions da policy instability
        # (es. C2 picco 100% -> fine 35%) al punto di partenza della fase seguente.
        ultima_fase = (i == len(FASI_CURRICULUM_V8) - 1)
        if not ultima_fase:
            best_path = Path(f"{dir_output}/best_{nome_fase}_seed{seme}/best_model.zip")
            if best_path.exists():
                modello = PPO.load(str(best_path), device="auto")
                print(f"[PPO-CNN-v8] Ricaricato best model: {best_path.name}")

    modello.save(str(percorso))
    print(f"\n[PPO-CNN-v8] Modello finale salvato: {percorso}.zip")
    _valuta_finale(modello, dir_dati, seme)


def _valuta_finale(modello, dir_dati, seme: int) -> None:
    """Valuta il modello finale su tutti i set Boxoban."""
    import numpy as np
    for diff, split in [("unfiltered", "test"), ("medium", "valid")]:
        test_path = DIR_DATI / diff / split
        if dir_dati is None or not test_path.exists():
            print(f"[PPO-CNN-v8] {diff}/{split} non trovato, saltato.")
            continue

        env = Monitor(AggiuntaCanale(SokobanEnv(
            directory_livelli=dir_dati,
            difficolta=diff,
            split=split,
        )))

        n_risolti = 0
        reward_totali = []
        n_ep = 100

        for _ in range(n_ep):
            obs, _ = env.reset()
            reward_ep = 0.0
            done = False
            while not done:
                azione, _ = modello.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = env.step(int(azione))
                reward_ep += float(r)
                done = terminated or truncated
            if terminated:
                n_risolti += 1
            reward_totali.append(reward_ep)

        env.close()
        print(
            f"\n[AgentePPO-v8] Valutazione ({diff}/{split}, {n_ep} episodi):\n"
            f"  Solve rate:        {n_risolti/n_ep*100:.1f}%\n"
            f"  Reward cumulativa: {float(np.mean(reward_totali)):.3f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training PPO v8: curriculum 10x10 fisso, 4 fasi su n_casse."
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed specifico (default: tutti i seed in SEEDS).")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Ambienti paralleli (default: 4).")
    parser.add_argument("--solo_valuta", action="store_true",
                        help="Salta il training e valuta il modello salvato.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_da_usare = [args.seed] if args.seed is not None else SEEDS

    print("=" * 60)
    print("AG-PPO v8 -- Curriculum 10x10 Fisso (solo n_casse varia)")
    print(f"  Seed:            {seed_da_usare}")
    print(f"  n_envs:          {args.n_envs}")
    print(f"  Scala Manhattan: {SCALA_MANHATTAN}")
    print(f"  Step penalty:    -0.01 (fisso in reward.py)")
    print(f"  Fasi:            {[f['nome'] for f in FASI_CURRICULUM_V8]}")
    print(f"  Timestep tot:    {sum(f['timestep'] for f in FASI_CURRICULUM_V8):,}")
    print(f"  max_step/fase:   {[f['max_step'] for f in FASI_CURRICULUM_V8]}")
    print(f"  ent_coef/fase:   {[f['ent_coef'] for f in FASI_CURRICULUM_V8]}")
    print(f"  CNN:             SokobanCNN(features_dim=256, norm=/7.0)")
    print("=" * 60)

    for seme in seed_da_usare:
        print(f"\n{'-' * 40}")
        print(f"SEED {seme}")
        print(f"{'-' * 40}")
        addestra_curriculum(seme=seme, n_envs=args.n_envs, solo_valuta=args.solo_valuta)

    print("\n[train_ppo_curriculum_cnn_v8] Tutti i seed completati.")


if __name__ == "__main__":
    main()
