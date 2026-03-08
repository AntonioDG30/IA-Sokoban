"""Script di training AG-PPO v2 con curriculum learning (Fase 2.3).

Addestra PPO in 3 fasi progressive con griglie crescenti:
    C1: 5x5,  2 casse, 300k step  (livelli generati proceduralmente)
    C2: 7x7,  3 casse, 400k step  (livelli generati proceduralmente)
    C3: 10x10, 4 casse, 300k step (dataset Boxoban unfiltered)

Con reward shaping basata su distanza Manhattan (scala=0.3).
I pesi del modello PPO vengono preservati tra le fasi (set_env).
Observation space sempre 10x10 float32 — SB3 non viene reinizializzato.

Uso:
    python experiments/train_ppo_curriculum.py
    python experiments/train_ppo_curriculum.py --seed 42
    python experiments/train_ppo_curriculum.py --seed 42 --solo_valuta

Modelli salvati in: models/ppo_v2/
"""

import argparse
import sys
from pathlib import Path
from typing import List

RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from sokoban_env import SokobanEnv
from experiments.config import (
    CONFIG_PPO, SEEDS, FASI_CURRICULUM, SCALA_MANHATTAN,
    DIR_DATI, DIR_LOG, DIR_MODELLI,
    percorso_modello_ppo_v2,
)


def _crea_env_fase(fase: dict, dir_dati: str, seme: int) -> Monitor:
    """Crea un SokobanEnv+Monitor per la fase curriculum specificata."""
    griglia = tuple(fase["griglia"])
    if fase["dataset"] == "generato":
        env = SokobanEnv(
            griglia_size=griglia,
            n_casse=fase["n_casse"],
            scala_manhattan=SCALA_MANHATTAN,
            seme=seme,
        )
    else:
        # Fase C3: Boxoban 10x10 (con shaping Manhattan attivo)
        env = SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split="train",
            griglia_size=griglia,
            n_casse=fase["n_casse"],
            scala_manhattan=SCALA_MANHATTAN,
            seme=seme,
        )
    return Monitor(env)


def _crea_vecenv_fase(fase: dict, dir_dati: str, seme: int, n_envs: int):
    """Crea un VecEnv (DummyVecEnv) per la fase curriculum specificata."""
    griglia = tuple(fase["griglia"])
    if fase["dataset"] == "generato":
        def _factory():
            return SokobanEnv(
                griglia_size=griglia,
                n_casse=fase["n_casse"],
                scala_manhattan=SCALA_MANHATTAN,
            )
    else:
        def _factory():
            return SokobanEnv(
                directory_livelli=dir_dati,
                difficolta="unfiltered",
                split="train",
                griglia_size=griglia,
                n_casse=fase["n_casse"],
                scala_manhattan=SCALA_MANHATTAN,
            )
    return make_vec_env(_factory, n_envs=n_envs, seed=seme)


def addestra_curriculum(seme: int, n_envs: int, solo_valuta: bool) -> None:
    """Esegue il training curriculum PPO per un singolo seed."""
    dir_dati = str(DIR_DATI) if DIR_DATI.exists() else None
    dir_output = str(DIR_MODELLI / "ppo_v2")
    percorso = percorso_modello_ppo_v2(seme)

    if solo_valuta:
        if not percorso.with_suffix(".zip").exists():
            print(f"[train_ppo_curriculum] Modello non trovato: {percorso}.zip")
            return
        # Carica e valuta solo su Boxoban 10x10
        env_test = Monitor(SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split="test",
        ))
        modello = PPO.load(str(percorso), env=env_test)
        _valuta_finale(modello, dir_dati, seme)
        return

    Path(dir_output).mkdir(parents=True, exist_ok=True)

    config = {k: v for k, v in CONFIG_PPO.items() if k != "tensorboard_log"}
    modello = None

    for i, fase in enumerate(FASI_CURRICULUM):
        nome_fase = fase["nome"]
        timestep_fase = fase["timestep"]
        prima_fase = (i == 0)

        print(f"\n{'=' * 60}")
        print(f"[PPO Curriculum] Seed {seme} | {nome_fase} | {timestep_fase:,} step")
        print(f"  Griglia: {fase['griglia']} | Casse: {fase['n_casse']} | Manhattan: {SCALA_MANHATTAN}")
        print(f"{'=' * 60}")

        # Ambiente training e validazione per questa fase
        env_train = _crea_vecenv_fase(fase, dir_dati, seme, n_envs)
        env_val   = _crea_env_fase(fase, dir_dati, seme)

        if prima_fase:
            # Prima fase: crea il modello PPO
            modello = PPO(
                env=env_train,
                seed=seme,
                tensorboard_log=str(DIR_LOG),
                **config,
            )
        else:
            # Fasi successive: sostituisce l'env mantenendo i pesi
            modello.set_env(env_train)

        # Callbacks
        callbacks: List[BaseCallback] = []

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
            name_prefix=f"ppo_curriculum_{nome_fase}_seed{seme}",
            verbose=0,
        )
        callbacks.append(ckpt_cb)

        modello.learn(
            total_timesteps=timestep_fase,
            callback=callbacks,
            tb_log_name=f"PPO_curriculum_seed{seme}",
            reset_num_timesteps=prima_fase,
            progress_bar=False,
        )

        print(f"[PPO Curriculum] {nome_fase} completata.")

    # Salva modello finale
    modello.save(str(percorso))
    print(f"\n[PPO Curriculum] Modello finale salvato: {percorso}.zip")

    # Valutazione finale su Boxoban 10x10
    _valuta_finale(modello, dir_dati, seme)


def _valuta_finale(modello, dir_dati, seme: int) -> None:
    """Valuta il modello finale su tutti i set Boxoban."""
    for diff, split in [("unfiltered", "test"), ("medium", "valid")]:
        test_path = DIR_DATI / diff / split
        if dir_dati is None or not test_path.exists():
            print(f"[PPO Curriculum] {diff}/{split} non trovato, saltato.")
            continue

        env = Monitor(SokobanEnv(
            directory_livelli=dir_dati,
            difficolta=diff,
            split=split,
        ))

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
        import numpy as np
        print(
            f"\n[AgentePPO-v2] Valutazione ({diff}/{split}, {n_ep} episodi):\n"
            f"  Solve rate:        {n_risolti/n_ep*100:.1f}%\n"
            f"  Reward cumulativa: {float(np.mean(reward_totali)):.3f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training PPO v2 con curriculum learning (Fase 2.3)."
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed specifico (default: tutti i seed).")
    parser.add_argument("--n_envs", type=int, default=4,
                        help="Ambienti paralleli (default: 4).")
    parser.add_argument("--solo_valuta", action="store_true",
                        help="Salta il training e valuta il modello salvato.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seed_da_usare = [args.seed] if args.seed is not None else SEEDS

    print("=" * 60)
    print("AG-PPO v2 — Curriculum Learning + Manhattan Shaping")
    print(f"  Seed:          {seed_da_usare}")
    print(f"  n_envs:        {args.n_envs}")
    print(f"  Scala Manhattan: {SCALA_MANHATTAN}")
    print(f"  Fasi:          {[f['nome'] for f in FASI_CURRICULUM]}")
    print(f"  Timestep tot:  {sum(f['timestep'] for f in FASI_CURRICULUM):,}")
    print("=" * 60)

    for seme in seed_da_usare:
        print(f"\n{'-' * 40}")
        print(f"SEED {seme}")
        print(f"{'-' * 40}")
        addestra_curriculum(seme=seme, n_envs=args.n_envs, solo_valuta=args.solo_valuta)

    print("\n[train_ppo_curriculum] Tutti i seed completati.")


if __name__ == "__main__":
    main()
