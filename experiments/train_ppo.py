"""Script di training per AG-PPO (Fase 1).

Addestra il baseline PPO su Boxoban (unfiltered) per tutti i seed definiti
in config.py e salva i checkpoint in models/ppo/.

Uso:
    python experiments/train_ppo.py
    python experiments/train_ppo.py --seed 42
    python experiments/train_ppo.py --seed 42 --timesteps 500000
    python experiments/train_ppo.py --n_envs 8

Logging TensorBoard:
    tensorboard --logdir logs/ppo
"""

import argparse
import sys
from pathlib import Path

# Aggiunge la radice del progetto al path per import dei moduli locali
RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from experiments.config import (
    CONFIG_PPO,
    SEEDS,
    TIMESTEPS_PPO,
    DIR_DATI,
    DIR_LOG,
    DIR_MODELLI,
    percorso_modello_ppo,
)
from agents.ppo_agent import AgentePPO


def _parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Training AG-PPO su Sokoban Boxoban."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed specifico da usare (default: tutti i seed di config.py).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS_PPO,
        help=f"Numero totale di timestep (default: {TIMESTEPS_PPO:,}).",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="Numero di ambienti paralleli (default: 4).",
    )
    parser.add_argument(
        "--difficolta",
        type=str,
        default="unfiltered",
        choices=["unfiltered", "medium", "hard"],
        help="Difficoltà del dataset (default: unfiltered).",
    )
    parser.add_argument(
        "--solo_valuta",
        action="store_true",
        help="Salta il training e valuta il modello già salvato.",
    )
    return parser.parse_args()


def addestra_un_seed(
    seme: int,
    timesteps: int,
    n_envs: int,
    difficolta: str,
    solo_valuta: bool,
) -> None:
    """Esegue training e/o valutazione per un singolo seed.

    Parametri:
        seme:        seed per la riproducibilità.
        timesteps:   numero totale di timestep.
        n_envs:      ambienti paralleli.
        difficolta:  set Boxoban.
        solo_valuta: se True, salta il training.
    """
    dir_livelli = str(DIR_DATI) if DIR_DATI.exists() else None
    percorso = percorso_modello_ppo(seme)

    agente = AgentePPO(
        config_ppo=CONFIG_PPO,
        directory_livelli=dir_livelli,
        difficolta=difficolta,
        n_envs=n_envs,
        seme=seme,
    )

    if not solo_valuta:
        agente.addestra(
            totale_timesteps=timesteps,
            dir_log=str(DIR_LOG),
            dir_modello=str(DIR_MODELLI / "ppo"),
            frequenza_eval=10_000,
            frequenza_checkpoint=100_000,
        )
        agente.salva(str(percorso))
    else:
        if not percorso.with_suffix(".zip").exists():
            print(f"[train_ppo] Modello non trovato: {percorso}.zip — esegui prima il training.")
            return
        agente.carica(str(percorso))

    # Valutazione finale su tutti e tre i set di difficoltà
    for split_test in [("unfiltered", "test"), ("medium", "valid"), ("hard", "test")]:
        diff_test, split = split_test
        # Salta se i dati non sono presenti
        test_dir = DIR_DATI / diff_test / split
        if dir_livelli is None or not test_dir.exists():
            print(f"[train_ppo] Dati {diff_test}/{split} non trovati, saltato.")
            continue
        agente.valuta(
            n_episodi=100,
            difficolta=diff_test,
            split=split,
        )


def main() -> None:
    args = _parse_args()

    seed_da_usare = [args.seed] if args.seed is not None else SEEDS

    print("=" * 60)
    print("AG-PPO — Training Sokoban")
    print(f"  Seed:       {seed_da_usare}")
    print(f"  Timesteps:  {args.timesteps:,}")
    print(f"  n_envs:     {args.n_envs}")
    print(f"  Difficoltà: {args.difficolta}")
    print(f"  Dati:       {DIR_DATI}")
    print("=" * 60)

    for seme in seed_da_usare:
        print(f"\n{'-' * 40}")
        print(f"SEED {seme}")
        print(f"{'-' * 40}")
        addestra_un_seed(
            seme=seme,
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            difficolta=args.difficolta,
            solo_valuta=args.solo_valuta,
        )

    print("\n[train_ppo] Tutti i seed completati.")


if __name__ == "__main__":
    main()
