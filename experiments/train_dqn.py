"""Script di training per AG-DQN (Fase 2).

Addestra il baseline DQN su Boxoban (unfiltered) per tutti i seed definiti
in config.py e salva i checkpoint in models/dqn/.

Uso:
    python experiments/train_dqn.py
    python experiments/train_dqn.py --seed 42
    python experiments/train_dqn.py --seed 42 --timesteps 500000
    python experiments/train_dqn.py --solo_valuta --seed 42

Logging TensorBoard:
    tensorboard --logdir logs/dqn
"""

import argparse
import sys
from pathlib import Path

# Aggiunge la radice del progetto al path per import dei moduli locali
RADICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RADICE))

from experiments.config import (
    CONFIG_DQN,
    SEEDS,
    TIMESTEPS_DQN,
    DIR_DATI,
    DIR_LOG,
    DIR_MODELLI,
    percorso_modello_dqn,
)
from agents.dqn_agent import AgenteDQN


def _parse_args() -> argparse.Namespace:
    """Analizza gli argomenti da riga di comando."""
    parser = argparse.ArgumentParser(
        description="Training AG-DQN su Sokoban Boxoban."
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
        default=TIMESTEPS_DQN,
        help=f"Numero totale di timestep (default: {TIMESTEPS_DQN:,}).",
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
    difficolta: str,
    solo_valuta: bool,
) -> None:
    """Esegue training e/o valutazione per un singolo seed.

    Parametri:
        seme:        seed per la riproducibilità.
        timesteps:   numero totale di timestep.
        difficolta:  set Boxoban.
        solo_valuta: se True, salta il training.
    """
    dir_livelli = str(DIR_DATI) if DIR_DATI.exists() else None
    percorso = percorso_modello_dqn(seme)

    agente = AgenteDQN(
        config_dqn=CONFIG_DQN,
        directory_livelli=dir_livelli,
        difficolta=difficolta,
        seme=seme,
    )

    if not solo_valuta:
        agente.addestra(
            totale_timesteps=timesteps,
            dir_log=str(DIR_LOG),
            dir_modello=str(DIR_MODELLI / "dqn"),
            frequenza_eval=10_000,
            frequenza_checkpoint=100_000,
        )
        agente.salva(str(percorso))
    else:
        if not percorso.with_suffix(".zip").exists():
            print(f"[train_dqn] Modello non trovato: {percorso}.zip — esegui prima il training.")
            return
        agente.carica(str(percorso))

    # Valutazione finale su tutti e tre i set di difficoltà
    for split_test in [("unfiltered", "test"), ("medium", "valid"), ("hard", "test")]:
        diff_test, split = split_test
        test_dir = DIR_DATI / diff_test / split
        if dir_livelli is None or not test_dir.exists():
            print(f"[train_dqn] Dati {diff_test}/{split} non trovati, saltato.")
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
    print("AG-DQN — Training Sokoban")
    print(f"  Seed:       {seed_da_usare}")
    print(f"  Timesteps:  {args.timesteps:,}")
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
            difficolta=args.difficolta,
            solo_valuta=args.solo_valuta,
        )

    print("\n[train_dqn] Tutti i seed completati.")


if __name__ == "__main__":
    main()
