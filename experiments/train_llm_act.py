"""Valutazione AG-LLM (policy diretta LLM) su curriculum C0->C5.

AG-LLM non ha training: il LLM viene chiamato ad ogni step per generare
l'azione. Questo script valuta l'agente su N_EPISODI_LLM_ACT episodi per
ogni fase del curriculum e salva i risultati in JSON.

Uso:
    python experiments/train_llm_act.py [--seed 42] [--provider ollama]
                                        [--dir-dati data/boxoban]

Risultati salvati in models/llm_act/risultati_seed{seed}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from experiments.config import (
    FASI_CURRICULUM_V9,
    DIR_DATI,
    N_EPISODI_LLM_ACT,
    PROVIDER_DEFAULT,
    crea_env_da_fase,
    percorso_risultati_llm_act,
)
from agents.llm_act_agent import AgenteAgLLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Valutazione AG-LLM su curriculum v9")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path data/boxoban")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------

def valuta_tutte_le_fasi(seed: int, provider: str, dir_dati: str) -> None:
    """Valuta AG-LLM su tutte le fasi C0->C5 e salva risultati JSON."""
    dir_dati_path = Path(dir_dati)
    percorso_out  = percorso_risultati_llm_act(seed)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)

    print("\n[AG-LLM] ========================================")
    print("[AG-LLM] Valutazione curriculum v9 — seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM] Episodi per fase: " + str(N_EPISODI_LLM_ACT))
    print("[AG-LLM] ========================================\n")

    agente = AgenteAgLLM(provider=provider, seme=seed)

    risultati_globali = {
        "seed":     seed,
        "provider": provider,
        "fasi":     {},
    }
    t_totale = time.time()

    for fase in FASI_CURRICULUM_V9:
        nome    = fase["nome"]
        max_s   = fase["max_step"]

        print("\n[AG-LLM] --- Fase: " + nome + " ---")

        # Usa split test per valutazione (generato non ha split, ma split ignorato)
        env = crea_env_da_fase(fase, str(dir_dati_path), seed, split="test")

        t0 = time.time()
        metriche = agente.valuta(
            env=env,
            n_episodi=N_EPISODI_LLM_ACT,
            max_step=max_s,
            nome_fase=nome,
        )
        elapsed = time.time() - t0
        metriche["elapsed_sec"] = round(elapsed, 1)

        risultati_globali["fasi"][nome] = metriche
        env.close()

        print("[AG-LLM] " + nome + " completata in "
              + str(round(elapsed / 60, 1)) + " min")

    elapsed_tot = time.time() - t_totale
    risultati_globali["elapsed_totale_sec"] = round(elapsed_tot, 1)

    # Salva JSON
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati_globali, f, indent=2, ensure_ascii=False)

    print("\n[AG-LLM] ========================================")
    print("[AG-LLM] Valutazione completata in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM] Risultati salvati: " + str(percorso_out))

    # Riepilogo
    print("\n[AG-LLM] Riepilogo solve rate per fase:")
    for nome, m in risultati_globali["fasi"].items():
        print("  " + nome.ljust(22) + str(m["solve_rate"]) + "%")
    print("[AG-LLM] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutte_le_fasi(
        seed=args.seed,
        provider=args.provider,
        dir_dati=args.dir_dati,
    )
