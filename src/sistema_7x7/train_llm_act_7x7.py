# Valutazione di AG-LLM-ACT sul curriculum 7x7 (C0->C2) — gemello di train_llm_act.py.
#
# AG-LLM-ACT non ha training: a ogni step il LLM genera l'azione. Valuta su
# N_EPISODI_LLM_ACT_7x7 episodi per fase, con checkpoint per il resume automatico in caso di
# interruzione. Identico a src/sistema_10x10/train_llm_act.py tranne che per: 3 fasi generate
# C0/C1/C2 invece di 6, l'ambiente SokobanEnv7x7 (7,7) e l'assenza di dataset Boxoban.
#
# Uso: python src/sistema_7x7/train_llm_act_7x7.py [--seed 42] [--provider ollama]
# Risultati in artifacts/models/7x7/llm_act/risultati_7x7_seed{seed}.json (checkpoint per-fase accanto).

import argparse
import json
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sistema_7x7.config_7x7 import (
    FASI_CURRICULUM_7x7,
    N_EPISODI_LLM_ACT_7x7,
    PROVIDER_DEFAULT,
    percorso_llm_act_7x7,
    crea_env_7x7,
)
from core.agenti_llm.llm_act_agent import AgenteAgLLM


# CLI

def _parse_args():
    """Legge --seed e --provider dalla riga di comando."""
    p = argparse.ArgumentParser(description="Valutazione AG-LLM-ACT su curriculum 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    return p.parse_args()


# HELPER PER IL CHECKPOINT — IDENTICI A train_llm_act.py

def _percorso_checkpoint(percorso_out: Path) -> Path:
    """
    Deriva il path del checkpoint da quello dei risultati, sostituendo 'risultati_' con
    'checkpoint_' (es. risultati_7x7_seed42.json -> checkpoint_7x7_seed42.json).
    """
    return percorso_out.parent / percorso_out.name.replace("risultati_", "checkpoint_")


def _carica_checkpoint(percorso_ckpt: Path) -> dict:
    """Restituisce il dizionario delle fasi già completate, o uno vuoto se il file non c'è."""
    if percorso_ckpt.exists():
        try:
            with open(percorso_ckpt, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _salva_checkpoint(percorso_ckpt: Path, fasi_completate: dict) -> None:
    """Scrive su disco le fasi già completate, così il processo si può riprendere dopo un'interruzione."""
    with open(percorso_ckpt, "w", encoding="utf-8") as f:
        json.dump(fasi_completate, f, indent=2, ensure_ascii=False)


# VALUTAZIONE — IDENTICA A train_llm_act.py

def valuta_tutte_le_fasi(seed: int, provider: str) -> None:
    """Valuta AG-LLM-ACT su tutte le fasi C0->C2 del 7x7 e salva i risultati in JSON."""
    percorso_out  = percorso_llm_act_7x7(seed)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)
    percorso_ckpt = _percorso_checkpoint(percorso_out)

    # Resume: recupera le fasi eventualmente già completate
    fasi_completate = _carica_checkpoint(percorso_ckpt)
    if fasi_completate:
        print("[AG-LLM-ACT-7x7] Checkpoint trovato -- riprendo da fase successiva.")
        print("[AG-LLM-ACT-7x7] Fasi gia' completate: " + ", ".join(fasi_completate.keys()))

    print("[AG-LLM-ACT-7x7] Valutazione curriculum 7x7 -- seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-ACT-7x7] Episodi per fase: " + str(N_EPISODI_LLM_ACT_7x7))

    agente = AgenteAgLLM(provider=provider, seme=seed)

    risultati_globali = {
        "seed":     seed,
        "provider": provider,
        "fasi":     dict(fasi_completate),
    }
    t_totale = time.time()

    for fase in FASI_CURRICULUM_7x7:
        nome  = fase["nome"]
        max_s = fase["max_step"]

        # Salta le fasi già fatte (resume)
        if nome in fasi_completate:
            print("[AG-LLM-ACT-7x7] " + nome + " -- gia' completata (skip)")
            continue

        print("\n[AG-LLM-ACT-7x7] Fase: " + nome)

        env = crea_env_7x7(fase, seed)
        t0  = time.time()

        metriche = agente.valuta(
            env=env,
            n_episodi=N_EPISODI_LLM_ACT_7x7,
            max_step=max_s,
            nome_fase=nome,
        )
        elapsed = time.time() - t0
        metriche["elapsed_sec"] = round(elapsed, 1)

        risultati_globali["fasi"][nome] = metriche
        fasi_completate[nome] = metriche
        env.close()

        # Salva il checkpoint dopo ogni fase
        _salva_checkpoint(percorso_ckpt, fasi_completate)

        print("[AG-LLM-ACT-7x7] " + nome + " completata in "
              + str(round(elapsed / 60, 1)) + " min")

    elapsed_tot = time.time() - t_totale
    risultati_globali["elapsed_totale_sec"] = round(elapsed_tot, 1)

    # Scrive il JSON finale
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati_globali, f, indent=2, ensure_ascii=False)

    # Tolto il checkpoint: la valutazione è completa
    if percorso_ckpt.exists():
        percorso_ckpt.unlink()

    print("[AG-LLM-ACT-7x7] Valutazione completata in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-ACT-7x7] Risultati salvati: " + str(percorso_out))

    print("\n[AG-LLM-ACT-7x7] Riepilogo solve rate per fase:")
    for nome, m in risultati_globali["fasi"].items():
        print("  " + nome.ljust(22) + str(m["solve_rate"]) + "%")


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutte_le_fasi(seed=args.seed, provider=args.provider)
