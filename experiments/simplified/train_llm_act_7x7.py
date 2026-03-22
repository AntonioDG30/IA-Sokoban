"""Valutazione AG-LLM su curriculum 7x7 (C0->C2) -- identico a train_llm_act.py.

AG-LLM non ha training: il LLM viene chiamato ad ogni step per generare
l'azione. Valuta su N_EPISODI_LLM_ACT_7x7 episodi per ogni fase.
Checkpoint per resume automatico se il processo viene interrotto.

Identico a experiments/train_llm_act.py eccetto:
    - Fasi: 3 fasi generate C0/C1/C2 invece di 6 fasi C0->C5
    - Env: SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
    - Nessun dataset Boxoban (solo livelli generati)

Uso:
    python experiments/simplified/train_llm_act_7x7.py [--seed 42] [--provider ollama]

Risultati salvati in models_7x7/llm_act/risultati_7x7_seed{seed}.json
Checkpoint per-fase: models_7x7/llm_act/checkpoint_7x7_seed{seed}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from experiments.simplified.config_7x7 import (
    FASI_CURRICULUM_7x7,
    N_EPISODI_LLM_ACT_7x7,
    PROVIDER_DEFAULT,
    percorso_llm_act_7x7,
    crea_env_7x7,
)
from agents.llm_act_agent import AgenteAgLLM


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Legge seed e provider LLM dalla riga di comando.

    Restituisce:
        namespace argparse con attributi seed e provider
    """
    p = argparse.ArgumentParser(description="Valutazione AG-LLM su curriculum 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers -- identici a train_llm_act.py
# ---------------------------------------------------------------------------

def _percorso_checkpoint(percorso_out: Path) -> Path:
    """Deriva il percorso del checkpoint dal percorso del file risultati finale.

    Parametri:
        percorso_out: path del JSON risultati (es. risultati_7x7_seed42.json)
    Restituisce:
        path del file checkpoint (es. checkpoint_7x7_seed42.json)
    """
    return percorso_out.parent / percorso_out.name.replace("risultati_", "checkpoint_")


def _carica_checkpoint(percorso_ckpt: Path) -> dict:
    """Restituisce dizionario fasi gia' completate, o vuoto se assente."""
    if percorso_ckpt.exists():
        try:
            with open(percorso_ckpt, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _salva_checkpoint(percorso_ckpt: Path, fasi_completate: dict) -> None:
    """Salva su disco le fasi gia' completate per consentire il resume.

    Parametri:
        percorso_ckpt:   path del file checkpoint JSON
        fasi_completate: dizionario {nome_fase: metriche}
    """
    with open(percorso_ckpt, "w", encoding="utf-8") as f:
        json.dump(fasi_completate, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Valutazione -- identica a train_llm_act.py
# ---------------------------------------------------------------------------

def valuta_tutte_le_fasi(seed: int, provider: str) -> None:
    """Valuta AG-LLM su tutte le fasi C0->C2 e salva risultati JSON."""
    percorso_out  = percorso_llm_act_7x7(seed)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)
    percorso_ckpt = _percorso_checkpoint(percorso_out)

    # Resume: carica fasi gia' completate
    fasi_completate = _carica_checkpoint(percorso_ckpt)
    if fasi_completate:
        print("[AG-LLM-7x7] Checkpoint trovato -- riprendo da fase successiva.")
        print("[AG-LLM-7x7] Fasi gia' completate: " + ", ".join(fasi_completate.keys()))

    print("\n[AG-LLM-7x7] ========================================")
    print("[AG-LLM-7x7] Valutazione curriculum 7x7 -- seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-7x7] Episodi per fase: " + str(N_EPISODI_LLM_ACT_7x7))
    print("[AG-LLM-7x7] ========================================\n")

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

        # Skip se gia' completata (resume)
        if nome in fasi_completate:
            print("[AG-LLM-7x7] " + nome + " -- gia' completata (skip)")
            continue

        print("\n[AG-LLM-7x7] --- Fase: " + nome + " ---")

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

        # Salva checkpoint dopo ogni fase
        _salva_checkpoint(percorso_ckpt, fasi_completate)

        print("[AG-LLM-7x7] " + nome + " completata in "
              + str(round(elapsed / 60, 1)) + " min")

    elapsed_tot = time.time() - t_totale
    risultati_globali["elapsed_totale_sec"] = round(elapsed_tot, 1)

    # Salva JSON finale
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati_globali, f, indent=2, ensure_ascii=False)

    # Rimuove checkpoint (completato)
    if percorso_ckpt.exists():
        percorso_ckpt.unlink()

    print("\n[AG-LLM-7x7] ========================================")
    print("[AG-LLM-7x7] Valutazione completata in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-7x7] Risultati salvati: " + str(percorso_out))

    print("\n[AG-LLM-7x7] Riepilogo solve rate per fase:")
    for nome, m in risultati_globali["fasi"].items():
        print("  " + nome.ljust(22) + str(m["solve_rate"]) + "%")
    print("[AG-LLM-7x7] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutte_le_fasi(seed=args.seed, provider=args.provider)
