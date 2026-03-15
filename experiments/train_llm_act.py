"""Valutazione AG-LLM (policy diretta LLM) su curriculum C0->C5.

AG-LLM non ha training: il LLM viene chiamato ad ogni step per generare
l'azione. Questo script valuta l'agente su N_EPISODI_LLM_ACT episodi per
ogni fase del curriculum e salva i risultati in JSON.

Uso:
    python experiments/train_llm_act.py [--seed 42] [--provider ollama]
                                        [--dir-dati data/boxoban]

Risultati salvati in models/llm_act/risultati_seed{seed}.json
Checkpoint per-fase: models/llm_act/checkpoint_seed{seed}.json
  (consente resume automatico se il processo viene interrotto)
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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _percorso_checkpoint(percorso_out: Path) -> Path:
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
    with open(percorso_ckpt, "w", encoding="utf-8") as f:
        json.dump(fasi_completate, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------

def valuta_tutte_le_fasi(seed: int, provider: str, dir_dati: str) -> None:
    """Valuta AG-LLM su tutte le fasi C0->C5 e salva risultati JSON."""
    dir_dati_path = Path(dir_dati)
    percorso_out  = percorso_risultati_llm_act(seed)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)
    percorso_ckpt = _percorso_checkpoint(percorso_out)

    # Resume: carica fasi gia' completate
    fasi_completate = _carica_checkpoint(percorso_ckpt)
    if fasi_completate:
        print("[AG-LLM] Checkpoint trovato — riprendo da fase successiva.")
        print("[AG-LLM] Fasi gia' completate: " + ", ".join(fasi_completate.keys()))

    print("\n[AG-LLM] ========================================")
    print("[AG-LLM] Valutazione curriculum v9 — seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM] Episodi per fase: " + str(N_EPISODI_LLM_ACT))
    print("[AG-LLM] ========================================\n")

    agente = AgenteAgLLM(provider=provider, seme=seed)

    risultati_globali = {
        "seed":     seed,
        "provider": provider,
        "fasi":     dict(fasi_completate),  # include gia' completate
    }
    t_totale = time.time()

    for fase in FASI_CURRICULUM_V9:
        nome  = fase["nome"]
        max_s = fase["max_step"]

        # Skip se gia' completata (resume)
        if nome in fasi_completate:
            print("[AG-LLM] " + nome + " — gia' completata (skip)")
            continue

        print("\n[AG-LLM] --- Fase: " + nome + " ---")

        # Sceglie split corretto per dataset:
        #   generato           -> "train" (split ignorato dal generatore)
        #   boxoban_medium     -> "valid" (medium non ha split test)
        #   boxoban_unfiltered -> "test"
        _dataset = fase.get("dataset", "generato")
        if _dataset == "boxoban_medium":
            _split = "valid"
        elif _dataset == "boxoban_unfiltered":
            _split = "test"
        else:
            _split = "train"
        env = crea_env_da_fase(fase, str(dir_dati_path), seed, split=_split)

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
        fasi_completate[nome] = metriche
        env.close()

        # Salva checkpoint dopo ogni fase
        _salva_checkpoint(percorso_ckpt, fasi_completate)

        print("[AG-LLM] " + nome + " completata in "
              + str(round(elapsed / 60, 1)) + " min")

    elapsed_tot = time.time() - t_totale
    risultati_globali["elapsed_totale_sec"] = round(elapsed_tot, 1)

    # Salva JSON finale
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati_globali, f, indent=2, ensure_ascii=False)

    # Rimuove checkpoint (completato)
    if percorso_ckpt.exists():
        percorso_ckpt.unlink()

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
