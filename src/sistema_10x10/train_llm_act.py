# Valutazione di AG-LLM-ACT (il LLM come policy diretta) sul curriculum C0->C5.
#
# AG-LLM-ACT non ha training: a ogni step il LLM genera l'azione. Questo script valuta
# l'agente su N_EPISODI_LLM_ACT episodi per ciascuna fase del curriculum e salva i risultati
# in JSON.
#
# Uso: python src/sistema_10x10/train_llm_act.py [--seed 42] [--provider ollama] [--dir-dati dataset/boxoban]
#
# Risultati in artifacts/models/10x10/llm_act/risultati_seed{seed}.json. Checkpoint per-fase in
# artifacts/models/10x10/llm_act/checkpoint_seed{seed}.json (consente il resume automatico se il processo
# viene interrotto).

import argparse
import json
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sistema_10x10.config import (
    FASI_CURRICULUM_V9,
    DIR_DATI,
    N_EPISODI_LLM_ACT,
    PROVIDER_DEFAULT,
    crea_env_da_fase,
    percorso_risultati_llm_act,
)
from core.agenti_llm.llm_act_agent import AgenteAgLLM


# CLI

def _parse_args():
    """Legge gli argomenti da riga di comando: --seed (int), --provider (str), --dir-dati (path)."""
    p = argparse.ArgumentParser(description="Valutazione AG-LLM-ACT su curriculum v9")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path dataset/boxoban")
    return p.parse_args()


# HELPER PER IL CHECKPOINT (RESUME)

def _percorso_checkpoint(percorso_out: Path) -> Path:
    """
    Deriva il percorso del checkpoint da quello dei risultati, sostituendo 'risultati_' con
    'checkpoint_' nello stesso folder: così basta un'unica variabile per trovare entrambi.
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
    """
    Sovrascrive il file checkpoint con le fasi già completate. Va chiamata dopo ogni fase per
    permettere il resume; ensure_ascii=False preserva eventuali caratteri non-ASCII.
    """
    with open(percorso_ckpt, "w", encoding="utf-8") as f:
        json.dump(fasi_completate, f, indent=2, ensure_ascii=False)


# VALUTAZIONE

def valuta_tutte_le_fasi(seed: int, provider: str, dir_dati: str) -> None:
    """Valuta AG-LLM-ACT su tutte le fasi C0->C5 e salva i risultati in JSON."""
    dir_dati_path = Path(dir_dati)
    percorso_out  = percorso_risultati_llm_act(seed)
    percorso_out.parent.mkdir(parents=True, exist_ok=True)
    percorso_ckpt = _percorso_checkpoint(percorso_out)

    # Resume: recupera le fasi eventualmente già completate
    fasi_completate = _carica_checkpoint(percorso_ckpt)
    if fasi_completate:
        print("[AG-LLM-ACT] Checkpoint trovato - riprendo da fase successiva.")
        print("[AG-LLM-ACT] Fasi gia' completate: " + ", ".join(fasi_completate.keys()))

    print("[AG-LLM-ACT] Valutazione curriculum v9 - seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-ACT] Episodi per fase: " + str(N_EPISODI_LLM_ACT))

    agente = AgenteAgLLM(provider=provider, seme=seed)

    risultati_globali = {
        "seed":     seed,
        "provider": provider,
        "fasi":     dict(fasi_completate),  # parte dalle fasi già completate
    }
    t_totale = time.time()

    for fase in FASI_CURRICULUM_V9:
        nome  = fase["nome"]
        max_s = fase["max_step"]

        # Salta le fasi già fatte (resume)
        if nome in fasi_completate:
            print("[AG-LLM-ACT] " + nome + " - gia' completata (skip)")
            continue

        print("\n[AG-LLM-ACT] Fase: " + nome)

        # Split corretto per ciascun dataset:
        #   generato           -> "train" (il generatore ignora lo split)
        #   boxoban_medium     -> "valid" (medium non ha lo split test)
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

        # Salva il checkpoint dopo ogni fase
        _salva_checkpoint(percorso_ckpt, fasi_completate)

        print("[AG-LLM-ACT] " + nome + " completata in "
              + str(round(elapsed / 60, 1)) + " min")

    elapsed_tot = time.time() - t_totale
    risultati_globali["elapsed_totale_sec"] = round(elapsed_tot, 1)

    # Scrive il JSON finale
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati_globali, f, indent=2, ensure_ascii=False)

    # Tolto il checkpoint: la valutazione è completa
    if percorso_ckpt.exists():
        percorso_ckpt.unlink()

    print("[AG-LLM-ACT] Valutazione completata in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-ACT] Risultati salvati: " + str(percorso_out))

    # Riepilogo del solve rate per fase
    print("\n[AG-LLM-ACT] Riepilogo solve rate per fase:")
    for nome, m in risultati_globali["fasi"].items():
        print("  " + nome.ljust(22) + str(m["solve_rate"]) + "%")


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutte_le_fasi(
        seed=args.seed,
        provider=args.provider,
        dir_dati=args.dir_dati,
    )
