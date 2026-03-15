"""Valutazione comparativa di tutti gli agenti su tutte le fasi curriculum v9.

Valuta AG-PPO, AG-DQN, AG-LLM, AG-LLM-GUIDE, AG-LLM-REW su:
    - Ogni fase del curriculum C0->C5 (split test/valid)
    - Set extra: Boxoban medium/valid e unfiltered/test (SET_VALUTAZIONE)

Agenti (5 totali):
    AG-PPO       : PPO baseline (traccia PDF)
    AG-DQN       : DQN baseline (traccia PDF)
    AG-LLM       : LLM direct policy a inference time (traccia PDF)
    AG-LLM-GUIDE : DQN addestrato via LfD con demo LLM (mail professore)
                   A inference time usa solo il DQN (no LLM).
    AG-LLM-REW   : PPO con reward LLM (traccia PDF + mail professore)

Modalita' di caricamento modelli:
    - Per le fasi C0->C5: carica il BEST model per fase (salvato da EvalCallback).
      Evita il catastrophic forgetting del modello finale.
      Percorso: models/{agente}/{nome_fase}/best/best_model.zip
    - Per SET_VALUTAZIONE (val-medium, val-unfiltered): usa il best model C5.
    - Fallback: se il best model non esiste, usa il modello finale.

Output:
    - Stampa tabella riepilogativa a terminale
    - Salva risultati in docs/report/risultati_comparativi_seed{seed}.json

Uso:
    python experiments/evaluate_all.py [--seed 42] [--dir-dati data/boxoban]
                                       [--no-llm]
    --no-llm: salta AG-LLM e AG-LLM-REW (utile prima del training LLM)
              AG-LLM-GUIDE NON viene saltato (inference senza LLM)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from experiments.config import (
    FASI_CURRICULUM_V9,
    SET_VALUTAZIONE,
    DIR_DATI,
    DIR_MODELLI,
    DIR_RISULTATI,
    N_EPISODI_VALUTAZIONE,
    PROVIDER_DEFAULT,
    N_EPISODI_LLM_ACT,
    crea_env_da_fase,
    percorso_modello_ppo,
    percorso_modello_dqn,
    percorso_risultati_llm_act,
    percorso_modello_llm_rew,
    percorso_modello_llm_guide,
)
from agents.llm_act_agent import AgenteAgLLM
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# Helper: best model per fase (evita catastrophic forgetting)
# ---------------------------------------------------------------------------

def _carica_best_o_finale(cls, base_dir: Path, nome_fase: str, fallback: Path):
    """Carica best model per la fase data. Se non esiste usa il modello finale.

    Args:
        cls:        Classe SB3 (PPO o DQN)
        base_dir:   Cartella base agente (es. models/ppo)
        nome_fase:  Nome fase curriculum (es. 'C0-1box-gen')
        fallback:   Path modello finale (senza estensione .zip)
    Returns:
        (modello, fonte) dove fonte e' 'best' o 'finale'
    """
    best = base_dir / nome_fase / "best" / "best_model.zip"
    if best.exists():
        return cls.load(str(best.with_suffix(""))), "best"
    if fallback.with_suffix(".zip").exists():
        return cls.load(str(fallback)), "finale"
    return None, "non trovato"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Valutazione comparativa tutti gli agenti")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path data/boxoban")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--no-llm",   action="store_true",
                   help="Salta AG-LLM e AG-LLM-REW (non AG-LLM-GUIDE: inference senza LLM)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers valutazione
# ---------------------------------------------------------------------------

def _valuta_rl(modello, env, n_episodi: int) -> Dict[str, Any]:
    """Valuta un modello SB3 (PPO o DQN) su n_episodi. Restituisce metriche."""
    n_risolti = 0
    mosse_risolti = []
    rewards = []
    casse_totali = []

    for _ in range(n_episodi):
        obs, _ = env.reset()
        reward_ep = 0.0
        step_ep = 0
        casse_finali = 0
        done = False
        while not done:
            azione, _ = modello.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(azione))
            reward_ep += float(r)
            step_ep += 1
            casse_finali = info.get("casse_su_target", 0)
            done = terminated or truncated
        if terminated:
            n_risolti += 1
            mosse_risolti.append(step_ep)
        rewards.append(reward_ep)
        casse_totali.append(casse_finali)

    solve_rate = n_risolti / n_episodi * 100
    return {
        "solve_rate":        round(solve_rate, 2),
        "mosse_medie":       round(float(np.mean(mosse_risolti)), 2) if mosse_risolti else 0.0,
        "reward_cumulativa": round(float(np.mean(rewards)), 4),
        "casse_su_target":   round(float(np.mean(casse_totali)), 3),
        "n_episodi":         n_episodi,
        "n_risolti":         n_risolti,
    }


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def valuta_tutti(seed: int, dir_dati: str, provider: str, no_llm: bool) -> None:
    dir_dati_path = Path(dir_dati)
    DIR_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("\n[evaluate_all] ========================================")
    print("[evaluate_all] Valutazione comparativa 5 agenti — seed=" + str(seed))
    print("[evaluate_all] Modalita': best model per fase (evita forgetting)")
    print("[evaluate_all] ========================================\n")

    risultati: Dict[str, Any] = {"seed": seed, "modalita": "best_per_fase", "agenti": {}}

    # Cartelle base modelli RL
    dir_ppo   = DIR_MODELLI / "ppo"
    dir_dqn   = DIR_MODELLI / "dqn"
    dir_guide = DIR_MODELLI / "llm_guide"
    dir_rew   = DIR_MODELLI / "llm_rew"

    # Fallback: modelli finali (usati se best non esiste)
    fb_ppo   = percorso_modello_ppo(seed)
    fb_dqn   = percorso_modello_dqn(seed)
    fb_guide = percorso_modello_llm_guide(seed)
    fb_rew   = percorso_modello_llm_rew(seed)

    # AG-LLM: carica risultati JSON pre-calcolati da train_llm_act.py
    agente_llm = None
    risultati_llm_precaricati = None
    if not no_llm:
        percorso_llm_json = percorso_risultati_llm_act(seed)
        if percorso_llm_json.exists():
            with open(percorso_llm_json, "r", encoding="utf-8") as f:
                risultati_llm_precaricati = json.load(f)
            print("[evaluate_all] AG-LLM       risultati JSON: " + str(percorso_llm_json))
        else:
            print("[evaluate_all] AG-LLM       JSON non trovato, valutazione live...")
            agente_llm = AgenteAgLLM(provider=provider, seme=seed)

    # ---------------------------------------------------------------
    # Valutazione per fase curriculum + set extra
    # Per SET_VALUTAZIONE usa il best model C5 (fase Boxoban piu' avanzata)
    # ---------------------------------------------------------------
    fasi_curriculum = list(FASI_CURRICULUM_V9)
    fasi_extra = [
        {**sv, "nome": sv["nome"], "max_step": 300, "n_casse": 4,
         "dataset": sv["dataset"], "timestep_ppo": 0, "timestep_dqn": 0, "ent_coef": 0}
        for sv in SET_VALUTAZIONE
    ]
    fasi_da_valutare = fasi_curriculum + fasi_extra

    # Fase da usare come riferimento per i set extra (best model C5)
    nome_fase_boxoban = "C5-4box-unfiltered"

    for fase in fasi_da_valutare:
        nome  = fase["nome"]
        max_s = fase["max_step"]
        split = fase.get("split", "test")
        # Per set extra usa il best model dell'ultima fase curriculum Boxoban
        nome_best = nome if nome in [f["nome"] for f in fasi_curriculum] else nome_fase_boxoban

        print("\n[evaluate_all] --- " + nome + " (split=" + split + ") ---")

        metriche_fase: Dict[str, Any] = {}

        # AG-PPO
        mod_ppo, fonte_ppo = _carica_best_o_finale(PPO, dir_ppo, nome_best, fb_ppo)
        if mod_ppo is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-PPO"] = _valuta_rl(mod_ppo, env, N_EPISODI_VALUTAZIONE)
            metriche_fase["AG-PPO"]["fonte_modello"] = fonte_ppo
            env.close()
            print("  AG-PPO       [" + fonte_ppo + "]: solve="
                  + str(metriche_fase["AG-PPO"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-PPO"]["reward_cumulativa"]))

        # AG-DQN
        mod_dqn, fonte_dqn = _carica_best_o_finale(DQN, dir_dqn, nome_best, fb_dqn)
        if mod_dqn is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-DQN"] = _valuta_rl(mod_dqn, env, N_EPISODI_VALUTAZIONE)
            metriche_fase["AG-DQN"]["fonte_modello"] = fonte_dqn
            env.close()
            print("  AG-DQN       [" + fonte_dqn + "]: solve="
                  + str(metriche_fase["AG-DQN"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-DQN"]["reward_cumulativa"]))

        # AG-LLM — risultati precaricati o valutazione live
        if not no_llm:
            if risultati_llm_precaricati is not None:
                m = risultati_llm_precaricati.get("fasi", {}).get(nome)
                if m:
                    metriche_fase["AG-LLM"] = m
                    print("  AG-LLM       [JSON]: solve=" + str(m["solve_rate"]) + "%"
                          + " | fallback=" + str(m.get("fallback_rate", "?")) + "%")
            elif agente_llm is not None:
                env = crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)
                metriche_fase["AG-LLM"] = agente_llm.valuta(
                    env, N_EPISODI_LLM_ACT, max_s, nome)
                env.close()
                print("  AG-LLM       [live]: solve="
                      + str(metriche_fase["AG-LLM"]["solve_rate"]) + "%")

        # AG-LLM-GUIDE — inference: solo DQN, nessuna chiamata LLM
        mod_guide, fonte_guide = _carica_best_o_finale(DQN, dir_guide, nome_best, fb_guide)
        if mod_guide is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-LLM-GUIDE"] = _valuta_rl(mod_guide, env, N_EPISODI_VALUTAZIONE)
            metriche_fase["AG-LLM-GUIDE"]["fonte_modello"] = fonte_guide
            env.close()
            print("  AG-LLM-GUIDE [" + fonte_guide + "]: solve="
                  + str(metriche_fase["AG-LLM-GUIDE"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-LLM-GUIDE"]["reward_cumulativa"]))

        # AG-LLM-REW — inference: solo la policy PPO, nessuna chiamata LLM
        if not no_llm:
            mod_rew, fonte_rew = _carica_best_o_finale(PPO, dir_rew, nome_best, fb_rew)
            if mod_rew is not None:
                env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
                metriche_fase["AG-LLM-REW"] = _valuta_rl(mod_rew, env, N_EPISODI_VALUTAZIONE)
                metriche_fase["AG-LLM-REW"]["fonte_modello"] = fonte_rew
                env.close()
                print("  AG-LLM-REW   [" + fonte_rew + "]: solve="
                      + str(metriche_fase["AG-LLM-REW"]["solve_rate"]) + "%"
                      + " | rew=" + str(metriche_fase["AG-LLM-REW"]["reward_cumulativa"]))

        risultati["agenti"][nome] = metriche_fase

    # ---------------------------------------------------------------
    # Tabella riepilogativa solve rate (%)
    # ---------------------------------------------------------------
    print("\n\n[evaluate_all] === RIEPILOGO SOLVE RATE (%) ===")
    agenti_nomi = ["AG-PPO", "AG-DQN", "AG-LLM", "AG-LLM-GUIDE", "AG-LLM-REW"]
    header = "Fase".ljust(24) + "".join(n.ljust(15) for n in agenti_nomi)
    print(header)
    print("-" * len(header))
    for nome, per_agente in risultati["agenti"].items():
        riga = nome.ljust(24)
        for ag in agenti_nomi:
            if ag in per_agente:
                riga += str(per_agente[ag]["solve_rate"]).ljust(15)
            else:
                riga += "N/A".ljust(15)
        print(riga)

    # ---------------------------------------------------------------
    # Salvataggio JSON
    # ---------------------------------------------------------------
    percorso_out = DIR_RISULTATI / ("risultati_comparativi_seed" + str(seed) + ".json")
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati, f, indent=2, ensure_ascii=False)
    print("\n[evaluate_all] Risultati salvati: " + str(percorso_out))


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutti(
        seed=args.seed,
        dir_dati=args.dir_dati,
        provider=args.provider,
        no_llm=args.no_llm,
    )
