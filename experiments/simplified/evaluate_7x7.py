"""Valutazione comparativa tutti gli agenti su curriculum 7x7 -- identico a evaluate_all.py.

Valuta AG-PPO, AG-DQN, AG-LLM-ACT, AG-LLM-GUIDE, AG-LLM-REW su:
    - Ogni fase del curriculum C0->C2 (3 fasi generate, nessun Boxoban)

Agenti (5 totali):
    AG-PPO       : PPO baseline (RecurrentPPO, CnnLstmPolicy)
    AG-DQN       : DQN baseline (CnnPolicy)
    AG-LLM-ACT   : LLM direct policy a inference time
    AG-LLM-GUIDE : DQN addestrato via LfD con demo LLM.
                   A inference time usa solo il DQN (no LLM).
    AG-LLM-REW   : PPO con reward LLM durante training.
                   A inference time usa solo la policy PPO (no LLM).

Identico a experiments/evaluate_all.py eccetto:
    - Env: SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
    - Fasi: 3 fasi generate C0/C1/C2 invece di 6 fasi C0->C5
    - Nessun dataset Boxoban (solo livelli generati)
    - Nessun SET_VALUTAZIONE extra

Modalita' di caricamento modelli:
    - Per le fasi C0->C2: carica il BEST model per fase (salvato da EvalCallback).
      Evita il catastrophic forgetting del modello finale.
      Percorso: models_7x7/{agente}/{nome_fase}/best/best_model.zip
    - Fallback: se il best model non esiste, usa il modello finale.

Output:
    - Stampa tabella riepilogativa a terminale
    - Salva risultati in docs/report/risultati_7x7_seed{seed}.json

Uso:
    python experiments/simplified/evaluate_7x7.py [--seed 42]
    python experiments/simplified/evaluate_7x7.py [--seed 42] [--no-llm]
    --no-llm: salta AG-LLM-ACT e AG-LLM-REW (utile prima del training LLM)
              AG-LLM-GUIDE NON viene saltato (inference senza LLM)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from experiments.simplified.config_7x7 import (
    FASI_CURRICULUM_7x7,
    DIR_MODELLI_7x7,
    DIR_RISULTATI_7x7,
    N_EPISODI_VALUTAZIONE_7x7,
    N_EPISODI_LLM_ACT_7x7,
    PROVIDER_DEFAULT,
    percorso_ppo_7x7,
    percorso_dqn_7x7,
    percorso_llm_guide_7x7,
    percorso_llm_rew_7x7,
    percorso_llm_act_7x7,
    crea_env_7x7,
)
from agents.llm_act_agent import AgenteAgLLM
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# Helper: best model per fase (evita catastrophic forgetting) -- identico a evaluate_all.py
# ---------------------------------------------------------------------------

def _carica_best_o_finale(cls, base_dir: Path, nome_fase: str, fallback: Path):
    """Carica best model per la fase data. Se non esiste usa il modello finale.

    Args:
        cls:        Classe SB3 (PPO o DQN)
        base_dir:   Cartella base agente (es. models_7x7/ppo)
        nome_fase:  Nome fase curriculum (es. 'C0-1box-7x7')
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
    p = argparse.ArgumentParser(description="Valutazione comparativa tutti gli agenti 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--no-llm",   action="store_true",
                   help="Salta AG-LLM-ACT e AG-LLM-REW (non AG-LLM-GUIDE: inference senza LLM)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers valutazione -- identici a evaluate_all.py
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
# Funzione principale -- identica a evaluate_all.py, adattata per 7x7
# ---------------------------------------------------------------------------

def valuta_tutti(seed: int, provider: str, no_llm: bool) -> None:
    DIR_RISULTATI_7x7.mkdir(parents=True, exist_ok=True)

    print("\n[evaluate_7x7] ========================================")
    print("[evaluate_7x7] Valutazione comparativa 5 agenti 7x7 -- seed=" + str(seed))
    print("[evaluate_7x7] Modalita': best model per fase (evita forgetting)")
    print("[evaluate_7x7] ========================================\n")

    risultati: Dict[str, Any] = {"seed": seed, "modalita": "best_per_fase", "agenti": {}}

    # Cartelle base modelli RL
    dir_ppo   = DIR_MODELLI_7x7 / "ppo"
    dir_dqn   = DIR_MODELLI_7x7 / "dqn"
    dir_guide = DIR_MODELLI_7x7 / "llm_guide"
    dir_rew   = DIR_MODELLI_7x7 / "llm_rew"

    # Fallback: modelli finali (usati se best non esiste)
    fb_ppo   = percorso_ppo_7x7(seed)
    fb_dqn   = percorso_dqn_7x7(seed)
    fb_guide = percorso_llm_guide_7x7(seed)
    fb_rew   = percorso_llm_rew_7x7(seed)

    # AG-LLM-ACT: carica risultati JSON pre-calcolati da train_llm_act_7x7.py
    agente_llm = None
    risultati_llm_precaricati = None
    if not no_llm:
        percorso_llm_json = percorso_llm_act_7x7(seed)
        if percorso_llm_json.exists():
            with open(percorso_llm_json, "r", encoding="utf-8") as f:
                risultati_llm_precaricati = json.load(f)
            print("[evaluate_7x7] AG-LLM-ACT   risultati JSON: " + str(percorso_llm_json))
        else:
            print("[evaluate_7x7] AG-LLM-ACT   JSON non trovato, valutazione live...")
            agente_llm = AgenteAgLLM(provider=provider, seme=seed)

    # ---------------------------------------------------------------
    # Valutazione per fase curriculum C0->C2
    # ---------------------------------------------------------------
    for fase in FASI_CURRICULUM_7x7:
        nome  = fase["nome"]
        max_s = fase["max_step"]

        print("\n[evaluate_7x7] --- " + nome + " ---")

        metriche_fase: Dict[str, Any] = {}

        # AG-PPO
        mod_ppo, fonte_ppo = _carica_best_o_finale(PPO, dir_ppo, nome, fb_ppo)
        if mod_ppo is not None:
            env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
            metriche_fase["AG-PPO"] = _valuta_rl(mod_ppo, env, N_EPISODI_VALUTAZIONE_7x7)
            metriche_fase["AG-PPO"]["fonte_modello"] = fonte_ppo
            env.close()
            print("  AG-PPO       [" + fonte_ppo + "]: solve="
                  + str(metriche_fase["AG-PPO"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-PPO"]["reward_cumulativa"]))

        # AG-DQN
        mod_dqn, fonte_dqn = _carica_best_o_finale(DQN, dir_dqn, nome, fb_dqn)
        if mod_dqn is not None:
            env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
            metriche_fase["AG-DQN"] = _valuta_rl(mod_dqn, env, N_EPISODI_VALUTAZIONE_7x7)
            metriche_fase["AG-DQN"]["fonte_modello"] = fonte_dqn
            env.close()
            print("  AG-DQN       [" + fonte_dqn + "]: solve="
                  + str(metriche_fase["AG-DQN"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-DQN"]["reward_cumulativa"]))

        # AG-LLM-ACT -- risultati precaricati o valutazione live
        if not no_llm:
            if risultati_llm_precaricati is not None:
                m = risultati_llm_precaricati.get("fasi", {}).get(nome)
                if m:
                    metriche_fase["AG-LLM-ACT"] = m
                    print("  AG-LLM-ACT   [JSON]: solve=" + str(m["solve_rate"]) + "%"
                          + " | fallback=" + str(m.get("fallback_rate", "?")) + "%")
            elif agente_llm is not None:
                env = crea_env_7x7(fase, seed)
                metriche_fase["AG-LLM-ACT"] = agente_llm.valuta(
                    env, N_EPISODI_LLM_ACT_7x7, max_s, nome)
                env.close()
                print("  AG-LLM-ACT   [live]: solve="
                      + str(metriche_fase["AG-LLM-ACT"]["solve_rate"]) + "%")

        # AG-LLM-GUIDE -- inference: solo DQN, nessuna chiamata LLM
        mod_guide, fonte_guide = _carica_best_o_finale(DQN, dir_guide, nome, fb_guide)
        if mod_guide is not None:
            env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
            metriche_fase["AG-LLM-GUIDE"] = _valuta_rl(mod_guide, env, N_EPISODI_VALUTAZIONE_7x7)
            metriche_fase["AG-LLM-GUIDE"]["fonte_modello"] = fonte_guide
            env.close()
            print("  AG-LLM-GUIDE [" + fonte_guide + "]: solve="
                  + str(metriche_fase["AG-LLM-GUIDE"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-LLM-GUIDE"]["reward_cumulativa"]))

        # AG-LLM-REW -- inference: solo la policy PPO, nessuna chiamata LLM
        if not no_llm:
            mod_rew, fonte_rew = _carica_best_o_finale(PPO, dir_rew, nome, fb_rew)
            if mod_rew is not None:
                env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
                metriche_fase["AG-LLM-REW"] = _valuta_rl(mod_rew, env, N_EPISODI_VALUTAZIONE_7x7)
                metriche_fase["AG-LLM-REW"]["fonte_modello"] = fonte_rew
                env.close()
                print("  AG-LLM-REW   [" + fonte_rew + "]: solve="
                      + str(metriche_fase["AG-LLM-REW"]["solve_rate"]) + "%"
                      + " | rew=" + str(metriche_fase["AG-LLM-REW"]["reward_cumulativa"]))

        risultati["agenti"][nome] = metriche_fase

    # ---------------------------------------------------------------
    # Tabella riepilogativa solve rate (%) -- identica a evaluate_all.py
    # ---------------------------------------------------------------
    print("\n\n[evaluate_7x7] === RIEPILOGO SOLVE RATE (%) ===")
    agenti_nomi = ["AG-PPO", "AG-DQN", "AG-LLM-ACT", "AG-LLM-GUIDE", "AG-LLM-REW"]
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
    # Salvataggio JSON -- identico a evaluate_all.py
    # ---------------------------------------------------------------
    percorso_out = DIR_RISULTATI_7x7 / ("risultati_7x7_seed" + str(seed) + ".json")
    with open(percorso_out, "w", encoding="utf-8") as f:
        json.dump(risultati, f, indent=2, ensure_ascii=False)
    print("\n[evaluate_7x7] Risultati salvati: " + str(percorso_out))


if __name__ == "__main__":
    args = _parse_args()
    valuta_tutti(
        seed=args.seed,
        provider=args.provider,
        no_llm=args.no_llm,
    )
