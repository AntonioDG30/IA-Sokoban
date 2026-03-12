"""Valutazione comparativa di tutti gli agenti su tutte le fasi curriculum v9.

Valuta AG-PPO, AG-DQN, AG-LLM, AG-LLM-REW su:
    - Ogni fase del curriculum C0->C5 (split test/valid)
    - Set extra: Boxoban medium/valid e unfiltered/test (SET_VALUTAZIONE)

Output:
    - Stampa tabella riepilogativa a terminale
    - Salva risultati in docs/report/risultati_comparativi_seed{seed}.json

Uso:
    python experiments/evaluate_all.py [--seed 42] [--dir-dati data/boxoban]
                                       [--no-llm]
    --no-llm: salta AG-LLM e AG-LLM-REW (utile prima del training LLM)
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
    DIR_RISULTATI,
    N_EPISODI_VALUTAZIONE,
    PROVIDER_DEFAULT,
    N_EPISODI_LLM_ACT,
    crea_env_da_fase,
    percorso_modello_ppo,
    percorso_modello_dqn,
    percorso_risultati_llm_act,
    percorso_modello_llm_rew,
)
from agents.llm_act_agent import AgenteAgLLM
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Valutazione comparativa tutti gli agenti")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path data/boxoban")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--no-llm",   action="store_true",
                   help="Salta agenti LLM (AG-LLM e AG-LLM-REW)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers valutazione
# ---------------------------------------------------------------------------

def _valuta_rl(modello, env, n_episodi: int) -> Dict[str, Any]:
    """Valuta un modello SB3 su n_episodi. Restituisce metriche."""
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
    print("[evaluate_all] Valutazione comparativa — seed=" + str(seed))
    print("[evaluate_all] ========================================\n")

    risultati: Dict[str, Any] = {"seed": seed, "agenti": {}}

    # ---------------------------------------------------------------
    # Carica modelli RL
    # ---------------------------------------------------------------
    percorso_ppo = percorso_modello_ppo(seed).with_suffix(".zip")
    percorso_dqn = percorso_modello_dqn(seed).with_suffix(".zip")
    percorso_rew = percorso_modello_llm_rew(seed).with_suffix(".zip")

    modello_ppo = None
    modello_dqn = None
    modello_rew = None

    if percorso_ppo.exists():
        modello_ppo = PPO.load(str(percorso_ppo.with_suffix("")))
        print("[evaluate_all] AG-PPO caricato: " + str(percorso_ppo))
    else:
        print("[evaluate_all] AG-PPO non trovato: " + str(percorso_ppo))

    if percorso_dqn.exists():
        modello_dqn = DQN.load(str(percorso_dqn.with_suffix("")))
        print("[evaluate_all] AG-DQN caricato: " + str(percorso_dqn))
    else:
        print("[evaluate_all] AG-DQN non trovato: " + str(percorso_dqn))

    if not no_llm and percorso_rew.exists():
        modello_rew = PPO.load(str(percorso_rew.with_suffix("")))
        print("[evaluate_all] AG-LLM-REW caricato: " + str(percorso_rew))
    elif not no_llm:
        print("[evaluate_all] AG-LLM-REW non trovato: " + str(percorso_rew))

    # AG-LLM: carica risultati JSON pre-calcolati da train_llm_act.py
    agente_llm = None
    risultati_llm_precaricati = None
    if not no_llm:
        percorso_llm_json = percorso_risultati_llm_act(seed)
        if percorso_llm_json.exists():
            with open(percorso_llm_json, "r", encoding="utf-8") as f:
                risultati_llm_precaricati = json.load(f)
            print("[evaluate_all] AG-LLM risultati pre-caricati: "
                  + str(percorso_llm_json))
        else:
            print("[evaluate_all] AG-LLM: risultati non trovati, valutazione live...")
            agente_llm = AgenteAgLLM(provider=provider, seme=seed)

    # ---------------------------------------------------------------
    # Valutazione per fase curriculum
    # ---------------------------------------------------------------
    fasi_da_valutare = list(FASI_CURRICULUM_V9) + [
        {**sv, "nome": sv["nome"], "max_step": 300, "n_casse": 4,
         "dataset": sv["dataset"], "timestep_ppo": 0, "timestep_dqn": 0, "ent_coef": 0}
        for sv in SET_VALUTAZIONE
    ]

    for fase in fasi_da_valutare:
        nome   = fase["nome"]
        max_s  = fase["max_step"]
        split  = fase.get("split", "test")

        print("\n[evaluate_all] --- " + nome + " (split=" + split + ") ---")

        metriche_fase: Dict[str, Any] = {}

        # AG-PPO — richiede AggiuntaCanale: (10,10) -> (1,10,10) per SokobanCNN
        if modello_ppo is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-PPO"] = _valuta_rl(modello_ppo, env, N_EPISODI_VALUTAZIONE)
            env.close()
            print("  AG-PPO:     solve=" + str(metriche_fase["AG-PPO"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-PPO"]["reward_cumulativa"]))

        # AG-DQN — richiede AggiuntaCanale: (10,10) -> (1,10,10) per SokobanCNN
        if modello_dqn is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-DQN"] = _valuta_rl(modello_dqn, env, N_EPISODI_VALUTAZIONE)
            env.close()
            print("  AG-DQN:     solve=" + str(metriche_fase["AG-DQN"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-DQN"]["reward_cumulativa"]))

        # AG-LLM
        if not no_llm:
            if risultati_llm_precaricati is not None:
                m = risultati_llm_precaricati.get("fasi", {}).get(nome)
                if m:
                    metriche_fase["AG-LLM"] = m
                    print("  AG-LLM:     solve=" + str(m["solve_rate"]) + "%"
                          + " | fallback=" + str(m.get("fallback_rate", "?")) + "%")
            elif agente_llm is not None:
                env = crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)
                metriche_fase["AG-LLM"] = agente_llm.valuta(
                    env, N_EPISODI_LLM_ACT, max_s, nome)
                env.close()
                print("  AG-LLM:     solve=" + str(metriche_fase["AG-LLM"]["solve_rate"]) + "%")

        # AG-LLM-REW — richiede AggiuntaCanale (inference: solo la policy PPO, niente LLM)
        if modello_rew is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-LLM-REW"] = _valuta_rl(modello_rew, env, N_EPISODI_VALUTAZIONE)
            env.close()
            print("  AG-LLM-REW: solve=" + str(metriche_fase["AG-LLM-REW"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-LLM-REW"]["reward_cumulativa"]))

        risultati["agenti"][nome] = metriche_fase

    # ---------------------------------------------------------------
    # Tabella riepilogativa
    # ---------------------------------------------------------------
    print("\n\n[evaluate_all] === RIEPILOGO SOLVE RATE (%) ===")
    agenti_nomi = ["AG-PPO", "AG-DQN", "AG-LLM", "AG-LLM-REW"]
    header = "Fase".ljust(24) + "".join(n.ljust(14) for n in agenti_nomi)
    print(header)
    print("-" * len(header))
    for nome, per_agente in risultati["agenti"].items():
        riga = nome.ljust(24)
        for ag in agenti_nomi:
            if ag in per_agente:
                riga += str(per_agente[ag]["solve_rate"]).ljust(14)
            else:
                riga += "N/A".ljust(14)
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
