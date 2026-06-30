# Valutazione comparativa di tutti gli agenti sul curriculum 7x7 — gemello di evaluate_all.py.
#
# Valuta AG-PPO, AG-DQN, AG-LLM-ACT, AG-LLM-GUIDE, AG-LLM-REW su ogni fase del curriculum
# C0->C2 (3 fasi generate, nessun Boxoban).
#
# I 5 agenti:
#   AG-PPO        PPO baseline (RecurrentPPO, CnnLstmPolicy)
#   AG-DQN        DQN baseline (CnnPolicy)
#   AG-LLM-ACT    il LLM come policy diretta a inference time
#   AG-LLM-GUIDE  DQN addestrato via LfD con demo del LLM; a inference solo il DQN
#   AG-LLM-REW    PPO con reward LLM durante il training; a inference solo la policy PPO
#
# Identico a src/sistema_10x10/evaluate_all.py tranne che per: l'ambiente SokobanEnv7x7 (7,7), 3
# fasi generate C0/C1/C2 invece di 6, nessun Boxoban e nessun SET_VALUTAZIONE extra.
#
# Caricamento dei modelli: per ogni fase il BEST model di quella fase (evita il catastrophic
# forgetting), con fallback al modello finale: artifacts/models/7x7/{agente}/{fase}/best/best_model.zip.
#
# Output: tabella riepilogativa a terminale + JSON in results/seed42/risultati_7x7_seed{seed}.json.
#
# Uso: python src/sistema_7x7/evaluate_7x7.py [--seed 42] [--no-llm]
#   --no-llm: salta AG-LLM-ACT e AG-LLM-REW (utile prima del training LLM); AG-LLM-GUIDE NON
#             viene saltato perché a inference non usa il LLM.

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO

from sistema_7x7.config_7x7 import (
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
from core.agenti_llm.llm_act_agent import AgenteAgLLM
from core.ambiente.cnn_wrapper import AggiuntaCanale


# HELPER: BEST MODEL PER FASE (EVITA IL CATASTROPHIC FORGETTING) — IDENTICO A evaluate_all.py

def _carica_best_o_finale(cls, base_dir: Path, nome_fase: str, fallback: Path):
    """
    Carica il best model della fase data; se non esiste, ripiega sul modello finale.

    cls è la classe SB3 (RecurrentPPO o DQN), base_dir la cartella dell'agente (es.
    artifacts/models/7x7/ppo), nome_fase il nome della fase, fallback il path del modello finale (senza
    .zip). Restituisce (modello, fonte) con fonte = 'best', 'finale' o 'non trovato' (None).
    """
    best = base_dir / nome_fase / "best" / "best_model.zip"
    if best.exists():
        return cls.load(str(best.with_suffix(""))), "best"
    if fallback.with_suffix(".zip").exists():
        return cls.load(str(fallback)), "finale"
    return None, "non trovato"


# CLI

def _parse_args():
    """
    Legge --seed, --provider e --no-llm dalla riga di comando.
    --no-llm è utile quando i modelli LLM-REW e LLM-ACT non sono ancora addestrati/valutati:
    così si vede comunque il confronto PPO vs DQN vs LLM-GUIDE.
    """
    p = argparse.ArgumentParser(description="Valutazione comparativa tutti gli agenti 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--no-llm",   action="store_true",
                   help="Salta AG-LLM-ACT e AG-LLM-REW (non AG-LLM-GUIDE: inference senza LLM)")
    return p.parse_args()


# HELPER DI VALUTAZIONE — IDENTICO A evaluate_all.py

def _valuta_rl(modello, env, n_episodi: int) -> Dict[str, Any]:
    """
    Valuta un modello SB3 (RecurrentPPO o DQN) su n_episodi con policy deterministica.

    Con predict(deterministic=True) la policy non esplora ma sceglie sempre l'azione con la
    stima più alta. Restituisce solve_rate, mosse_medie (solo episodi risolti),
    reward_cumulativa media, casse_su_target medie, n_episodi e n_risolti.
    """
    # Le policy ricorrenti (RecurrentPPO: AG-PPO e AG-LLM-REW) hanno uno stato nascosto LSTM
    # che va propagato fra gli step: predict() restituisce lo stato aggiornato e va ripassato
    # alla chiamata successiva, azzerandolo solo all'inizio di ogni episodio (episode_start).
    # Senza questa propagazione la LSTM viene resettata ad ogni step e la policy e' valutata
    # in regime markoviano (memoria assente), deprimendone artificialmente le prestazioni.
    # Per i modelli non ricorrenti (DQN) lo stato non serve: si usa la chiamata semplice.
    e_ricorrente = isinstance(modello, RecurrentPPO)

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
        stato_lstm = None
        inizio_episodio = np.ones((1,), dtype=bool)
        while not done:
            if e_ricorrente:
                azione, stato_lstm = modello.predict(
                    obs, state=stato_lstm, episode_start=inizio_episodio, deterministic=True)
            else:
                azione, _ = modello.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(azione))
            reward_ep += float(r)
            step_ep += 1
            casse_finali = info.get("casse_su_target", 0)
            inizio_episodio = np.array([terminated or truncated])
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


# FUNZIONE PRINCIPALE — COME evaluate_all.py, ADATTATA AL 7x7

def valuta_tutti(seed: int, provider: str, no_llm: bool) -> None:
    """
    Valuta i 5 agenti sul curriculum 7x7 e salva i risultati in JSON.

    Per ogni agente/fase carica il best model di quella fase (evita il catastrophic
    forgetting del modello finale); per AG-LLM-ACT usa il JSON precalcolato se esiste. seed
    serve a ritrovare i modelli, provider è per la valutazione live di AG-LLM-ACT, no_llm=True
    salta AG-LLM-ACT e AG-LLM-REW.
    """
    DIR_RISULTATI_7x7.mkdir(parents=True, exist_ok=True)

    print("[evaluate_7x7] Valutazione comparativa 5 agenti 7x7 -- seed=" + str(seed))
    print("[evaluate_7x7] Modalita': best model per fase (evita forgetting)")

    risultati: Dict[str, Any] = {"seed": seed, "modalita": "best_per_fase", "agenti": {}}

    # Cartelle base dei modelli RL
    dir_ppo   = DIR_MODELLI_7x7 / "ppo"
    dir_dqn   = DIR_MODELLI_7x7 / "dqn"
    dir_guide = DIR_MODELLI_7x7 / "llm_guide"
    dir_rew   = DIR_MODELLI_7x7 / "llm_rew"

    # Modelli finali, usati come fallback se il best non esiste
    fb_ppo   = percorso_ppo_7x7(seed)
    fb_dqn   = percorso_dqn_7x7(seed)
    fb_guide = percorso_llm_guide_7x7(seed)
    fb_rew   = percorso_llm_rew_7x7(seed)

    # AG-LLM-ACT: carica i risultati JSON precalcolati da train_llm_act_7x7.py (se ci sono)
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

    # Valutazione per ogni fase del curriculum C0->C2
    for fase in FASI_CURRICULUM_7x7:
        nome  = fase["nome"]
        max_s = fase["max_step"]

        print("\n[evaluate_7x7] " + nome)

        metriche_fase: Dict[str, Any] = {}

        # AG-PPO
        mod_ppo, fonte_ppo = _carica_best_o_finale(RecurrentPPO, dir_ppo, nome, fb_ppo)
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

        # AG-LLM-ACT — metriche da JSON precalcolato oppure valutazione live
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

        # AG-LLM-GUIDE — a inference agisce solo il DQN, nessuna chiamata al LLM
        mod_guide, fonte_guide = _carica_best_o_finale(DQN, dir_guide, nome, fb_guide)
        if mod_guide is not None:
            env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
            metriche_fase["AG-LLM-GUIDE"] = _valuta_rl(mod_guide, env, N_EPISODI_VALUTAZIONE_7x7)
            metriche_fase["AG-LLM-GUIDE"]["fonte_modello"] = fonte_guide
            env.close()
            print("  AG-LLM-GUIDE [" + fonte_guide + "]: solve="
                  + str(metriche_fase["AG-LLM-GUIDE"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-LLM-GUIDE"]["reward_cumulativa"]))

        # AG-LLM-REW — a inference agisce solo la policy PPO, nessuna chiamata al LLM
        if not no_llm:
            mod_rew, fonte_rew = _carica_best_o_finale(RecurrentPPO, dir_rew, nome, fb_rew)
            if mod_rew is not None:
                env = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))
                metriche_fase["AG-LLM-REW"] = _valuta_rl(mod_rew, env, N_EPISODI_VALUTAZIONE_7x7)
                metriche_fase["AG-LLM-REW"]["fonte_modello"] = fonte_rew
                env.close()
                print("  AG-LLM-REW   [" + fonte_rew + "]: solve="
                      + str(metriche_fase["AG-LLM-REW"]["solve_rate"]) + "%"
                      + " | rew=" + str(metriche_fase["AG-LLM-REW"]["reward_cumulativa"]))

        risultati["agenti"][nome] = metriche_fase

    # TABELLA RIEPILOGATIVA DEL SOLVE RATE (%) — IDENTICA A evaluate_all.py
    print("\n\n[evaluate_7x7] Riepilogo solve rate (%):")
    agenti_nomi = ["AG-PPO", "AG-DQN", "AG-LLM-ACT", "AG-LLM-GUIDE", "AG-LLM-REW"]
    header = "Fase".ljust(24) + "".join(n.ljust(15) for n in agenti_nomi)
    print(header)
    for nome, per_agente in risultati["agenti"].items():
        riga = nome.ljust(24)
        for ag in agenti_nomi:
            if ag in per_agente:
                riga += str(per_agente[ag]["solve_rate"]).ljust(15)
            else:
                riga += "N/A".ljust(15)
        print(riga)

    # SALVATAGGIO DEL JSON — IDENTICO A evaluate_all.py
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
