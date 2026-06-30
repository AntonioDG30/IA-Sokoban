# Valutazione comparativa di tutti gli agenti su tutte le fasi del curriculum v9.
#
# Valuta AG-PPO, AG-DQN, AG-LLM-ACT, AG-LLM-GUIDE, AG-LLM-REW su:
#   - ogni fase del curriculum C0->C5 (split test/valid);
#   - i set extra Boxoban medium/valid e unfiltered/test (SET_VALUTAZIONE).
#
# I 5 agenti:
#   AG-PPO        PPO baseline (traccia PDF)
#   AG-DQN        DQN baseline (traccia PDF)
#   AG-LLM-ACT    il LLM come policy diretta a inference time (traccia PDF)
#   AG-LLM-GUIDE  DQN addestrato via LfD con demo del LLM (mail prof); a inference solo DQN
#   AG-LLM-REW    PPO con reward LLM (traccia PDF + mail prof)
#
# Caricamento dei modelli:
#   - per le fasi C0->C5 carica il BEST model di quella fase (salvato da EvalCallback), così
#     evita il catastrophic forgetting del modello finale: artifacts/models/10x10/{agente}/{fase}/best/best_model.zip;
#   - per i set extra usa il best model della fase C5 (la Boxoban più avanzata);
#   - fallback: se il best non esiste, usa il modello finale.
#
# Output: tabella riepilogativa a terminale + JSON in results/seed42/risultati_comparativi_seed{seed}.json.
#
# Uso: python src/sistema_10x10/evaluate_all.py [--seed 42] [--dir-dati dataset/boxoban] [--no-llm]
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

from sistema_10x10.config import (
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
from core.agenti_llm.llm_act_agent import AgenteAgLLM
from core.ambiente.cnn_wrapper import AggiuntaCanale


# HELPER: BEST MODEL PER FASE (EVITA IL CATASTROPHIC FORGETTING)

def _carica_best_o_finale(cls, base_dir: Path, nome_fase: str, fallback: Path):
    """
    Carica il best model della fase data; se non esiste, ripiega sul modello finale.

    cls è la classe SB3 con cui caricare (RecurrentPPO o DQN), base_dir la cartella
    dell'agente (es. artifacts/models/10x10/ppo), nome_fase il nome della fase del curriculum, fallback il
    path del modello finale (senza .zip). Restituisce (modello, fonte) con fonte = 'best',
    'finale' oppure 'non trovato' (modello None).
    """
    best = base_dir / nome_fase / "best" / "best_model.zip"
    if best.exists():
        return cls.load(str(best.with_suffix(""))), "best"
    if fallback.with_suffix(".zip").exists():
        return cls.load(str(fallback)), "finale"
    return None, "non trovato"


# CLI

def _parse_args():
    """Legge gli argomenti da riga di comando: --seed, --dir-dati, --provider e --no-llm."""
    p = argparse.ArgumentParser(description="Valutazione comparativa tutti gli agenti")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path dataset/boxoban")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--no-llm",   action="store_true",
                   help="Salta AG-LLM-ACT e AG-LLM-REW (non AG-LLM-GUIDE: inference senza LLM)")
    return p.parse_args()


# HELPER DI VALUTAZIONE

def _valuta_rl(modello, env, n_episodi: int) -> Dict[str, Any]:
    """
    Valuta un modello SB3 (RecurrentPPO o DQN) su n_episodi con policy deterministica.
    Restituisce solve_rate, mosse_medie (solo episodi risolti), reward_cumulativa media,
    casse_su_target medie, n_episodi e n_risolti.
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


# FUNZIONE PRINCIPALE

def valuta_tutti(seed: int, dir_dati: str, provider: str, no_llm: bool) -> None:
    """
    Valuta i 5 agenti su tutte le fasi C0->C5 e sui set extra, salvando il JSON comparativo.

    Per ogni fase carica il best model di quella fase (evita il catastrophic forgetting del
    modello finale). seed serve a ritrovare i modelli salvati; provider è usato per la
    valutazione live di AG-LLM-ACT se manca il JSON precalcolato; con no_llm=True saltano
    AG-LLM-ACT e AG-LLM-REW.
    """
    dir_dati_path = Path(dir_dati)
    DIR_RISULTATI.mkdir(parents=True, exist_ok=True)

    print("[evaluate_all] Valutazione comparativa 5 agenti - seed=" + str(seed))
    print("[evaluate_all] Modalita': best model per fase (evita forgetting)")

    risultati: Dict[str, Any] = {"seed": seed, "modalita": "best_per_fase", "agenti": {}}

    # Cartelle base dei modelli RL
    dir_ppo   = DIR_MODELLI / "ppo"
    dir_dqn   = DIR_MODELLI / "dqn"
    dir_guide = DIR_MODELLI / "llm_guide"
    dir_rew   = DIR_MODELLI / "llm_rew"

    # Modelli finali, usati come fallback se il best non esiste
    fb_ppo   = percorso_modello_ppo(seed)
    fb_dqn   = percorso_modello_dqn(seed)
    fb_guide = percorso_modello_llm_guide(seed)
    fb_rew   = percorso_modello_llm_rew(seed)

    # AG-LLM-ACT: carica i risultati JSON precalcolati da train_llm_act.py (se ci sono)
    agente_llm = None
    risultati_llm_precaricati = None
    if not no_llm:
        percorso_llm_json = percorso_risultati_llm_act(seed)
        if percorso_llm_json.exists():
            with open(percorso_llm_json, "r", encoding="utf-8") as f:
                risultati_llm_precaricati = json.load(f)
            print("[evaluate_all] AG-LLM-ACT      risultati JSON: " + str(percorso_llm_json))
        else:
            print("[evaluate_all] AG-LLM-ACT      JSON non trovato, valutazione live...")
            agente_llm = AgenteAgLLM(provider=provider, seme=seed)

    # Fasi del curriculum + set extra (questi ultimi riusano il best model C5).
    # I set extra prendono la forma di una "fase" con i campi minimi richiesti dal loop.
    fasi_curriculum = list(FASI_CURRICULUM_V9)
    fasi_extra = [
        {**sv, "nome": sv["nome"], "max_step": 300, "n_casse": 4,
         "dataset": sv["dataset"], "timestep_ppo": 0, "timestep_dqn": 0, "ent_coef": 0}
        for sv in SET_VALUTAZIONE
    ]
    fasi_da_valutare = fasi_curriculum + fasi_extra

    # Fase di riferimento per i set extra: il best model dell'ultima fase Boxoban
    nome_fase_boxoban = "C5-4box-unfiltered"

    for fase in fasi_da_valutare:
        nome  = fase["nome"]
        max_s = fase["max_step"]
        split = fase.get("split", "test")
        # Per i set extra si usa il best model dell'ultima fase del curriculum (C5)
        nome_best = nome if nome in [f["nome"] for f in fasi_curriculum] else nome_fase_boxoban

        print("\n[evaluate_all] " + nome + " (split=" + split + ")")

        metriche_fase: Dict[str, Any] = {}

        # AG-PPO
        mod_ppo, fonte_ppo = _carica_best_o_finale(RecurrentPPO, dir_ppo, nome_best, fb_ppo)
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

        # AG-LLM-ACT — metriche da JSON precalcolato oppure valutazione live
        if not no_llm:
            if risultati_llm_precaricati is not None:
                m = risultati_llm_precaricati.get("fasi", {}).get(nome)
                if m:
                    metriche_fase["AG-LLM-ACT"] = m
                    print("  AG-LLM-ACT      [JSON]: solve=" + str(m["solve_rate"]) + "%"
                          + " | fallback=" + str(m.get("fallback_rate", "?")) + "%")
            elif agente_llm is not None:
                env = crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)
                metriche_fase["AG-LLM-ACT"] = agente_llm.valuta(
                    env, N_EPISODI_LLM_ACT, max_s, nome)
                env.close()
                print("  AG-LLM-ACT      [live]: solve="
                      + str(metriche_fase["AG-LLM-ACT"]["solve_rate"]) + "%")

        # AG-LLM-GUIDE — a inference agisce solo il DQN, nessuna chiamata al LLM
        mod_guide, fonte_guide = _carica_best_o_finale(DQN, dir_guide, nome_best, fb_guide)
        if mod_guide is not None:
            env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
            metriche_fase["AG-LLM-GUIDE"] = _valuta_rl(mod_guide, env, N_EPISODI_VALUTAZIONE)
            metriche_fase["AG-LLM-GUIDE"]["fonte_modello"] = fonte_guide
            env.close()
            print("  AG-LLM-GUIDE [" + fonte_guide + "]: solve="
                  + str(metriche_fase["AG-LLM-GUIDE"]["solve_rate"]) + "%"
                  + " | rew=" + str(metriche_fase["AG-LLM-GUIDE"]["reward_cumulativa"]))

        # AG-LLM-REW — a inference agisce solo la policy PPO, nessuna chiamata al LLM
        if not no_llm:
            mod_rew, fonte_rew = _carica_best_o_finale(RecurrentPPO, dir_rew, nome_best, fb_rew)
            if mod_rew is not None:
                env = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split=split)))
                metriche_fase["AG-LLM-REW"] = _valuta_rl(mod_rew, env, N_EPISODI_VALUTAZIONE)
                metriche_fase["AG-LLM-REW"]["fonte_modello"] = fonte_rew
                env.close()
                print("  AG-LLM-REW   [" + fonte_rew + "]: solve="
                      + str(metriche_fase["AG-LLM-REW"]["solve_rate"]) + "%"
                      + " | rew=" + str(metriche_fase["AG-LLM-REW"]["reward_cumulativa"]))

        risultati["agenti"][nome] = metriche_fase

    # TABELLA RIEPILOGATIVA DEL SOLVE RATE (%)
    print("\n\n[evaluate_all] Riepilogo solve rate (%):")
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

    # SALVATAGGIO DEL JSON COMPARATIVO
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
