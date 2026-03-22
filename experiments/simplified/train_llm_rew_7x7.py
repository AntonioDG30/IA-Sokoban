"""Training AG-LLM-REW su curriculum 7x7 (C0->C2) -- identico a train_ppo_llm_rew.py.

RecurrentPPO con reward aumentata da LLM su curriculum 7x7 C0->C2 (v10).
Il LLM valuta l'azione eseguita confrontando griglia pre e post mossa.
Viene chiamato quando il giocatore era adiacente a una cassa (~20% degli step).
A inference time solo la policy RecurrentPPO agisce: il LLM non serve piu'.

Identico a experiments/train_ppo_llm_rew.py eccetto:
    - Env: SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
    - Fasi: 3 fasi generate C0/C1/C2 invece di 6 fasi C0->C5
    - Nessun dataset Boxoban (solo livelli generati)

n_envs=1: Ollama single-threaded, chiamate sequenziali.

Uso:
    python experiments/simplified/train_llm_rew_7x7.py [--seed 42] [--provider ollama]

Modello salvato in models_7x7/llm_rew/llm_rew_7x7_seed{seed}.zip
"""

import argparse
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from experiments.simplified.config_7x7 import (
    FASI_CURRICULUM_7x7,
    CONFIG_PPO_7x7,
    DIR_MODELLI_7x7,
    DIR_LOG_7x7,
    N_EPISODI_VALUTAZIONE_7x7,
    MAX_RIPETIZIONI_FASE_7x7,
    SOGLIE_7x7,
    SOGLIA_RISOLTO_7x7,
    LAMBDA_LLM_7x7,
    PROVIDER_DEFAULT,
    percorso_llm_rew_7x7,
    crea_env_7x7,
)
from agents.llm_reward_agent import AgenteRicompensaLLM, RicompensaLLM
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# Curriculum adattivo -- IDENTICO a train_ppo_llm_rew.py
# ---------------------------------------------------------------------------

import numpy as np

_SOGLIA_RISOLTO = SOGLIA_RISOLTO_7x7


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """Legge il massimo solve rate (%) dai log di valutazione SB3."""
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        return float((d["results"] >= _SOGLIA_RISOLTO).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Helper: recupera il RicompensaLLM dalla catena di wrapper -- identico
# ---------------------------------------------------------------------------

def _ottieni_wrapper_llm(env) -> RicompensaLLM:
    """Naviga la catena di wrapper per trovare l'istanza di RicompensaLLM.

    Serve per leggere le statistiche LLM (n_chiamate, cache_hit) dopo il training.
    La catena tipica e': Monitor -> AggiuntaCanale -> RicompensaLLM -> SokobanEnv7x7.
    Il limite di 5 iterazioni previene loop infiniti su catene malformate.

    Parametri:
        env: ambiente (potenzialmente avvolto in piu' wrapper)
    Restituisce:
        istanza RicompensaLLM trovata, oppure None se assente nella catena
    """
    curr = env
    for _ in range(5):
        if isinstance(curr, RicompensaLLM):
            return curr
        if hasattr(curr, "env"):
            curr = curr.env
        else:
            break
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Legge seed e provider LLM dalla riga di comando.

    Restituisce:
        namespace argparse con attributi seed e provider
    """
    p = argparse.ArgumentParser(description="Training AG-LLM-REW curriculum 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int, provider: str) -> None:
    """Addestra AG-LLM-REW con curriculum 7x7 C0->C2.

    Identico a train_ppo_llm_rew.py::addestra_curriculum() eccetto:
        - Usa SokobanEnv7x7 (7,7) invece di SokobanEnv (10,10)
        - 3 fasi generate invece di 6
        - Percorsi modelli/log in models_7x7/ e logs_7x7/
    """
    dir_modello = percorso_llm_rew_7x7(seed).parent
    dir_log_rew = DIR_LOG_7x7 / "llm_rew" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_rew.mkdir(parents=True, exist_ok=True)

    print("\n[AG-LLM-REW-7x7] ========================================")
    print("[AG-LLM-REW-7x7] Curriculum 7x7 -- seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-REW-7x7] LAMBDA_LLM=" + str(LAMBDA_LLM_7x7)
          + " | trigger=solo_push (~5% step)")
    print("[AG-LLM-REW-7x7] n_envs=1 (Ollama single-threaded)")
    print("[AG-LLM-REW-7x7] ========================================\n")

    # Agente LLM-REW -- identico a train_ppo_llm_rew.py
    agente = AgenteRicompensaLLM(
        provider=provider,
        lambda_llm=LAMBDA_LLM_7x7,
    )

    # Policy kwargs: CNN custom + LSTM -- identici a train_ppo.py/train_ppo_llm_rew.py
    policy_kwargs = dict(
        features_extractor_class=SokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,
    )

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_7x7):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        ent    = fase["ent_coef"]

        print("\n[AG-LLM-REW-7x7] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print("[AG-LLM-REW-7x7] Timestep=" + str(ts_ppo) + " | max_step=" + str(fase["max_step"])
              + " | ent_coef=" + str(ent))

        # Ambiente di training -- identico a train_ppo_llm_rew.py
        # SokobanEnv7x7 -> RicompensaLLM -> AggiuntaCanale -> Monitor
        base_train = crea_env_7x7(fase, seed)
        env_train  = Monitor(AggiuntaCanale(agente.avvolgi_env(base_train)))

        # Validazione senza wrapper LLM (valuta la policy pura)
        env_val = Monitor(AggiuntaCanale(crea_env_7x7(fase, seed)))

        config = {**CONFIG_PPO_7x7, "ent_coef": ent}
        config.pop("verbose", None)

        if modello is None:
            modello = RecurrentPPO(
                policy="CnnLstmPolicy",
                env=env_train,
                seed=seed,
                tensorboard_log=str(dir_log_rew),
                policy_kwargs=policy_kwargs,
                verbose=1,
                **config,
            )
        else:
            modello.set_env(env_train)
            modello.ent_coef = ent

        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=max(20_000 // 1, 1),   # n_envs=1 come l'originale
                n_eval_episodes=N_EPISODI_VALUTAZIONE_7x7,
                deterministic=True,
                render=False,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=100_000,
                save_path=str(dir_fase / "checkpoints"),
                name_prefix="llm_rew_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # --- Curriculum adattivo -- identico a train_ppo_llm_rew.py ---
        # LLM-REW usa stesso curriculum dell'originale (no ripetizioni adattive):
        # il bottleneck e' il tempo LLM, non il convergere del RL.
        soglia = SOGLIE_7x7.get(nome, 0.0)
        t_fase = time.time()
        t0 = time.time()
        modello.learn(
            total_timesteps=ts_ppo,
            callback=callbacks,
            tb_log_name="LLM_REW_" + nome + "_seed" + str(seed),
            reset_num_timesteps=(i_fase == 0),
        )
        elapsed = time.time() - t0

        # Statistiche LLM -- identiche a train_ppo_llm_rew.py
        wrapper = _ottieni_wrapper_llm(env_train)
        if wrapper is not None:
            s = wrapper.statistiche_llm
            print("[AG-LLM-REW-7x7] " + nome + " -- LLM calls=" + str(s["n_chiamate"])
                  + " | cache_hit=" + str(s["n_cache_hit"])
                  + " | cache_size=" + str(s["cache_size"])
                  + " | elapsed=" + str(round(elapsed / 60, 1)) + "min")
        else:
            print("[AG-LLM-REW-7x7] " + nome + " completata in "
                  + str(round(elapsed / 60, 1)) + " min")

        print(
            "[AG-LLM-REW-7x7] Fase " + nome + " totale: "
            + str(round((time.time() - t_fase) / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salva modello finale
    percorso_finale = str(percorso_llm_rew_7x7(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("\n[AG-LLM-REW-7x7] ========================================")
    print("[AG-LLM-REW-7x7] Training completato in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-REW-7x7] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-LLM-REW-7x7] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, provider=args.provider)
