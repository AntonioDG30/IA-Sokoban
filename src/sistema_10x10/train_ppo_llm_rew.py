# Training di AG-LLM-REW: RecurrentPPO con reward aumentata dal LLM, su curriculum C0->C5.
#
# Il LLM valuta l'azione confrontando la griglia prima e dopo la mossa; viene chiamato solo
# quando una cassa viene effettivamente spinta (~5% degli step). A inference time agisce solo
# la policy RecurrentPPO: il LLM non serve più. Si usa RecurrentPPO (CnnLstmPolicy, sb3-contrib)
# come AG-PPO, per un confronto equo.
#
# Catena di wrapping:
#   SokobanEnv -> RicompensaLLM -> AggiuntaCanale -> Monitor -> RecurrentPPO (CnnLstmPolicy)
#
# Curriculum identico ad AG-PPO (FASI_CURRICULUM_V9, stesso budget di step per un confronto
# equo a parità di esperienza raccolta):
#   C0: 1 cassa, generata            (600K step)
#   C1: 2 casse, generata            (1M step)
#   C2: 3 casse, generata            (1.5M step)
#   C3: 4 casse, generata            (2M step)
#   C4: 4 casse, Boxoban medium      (2M step)
#   C5: 4 casse, Boxoban unfiltered  (2M step)
#   Totale: ~9.1M step (una sola passata per fase, nessuna ripetizione adattiva)
#
# Trigger del LLM: solo spinta effettiva (info['cassa_spostata']), ~5% degli step; la cache
# (obs_pre, action) riduce ancora le chiamate effettive. Con qwen3:14b su Ollama (~0.21 s a
# chiamata, a regime) il training gira in una sessione dedicata, con n_envs=1 perché Ollama è
# single-threaded e serve le chiamate in sequenza.
#
# Uso: python src/sistema_10x10/train_ppo_llm_rew.py [--seed 42] [--provider ollama] [--dir-dati dataset/boxoban]
# Modello finale salvato in artifacts/models/10x10/llm_rew/llm_rew_seed{seed}.zip

import argparse
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from sistema_10x10.config import (
    FASI_CURRICULUM_V9,
    CONFIG_PPO,
    DIR_DATI,
    DIR_LOG,
    N_EPISODI_VALUTAZIONE,
    PROVIDER_DEFAULT,
    LAMBDA_LLM,
    crea_env_da_fase,
    percorso_modello_llm_rew,
)
from core.agenti_llm.llm_reward_agent import AgenteRicompensaLLM, RicompensaLLM
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale


# CLI

def _parse_args():
    """Legge gli argomenti da riga di comando: --seed (int), --provider (str), --dir-dati (path)."""
    p = argparse.ArgumentParser(description="Training AG-LLM-REW curriculum v9")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path dataset/boxoban")
    return p.parse_args()


# HELPER: RECUPERA IL RicompensaLLM DALLA CATENA DI WRAPPER

def _ottieni_wrapper_llm(env) -> RicompensaLLM:
    """Risale la catena di wrapper (.env) per trovare l'istanza RicompensaLLM, o None."""
    curr = env
    for _ in range(5):
        if isinstance(curr, RicompensaLLM):
            return curr
        if hasattr(curr, "env"):
            curr = curr.env
        else:
            break
    return None


# TRAINING

def addestra_curriculum(seed: int, provider: str, dir_dati: str) -> None:
    """Addestra AG-LLM-REW con il curriculum completo C0->C5."""
    dir_dati_path  = Path(dir_dati)
    dir_modello    = percorso_modello_llm_rew(seed).parent
    dir_log_rew    = DIR_LOG / "llm_rew" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_rew.mkdir(parents=True, exist_ok=True)

    print("[AG-LLM-REW] Curriculum v9 - seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-REW] LAMBDA_LLM=" + str(LAMBDA_LLM)
          + " | trigger=solo_push (~5% step)")
    print("[AG-LLM-REW] n_envs=1 (Ollama single-threaded)")

    # Agente LLM-REW: il client viene riusato tra le fasi (warm-up una volta sola)
    agente = AgenteRicompensaLLM(
        provider=provider,
        lambda_llm=LAMBDA_LLM,
    )

    # Policy kwargs: CNN custom + LSTM, identici ad AG-PPO per un confronto equo
    policy_kwargs = dict(
        features_extractor_class=SokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=True,
        enable_critic_lstm=False,  # obbligatorio quando shared_lstm=True
    )

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        max_s  = fase["max_step"]
        ent    = fase["ent_coef"]

        print("\n[AG-LLM-REW] Fase " + str(i_fase) + ": " + nome)
        print("[AG-LLM-REW] Timestep=" + str(ts_ppo) + " | max_step=" + str(max_s)
              + " | ent_coef=" + str(ent))

        # Ambiente di training:
        #   SokobanEnv (10,10) -> RicompensaLLM (legge l'obs 2D per il LLM) -> AggiuntaCanale (1,10,10) -> Monitor
        base_train = crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")
        env_train  = Monitor(AggiuntaCanale(agente.avvolgi_env(base_train)))

        # Validazione senza wrapper LLM: misura la policy pura
        env_val = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")))

        config = {**CONFIG_PPO, "ent_coef": ent}
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

        # Callback della fase
        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=20_000,
                n_eval_episodes=N_EPISODI_VALUTAZIONE,
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

        t0 = time.time()
        modello.learn(
            total_timesteps=ts_ppo,
            callback=callbacks,
            tb_log_name="LLM_REW_" + nome + "_seed" + str(seed),
            reset_num_timesteps=(i_fase == 0),
        )
        elapsed = time.time() - t0

        # Statistiche d'uso del LLM in questa fase (chiamate, cache hit, dimensione cache)
        wrapper = _ottieni_wrapper_llm(env_train)
        if wrapper is not None:
            s = wrapper.statistiche_llm
            print("[AG-LLM-REW] " + nome + " - LLM calls=" + str(s["n_chiamate"])
                  + " | cache_hit=" + str(s["n_cache_hit"])
                  + " | cache_size=" + str(s["cache_size"])
                  + " | elapsed=" + str(round(elapsed / 60, 1)) + "min")
        else:
            print("[AG-LLM-REW] " + nome + " completata in "
                  + str(round(elapsed / 60, 1)) + " min")

        env_train.close()
        env_val.close()

    # Salvataggio del modello finale
    percorso_finale = str(percorso_modello_llm_rew(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("[AG-LLM-REW] Training completato in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-REW] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(
        seed=args.seed,
        provider=args.provider,
        dir_dati=args.dir_dati,
    )
