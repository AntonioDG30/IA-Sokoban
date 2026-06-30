# Training di AG-LLM-REW sul curriculum 7x7 (C0->C2) — gemello di train_ppo_llm_rew.py.
#
# RecurrentPPO con reward aumentata dal LLM. Il LLM valuta l'azione confrontando la griglia
# prima e dopo la mossa; viene chiamato solo quando una cassa viene effettivamente spinta
# (~5% degli step). A inference time agisce solo la policy RecurrentPPO: il LLM non serve più.
#
# Identico a src/sistema_10x10/train_ppo_llm_rew.py tranne che per: l'ambiente SokobanEnv7x7 (7,7)
# al posto di SokobanEnv (10,10), 3 fasi generate C0/C1/C2 invece di 6 e l'assenza di dataset
# Boxoban. n_envs=1 perché Ollama è single-threaded e serve le chiamate in sequenza.
#
# Uso: python src/sistema_7x7/train_llm_rew_7x7.py [--seed 42] [--provider ollama]
# Modello finale salvato in artifacts/models/7x7/llm_rew/llm_rew_7x7_seed{seed}.zip

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

from sistema_7x7.config_7x7 import (
    FASI_CURRICULUM_7x7,
    CONFIG_PPO_7x7,
    DIR_LOG_7x7,
    N_EPISODI_VALUTAZIONE_7x7,
    SOGLIE_7x7,
    LAMBDA_LLM_7x7,
    PROVIDER_DEFAULT,
    percorso_llm_rew_7x7,
    crea_env_7x7,
)
from core.agenti_llm.llm_reward_agent import AgenteRicompensaLLM, RicompensaLLM
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale


# HELPER: RECUPERA IL RicompensaLLM DALLA CATENA DI WRAPPER — IDENTICO

def _ottieni_wrapper_llm(env) -> RicompensaLLM:
    """
    Risale la catena di wrapper (.env) per trovare l'istanza di RicompensaLLM.

    Serve a leggere le statistiche del LLM (n_chiamate, cache_hit) dopo il training. La catena
    tipica è Monitor -> AggiuntaCanale -> RicompensaLLM -> SokobanEnv7x7; il limite di 5
    iterazioni evita loop su catene malformate. Restituisce il wrapper o None se assente.
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


# CLI

def _parse_args():
    """Legge --seed e --provider dalla riga di comando."""
    p = argparse.ArgumentParser(description="Training AG-LLM-REW curriculum 7x7")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    return p.parse_args()


# TRAINING

def addestra_curriculum(seed: int, provider: str) -> None:
    """
    Addestra AG-LLM-REW con il curriculum 7x7 C0->C2.
    Stessa logica di train_ppo_llm_rew.py::addestra_curriculum() ma con SokobanEnv7x7 (7,7),
    3 fasi generate invece di 6 e percorsi in artifacts/models/7x7/ e artifacts/logs/7x7/.
    """
    dir_modello = percorso_llm_rew_7x7(seed).parent
    dir_log_rew = DIR_LOG_7x7 / "llm_rew" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_rew.mkdir(parents=True, exist_ok=True)

    print("[AG-LLM-REW-7x7] Curriculum 7x7 -- seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-REW-7x7] LAMBDA_LLM=" + str(LAMBDA_LLM_7x7)
          + " | trigger=solo_push (~5% step)")
    print("[AG-LLM-REW-7x7] n_envs=1 (Ollama single-threaded)")

    # Agente LLM-REW (come in train_ppo_llm_rew.py): client riusato tra le fasi
    agente = AgenteRicompensaLLM(
        provider=provider,
        lambda_llm=LAMBDA_LLM_7x7,
    )

    # Policy kwargs: CNN custom + LSTM, identici a train_ppo.py / train_ppo_llm_rew.py
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

        print("\n[AG-LLM-REW-7x7] Fase " + str(i_fase) + ": " + nome)
        print("[AG-LLM-REW-7x7] Timestep=" + str(ts_ppo) + " | max_step=" + str(fase["max_step"])
              + " | ent_coef=" + str(ent))

        # Ambiente di training (come train_ppo_llm_rew.py):
        # SokobanEnv7x7 -> RicompensaLLM -> AggiuntaCanale -> Monitor
        base_train = crea_env_7x7(fase, seed)
        env_train  = Monitor(AggiuntaCanale(agente.avvolgi_env(base_train)))

        # Validazione senza wrapper LLM: misura la policy pura
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

        # Curriculum come train_ppo_llm_rew.py: una sola passata per fase, nessuna ripetizione
        # adattiva (qui il collo di bottiglia è il tempo del LLM, non la convergenza dell'RL).
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

        # Statistiche d'uso del LLM (come train_ppo_llm_rew.py)
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

    # Salvataggio del modello finale
    percorso_finale = str(percorso_llm_rew_7x7(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("[AG-LLM-REW-7x7] Training completato in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-REW-7x7] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, provider=args.provider)
