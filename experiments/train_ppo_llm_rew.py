"""Training AG-LLM-REW: PPO con reward aumentata da LLM su curriculum C0->C5 (v9).

Il LLM valuta l'azione eseguita confrontando griglia pre e post mossa.
Viene chiamato quando il giocatore era adiacente a una cassa (~20% degli step).
A inference time solo la policy PPO agisce: il LLM non serve piu'.

Architettura:
    SokobanEnv -> RicompensaLLM -> Monitor -> PPO (CnnPolicy)

Curriculum v9 (stesso di AG-PPO per confronto equo):
    C0: 1 cassa, griglia generata    (300K step)
    C1: 2 casse, griglia generata    (500K step)
    C2: 3 casse, griglia generata    (800K step)
    C3: 4 casse, griglia generata    (1M step)
    C4: 4 casse, Boxoban medium      (1.2M step)
    C5: 4 casse, Boxoban unfiltered  (1.5M step)
    Totale: ~5.3M step

Stima chiamate LLM: ~20% step adiacenti * 5.3M / VecEnv=1 = ~1.06M
Con cache (obs_pre,action): dipende dalla varieta' degli stati (~300K effettive).
Con qwen3:14b Ollama ~0.21s/call warm: ~17.5h (run notturna).

n_envs=1: Ollama single-threaded, chiamate sequenziali.

Uso:
    python experiments/train_ppo_llm_rew.py [--seed 42] [--provider ollama]
                                            [--dir-dati data/boxoban]

Modello salvato in models/llm_rew/llm_rew_seed{seed}.zip
"""

import argparse
import sys
import time
from pathlib import Path

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from experiments.config import (
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
from agents.llm_reward_agent import AgenteRicompensaLLM, RicompensaLLM
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Training AG-LLM-REW curriculum v9")
    p.add_argument("--seed",     type=int, default=42,              help="Seed fisso")
    p.add_argument("--provider", type=str, default=PROVIDER_DEFAULT, help="Provider LLM")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI),   help="Path data/boxoban")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helper: recupera il RicompensaLLM dalla catena di wrapper
# ---------------------------------------------------------------------------

def _ottieni_wrapper_llm(env) -> RicompensaLLM:
    """Naviga la catena wrapper per trovare RicompensaLLM."""
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
# Training
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int, provider: str, dir_dati: str) -> None:
    """Addestra AG-LLM-REW con curriculum completo C0->C5."""
    dir_dati_path  = Path(dir_dati)
    dir_modello    = percorso_modello_llm_rew(seed).parent
    dir_log_rew    = DIR_LOG / "llm_rew" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_rew.mkdir(parents=True, exist_ok=True)

    print("\n[AG-LLM-REW] ========================================")
    print("[AG-LLM-REW] Curriculum v9 — seed=" + str(seed)
          + " | provider=" + provider)
    print("[AG-LLM-REW] LAMBDA_LLM=" + str(LAMBDA_LLM)
          + " | trigger=adiacente_cassa (~20% step)")
    print("[AG-LLM-REW] n_envs=1 (Ollama single-threaded)")
    print("[AG-LLM-REW] ========================================\n")

    # Agente LLM-REW: client riutilizzato tra le fasi (warm-up una volta sola)
    agente = AgenteRicompensaLLM(
        provider=provider,
        lambda_llm=LAMBDA_LLM,
        solo_adiacente=True,
    )

    policy_kwargs = dict(
        features_extractor_class=SokobanCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        max_s  = fase["max_step"]
        ent    = fase["ent_coef"]

        print("\n[AG-LLM-REW] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print("[AG-LLM-REW] Timestep=" + str(ts_ppo) + " | max_step=" + str(max_s)
              + " | ent_coef=" + str(ent))

        # Ambiente di training:
        #   SokobanEnv (10,10) -> RicompensaLLM (usa obs 2D per LLM) -> AggiuntaCanale (1,10,10) -> Monitor
        base_train = crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")
        env_train  = Monitor(AggiuntaCanale(agente.avvolgi_env(base_train)))

        # Validazione senza wrapper LLM (valuta la policy pura)
        env_val = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")))

        config = {**CONFIG_PPO, "ent_coef": ent}
        config.pop("verbose", None)

        if modello is None:
            modello = PPO(
                policy="CnnPolicy",
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

        # Callbacks
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

        # Statistiche LLM
        wrapper = _ottieni_wrapper_llm(env_train)
        if wrapper is not None:
            s = wrapper.statistiche_llm
            print("[AG-LLM-REW] " + nome + " — LLM calls=" + str(s["n_chiamate"])
                  + " | cache_hit=" + str(s["n_cache_hit"])
                  + " | cache_size=" + str(s["cache_size"])
                  + " | elapsed=" + str(round(elapsed / 60, 1)) + "min")
        else:
            print("[AG-LLM-REW] " + nome + " completata in "
                  + str(round(elapsed / 60, 1)) + " min")

        env_train.close()
        env_val.close()

    # Salva modello finale
    percorso_finale = str(percorso_modello_llm_rew(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("\n[AG-LLM-REW] ========================================")
    print("[AG-LLM-REW] Training completato in "
          + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-LLM-REW] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-LLM-REW] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(
        seed=args.seed,
        provider=args.provider,
        dir_dati=args.dir_dati,
    )
