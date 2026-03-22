"""Training AG-PPO con curriculum learning C0->C5 (v10).

Usa RecurrentPPO (CnnLstmPolicy, sb3-contrib) invece di PPO plain:
    - LSTM nascosta 256 unita' permette di ricordare sequenze di azioni precedenti
    - Cruciale per puzzle Sokoban: pianificare push richiede memoria degli step
    - Compatible con VecEnv e curriculum learning (set_env() funziona normalmente)

Reward shaping v10 (entrambe le opzioni attive):
    - Option A: shaping giocatore->cassa (scala_player_box=0.1)
    - SCALA_MANHATTAN=0.3: shaping casse->target (safe, necessario su livelli infiniti)

Curriculum adattivo in 6 fasi progressive (FASI_CURRICULUM_V9):
    C0: 1 cassa, griglia generata    (600K step base, soglia 15% solve rate)
    C1: 2 casse, griglia generata    (1M step base,   soglia 10%)
    C2: 3 casse, griglia generata    (1.5M step base, soglia  5%)
    C3: 4 casse, griglia generata    (2M step base,   soglia  3%)
    C4: 4 casse, Boxoban medium      (2M step base,   soglia  2%)
    C5: 4 casse, Boxoban unfiltered  (2M step base,   nessuna soglia)
    Totale: ~9.1M step (piu' ripetizioni se la soglia non e' raggiunta)

Uso:
    python experiments/train_ppo.py [--seed 42] [--dir-dati data/boxoban]

Il modello viene salvato in models/ppo/ppo_seed{seed}.zip
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from experiments.config import (
    FASI_CURRICULUM_V9,
    CONFIG_PPO,
    DIR_DATI,
    DIR_LOG,
    N_ENVS_PPO,
    N_EPISODI_VALUTAZIONE,
    crea_env_da_fase,
    percorso_modello_ppo,
)
from sokoban_env.sokoban_cnn import SokobanCNN
from sokoban_env.cnn_wrapper import AggiuntaCanale


# ---------------------------------------------------------------------------
# Curriculum adattivo — soglie e parametri ripetizione
# ---------------------------------------------------------------------------

# Solve rate minimo (%) richiesto per avanzare alla fase successiva.
# Se non raggiunto entro MAX_RIPETIZIONI_FASE ripetizioni, si avanza comunque.
SOGLIE_CURRICULUM = {
    "C0-1box-gen":        15.0,   # 1 cassa: obiettivo ragionevole
    "C1-2box-gen":        10.0,   # 2 casse: difficile, soglia piu' bassa
    "C2-3box-gen":         5.0,   # 3 casse: molto difficile
    "C3-4box-gen":         3.0,   # 4 casse generate: quasi impossibile al 100%
    "C4-4box-medium":      2.0,   # Boxoban medium: dati reali piu' vari
    "C5-4box-unfiltered":  0.0,   # Ultima fase: nessuna soglia, si conclude sempre
}

# Massimo numero di ripetizioni aggiuntive per fase (oltre alla prima esecuzione)
MAX_RIPETIZIONI_FASE = 2   # => max 3x budget per fase


# Soglia minima per considerare un episodio "risolto" (livello completato).
# Con reward shaping attivo (SCALA_MANHATTAN=0.3, SCALA_PLAYER_BOX=0.1) la reward
# puo' essere positiva anche senza completare il livello (shaping su avvicinamento).
# Soglia 9.0 e' sicura perche':
#   - Min reward episodio RISOLTO  = 10.0 (BONUS_COMPLETAMENTO) + n_casse - step_penalty >= 9.5
#   - Max reward episodio NON RISOLTO con shaping < 9.0 (verificato empiricamente)
# NON usare > 0: con shaping produce falsi positivi (~20% inflate vs ~0% reale).
_SOGLIA_RISOLTO = 9.0


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """Legge il massimo solve rate (%) dai log di valutazione SB3 (evaluations.npz).

    Restituisce 0.0 se il file non esiste o e' corrotto.
    Usa _SOGLIA_RISOLTO (9.0) per distinguere episodi completati da episodi con
    solo reward shaping positivo (falsi positivi con threshold > 0).
    """
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        # results shape: (n_eval, n_eval_episodes) — ogni riga e' un evaluation step
        return float((d["results"] >= _SOGLIA_RISOLTO).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Legge gli argomenti da riga di comando.

    Restituisce:
        Namespace con seed (int) e dir_dati (str percorso a data/boxoban).
    """
    p = argparse.ArgumentParser(description="Training AG-PPO curriculum v9")
    p.add_argument("--seed",     type=int, default=42,           help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI), help="Path data/boxoban")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def addestra_curriculum(seed: int, dir_dati: str) -> None:
    """Addestra AG-PPO con curriculum adattivo completo C0->C5.

    Per ogni fase crea un nuovo VecEnv, aggiorna il modello RecurrentPPO
    (o lo crea alla prima fase) e ripete la fase fino a MAX_RIPETIZIONI_FASE
    volte se la soglia solve rate non e' raggiunta. Salva il modello finale
    in models/ppo/ppo_seed{seed}.zip.

    Parametri:
        seed:     seed per la riproducibilita' del training e della generazione livelli.
        dir_dati: percorso alla directory data/boxoban/ contenente i dataset Boxoban.
    """
    dir_dati_path = Path(dir_dati)
    dir_modello   = percorso_modello_ppo(seed).parent
    dir_log_ppo   = DIR_LOG / "ppo" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_ppo.mkdir(parents=True, exist_ok=True)

    print("\n[AG-PPO] ========================================")
    print("[AG-PPO] Curriculum v9 — seed=" + str(seed))
    print("[AG-PPO] ========================================\n")

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        max_s  = fase["max_step"]
        ent    = fase["ent_coef"]

        print("\n[AG-PPO] --- Fase " + str(i_fase) + ": " + nome + " ---")
        print("[AG-PPO] Timestep=" + str(ts_ppo) + " | max_step=" + str(max_s)
              + " | ent_coef=" + str(ent))

        # Ambiente di training (VecEnv parallelo)
        # AggiuntaCanale: (10,10) -> (1,10,10) richiesto da SokobanCNN (channels-first)
        def _factory_train(fase=fase):
            return Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")))

        env_train = make_vec_env(_factory_train, n_envs=N_ENVS_PPO, seed=seed)

        # Ambiente di validazione (singolo)
        env_val = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")))

        # Policy kwargs: CNN custom + LSTM (v10 Option B)
        # CnnLstmPolicy = SokobanCNN features extractor + LSTM 256 unita'
        # LSTM permette di ricordare sequenze di azioni: essenziale per Sokoban
        # (pianificare push richiede memoria degli step precedenti).
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,  # richiesto quando shared_lstm=True
        )

        # Crea o aggiorna il modello (set_env per fase successiva)
        config = {**CONFIG_PPO, "ent_coef": ent}
        config.pop("verbose", None)

        if modello is None:
            modello = RecurrentPPO(
                policy="CnnLstmPolicy",
                env=env_train,
                seed=seed,
                tensorboard_log=str(dir_log_ppo),
                policy_kwargs=policy_kwargs,
                verbose=1,
                **config,
            )
        else:
            modello.set_env(env_train)
            modello.ent_coef = ent

        # Callbacks per questa fase (riusati tra le ripetizioni per accumulare log)
        dir_fase = dir_modello / nome
        dir_fase.mkdir(exist_ok=True)
        callbacks = [
            EvalCallback(
                env_val,
                best_model_save_path=str(dir_fase / "best"),
                log_path=str(dir_fase / "eval_logs"),
                eval_freq=max(20_000 // N_ENVS_PPO, 1),
                n_eval_episodes=N_EPISODI_VALUTAZIONE,
                deterministic=True,
                render=False,
                verbose=0,
            ),
            CheckpointCallback(
                save_freq=max(100_000 // N_ENVS_PPO, 1),
                save_path=str(dir_fase / "checkpoints"),
                name_prefix="ppo_" + nome + "_seed" + str(seed),
                verbose=0,
            ),
        ]

        # --- Curriculum adattivo ---
        # Ripete la fase fino a MAX_RIPETIZIONI_FASE volte se il solve rate
        # non raggiunge la soglia. Alla terza ripetizione avanza comunque.
        soglia = SOGLIE_CURRICULUM.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE + 1):
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_ppo,
                callback=callbacks,
                tb_log_name="PPO_" + nome + "_seed" + str(seed),
                # reset_num_timesteps solo alla primissima fase (rep 0 di C0)
                reset_num_timesteps=(i_fase == 0 and rep == 0),
            )
            elapsed_rep = time.time() - t0

            max_sr = _leggi_max_solve_rate(dir_fase / "eval_logs")
            print(
                "[AG-PPO] Fase " + nome + " rep " + str(rep)
                + " completata in " + str(round(elapsed_rep / 60, 1)) + " min"
                + " | max_solve_rate=" + str(round(max_sr, 1)) + "%"
                + " | soglia=" + str(soglia) + "%"
            )

            # Avanza se la soglia e' raggiunta o se sono esaurite le ripetizioni
            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-PPO] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) — avanzamento forzato"
                    )
                break

            print(
                "[AG-PPO] Solve rate " + str(round(max_sr, 1)) + "% < soglia "
                + str(soglia) + "% — ripetizione " + str(rep + 1)
                + "/" + str(MAX_RIPETIZIONI_FASE)
            )

        elapsed_tot_fase = time.time() - t_fase
        print(
            "[AG-PPO] Fase " + nome + " totale: "
            + str(round(elapsed_tot_fase / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salva modello finale
    percorso_finale = str(percorso_modello_ppo(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("\n[AG-PPO] ========================================")
    print("[AG-PPO] Training completato in " + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-PPO] Modello salvato: " + percorso_finale + ".zip")
    print("[AG-PPO] ========================================\n")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, dir_dati=args.dir_dati)
