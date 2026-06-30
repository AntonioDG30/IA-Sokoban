# Training di AG-PPO con curriculum learning C0->C5.
#
# Usa RecurrentPPO (CnnLstmPolicy, sb3-contrib) invece del PPO semplice:
#   - una LSTM nascosta da 256 unità ricorda le sequenze di azioni precedenti;
#   - è utile in Sokoban, dove pianificare una spinta richiede memoria degli step;
#   - resta compatibile con VecEnv e con il curriculum (set_env() funziona normalmente).
#
# Reward shaping attivo su entrambe le componenti:
#   - giocatore->cassa (scala_player_box=0.1);
#   - casse->target via distanza Manhattan/Ungherese (SCALA_MANHATTAN=0.3).
#
# Curriculum adattivo a 6 fasi (FASI_CURRICULUM_V9):
#   C0: 1 cassa, generata            (600K step base, soglia 15% solve rate)
#   C1: 2 casse, generata            (1M step base,   soglia 10%)
#   C2: 3 casse, generata            (1.5M step base, soglia  5%)
#   C3: 4 casse, generata            (2M step base,   soglia  3%)
#   C4: 4 casse, Boxoban medium      (2M step base,   soglia  2%)
#   C5: 4 casse, Boxoban unfiltered  (2M step base,   nessuna soglia)
#   Totale: ~9.1M step (di più se una fase viene ripetuta perché sotto soglia)
#
# Uso: python src/sistema_10x10/train_ppo.py [--seed 42] [--dir-dati dataset/boxoban]
# Il modello finale viene salvato in artifacts/models/10x10/ppo/ppo_seed{seed}.zip

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

from sistema_10x10.config import (
    FASI_CURRICULUM_V9,
    CONFIG_PPO,
    DIR_DATI,
    DIR_LOG,
    N_ENVS_PPO,
    N_EPISODI_VALUTAZIONE,
    crea_env_da_fase,
    percorso_modello_ppo,
)
from core.ambiente.sokoban_cnn import SokobanCNN
from core.ambiente.cnn_wrapper import AggiuntaCanale


# CURRICULUM ADATTIVO — SOGLIE E PARAMETRI DI RIPETIZIONE

# Solve rate minimo (%) per avanzare alla fase successiva. Se non viene raggiunto entro
# MAX_RIPETIZIONI_FASE ripetizioni, si avanza comunque.
SOGLIE_CURRICULUM = {
    "C0-1box-gen":        15.0,   # 1 cassa: obiettivo ragionevole
    "C1-2box-gen":        10.0,   # 2 casse: difficile, soglia più bassa
    "C2-3box-gen":         5.0,   # 3 casse: molto difficile
    "C3-4box-gen":         3.0,   # 4 casse generate: quasi impossibile al 100%
    "C4-4box-medium":      2.0,   # Boxoban medium: dati reali più vari
    "C5-4box-unfiltered":  0.0,   # ultima fase: nessuna soglia, si conclude sempre
}

# Ripetizioni aggiuntive massime per fase, oltre alla prima esecuzione
MAX_RIPETIZIONI_FASE = 2   # => budget massimo 3x per fase

# Reward soglia oltre la quale un episodio conta come "risolto".
# Con il reward shaping attivo (SCALA_MANHATTAN=0.3, SCALA_PLAYER_BOX=0.1) un episodio può
# avere reward positiva anche senza completare il livello (premio dell'avvicinamento). La
# soglia 9.0 è sicura perché:
#   - un episodio RISOLTO ha sempre reward >= 9.5 (BONUS_COMPLETAMENTO +10, più le casse);
#   - un episodio NON risolto non arriva a 9.0 nemmeno nel caso migliore di shaping.
# NON usare > 0: con lo shaping produrrebbe falsi positivi (~20% gonfiato vs ~0% reale).
_SOGLIA_RISOLTO = 9.0


def _leggi_max_solve_rate(dir_eval_logs: Path) -> float:
    """
    Legge dal file evaluations.npz di SB3 il massimo solve rate (%) raggiunto.

    Conta come risolti gli episodi con reward >= _SOGLIA_RISOLTO (9.0), distinguendoli da
    quelli con solo shaping positivo. Restituisce 0.0 se il file manca o è corrotto.
    """
    npz = dir_eval_logs / "evaluations.npz"
    if not npz.exists():
        return 0.0
    try:
        d = np.load(str(npz))
        # results ha shape (n_eval, n_eval_episodes): ogni riga è una valutazione
        return float((d["results"] >= _SOGLIA_RISOLTO).mean(axis=1).max() * 100)
    except Exception:
        return 0.0


# CLI

def _parse_args():
    """Legge gli argomenti da riga di comando: --seed (int) e --dir-dati (path a dataset/boxoban)."""
    p = argparse.ArgumentParser(description="Training AG-PPO curriculum v9")
    p.add_argument("--seed",     type=int, default=42,           help="Seed fisso")
    p.add_argument("--dir-dati", type=str, default=str(DIR_DATI), help="Path dataset/boxoban")
    return p.parse_args()


# TRAINING

def addestra_curriculum(seed: int, dir_dati: str) -> None:
    """
    Addestra AG-PPO con il curriculum adattivo completo C0->C5.

    Per ogni fase costruisce un nuovo VecEnv, crea il modello RecurrentPPO (alla prima fase)
    o gli cambia ambiente con set_env(), e ripete la fase fino a MAX_RIPETIZIONI_FASE volte
    se il solve rate resta sotto soglia. Alla fine salva il modello in
    artifacts/models/10x10/ppo/ppo_seed{seed}.zip.
    """
    dir_dati_path = Path(dir_dati)
    dir_modello   = percorso_modello_ppo(seed).parent
    dir_log_ppo   = DIR_LOG / "ppo" / f"seed{seed}"
    dir_modello.mkdir(parents=True, exist_ok=True)
    dir_log_ppo.mkdir(parents=True, exist_ok=True)

    print("[AG-PPO] Curriculum v9 - seed=" + str(seed))

    modello = None
    t_totale = time.time()

    for i_fase, fase in enumerate(FASI_CURRICULUM_V9):
        nome   = fase["nome"]
        ts_ppo = fase["timestep_ppo"]
        max_s  = fase["max_step"]
        ent    = fase["ent_coef"]

        print("\n[AG-PPO] Fase " + str(i_fase) + ": " + nome)
        print("[AG-PPO] Timestep=" + str(ts_ppo) + " | max_step=" + str(max_s)
              + " | ent_coef=" + str(ent))

        # Ambiente di training (VecEnv parallelo).
        # AggiuntaCanale: (10,10) -> (1,10,10), il formato channels-first richiesto da SokobanCNN
        def _factory_train(fase=fase):
            return Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="train")))

        env_train = make_vec_env(_factory_train, n_envs=N_ENVS_PPO, seed=seed)

        # Ambiente di validazione (singolo)
        env_val = Monitor(AggiuntaCanale(crea_env_da_fase(fase, str(dir_dati_path), seed, split="valid")))

        # Policy kwargs: CNN custom + LSTM.
        # CnnLstmPolicy = SokobanCNN come estrattore di feature + LSTM da 256 unità, che
        # dà alla policy memoria delle azioni precedenti (essenziale per pianificare le spinte).
        policy_kwargs = dict(
            features_extractor_class=SokobanCNN,
            features_extractor_kwargs=dict(features_dim=256),
            lstm_hidden_size=256,
            n_lstm_layers=1,
            shared_lstm=True,
            enable_critic_lstm=False,  # obbligatorio quando shared_lstm=True
        )

        # Crea il modello alla prima fase, poi nelle successive cambia solo ambiente e ent_coef
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

        # Callback della fase (riusati tra le ripetizioni così i log si accumulano)
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

        # Curriculum adattivo: ripete la fase fino a MAX_RIPETIZIONI_FASE volte finché il
        # solve rate non raggiunge la soglia; alla terza ripetizione avanza comunque.
        soglia = SOGLIE_CURRICULUM.get(nome, 0.0)
        t_fase = time.time()

        for rep in range(MAX_RIPETIZIONI_FASE + 1):
            t0 = time.time()
            modello.learn(
                total_timesteps=ts_ppo,
                callback=callbacks,
                tb_log_name="PPO_" + nome + "_seed" + str(seed),
                # reset_num_timesteps solo alla primissima esecuzione (rep 0 di C0)
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

            # Avanza se la soglia è raggiunta o se le ripetizioni sono esaurite
            if max_sr >= soglia or rep >= MAX_RIPETIZIONI_FASE:
                if rep > 0 and max_sr < soglia:
                    print(
                        "[AG-PPO] Soglia " + str(soglia) + "% non raggiunta"
                        + " (" + str(round(max_sr, 1)) + "%) - avanzamento forzato"
                    )
                break

            print(
                "[AG-PPO] Solve rate " + str(round(max_sr, 1)) + "% < soglia "
                + str(soglia) + "% - ripetizione " + str(rep + 1)
                + "/" + str(MAX_RIPETIZIONI_FASE)
            )

        elapsed_tot_fase = time.time() - t_fase
        print(
            "[AG-PPO] Fase " + nome + " totale: "
            + str(round(elapsed_tot_fase / 60, 1)) + " min"
        )

        env_train.close()
        env_val.close()

    # Salvataggio del modello finale
    percorso_finale = str(percorso_modello_ppo(seed))
    Path(percorso_finale).parent.mkdir(parents=True, exist_ok=True)
    modello.save(percorso_finale)

    elapsed_tot = time.time() - t_totale
    print("[AG-PPO] Training completato in " + str(round(elapsed_tot / 3600, 2)) + " ore")
    print("[AG-PPO] Modello salvato: " + percorso_finale + ".zip")


if __name__ == "__main__":
    args = _parse_args()
    addestra_curriculum(seed=args.seed, dir_dati=args.dir_dati)
