# Configurazione centrale degli esperimenti.
#
# Tutti gli iperparametri, i percorsi e i seed sono definiti qui: gli script di training
# importano da questo modulo, nessun valore è hardcoded altrove.
#
# Curriculum adattivo a sei fasi (griglia 10x10 fissa):
#   C0: 1 cassa generata proceduralmente
#   C1: 2 casse generate proceduralmente
#   C2: 3 casse generate proceduralmente
#   C3: 4 casse generate proceduralmente
#   C4: 4 casse Boxoban medium      (~450K livelli, difficoltà intermedia)
#   C5: 4 casse Boxoban unfiltered  (~900K livelli, benchmark finale)

from pathlib import Path

# PERCORSI BASE DEL PROGETTO

# Radice del repo, risalendo da questo file: src/sistema_10x10/config.py -> root (3 livelli)
RADICE_PROGETTO = Path(__file__).resolve().parent.parent.parent

DIR_DATI      = RADICE_PROGETTO / "dataset" / "boxoban"   # dataset Boxoban
DIR_MODELLI   = RADICE_PROGETTO / "artifacts" / "models" / "10x10"             # checkpoint salvati
DIR_LOG       = RADICE_PROGETTO / "artifacts" / "logs" / "10x10"               # log TensorBoard
DIR_RISULTATI = RADICE_PROGETTO / "results" / "seed42"    # JSON con le metriche finali

# FASI DEL CURRICULUM ADATTIVO (VERSIONE DEFINITIVA)
# Il seed di riproducibilità è 42 (i seed 123 e 456 erano pianificati ma non eseguiti per
# vincoli di tempo, vedi Cap8); gli script lo ricevono da riga di comando con default 42.

FASI_CURRICULUM_V9 = [
    # C0: 1 cassa — livelli generati su griglia 10x10; budget più alto perché i livelli
    # 10x10 reali sono più difficili dei 5x5 usati nei primi test
    {
        "nome":         "C0-1box-gen",
        "n_casse":      1,
        "dataset":      "generato",
        "timestep_ppo": 600_000,
        "timestep_dqn": 400_000,
        "max_step":     120,
        "ent_coef":     0.01,
    },
    # C1: 2 casse — lo spazio degli stati cresce in modo cubico, budget raddoppiato
    {
        "nome":         "C1-2box-gen",
        "n_casse":      2,
        "dataset":      "generato",
        "timestep_ppo": 1_000_000,
        "timestep_dqn": 700_000,
        "max_step":     150,
        "ent_coef":     0.01,
    },
    # C2: 3 casse — budget proporzionale alla difficoltà crescente
    {
        "nome":         "C2-3box-gen",
        "n_casse":      3,
        "dataset":      "generato",
        "timestep_ppo": 1_500_000,
        "timestep_dqn": 1_000_000,
        "max_step":     200,
        "ent_coef":     0.02,
    },
    # C3: 4 casse generate — stessa difficoltà di Boxoban ma su livelli controllati
    {
        "nome":         "C3-4box-gen",
        "n_casse":      4,
        "dataset":      "generato",
        "timestep_ppo": 2_000_000,
        "timestep_dqn": 1_400_000,
        "max_step":     250,
        "ent_coef":     0.02,
    },
    # C4: Boxoban medium — dataset eterogeneo, distribuzione più variegata.
    # Usa split="valid" perché boxoban_medium non ha una cartella test/
    {
        "nome":         "C4-4box-medium",
        "n_casse":      4,
        "dataset":      "boxoban_medium",
        "split":        "valid",
        "timestep_ppo": 2_000_000,
        "timestep_dqn": 1_200_000,
        "max_step":     300,
        "ent_coef":     0.03,
    },
    # C5: Boxoban unfiltered — benchmark finale su 900K livelli reali.
    # Usa split="test" per valutare su dati mai visti durante il training
    {
        "nome":         "C5-4box-unfiltered",
        "n_casse":      4,
        "dataset":      "boxoban_unfiltered",
        "split":        "test",
        "timestep_ppo": 2_000_000,
        "timestep_dqn": 1_300_000,
        "max_step":     300,
        "ent_coef":     0.03,
    },
]

# Budget totale: 9,1M step per AG-PPO e 6M per AG-DQN (somma dei budget per fase qui sopra).

# PARAMETRI AMBIENTE

# Griglia sempre 10x10 per tutto il training: evita che la CNN impari la posizione
# assoluta degli oggetti invece della struttura del puzzle
GRIGLIA_SIZE = (10, 10)

# Reward shaping Manhattan: incentiva l'avvicinamento delle casse ai target. 0.3 è scelto
# sperimentalmente, abbastanza da dare gradiente utile ma senza dominare il bonus +10
SCALA_MANHATTAN = 0.3

# Reward shaping giocatore->cassa: spinge il giocatore verso le casse prima di spingerle.
# Delta-based: oscillando avanti e indietro il guadagno netto è zero, quindi non hackabile
SCALA_PLAYER_BOX = 0.1

# Set di valutazione extra usati da evaluate_all.py per il confronto finale
SET_VALUTAZIONE = [
    {"nome": "val-medium",     "dataset": "boxoban_medium",     "split": "valid"},
    {"nome": "val-unfiltered", "dataset": "boxoban_unfiltered", "split": "test"},
]

# Episodi per agente su ogni set di valutazione
N_EPISODI_VALUTAZIONE = 100

# PPO usa SubprocVecEnv con 8 ambienti paralleli per raccogliere traiettorie più in fretta;
# DQN non supporta ambienti vettorizzati e ne usa uno solo (cablato negli script di training)
N_ENVS_PPO = 8

# IPERPARAMETRI PPO (usati da AG-PPO e AG-LLM-REW)

CONFIG_PPO = {
    "learning_rate": 3e-4,    # tasso di apprendimento di Adam
    "n_steps":       2048,    # step raccolti per aggiornamento, per ambiente
    "batch_size":    64,      # dimensione del mini-batch di ogni update SGD
    "n_epochs":      10,      # passate sull'intero buffer a ogni update
    "gamma":         0.99,    # fattore di sconto delle reward future
    "gae_lambda":    0.95,    # lambda della GAE (trade-off bias-varianza)
    "clip_range":    0.2,     # clipping del ratio PPO (valore standard)
    "ent_coef":      0.01,    # coefficiente di entropia: sovrascritto per fase
    "verbose":       1,
    "device":        "auto",  # GPU se disponibile, altrimenti CPU
}

# IPERPARAMETRI DQN (usati da AG-DQN e AG-LLM-GUIDE)

CONFIG_DQN = {
    "learning_rate":          1e-4,    # tasso di apprendimento di Adam
    "buffer_size":            100_000, # dimensione del replay buffer (non azzerato tra fasi)
    "learning_starts":        1_000,   # step da raccogliere prima di iniziare gli update
    "batch_size":             32,      # campioni per update prelevati dal replay buffer
    "gamma":                  0.99,    # fattore di sconto
    "exploration_fraction":   0.15,    # frazione di training in cui epsilon decresce
    "exploration_final_eps":  0.05,    # epsilon minimo dopo la fase di esplorazione
    "target_update_interval": 1_000,   # ogni quanti step sincronizzare la target network
    "verbose":                1,
    "device":                 "auto",
}

# CONFIGURAZIONE LLM (SOLO OLLAMA LOCALE)

# Supportiamo solo Ollama in locale: niente API esterne né chiavi segrete. Il client
# (ClienteLLM) parla con l'API nativa di Ollama (/api/chat e /api/ps) sull'host cablato
# localhost:11434; di questa configurazione legge solo il nome del modello.
CONFIG_LLM = {
    "ollama": {
        "model": "qwen3:14b-q4_K_M",
    },
}

PROVIDER_DEFAULT  = "ollama"
MAX_TOKENS_AZIONE = 10    # token massimi per la risposta-azione (es. "sinistra" = 1 token)
MAX_TOKENS_REWARD = 10    # token massimi per la risposta-punteggio (es. "2" = 1 token)
TIMEOUT_LLM_SEC   = 10    # timeout di una singola chiamata HTTP al LLM
MAX_RETRY_LLM     = 3     # tentativi in caso di errore di rete

# Peso del segnale LLM nella reward di AG-LLM-REW: 0.3 bilancia il contributo del LLM
# senza farlo dominare sul bonus di completamento (+10)
LAMBDA_LLM = 0.3

# Episodi di valutazione per AG-LLM-ACT (non ha training: solo inference)
N_EPISODI_LLM_ACT = 100

# FUNZIONI PER I PERCORSI DEI MODELLI

def percorso_modello_ppo(seed: int) -> Path:
    """Percorso del checkpoint finale di AG-PPO per il seed dato."""
    return DIR_MODELLI / "ppo" / f"ppo_seed{seed}"


def percorso_modello_dqn(seed: int) -> Path:
    """Percorso del checkpoint finale di AG-DQN per il seed dato."""
    return DIR_MODELLI / "dqn" / f"dqn_seed{seed}"


def percorso_risultati_llm_act(seed: int) -> Path:
    """Percorso del JSON dei risultati di AG-LLM-ACT (non salva nessun modello)."""
    return DIR_MODELLI / "llm_act" / f"risultati_seed{seed}.json"


def percorso_modello_llm_rew(seed: int) -> Path:
    """Percorso del checkpoint finale di AG-LLM-REW per il seed dato."""
    return DIR_MODELLI / "llm_rew" / f"llm_rew_seed{seed}"


def percorso_modello_llm_guide(seed: int) -> Path:
    """
    Percorso del checkpoint finale di AG-LLM-GUIDE per il seed dato.
    A inference time il LLM non serve: agisce solo il DQN addestrato via LfD.
    """
    return DIR_MODELLI / "llm_guide" / f"llm_guide_seed{seed}"


# FUNZIONE DI UTILITÀ: CREA UN SokobanEnv DALLA CONFIGURAZIONE DI UNA FASE

def crea_env_da_fase(fase: dict, dir_dati: str, seme: int, split: str = "train"):
    """
    Crea il SokobanEnv configurato per una fase del curriculum.

    In base al campo 'dataset' della fase sceglie il generatore procedurale (per le fasi
    generate) o il caricatore Boxoban, e applica i parametri di reward shaping di questo
    file (SCALA_MANHATTAN, SCALA_PLAYER_BOX). split conta solo per i dataset Boxoban.
    Restituisce il SokobanEnv nudo, senza Monitor né AggiuntaCanale.
    """
    import sys
    sys.path.insert(0, str(RADICE_PROGETTO))
    from core.ambiente import SokobanEnv

    n_casse  = fase["n_casse"]
    max_step = fase["max_step"]
    dataset  = fase["dataset"]

    if dataset == "generato":
        # usa_generatore=True forza il GeneratoreLivelli anche su griglia 10x10; senza questo
        # flag SokobanEnv userebbe il CaricatoreLivelli con i soli 3 livelli builtin di
        # fallback invece dei livelli procedurali del curriculum
        return SokobanEnv(
            griglia_size=GRIGLIA_SIZE,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            scala_player_box=SCALA_PLAYER_BOX,
            max_step=max_step,
            seme=seme,
            usa_generatore=True,
        )
    elif dataset == "boxoban_medium":
        return SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="medium",
            split=split,
            griglia_size=GRIGLIA_SIZE,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            scala_player_box=SCALA_PLAYER_BOX,
            max_step=max_step,
            seme=seme,
        )
    elif dataset == "boxoban_unfiltered":
        return SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split=split,
            griglia_size=GRIGLIA_SIZE,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            scala_player_box=SCALA_PLAYER_BOX,
            max_step=max_step,
            seme=seme,
        )
    else:
        raise ValueError("Dataset non riconosciuto: " + dataset)
