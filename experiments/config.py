"""Configurazione centrale degli esperimenti.

Tutti gli iperparametri, i percorsi e i seed sono definiti qui.
Gli script di training importano da questo modulo — nessun valore
e' hardcoded altrove.

Curriculum adattivo a sei fasi (10x10 fisso):
    C0: 1 cassa generata proceduralmente
    C1: 2 casse generate proceduralmente
    C2: 3 casse generate proceduralmente
    C3: 4 casse generate proceduralmente
    C4: 4 casse Boxoban medium  (450K livelli, difficolta' intermedia)
    C5: 4 casse Boxoban unfiltered (900K livelli, benchmark finale)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi base del progetto
# ---------------------------------------------------------------------------

# Radice del progetto calcolata dinamicamente rispetto a questo file
RADICE_PROGETTO = Path(__file__).resolve().parent.parent

DIR_DATI      = RADICE_PROGETTO / "data" / "boxoban"   # dataset Boxoban
DIR_MODELLI   = RADICE_PROGETTO / "models"             # checkpoint salvati
DIR_LOG       = RADICE_PROGETTO / "logs"               # log TensorBoard
DIR_RISULTATI = RADICE_PROGETTO / "docs" / "report"    # JSON metriche finali

# ---------------------------------------------------------------------------
# Seed di riproducibilita'
# ---------------------------------------------------------------------------

# Unico seed usato in tutto il progetto; i seed 123 e 456 erano pianificati
# ma non eseguiti per vincoli di tempo (vedi Cap8 — sviluppi futuri)
SEED_DEFAULT = 42

# ---------------------------------------------------------------------------
# Fasi del curriculum adattivo (versione definitiva)
# ---------------------------------------------------------------------------

FASI_CURRICULUM_V9 = [
    # C0: 1 cassa — livelli generati su griglia 10x10, budget aumentato perche'
    # i livelli 10x10 reali sono piu' difficili dei livelli 5x5 dei test iniziali
    {
        "nome":         "C0-1box-gen",
        "n_casse":      1,
        "dataset":      "generato",
        "timestep_ppo": 600_000,
        "timestep_dqn": 400_000,
        "max_step":     120,
        "ent_coef":     0.01,
    },
    # C1: 2 casse — lo spazio degli stati cresce cubicamente, budget raddoppiato
    {
        "nome":         "C1-2box-gen",
        "n_casse":      2,
        "dataset":      "generato",
        "timestep_ppo": 1_000_000,
        "timestep_dqn": 700_000,
        "max_step":     150,
        "ent_coef":     0.01,
    },
    # C2: 3 casse — budget proporzionale alla difficolta' crescente
    {
        "nome":         "C2-3box-gen",
        "n_casse":      3,
        "dataset":      "generato",
        "timestep_ppo": 1_500_000,
        "timestep_dqn": 1_000_000,
        "max_step":     200,
        "ent_coef":     0.02,
    },
    # C3: 4 casse generate — stessa difficolta' di Boxoban ma livelli controllati
    {
        "nome":         "C3-4box-gen",
        "n_casse":      4,
        "dataset":      "generato",
        "timestep_ppo": 2_000_000,
        "timestep_dqn": 1_400_000,
        "max_step":     250,
        "ent_coef":     0.02,
    },
    # C4: Boxoban medium — dataset eterogeneo, distribuzione piu' variegata.
    # Usa split="valid" perche' boxoban_medium non ha cartella test/
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

# Budget totale calcolato come somma dei budget per fase
TIMESTEPS_TOTALI_PPO = sum(f["timestep_ppo"] for f in FASI_CURRICULUM_V9)  # 9_100_000
TIMESTEPS_TOTALI_DQN = sum(f["timestep_dqn"] for f in FASI_CURRICULUM_V9)  # 6_000_000

# ---------------------------------------------------------------------------
# Parametri ambiente
# ---------------------------------------------------------------------------

# Griglia sempre 10x10 per tutto il training: elimina il problema di
# CNN che impara la posizione assoluta anziche' la struttura del puzzle
GRIGLIA_SIZE = (10, 10)

# Reward shaping Manhattan: incentiva l'avvicinamento delle casse ai target.
# 0.3 e' stato scelto sperimentalmente: abbastanza da dare gradiente utile,
# abbastanza basso da non dominare sul bonus di completamento (+10)
SCALA_MANHATTAN = 0.3

# Reward shaping giocatore->cassa: incentiva il giocatore ad avvicinarsi
# alle casse prima di spingerle. Delta-based: oscillare da net=0, non hackabile
SCALA_PLAYER_BOX = 0.1

# Set di valutazione aggiuntivi usati da evaluate_all.py per il confronto finale
SET_VALUTAZIONE = [
    {"nome": "val-medium",     "dataset": "boxoban_medium",     "split": "valid"},
    {"nome": "val-unfiltered", "dataset": "boxoban_unfiltered", "split": "test"},
]

# Episodi per agente per ogni set di valutazione
N_EPISODI_VALUTAZIONE = 100

# PPO usa SubprocVecEnv con 8 ambienti paralleli per raccogliere traiettorie piu' velocemente.
# DQN non supporta ambienti vettorizzati: usa un solo ambiente
N_ENVS_PPO = 8
N_ENVS_DQN = 1

# ---------------------------------------------------------------------------
# Iperparametri PPO (usati da AG-PPO e AG-LLM-REW)
# ---------------------------------------------------------------------------

CONFIG_PPO = {
    "learning_rate": 3e-4,    # tasso di apprendimento Adam
    "n_steps":       2048,    # step per aggiornamento per environment
    "batch_size":    64,      # mini-batch per ogni update SGD
    "n_epochs":      10,      # passate sull'intero buffer per update
    "gamma":         0.99,    # fattore di sconto reward futura
    "gae_lambda":    0.95,    # lambda per GAE (bias-variance trade-off)
    "clip_range":    0.2,     # clipping ratio PPO (valore standard)
    "ent_coef":      0.01,    # coefficiente entropia: sovrascritta per fase
    "verbose":       1,
    "device":        "auto",  # GPU se disponibile, altrimenti CPU
}

# ---------------------------------------------------------------------------
# Iperparametri DQN (usato da AG-DQN e AG-LLM-GUIDE)
# ---------------------------------------------------------------------------

CONFIG_DQN = {
    "learning_rate":          1e-4,   # tasso di apprendimento Adam
    "buffer_size":            100_000, # dimensione replay buffer (non azzerato tra fasi)
    "learning_starts":        1_000,  # step prima di iniziare gli aggiornamenti
    "batch_size":             32,     # campioni per aggiornamento dalla replay memory
    "gamma":                  0.99,   # fattore di sconto
    "exploration_fraction":   0.15,   # frazione di training con epsilon decrescente
    "exploration_final_eps":  0.05,   # epsilon minimo dopo la fase di esplorazione
    "target_update_interval": 1_000,  # ogni quanti step aggiornare la target network
    "verbose":                1,
    "device":                 "auto",
}

# ---------------------------------------------------------------------------
# Configurazione LLM (solo Ollama locale)
# ---------------------------------------------------------------------------

# Solo Ollama e' supportato: non usiamo API esterne per non dipendere da chiavi
CONFIG_LLM = {
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "model":       "qwen3:14b-q4_K_M",
        "api_key_env": None,
    },
}

PROVIDER_DEFAULT  = "ollama"
MAX_TOKENS_AZIONE = 20    # token massimi per la risposta azione (es. "sinistra" = 1 token)
MAX_TOKENS_REWARD = 10    # token massimi per la risposta punteggio (es. "2" = 1 token)
TIMEOUT_LLM_SEC   = 10    # timeout per singola chiamata HTTP
MAX_RETRY_LLM     = 3     # tentativi in caso di errore di rete

# Peso del segnale LLM nella reward di AG-LLM-REW: 0.3 bilancia
# il contributo LLM senza dominare sulla reward di completamento (+10)
LAMBDA_LLM = 0.3

# Episodi di valutazione per AG-LLM (che non ha training, solo inference)
N_EPISODI_LLM_ACT = 100

# ---------------------------------------------------------------------------
# Funzioni per i percorsi dei modelli
# ---------------------------------------------------------------------------

def percorso_modello_ppo(seed: int) -> Path:
    """Restituisce il percorso del checkpoint finale di AG-PPO per il seed dato."""
    return DIR_MODELLI / "ppo" / f"ppo_seed{seed}"


def percorso_modello_dqn(seed: int) -> Path:
    """Restituisce il percorso del checkpoint finale di AG-DQN per il seed dato."""
    return DIR_MODELLI / "dqn" / f"dqn_seed{seed}"


def percorso_risultati_llm_act(seed: int) -> Path:
    """Restituisce il percorso del JSON risultati di AG-LLM (nessun modello salvato)."""
    return DIR_MODELLI / "llm_act" / f"risultati_seed{seed}.json"


def percorso_modello_llm_rew(seed: int) -> Path:
    """Restituisce il percorso del checkpoint finale di AG-LLM-REW per il seed dato."""
    return DIR_MODELLI / "llm_rew" / f"llm_rew_seed{seed}"


def percorso_modello_llm_guide(seed: int) -> Path:
    """Restituisce il percorso del checkpoint finale di AG-LLM-GUIDE per il seed dato.

    A inference time il LLM non serve: solo il DQN addestrato via LfD agisce.
    """
    return DIR_MODELLI / "llm_guide" / f"llm_guide_seed{seed}"


# ---------------------------------------------------------------------------
# Funzione di utilita': crea SokobanEnv dalla configurazione di una fase
# ---------------------------------------------------------------------------

def crea_env_da_fase(fase: dict, dir_dati: str, seme: int, split: str = "train"):
    """Crea un SokobanEnv configurato per la fase del curriculum specificata.

    Seleziona automaticamente GeneratoreLivelli o CaricatoreLivelli in base
    al campo 'dataset' della fase. Applica i parametri di reward shaping
    definiti in questo file (SCALA_MANHATTAN, SCALA_PLAYER_BOX).

    Parametri:
        fase:     dizionario da FASI_CURRICULUM_V9 con nome, n_casse, dataset, max_step.
        dir_dati: percorso stringa a data/boxoban/ (None per usare livelli builtin).
        seme:     seed per riproducibilita'.
        split:    'train' | 'valid' | 'test' (rilevante solo per dataset Boxoban).

    Restituisce:
        SokobanEnv non avvolto (senza Monitor o AggiuntaCanale).
    """
    import sys
    sys.path.insert(0, str(RADICE_PROGETTO))
    from sokoban_env import SokobanEnv

    n_casse  = fase["n_casse"]
    max_step = fase["max_step"]
    dataset  = fase["dataset"]

    if dataset == "generato":
        # usa_generatore=True forza GeneratoreLivelli anche su griglia 10x10.
        # Senza questo flag, SokobanEnv userebbe il CaricatoreLivelli con i soli
        # 3 livelli builtin di fallback invece dei livelli procedurali del curriculum
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
