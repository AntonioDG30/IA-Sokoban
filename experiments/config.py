"""Configurazione centrale degli esperimenti.

Tutti gli iperparametri, i percorsi e i seed sono definiti qui.
I training script importano da questo file — nessun valore hardcoded altrove.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

# Radice del progetto (directory che contiene experiments/, sokoban_env/, ...)
RADICE_PROGETTO = Path(__file__).resolve().parent.parent

DIR_DATI     = RADICE_PROGETTO / "data" / "boxoban"
DIR_MODELLI  = RADICE_PROGETTO / "models"
DIR_LOG      = RADICE_PROGETTO / "logs"
DIR_RISULTATI = RADICE_PROGETTO / "docs" / "report"

# ---------------------------------------------------------------------------
# Seed per riproducibilità
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]

# ---------------------------------------------------------------------------
# Configurazione ambiente
# ---------------------------------------------------------------------------

CONFIG_ENV = {
    "difficolta_train": "unfiltered",
    "split_train":      "train",
    "difficolta_val":   "unfiltered",
    "split_val":        "valid",
    "max_step":         120,
}

# Set di test (valutazione comparativa finale)
SET_DI_TEST = [
    {"difficolta": "unfiltered", "split": "test"},
    {"difficolta": "medium",     "split": "valid"},
    {"difficolta": "hard",       "split": "test"},
]

# Numero di livelli per ogni split (None = tutti)
N_LIVELLI_TRAIN = 10_000
N_LIVELLI_VAL   = 1_000
N_LIVELLI_TEST  = 1_000

# ---------------------------------------------------------------------------
# Iperparametri PPO (AG-PPO e AG-LLM-REW)
# ---------------------------------------------------------------------------

CONFIG_PPO = {
    "policy":          "MlpPolicy",
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      64,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,
    "verbose":         1,
    "device":          "auto",
    "tensorboard_log": str(DIR_LOG / "ppo"),
}

TIMESTEPS_PPO = 1_000_000

# ---------------------------------------------------------------------------
# Iperparametri DQN (AG-DQN)
# ---------------------------------------------------------------------------

CONFIG_DQN = {
    "policy":                  "MlpPolicy",
    "learning_rate":           1e-4,
    "buffer_size":             100_000,
    "learning_starts":         1_000,
    "batch_size":              32,
    "gamma":                   0.99,
    "exploration_fraction":    0.1,
    "exploration_final_eps":   0.05,
    "target_update_interval":  1_000,
    "verbose":                 1,
    "device":                  "auto",
    "tensorboard_log":         str(DIR_LOG / "dqn"),
}

TIMESTEPS_DQN = 1_000_000

# ---------------------------------------------------------------------------
# Configurazione LLM (AG-LLM-GUIDE e AG-LLM-REW)
# ---------------------------------------------------------------------------

CONFIG_LLM = {
    # Provider primario (Groq) — richiede GROQ_API_KEY nell'ambiente
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "model":       "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    # Provider secondario (Mistral) — richiede MISTRAL_API_KEY nell'ambiente
    "mistral": {
        "base_url":    "https://api.mistral.ai/v1",
        "model":       "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
    },
    # Provider locale (Ollama) — zero costi, nessun rate limit
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "model":       "llama3.1:8b",
        "api_key_env": None,
    },
}

# Provider di default per ogni agente LLM
PROVIDER_LLM_GUIDE = "groq"    # AG-LLM-GUIDE: Groq, modello grande, raccolta demo
PROVIDER_LLM_REW   = "ollama"  # AG-LLM-REW: Ollama, volume alto, nessun rate limit

# Parametri chiamate LLM
MAX_TOKENS_AZIONE = 20    # risposta azione: una parola (su/giu/sinistra/destra)
MAX_TOKENS_REWARD = 10    # risposta reward: un numero float
TIMEOUT_LLM_SEC   = 10   # timeout per singola chiamata API
MAX_RETRY_LLM     = 3    # tentativi in caso di errore

# Numero massimo di episodi per raccolta dimostrazioni AG-LLM-GUIDE
N_EPISODI_LLM_GUIDE = 500

# ---------------------------------------------------------------------------
# Configurazione valutazione
# ---------------------------------------------------------------------------

N_EPISODI_VALUTAZIONE  = 500      # episodi per agente per set di test
INTERVALLO_VALUTAZIONE = 100_000  # ogni N timesteps durante il training

# ---------------------------------------------------------------------------
# Percorsi checkpoint modelli
# ---------------------------------------------------------------------------

def percorso_modello_ppo(seed: int) -> Path:
    """Checkpoint AG-PPO v8 (curriculum 10x10 fisso)."""
    return DIR_MODELLI / "ppo_v8" / f"ppo_cnn_v8_seed{seed}"

def percorso_modello_dqn(seed: int) -> Path:
    """Checkpoint AG-DQN baseline."""
    return DIR_MODELLI / "dqn" / f"dqn_seed{seed}"

def percorso_modello_llm_guide(seed: int) -> Path:
    """Checkpoint AG-LLM-GUIDE (DQN addestrato su demo LLM)."""
    return DIR_MODELLI / "llm_guide" / f"llm_guide_seed{seed}"

def percorso_modello_llm_rew(seed: int) -> Path:
    """Checkpoint AG-LLM-REW (PPO con reward LLM)."""
    return DIR_MODELLI / "llm_rew" / f"llm_rew_seed{seed}"

# ---------------------------------------------------------------------------
# Curriculum learning v8 (AG-PPO baseline)
# ---------------------------------------------------------------------------
#
# Griglia SEMPRE 10x10 — elimina la causa radice dei fallimenti v3-v7:
#   padding_a_10x10() centrava le griglie piccole con offset variabile,
#   la CNN imparava feature di posizione assoluta non trasferibili.
#   Con griglia=(10,10) nativa non viene applicato alcun offset.
#
# Curriculum varia SOLO n_casse (1->2->3->4).
# Fase finale (Boxoban 10x10/4box) identica per struttura alle precedenti.
# Tra ogni fase: ricarica il best model (evita regressioni da policy instability).
#
# Struttura (4 fasi, 4.0M step):
#   C0: 10x10/1box  500k  max_step=150  ent_coef=0.01
#   C1: 10x10/2box  800k  max_step=200  ent_coef=0.01
#   C2: 10x10/3box  1.2M  max_step=250  ent_coef=0.02
#   C3: 10x10/4box  1.5M  max_step=300  ent_coef=0.03  (Boxoban dataset)
#
# Risultati seed42: C0=100%, C1=100%, C2=100%(picco), C3=in corso da best_C2

SCALA_MANHATTAN = 1.5  # reward shaping: scala fattore distanza Manhattan (Ungherese)

FASI_CURRICULUM_V8 = [
    {
        "nome":     "C0-10x10-1box",
        "griglia":  (10, 10),
        "n_casse":  1,
        "timestep": 500_000,
        "max_step": 150,
        "ent_coef": 0.01,
        "dataset":  "generato",
    },
    {
        "nome":     "C1-10x10-2box",
        "griglia":  (10, 10),
        "n_casse":  2,
        "timestep": 800_000,
        "max_step": 200,
        "ent_coef": 0.01,
        "dataset":  "generato",
    },
    {
        "nome":     "C2-10x10-3box",
        "griglia":  (10, 10),
        "n_casse":  3,
        "timestep": 1_200_000,
        "max_step": 250,
        "ent_coef": 0.02,
        "dataset":  "generato",
    },
    {
        "nome":     "C3-10x10-4box",
        "griglia":  (10, 10),
        "n_casse":  4,
        "timestep": 1_500_000,
        "max_step": 300,
        "ent_coef": 0.03,
        "dataset":  "boxoban",
    },
]

TIMESTEPS_CURRICULUM_V8 = sum(f["timestep"] for f in FASI_CURRICULUM_V8)  # 4_000_000
