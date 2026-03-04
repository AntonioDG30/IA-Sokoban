"""Configurazione centrale degli esperimenti.

Tutti gli iperparametri, i percorsi e i seed sono definiti qui.
I training script importano da questo file — nessun valore hardcoded altrove.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

# Radice del progetto (directory che contiene experiments/, sokoban_env/, ...)
RADICE_PROGETTO = Path(__file__).resolve().parent.parent

DIR_DATI = RADICE_PROGETTO / "data" / "boxoban"
DIR_MODELLI = RADICE_PROGETTO / "models"
DIR_LOG = RADICE_PROGETTO / "logs"
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
    "split_train": "train",
    "difficolta_val": "unfiltered",
    "split_val": "valid",
    "max_step": 120,
}

# Set di test (per valutazione comparativa finale)
SET_DI_TEST = [
    {"difficolta": "unfiltered", "split": "test"},
    {"difficolta": "medium",     "split": "valid"},
    {"difficolta": "hard",       "split": "test"},
]

# Numero di livelli da usare per ogni split (None = tutti)
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
    "tensorboard_log": str(DIR_LOG / "ppo"),
}

TIMESTEPS_PPO = 1_000_000

# ---------------------------------------------------------------------------
# Iperparametri DQN (AG-DQN)
# ---------------------------------------------------------------------------

CONFIG_DQN = {
    "policy":                    "MlpPolicy",
    "learning_rate":             1e-4,
    "buffer_size":               100_000,
    "learning_starts":           1_000,
    "batch_size":                32,
    "gamma":                     0.99,
    "exploration_fraction":      0.1,
    "exploration_final_eps":     0.05,
    "target_update_interval":    1_000,
    "verbose":                   1,
    "tensorboard_log":           str(DIR_LOG / "dqn"),
}

TIMESTEPS_DQN = 1_000_000

# ---------------------------------------------------------------------------
# Configurazione LLM (AG-LLM-ACT e AG-LLM-REW)
# ---------------------------------------------------------------------------

CONFIG_LLM = {
    # Provider primario (Groq) — richiede GROQ_API_KEY nell'ambiente
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model":    "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    # Provider secondario (Mistral) — richiede MISTRAL_API_KEY nell'ambiente
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "model":    "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
    },
    # Provider locale (Ollama) — zero costi, nessun rate limit
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model":    "llama3.1:8b",
        "api_key_env": None,
    },
}

# Provider di default per ogni agente LLM
PROVIDER_LLM_ACT = "groq"    # Groq: veloce, modello grande
PROVIDER_LLM_REW = "ollama"  # Ollama: volume alto, nessun rate limit

# Parametri chiamate LLM
MAX_TOKENS_AZIONE  = 20     # L'LLM deve rispondere con una parola (su/giù/...)
MAX_TOKENS_REWARD  = 10     # L'LLM deve rispondere con un numero float
TIMEOUT_LLM_SEC    = 10     # Timeout per singola chiamata API
MAX_RETRY_LLM      = 3      # Tentativi in caso di errore

# Numero massimo di episodi per AG-LLM-ACT (limitato da rate limit)
N_EPISODI_LLM_ACT = 200

# ---------------------------------------------------------------------------
# Configurazione valutazione
# ---------------------------------------------------------------------------

N_EPISODI_VALUTAZIONE = 500   # episodi per agente per set di test
INTERVALLO_VALUTAZIONE = 100_000  # ogni N timesteps durante il training

# ---------------------------------------------------------------------------
# Percorsi checkpoint modelli
# ---------------------------------------------------------------------------

def percorso_modello_ppo(seed: int) -> Path:
    """Restituisce il percorso del checkpoint PPO per un dato seed."""
    return DIR_MODELLI / "ppo" / f"ppo_seed{seed}"

def percorso_modello_dqn(seed: int) -> Path:
    """Restituisce il percorso del checkpoint DQN per un dato seed."""
    return DIR_MODELLI / "dqn" / f"dqn_seed{seed}"

def percorso_modello_llm_rew(seed: int) -> Path:
    """Restituisce il percorso del checkpoint PPO+LLM-reward per un dato seed."""
    return DIR_MODELLI / "ppo" / f"llm_rew_seed{seed}"
