"""Configurazione centrale degli esperimenti.

Tutti gli iperparametri, i percorsi e i seed sono definiti qui.
I training script importano da questo file — nessun valore hardcoded altrove.

Curriculum v9 (6 fasi, 10x10 fisso):
    C0: 1-box generato
    C1: 2-box generato
    C2: 3-box generato
    C3: 4-box generato
    C4: 4-box Boxoban medium  (difficolta' intermedia, 450K livelli)
    C5: 4-box Boxoban unfiltered (benchmark finale, 900K livelli)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

RADICE_PROGETTO = Path(__file__).resolve().parent.parent

DIR_DATI      = RADICE_PROGETTO / "data" / "boxoban"
DIR_MODELLI   = RADICE_PROGETTO / "models"
DIR_LOG       = RADICE_PROGETTO / "logs"
DIR_RISULTATI = RADICE_PROGETTO / "docs" / "report"

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

SEED_DEFAULT = 42   # seed attivo per questo progetto (seed unico, riproducibile)

# ---------------------------------------------------------------------------
# Curriculum v9 — unica versione attiva
# ---------------------------------------------------------------------------
#
# Griglia sempre 10x10. Curriculum varia n_casse (1->4) e poi dataset.
# dataset: "generato" | "boxoban_medium" | "boxoban_unfiltered"
#
# timestep_ppo: step per AG-PPO e AG-LLM-REW
# timestep_dqn: step per AG-DQN (meno perche' DQN no VecEnv)
# ent_coef: solo per PPO

FASI_CURRICULUM_V9 = [
    # C0: 1 cassa generata — livelli 10x10 reali piu' difficili dei fallback,
    #     budget raddoppiato rispetto alla stima iniziale.
    {
        "nome":           "C0-1box-gen",
        "n_casse":        1,
        "dataset":        "generato",
        "timestep_ppo":   600_000,
        "timestep_dqn":   400_000,
        "max_step":       120,
        "ent_coef":       0.01,
    },
    # C1: 2 casse generate — complessita' maggiore, budget 2x.
    {
        "nome":           "C1-2box-gen",
        "n_casse":        2,
        "dataset":        "generato",
        "timestep_ppo":   1_000_000,
        "timestep_dqn":   700_000,
        "max_step":       150,
        "ent_coef":       0.01,
    },
    # C2: 3 casse generate — budget 2x circa.
    {
        "nome":           "C2-3box-gen",
        "n_casse":        3,
        "dataset":        "generato",
        "timestep_ppo":   1_500_000,
        "timestep_dqn":   1_000_000,
        "max_step":       200,
        "ent_coef":       0.02,
    },
    # C3: 4 casse generate — budget 2x.
    {
        "nome":           "C3-4box-gen",
        "n_casse":        4,
        "dataset":        "generato",
        "timestep_ppo":   2_000_000,
        "timestep_dqn":   1_400_000,
        "max_step":       250,
        "ent_coef":       0.02,
    },
    # C4: 4 casse Boxoban medium — dataset eterogeneo, budget aumentato.
    {
        "nome":           "C4-4box-medium",
        "n_casse":        4,
        "dataset":        "boxoban_medium",
        "timestep_ppo":   2_000_000,
        "timestep_dqn":   1_200_000,
        "max_step":       300,
        "ent_coef":       0.03,
    },
    # C5: 4 casse Boxoban unfiltered — benchmark finale.
    {
        "nome":           "C5-4box-unfiltered",
        "n_casse":        4,
        "dataset":        "boxoban_unfiltered",
        "timestep_ppo":   2_000_000,
        "timestep_dqn":   1_300_000,
        "max_step":       300,
        "ent_coef":       0.03,
    },
]

TIMESTEPS_TOTALI_PPO = sum(f["timestep_ppo"] for f in FASI_CURRICULUM_V9)  # 9_100_000
TIMESTEPS_TOTALI_DQN = sum(f["timestep_dqn"] for f in FASI_CURRICULUM_V9)  # 6_000_000

# ---------------------------------------------------------------------------
# Ambiente — parametri comuni
# ---------------------------------------------------------------------------

GRIGLIA_SIZE    = (10, 10)   # fissa per tutto il progetto
SCALA_MANHATTAN = 2.0        # v9: reward shaping Manhattan aumentato da 1.5 -> 2.0
                              # Gradiente piu' forte verso i target per compensare
                              # la maggiore difficolta' dei livelli 10x10 reali.

# Set di valutazione finale (usato da evaluate_all.py)
# Ogni agente viene valutato su tutti i livelli di tutte le fasi + questi set extra
SET_VALUTAZIONE = [
    {"nome": "val-medium",      "dataset": "boxoban_medium",      "split": "valid"},
    {"nome": "val-unfiltered",  "dataset": "boxoban_unfiltered",  "split": "test"},
]

N_EPISODI_VALUTAZIONE = 100   # episodi per agente per set di valutazione

# Ambienti paralleli per VecEnv
N_ENVS_PPO = 8   # PPO beneficia di molti env paralleli (VecEnv)
N_ENVS_DQN = 1   # DQN usa env singolo (no VecEnv nativo)

# ---------------------------------------------------------------------------
# Iperparametri PPO (AG-PPO e AG-LLM-REW)
# ---------------------------------------------------------------------------

CONFIG_PPO = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      64,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,   # sovrascritta per fase in training script
    "verbose":         1,
    "device":          "auto",
}

# ---------------------------------------------------------------------------
# Iperparametri DQN (AG-DQN)
# ---------------------------------------------------------------------------

CONFIG_DQN = {
    "learning_rate":            1e-4,
    "buffer_size":              100_000,
    "learning_starts":          1_000,
    "batch_size":               32,
    "gamma":                    0.99,
    "exploration_fraction":     0.15,
    "exploration_final_eps":    0.05,
    "target_update_interval":   1_000,
    "verbose":                  1,
    "device":                   "auto",
}

# ---------------------------------------------------------------------------
# Configurazione LLM
# ---------------------------------------------------------------------------

CONFIG_LLM = {
    "ollama": {
        "base_url":    "http://localhost:11434/v1",
        "model":       "qwen3:14b-q4_K_M",
        "api_key_env": None,
    },
    "groq": {
        "base_url":    "https://api.groq.com/openai/v1",
        "model":       "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "mistral": {
        "base_url":    "https://api.mistral.ai/v1",
        "model":       "mistral-small-latest",
        "api_key_env": "MISTRAL_API_KEY",
    },
}

PROVIDER_DEFAULT   = "ollama"
MAX_TOKENS_AZIONE  = 20    # AG-LLM: token risposta azione
MAX_TOKENS_REWARD  = 10    # AG-LLM-REW: token risposta punteggio
TIMEOUT_LLM_SEC    = 10    # timeout singola chiamata
MAX_RETRY_LLM      = 3     # tentativi su errore

# AG-LLM-REW: parametri reward shaping LLM
LAMBDA_LLM = 0.3           # scala segnale LLM (addizionale a reward default)

# AG-LLM: episodi per valutazione per fase
N_EPISODI_LLM_ACT = 100

# ---------------------------------------------------------------------------
# Percorsi modelli (funzioni)
# ---------------------------------------------------------------------------

def percorso_modello_ppo(seed: int) -> Path:
    """Checkpoint AG-PPO (curriculum v9)."""
    return DIR_MODELLI / "ppo" / f"ppo_seed{seed}"

def percorso_modello_dqn(seed: int) -> Path:
    """Checkpoint AG-DQN (curriculum v9)."""
    return DIR_MODELLI / "dqn" / f"dqn_seed{seed}"

def percorso_risultati_llm_act(seed: int) -> Path:
    """Risultati JSON AG-LLM (direct policy, no modello salvato)."""
    return DIR_MODELLI / "llm_act" / f"risultati_seed{seed}.json"

def percorso_modello_llm_rew(seed: int) -> Path:
    """Checkpoint AG-LLM-REW (PPO con reward LLM)."""
    return DIR_MODELLI / "llm_rew" / f"llm_rew_seed{seed}"

# ---------------------------------------------------------------------------
# Helper: crea SokobanEnv dalla configurazione di una fase
# ---------------------------------------------------------------------------

def crea_env_da_fase(fase: dict, dir_dati: str, seme: int, split: str = "train"):
    """Crea SokobanEnv configurato per la fase specificata.

    Parametri:
        fase:     dizionario da FASI_CURRICULUM_V9
        dir_dati: percorso a data/boxoban/ (str o None)
        seme:     seed per riproducibilita'
        split:    "train" | "valid" | "test" (solo per dataset boxoban)

    Ritorna:
        SokobanEnv non avvolto (senza Monitor/AggiuntaCanale)
    """
    import sys
    sys.path.insert(0, str(RADICE_PROGETTO))
    from sokoban_env import SokobanEnv

    griglia = GRIGLIA_SIZE
    n_casse = fase["n_casse"]
    max_step = fase["max_step"]
    dataset = fase["dataset"]

    if dataset == "generato":
        # usa_generatore=True: forza GeneratoreLivelli anche su griglia 10x10.
        # Senza questo flag, SokobanEnv userebbe CaricatoreLivelli con i 3 livelli
        # built-in di fallback invece dei livelli procedurali del curriculum.
        return SokobanEnv(
            griglia_size=griglia,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            max_step=max_step,
            seme=seme,
            usa_generatore=True,
        )
    elif dataset == "boxoban_medium":
        return SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="medium",
            split=split,
            griglia_size=griglia,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            max_step=max_step,
            seme=seme,
        )
    elif dataset == "boxoban_unfiltered":
        return SokobanEnv(
            directory_livelli=dir_dati,
            difficolta="unfiltered",
            split=split,
            griglia_size=griglia,
            n_casse=n_casse,
            scala_manhattan=SCALA_MANHATTAN,
            max_step=max_step,
            seme=seme,
        )
    else:
        raise ValueError("Dataset sconosciuto: " + dataset)
