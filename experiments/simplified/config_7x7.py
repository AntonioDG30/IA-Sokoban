"""Configurazione curriculum semplificato 7x7 -- MODULO AUTONOMO.

Scopo: dimostrare la capacita' degli agenti su griglie semplici rispetto
a Boxoban 10x10. Observation space NATIVO (7,7) -- nessun padding offset.

Non modifica ne' dipende da experiments/config.py (progetto 10x10 Boxoban).
Separazione netta:
    Env:      experiments/simplified/sokoban_gym_7x7.py
    Config:   experiments/simplified/config_7x7.py  (questo file)
    Modelli:  models_7x7/{ppo,dqn,llm_act,llm_guide,llm_rew}/
    Log:      logs_7x7/{ppo,dqn,...}/
    Script:   experiments/simplified/train_*.py

Curriculum v7x7 (3 fasi, solo livelli generati proceduralmente):
    C0: 1-box generato  (400K step PPO base, soglia 50% solve rate)
    C1: 2-box generato  (700K step PPO base, soglia 15%)
    C2: 3-box generato  (1M step PPO base,   nessuna soglia)
    Totale base: 2.1M PPO / 1.48M DQN
    Totale worst case (3x per fase): 6.3M PPO / 4.44M DQN

Identico agli agenti 10x10 (FASI_CURRICULUM_V9) eccetto:
    - Griglia 7x7 nativa (nessun padding)
    - Solo 3 fasi generate (nessun dataset Boxoban)
    - Budget scalato (~67% dell'originale per le prime 3 fasi generative)
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Percorsi -- SEPARATI dal progetto principale
# ---------------------------------------------------------------------------

RADICE_PROGETTO   = Path(__file__).resolve().parent.parent.parent
DIR_MODELLI_7x7   = RADICE_PROGETTO / "models_7x7"
DIR_LOG_7x7       = RADICE_PROGETTO / "logs_7x7"
DIR_RISULTATI_7x7 = RADICE_PROGETTO / "docs" / "report"

# ---------------------------------------------------------------------------
# Curriculum 7x7 -- 3 fasi, solo livelli generati proceduralmente
# ---------------------------------------------------------------------------
#
# Budget calibrato su ~67% dell'originale per le prime 3 fasi generative C0/C1/C2
# (griglia 7x7 ha spazio degli stati piu' piccolo, converge prima).
#
# timestep_ppo: step per AG-PPO e AG-LLM-REW (N_ENVS_PPO_7x7=8 VecEnv)
# timestep_dqn: step per AG-DQN e AG-LLM-GUIDE (N_ENVS_DQN_7x7=1)
# ent_coef: solo per PPO (identico all'originale)

FASI_CURRICULUM_7x7 = [
    # C0: 1 cassa generata -- griglia 7x7 nativa, max_step ridotto.
    {
        "nome":         "C0-1box-7x7",
        "n_casse":      1,
        "timestep_ppo": 400_000,
        "timestep_dqn": 280_000,
        "max_step":     80,
        "ent_coef":     0.01,
    },
    # C1: 2 casse generate -- complessita' maggiore.
    {
        "nome":         "C1-2box-7x7",
        "n_casse":      2,
        "timestep_ppo": 700_000,
        "timestep_dqn": 500_000,
        "max_step":     120,
        "ent_coef":     0.01,
    },
    # C2: 3 casse generate -- benchmark finale 7x7.
    {
        "nome":         "C2-3box-7x7",
        "n_casse":      3,
        "timestep_ppo": 1_000_000,
        "timestep_dqn": 700_000,
        "max_step":     180,
        "ent_coef":     0.02,
    },
]

TIMESTEPS_TOTALI_PPO_7x7 = sum(f["timestep_ppo"] for f in FASI_CURRICULUM_7x7)  # 2_100_000
TIMESTEPS_TOTALI_DQN_7x7 = sum(f["timestep_dqn"] for f in FASI_CURRICULUM_7x7)  # 1_480_000

# ---------------------------------------------------------------------------
# Parametri curriculum adattivo -- IDENTICI agli agenti 10x10
# ---------------------------------------------------------------------------

# Ambienti paralleli per VecEnv (identico all'originale N_ENVS_PPO=8)
N_ENVS_PPO_7x7 = 8   # PPO beneficia di molti env paralleli (VecEnv)
N_ENVS_DQN_7x7 = 1   # DQN usa env singolo (no VecEnv nativo)

# Massimo numero di ripetizioni aggiuntive per fase (oltre alla prima).
# Identico all'originale: => max 3x budget per fase.
MAX_RIPETIZIONI_FASE_7x7 = 2   # => max 3x budget per fase

# Soglie solve rate (%) per avanzare alla fase successiva.
# Calibrate per la difficolta' reale di 7x7 (piu' semplice del 10x10).
SOGLIE_7x7 = {
    "C0-1box-7x7": 50.0,   # 1 cassa 7x7: facilmente raggiungibile
    "C1-2box-7x7": 15.0,   # 2 casse: obiettivo realistico
    "C2-3box-7x7":  0.0,   # ultima fase: nessuna soglia
}

# Soglia reward per "episodio risolto" -- identica all'originale (_SOGLIA_RISOLTO=9.0).
# Con reward shaping attivo, soglia 9.0 sicura:
#   min reward RISOLTO  = 10.0 + n_casse - step_penalty >= 9.5
#   max reward NON RISOLTO con shaping < 9.0
SOGLIA_RISOLTO_7x7 = 9.0

N_EPISODI_VALUTAZIONE_7x7 = 100   # identico all'originale (N_EPISODI_VALUTAZIONE=100)
N_EPISODI_LLM_ACT_7x7     = 100   # identico all'originale (N_EPISODI_LLM_ACT=100)

# Demo LLM per fase (raccolta LfD per AG-LLM-GUIDE).
# Identico all'originale (N_DEMO_LLM_FASE = 30 in train_llm_guide.py).
N_DEMO_LLM_7x7 = 30

# ---------------------------------------------------------------------------
# Iperparametri PPO (AG-PPO e AG-LLM-REW) -- IDENTICI all'originale
# ---------------------------------------------------------------------------

CONFIG_PPO_7x7 = {
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
# Iperparametri DQN (AG-DQN e AG-LLM-GUIDE) -- IDENTICI all'originale
# ---------------------------------------------------------------------------

CONFIG_DQN_7x7 = {
    "learning_rate":            1e-4,
    "buffer_size":              100_000,   # identico all'originale
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

LAMBDA_LLM_7x7  = 0.3    # identico all'originale (LAMBDA_LLM = 0.3)
PROVIDER_DEFAULT = "ollama"

# ---------------------------------------------------------------------------
# Percorsi modelli
# ---------------------------------------------------------------------------

def percorso_ppo_7x7(seed: int) -> Path:
    """Restituisce il path del modello PPO finale per il seed dato (senza .zip).

    Parametri:
        seed: seed usato durante il training
    """
    return DIR_MODELLI_7x7 / "ppo" / f"ppo_7x7_seed{seed}"

def percorso_dqn_7x7(seed: int) -> Path:
    """Restituisce il path del modello DQN finale per il seed dato (senza .zip).

    Parametri:
        seed: seed usato durante il training
    """
    return DIR_MODELLI_7x7 / "dqn" / f"dqn_7x7_seed{seed}"

def percorso_llm_act_7x7(seed: int) -> Path:
    """Restituisce il path del JSON risultati AG-LLM-ACT per il seed dato.

    Parametri:
        seed: seed usato durante la valutazione
    """
    return DIR_MODELLI_7x7 / "llm_act" / f"risultati_7x7_seed{seed}.json"

def percorso_llm_guide_7x7(seed: int) -> Path:
    """Restituisce il path del modello LLM-GUIDE finale per il seed dato (senza .zip).

    Parametri:
        seed: seed usato durante il training
    """
    return DIR_MODELLI_7x7 / "llm_guide" / f"llm_guide_7x7_seed{seed}"

def percorso_llm_rew_7x7(seed: int) -> Path:
    """Restituisce il path del modello LLM-REW finale per il seed dato (senza .zip).

    Parametri:
        seed: seed usato durante il training
    """
    return DIR_MODELLI_7x7 / "llm_rew" / f"llm_rew_7x7_seed{seed}"

# ---------------------------------------------------------------------------
# Factory ambiente 7x7
# ---------------------------------------------------------------------------

def crea_env_7x7(fase: dict, seme: int):
    """Crea SokobanEnv7x7 per la fase specificata.

    Equivalente di crea_env_da_fase() del progetto 10x10, ma per 7x7:
        - Observation space: Box(0,6,(7,7),float32) -- NATIVO, nessun padding.
        - Reward shaping: scala_manhattan=0.3, scala_player_box=0.1 (identico).
        - Solo livelli generati proceduralmente (nessun dataset Boxoban).

    Parametri:
        fase: dizionario da FASI_CURRICULUM_7x7
        seme: seed per riproducibilita'
    """
    from experiments.simplified.sokoban_gym_7x7 import SokobanEnv7x7
    return SokobanEnv7x7(
        n_casse=fase["n_casse"],
        max_step=fase["max_step"],
        seme=seme,
    )
