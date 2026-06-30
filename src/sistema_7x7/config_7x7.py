# Configurazione del curriculum semplificato 7x7 — MODULO AUTONOMO.
#
# Scopo: mostrare la capacità degli agenti su griglie semplici a confronto con Boxoban 10x10.
# Observation space NATIVO (7,7), senza alcun offset di padding. Non modifica né dipende da
# src/sistema_10x10/config.py (il progetto 10x10). Separazione netta:
#   Env:     src/sistema_7x7/sokoban_gym_7x7.py
#   Config:  src/sistema_7x7/config_7x7.py  (questo file)
#   Modelli: artifacts/models/7x7/{ppo,dqn,llm_act,llm_guide,llm_rew}/
#   Log:     artifacts/logs/7x7/{ppo,dqn,...}/
#   Script:  src/sistema_7x7/train_*.py
#
# Curriculum 7x7 (3 fasi, solo livelli generati proceduralmente):
#   C0: 1 cassa generata  (400K step PPO base, soglia 50% solve rate)
#   C1: 2 casse generate  (700K step PPO base, soglia 15%)
#   C2: 3 casse generate  (1M step PPO base,   nessuna soglia)
#   Totale base: 2.1M PPO / 1.48M DQN; worst case (3x per fase): 6.3M PPO / 4.44M DQN
#
# Identico agli agenti 10x10 (FASI_CURRICULUM_V9) tranne: griglia 7x7 nativa (nessun padding),
# sole 3 fasi generate (nessun Boxoban) e budget scalato a ~67% per le prime 3 fasi generative.

from pathlib import Path

# PERCORSI — SEPARATI DAL PROGETTO PRINCIPALE

RADICE_PROGETTO   = Path(__file__).resolve().parent.parent.parent
DIR_MODELLI_7x7   = RADICE_PROGETTO / "artifacts" / "models" / "7x7"
DIR_LOG_7x7       = RADICE_PROGETTO / "artifacts" / "logs" / "7x7"
DIR_RISULTATI_7x7 = RADICE_PROGETTO / "results" / "seed42"

# CURRICULUM 7x7 — 3 FASI, SOLO LIVELLI GENERATI PROCEDURALMENTE
#
# Budget calibrato a ~67% dell'originale per le prime 3 fasi generative C0/C1/C2 (la griglia
# 7x7 ha uno spazio degli stati più piccolo e converge prima).
#   timestep_ppo: step per AG-PPO e AG-LLM-REW (N_ENVS_PPO_7x7=8 in VecEnv)
#   timestep_dqn: step per AG-DQN e AG-LLM-GUIDE (1 solo ambiente)
#   ent_coef: solo per PPO (identico all'originale)

FASI_CURRICULUM_7x7 = [
    # C0: 1 cassa generata — griglia 7x7 nativa, max_step ridotto
    {
        "nome":         "C0-1box-7x7",
        "n_casse":      1,
        "timestep_ppo": 400_000,
        "timestep_dqn": 280_000,
        "max_step":     80,
        "ent_coef":     0.01,
    },
    # C1: 2 casse generate — complessità maggiore
    {
        "nome":         "C1-2box-7x7",
        "n_casse":      2,
        "timestep_ppo": 700_000,
        "timestep_dqn": 500_000,
        "max_step":     120,
        "ent_coef":     0.01,
    },
    # C2: 3 casse generate — benchmark finale del sistema 7x7
    {
        "nome":         "C2-3box-7x7",
        "n_casse":      3,
        "timestep_ppo": 1_000_000,
        "timestep_dqn": 700_000,
        "max_step":     180,
        "ent_coef":     0.02,
    },
]

# Budget totale: 2,1M step per AG-PPO e 1,48M per AG-DQN (somma dei budget per fase qui sopra).

# PARAMETRI DEL CURRICULUM ADATTIVO — IDENTICI AGLI AGENTI 10x10

# Ambienti paralleli per VecEnv (come l'originale N_ENVS_PPO=8). DQN usa 1 solo ambiente,
# cablato negli script di training (niente VecEnv nativo).
N_ENVS_PPO_7x7 = 8   # PPO beneficia di molti ambienti paralleli (VecEnv)

# Ripetizioni aggiuntive massime per fase, oltre alla prima (come l'originale: => max 3x budget)
MAX_RIPETIZIONI_FASE_7x7 = 2   # => budget massimo 3x per fase

# Soglie di solve rate (%) per avanzare di fase, calibrate sulla reale difficoltà del 7x7
# (più semplice del 10x10)
SOGLIE_7x7 = {
    "C0-1box-7x7": 50.0,   # 1 cassa su 7x7: facilmente raggiungibile
    "C1-2box-7x7": 15.0,   # 2 casse: obiettivo realistico
    "C2-3box-7x7":  0.0,   # ultima fase: nessuna soglia
}

# Reward minima per dire "episodio risolto" — identica all'originale (9.0). Con il reward
# shaping attivo è sicura: un episodio risolto vale sempre >= 9.5, uno non risolto < 9.0.
SOGLIA_RISOLTO_7x7 = 9.0

N_EPISODI_VALUTAZIONE_7x7 = 100   # come l'originale (N_EPISODI_VALUTAZIONE=100)
N_EPISODI_LLM_ACT_7x7     = 100   # come l'originale (N_EPISODI_LLM_ACT=100)

# Demo del LLM per fase nella raccolta LfD di AG-LLM-GUIDE (come l'originale N_DEMO_LLM_FASE=30)
N_DEMO_LLM_7x7 = 30

# IPERPARAMETRI PPO (AG-PPO e AG-LLM-REW) — IDENTICI ALL'ORIGINALE

CONFIG_PPO_7x7 = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      64,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.01,   # sovrascritta per fase nello script di training
    "verbose":         1,
    "device":          "auto",
}

# IPERPARAMETRI DQN (AG-DQN e AG-LLM-GUIDE) — IDENTICI ALL'ORIGINALE

CONFIG_DQN_7x7 = {
    "learning_rate":            1e-4,
    "buffer_size":              100_000,   # come l'originale
    "learning_starts":          1_000,
    "batch_size":               32,
    "gamma":                    0.99,
    "exploration_fraction":     0.15,
    "exploration_final_eps":    0.05,
    "target_update_interval":   1_000,
    "verbose":                  1,
    "device":                   "auto",
}

# CONFIGURAZIONE LLM

LAMBDA_LLM_7x7  = 0.3    # come l'originale (LAMBDA_LLM = 0.3)
PROVIDER_DEFAULT = "ollama"

# PERCORSI DEI MODELLI

def percorso_ppo_7x7(seed: int) -> Path:
    """Path del modello PPO finale 7x7 per il seed dato (senza .zip)."""
    return DIR_MODELLI_7x7 / "ppo" / f"ppo_7x7_seed{seed}"

def percorso_dqn_7x7(seed: int) -> Path:
    """Path del modello DQN finale 7x7 per il seed dato (senza .zip)."""
    return DIR_MODELLI_7x7 / "dqn" / f"dqn_7x7_seed{seed}"

def percorso_llm_act_7x7(seed: int) -> Path:
    """Path del JSON dei risultati AG-LLM-ACT 7x7 per il seed dato."""
    return DIR_MODELLI_7x7 / "llm_act" / f"risultati_7x7_seed{seed}.json"

def percorso_llm_guide_7x7(seed: int) -> Path:
    """Path del modello LLM-GUIDE finale 7x7 per il seed dato (senza .zip)."""
    return DIR_MODELLI_7x7 / "llm_guide" / f"llm_guide_7x7_seed{seed}"

def percorso_llm_rew_7x7(seed: int) -> Path:
    """Path del modello LLM-REW finale 7x7 per il seed dato (senza .zip)."""
    return DIR_MODELLI_7x7 / "llm_rew" / f"llm_rew_7x7_seed{seed}"

# FACTORY DELL'AMBIENTE 7x7

def crea_env_7x7(fase: dict, seme: int):
    """
    Crea il SokobanEnv7x7 per la fase data.

    È l'equivalente di crea_env_da_fase() del progetto 10x10, ma per il 7x7: observation
    space nativo Box(0,6,(7,7),float32) senza padding, stesso reward shaping
    (scala_manhattan=0.3, scala_player_box=0.1) e solo livelli generati proceduralmente.
    """
    from sistema_7x7.sokoban_gym_7x7 import SokobanEnv7x7
    return SokobanEnv7x7(
        n_casse=fase["n_casse"],
        max_step=fase["max_step"],
        seme=seme,
    )
