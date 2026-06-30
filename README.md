# IA-Sokoban: Reinforcement Learning e LLM applicati a Sokoban

> Progetto universitario per il corso di Intelligenza Artificiale
> Università degli Studi di Salerno, A.A. 2025/2026

---

## Indice

- [Panoramica](#panoramica)
- [Il problema: Sokoban](#il-problema-sokoban)
- [Agenti implementati](#agenti-implementati)
- [Risultati](#risultati)
- [Struttura del repository](#struttura-del-repository)
- [Installazione](#installazione)
- [Utilizzo](#utilizzo)
- [Dataset](#dataset)
- [Stack tecnologico](#stack-tecnologico)
- [Test](#test)

---

## Panoramica

Il progetto confronta cinque approcci per risolvere Sokoban su griglie 10×10 con 4 casse e 4 target:

| Agente | Paradigma | Algoritmo |
|---|---|---|
| AG-PPO | RL puro | RecurrentPPO + LSTM (sb3-contrib) |
| AG-DQN | RL puro | DQN (Stable Baselines 3) |
| AG-LLM-ACT | LLM come policy diretta | qwen3:14b — nessun training RL, inference pura |
| AG-LLM-GUIDE | RL guidato da LLM | DQfD: il LLM raccoglie demo, il DQN impara da esse |
| AG-LLM-REW | RL con reward aumentata da LLM | RecurrentPPO con reward shaping via LLM |

I quattro agenti RL usano curriculum learning progressivo, partendo da livelli generati con 1 cassa fino al dataset Boxoban reale con 4 casse. Il LLM usato è qwen3:14b-q4_K_M via Ollama locale. Per AG-PPO, AG-DQN, AG-LLM-GUIDE e AG-LLM-REW: a inference time non serve più il LLM, agisce solo il modello RL. Per AG-LLM-ACT: il LLM è la policy a inference time, senza alcun training RL.

---

## Il problema: Sokoban

Sokoban è un puzzle giapponese PSPACE-completo (Culberson, 1997). Il giocatore si muove su una griglia e spinge casse sui target. Non può tirarle. Una cassa bloccata in un angolo rende il livello irrisolvibile senza possibilità di recupero.

```
#########
# . .   #
#  $  $ #
#   @   #
#########
```

La configurazione usata è quella standard del dataset [DeepMind Boxoban](https://github.com/google-deepmind/boxoban-levels): griglia 10×10, 4 casse (`$`), 4 target (`.`). Lo spazio delle configurazioni di griglia è enorme (fino a ~8¹⁰⁰ assegnazioni di celle), mentre quello degli stati effettivamente raggiungibili con 4 casse supera 10⁸. Le azioni sono 4: su, giù, sinistra, destra.

---

## Agenti implementati

### AG-PPO

L'agente principale. Usa `RecurrentPPO` con `CnnLstmPolicy`: una CNN custom a 3 layer estrae 256 feature dalla griglia, poi una LSTM con 256 unità mantiene memoria tra i passi dell'episodio. Il training segue un curriculum a 6 fasi (C0→C5), trasferendo i pesi da una fase alla successiva senza ripartire da zero.

```bash
python src/sistema_10x10/train_ppo.py --seed 42
```

### AG-DQN

L'agente di confronto. Usa `DQN` con la stessa CNN di AG-PPO, experience replay buffer e target network. Il curriculum copre le stesse 6 fasi di AG-PPO (C0→C5), con budget di step ridotti. L'esplorazione è epsilon-greedy con decadimento sui primi 15% dei timestep.

```bash
python src/sistema_10x10/train_dqn.py --seed 42
```

### AG-LLM-GUIDE

Il LLM (qwen3:14b) gioca durante il training e raccoglie dimostrazioni. Il DQN impara da esse seguendo il paradigma DQfD (Hester et al., 2018):

1. Il LLM gioca N episodi per fase e salva le transizioni `(s, a, r, s', done)`
2. Le transizioni vengono caricate nel replay buffer DQN prima di avviare il training
3. Il DQN si addestra sia sulle demo che sulla propria esperienza generata online

A inference time il LLM non serve più. Solo il DQN agisce.

```bash
python src/sistema_10x10/train_llm_guide.py --seed 42
```

### AG-LLM-REW

Il LLM valuta ogni spinta di cassa confrontando la griglia prima e dopo la mossa e assegna un punteggio da 0 a 3. Il punteggio viene aggiunto alla reward standard:

```
r_aug = r_env + λ · score_norm    (λ = 0.3)
```

Il wrapper `RicompensaLLM` chiama il LLM solo quando una cassa viene effettivamente spostata, circa il 5% degli step. Questo riduce le chiamate di circa 20 volte rispetto a chiamare il LLM ad ogni step. Una cache `(stato_pre, azione) → score` evita di ripetere valutazioni già calcolate.

```bash
python src/sistema_10x10/train_ppo_llm_rew.py --seed 42
```

---

## Risultati

Valutazione su 100 episodi, seed 42. Per ogni agente viene usato il checkpoint migliore selezionato da `EvalCallback` durante il training.

**Sistema 10×10 (Boxoban):**

| Agente | C0 — 1 cassa | C1 — 2 casse | C5 — Boxoban | Reward C5 |
|---|---|---|---|---|
| AG-PPO | 3% | 0% | 0% | -1.186 |
| AG-DQN | 0% | 0% | 0% | -1.531 |
| AG-LLM-ACT | 2% | 0% | 0% | -1.375 |
| AG-LLM-GUIDE | 1% | 0% | 0% | -1.473 |
| AG-LLM-REW | 3% | 0% | 0% | -1.368 |

**Sistema 7×7 (curriculum semplificato), fase C0:**

| Agente | C0 — 1 cassa | C1 — 2 casse |
|---|---|---|
| **AG-PPO** | **16%** | **1%** |
| AG-DQN | 11% | 0% |
| AG-LLM-ACT | 9% (zero-shot) | 0% |
| AG-LLM-REW | 9% | 0% |
| AG-LLM-GUIDE | 6% | 0% |

Nessun agente risolve Boxoban 10×10 in modo consistente. Il problema è PSPACE-completo e i livelli del dataset sono pensati per giocatori umani esperti: 0% su Boxoban non sorprende. Quello che conta è il confronto tra agenti sulle stesse condizioni.

Sul sistema 7×7, dove il curriculum produce risoluzioni concrete, **AG-PPO (RecurrentPPO + LSTM) è il migliore con il 16%** su una cassa ed è l'unico a superare lo 0% su due casse (1%): la memoria ricorrente aiuta. Notevole anche AG-LLM-ACT, che senza alcun training raggiunge il 9% zero-shot. Sul 10×10 nessuno supera la prima fase: il gap tra livelli generati e Boxoban reali è troppo ampio per il budget disponibile (9.1M step, circa 220 volte meno del benchmark DeepMind da 2B step).

---

## Struttura del repository

```
IA-Sokoban/
│
├── src/                            # Codice sorgente
│   ├── core/                       # Componenti condivisi tra 10×10 e 7×7
│   │   ├── ambiente/               # Ambiente Sokoban (Gymnasium)
│   │   │   ├── sokoban_gym.py       # SokobanEnv: reset, step, render
│   │   │   ├── game_logic.py        # Fisica del gioco: mosse e vittoria
│   │   │   ├── reward.py            # Reward shaping (Manhattan + Ungherese)
│   │   │   ├── level_generator.py   # Generatore livelli procedurali
│   │   │   ├── level_loader.py      # Loader dataset Boxoban
│   │   │   ├── sokoban_cnn.py        # SokobanCNN: feature extractor custom
│   │   │   ├── cnn_wrapper.py        # AggiuntaCanale: (H,W) -> (1,H,W)
│   │   │   └── renderer.py           # Rendering Pygame
│   │   ├── agenti_llm/              # I tre agenti basati su LLM
│   │   │   ├── llm_act_agent.py      # AG-LLM-ACT: LLM come policy diretta
│   │   │   ├── llm_guide_agent.py    # AG-LLM-GUIDE: raccolta demo (LfD)
│   │   │   └── llm_reward_agent.py   # AG-LLM-REW: wrapper reward LLM
│   │   └── llm/                     # Integrazione LLM
│   │       ├── llm_client.py         # ClienteLLM con http.client keep-alive
│   │       └── sokoban_prompt.py     # Prompt, parser, griglia ASCII
│   │
│   ├── sistema_10x10/              # Sistema principale (Boxoban 10×10)
│   │   ├── config.py                # Iperparametri, curriculum, path
│   │   ├── train_ppo.py  train_dqn.py  train_llm_act.py
│   │   ├── train_llm_guide.py  train_ppo_llm_rew.py
│   │   └── evaluate_all.py          # Valutazione comparativa 10×10
│   │
│   └── sistema_7x7/                # Sistema semplificato (griglia 7×7 nativa)
│       ├── config_7x7.py  sokoban_gym_7x7.py
│       ├── train_ppo_7x7.py  ...  train_llm_rew_7x7.py
│       └── evaluate_7x7.py          # Valutazione comparativa 7×7
│
├── tests/                          # Test unitari (pytest): 64 test
├── conftest.py                     # Mette src/ sul sys.path per i test
│
├── dataset/                        # download_boxoban.py versionato; dati boxoban/ (gitignored)
├── artifacts/                      # Generati dal training (gitignored)
│   ├── models/{10x10,7x7}/         # Checkpoint, per sistema
│   └── logs/{10x10,7x7}/           # Log TensorBoard, per sistema
├── results/seed42/                 # JSON dei risultati (versionati)
│
├── docs/                           # Relazione, presentazione e traccia (PDF)
│
├── requirements.txt
└── README.md
```

---

## Installazione

**Prerequisiti:**
- Python 3.13
- [Ollama](https://ollama.com/) installato e in esecuzione
- GPU con CUDA consigliata (testato su RTX 4070, CUDA 12.6)

```bash
git clone https://github.com/AntonioDG30/IA-Sokoban
cd IA-Sokoban

python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**Setup LLM:**

```bash
# circa 9 GB di download
ollama pull qwen3:14b-q4_K_M
ollama serve
```

**Dataset Boxoban:**

```bash
python dataset/download_boxoban.py
```

Lo script clona [google-deepmind/boxoban-levels](https://github.com/google-deepmind/boxoban-levels) in `dataset/boxoban/` (fallback su ZIP se git non è disponibile). I dati non sono versionati: nel repository è incluso solo lo script di download.

---

## Utilizzo

### Training

```bash
# AG-PPO (~5h su RTX 4070)
python src/sistema_10x10/train_ppo.py --seed 42

# AG-DQN (~3h)
python src/sistema_10x10/train_dqn.py --seed 42

# AG-LLM-GUIDE (richiede Ollama attivo, ~4h)
python src/sistema_10x10/train_llm_guide.py --seed 42

# AG-LLM-REW (richiede Ollama attivo, ~10h)
python src/sistema_10x10/train_ppo_llm_rew.py --seed 42
```

Tutti gli script accettano `--seed`; quelli del sistema 10×10 anche `--dir-dati` (path al dataset Boxoban):

```bash
python src/sistema_10x10/train_ppo.py --seed 123 --dir-dati dataset/boxoban
```

### Valutazione comparativa

```bash
python src/sistema_10x10/evaluate_all.py --seed 42
# salva i risultati in results/seed42/risultati_comparativi_seed42.json
```

### TensorBoard

```bash
tensorboard --logdir artifacts/logs/10x10/
```

---

## Dataset

Il progetto usa il dataset Boxoban di DeepMind (Guez et al., ICML 2019), licenza Apache 2.0.

| Split | Livelli | Uso |
|---|---|---|
| `unfiltered/train` | ~900.000 | C3, C5 — training |
| `unfiltered/test` | 1.000 | C5 — valutazione |
| `medium/train` | ~500.000 | C4 — training |
| `medium/valid` | 1.000 | C4 — valutazione |
| `hard` | 3.332 | Non usato |

I livelli delle fasi C0-C2 vengono generati a runtime da `level_generator.py` e non richiedono download.

---

## Stack tecnologico

| Componente | Libreria | Versione |
|---|---|---|
| Ambiente RL | Gymnasium | 1.2.3 |
| Algoritmi RL | Stable Baselines 3 | 2.7.1 |
| RecurrentPPO | sb3-contrib | 2.7.1 |
| Deep learning | PyTorch | 2.10.0+cu126 |
| Rendering | Pygame | 2.6.1 |
| Numerics | NumPy, SciPy | 2.3.5, 1.17.1 |
| LLM locale | Ollama + qwen3:14b-q4_K_M | |
| LLM client | http.client (stdlib) | |
| Logging | TensorBoard | 2.20.0 |
| Test | pytest | 9.0.2 |

Il client LLM usa `http.client` con connessioni keep-alive invece dell'SDK OpenAI. La latenza per chiamata scende da ~2.3 s (nuova connessione TCP a ogni chiamata) a ~0.14–0.21 s a regime, rilevante quando se ne fanno centinaia di migliaia durante il training.

---

## Test

```bash
pytest tests/ -v

# per modulo
pytest tests/test_env.py tests/test_game_logic.py -v
pytest tests/test_llm_integration.py -v
```

64 test totali (36 sull'ambiente, 28 sull'integrazione LLM). Senza un server Ollama attivo ne passano 61 e 3 vengono saltati automaticamente (quelli che istanziano il client LLM reale).

---

## Riferimenti principali

- Culberson (1997) — *Sokoban is PSPACE-complete*
- Guez et al. (2019) — *An Investigation of Model-Free Planning* — [Boxoban dataset](https://github.com/google-deepmind/boxoban-levels)
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Hester et al. (2018) — *Deep Q-learning from Demonstrations*
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- Vaswani et al. (2017) — *Attention Is All You Need*
- Bengio et al. (2009) — *Curriculum Learning*

Bibliografia completa nella relazione: [`docs/Relazione_Progetto_IA_AntonioDiGiorgio.pdf`](docs/Relazione_Progetto_IA_AntonioDiGiorgio.pdf).

---

## Licenza

Progetto accademico, Università degli Studi di Salerno.
Il dataset Boxoban è distribuito sotto licenza [Apache 2.0](https://github.com/google-deepmind/boxoban-levels/blob/master/LICENSE).
