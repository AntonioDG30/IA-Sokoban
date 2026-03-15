"""Ambiente Sokoban compatibile con Gymnasium.

Implementa l'API completa Gymnasium:
    reset()  -> (obs, info)
    step()   -> (obs, reward, terminated, truncated, info)
    render() -> None oppure np.ndarray
    close()  -> None

Spazio di osservazione (fisso 10x10 per compatibilita' SB3):
    Box(low=0, high=7, shape=(10, 10), dtype=float32)
    Per griglie piu' piccole (curriculum learning) viene applicato padding con
    valore 7 (PADDING), distinto dai muri reali (0) e da tutte le celle di gioco (1-6).

Spazio delle azioni:
    Discrete(4)  -- 0=su, 1=giu, 2=sinistra, 3=destra

Parametri costruttore:
    directory_livelli  percorso alla dir con i dati Boxoban (solo se 10x10).
    difficolta         'unfiltered', 'medium', 'hard'.
    split              'train', 'valid', 'test'.
    render_mode        None, 'human', 'rgb_array'.
    max_step           numero massimo di step per episodio (default 120).
    seme               seme per la randomizzazione dei livelli.
    griglia_size       (righe, colonne) della griglia. Default (10, 10).
                       Se != (10,10) usa GeneratoreLivelli (curriculum).
    n_casse            numero di casse (solo per griglia_size != 10x10).
    scala_manhattan    fattore scala reward shaping Manhattan (0.0 = off).
    scala_player_box   fattore scala reward shaping giocatore->cassa (0.0 = off). [v10]
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

from sokoban_env.game_logic import (
    applica_mossa,
    controlla_vittoria,
    conta_casse_su_target,
    GIOCATORE, GIOCATORE_SU_TARGET,
    CASSA, CASSA_SU_TARGET,
    NOMI_AZIONI,
)
from sokoban_env.reward import calcola_reward
from sokoban_env.level_loader import CaricatoreLivelli
from sokoban_env.level_generator import GeneratoreLivelli, padding_a_10x10
from sokoban_env.renderer import RendererSokoban

# Numero massimo di step per episodio (coerente con gym-sokoban e Boxoban)
MAX_STEP_PER_EPISODIO = 120


class SokobanEnv(gymnasium.Env):
    """Ambiente Sokoban custom compatibile con Gymnasium.

    Carica livelli dal dataset DeepMind Boxoban (o dai livelli integrati
    se i dati non sono disponibili) e implementa la logica di gioco completa.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        directory_livelli: Optional[str] = None,
        difficolta: str = "unfiltered",
        split: str = "train",
        render_mode: Optional[str] = None,
        max_step: int = MAX_STEP_PER_EPISODIO,
        seme: Optional[int] = None,
        griglia_size: Tuple[int, int] = (10, 10),
        n_casse: int = 4,
        scala_manhattan: float = 0.0,
        scala_player_box: float = 0.0,
        usa_generatore: bool = False,
    ):
        """Crea l'ambiente Sokoban.

        Parametri:
            directory_livelli: path a data/boxoban/ (solo dataset Boxoban).
            difficolta:        'unfiltered' | 'medium' | 'hard'.
            split:             'train' | 'valid' | 'test'.
            render_mode:       None | 'human' | 'rgb_array'.
            max_step:          step massimi per episodio.
            seme:              seed per riproducibilita'.
            griglia_size:      (righe, colonne). Default (10, 10).
            n_casse:           numero casse (solo curriculum generato).
            scala_manhattan:   fattore reward shaping Manhattan. 0.0 = off.
            scala_player_box:  fattore reward shaping giocatore->cassa. 0.0 = off. [v10]
            usa_generatore:    True = usa GeneratoreLivelli anche su griglia 10x10.
                               Necessario per fasi curriculum 'generato' con griglie fisse.
        """
        super().__init__()

        # Validazione render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode non valido: '{render_mode}'. "
                f"Opzioni: {self.metadata['render_modes']}"
            )

        self.render_mode = render_mode
        self.max_step = max_step
        self.griglia_size = griglia_size
        self.scala_manhattan = scala_manhattan
        self.scala_player_box = scala_player_box

        # Spazio di osservazione: sempre 10x10 float32 (con padding per griglie piu' piccole).
        # high=7.0 perche' il valore di padding e' 7 (distinto da MURO=0 e celle 1-6).
        # Fisso per compatibilita' SB3 — il modello non viene reinizializzato tra fasi curriculum.
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=7.0, shape=(10, 10), dtype=np.float32
        )

        # Spazio delle azioni: 4 direzioni discrete
        self.action_space = gymnasium.spaces.Discrete(4)

        # Sorgente livelli: GeneratoreLivelli per curriculum, CaricatoreLivelli per Boxoban.
        # usa_generatore=True forza il generatore anche su griglia 10x10 (fasi 'generato' v9).
        # Fallback automatico: qualsiasi griglia != 10x10 usa sempre il generatore.
        self._usa_generatore = usa_generatore or (griglia_size != (10, 10))
        if self._usa_generatore:
            self._generatore = GeneratoreLivelli(seme=seme)
            self._n_casse = n_casse
            self._caricatore = None
        else:
            self._generatore = None
            self._n_casse = n_casse
            self._caricatore = CaricatoreLivelli(
                directory_base=directory_livelli,
                difficolta=difficolta,
                split=split,
                seme=seme,
            )

        # Renderer (lazy — creato solo se necessario)
        self._renderer: Optional[RendererSokoban] = None

        # Stato interno
        self._griglia: Optional[np.ndarray] = None
        self._step_corrente: int = 0
        self._seme = seme

        # RNG per azioni random (usato da Gymnasium)
        self.np_random: np.random.Generator = np.random.default_rng(seme)

    # ------------------------------------------------------------------
    # API Gymnasium obbligatoria
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resetta l'ambiente caricando un nuovo livello casuale.

        Parametri:
            seed:    seme per la selezione del livello (opzionale).
            options: dizionario opzionale con chiave 'indice_livello'
                     per selezionare un livello specifico.

        Restituisce:
            (osservazione, info)
        """
        super().reset(seed=seed)

        # Selezione / generazione livello
        if self._usa_generatore:
            # Curriculum: genera livello procedurale per la griglia corrente
            self._griglia = self._generatore.genera(
                righe=self.griglia_size[0],
                colonne=self.griglia_size[1],
                n_casse=self._n_casse,
            )
        elif options is not None and "indice_livello" in options:
            indice = int(options["indice_livello"])
            self._griglia = self._caricatore.ottieni(indice)
        else:
            self._griglia = self._caricatore.casuale()

        self._step_corrente = 0

        info = self._costruisci_info(mossa_eseguita=False, cassa_spostata=False)
        return self._osservazione(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Esegue un'azione nell'ambiente.

        Parametri:
            action: intero in {0,1,2,3}.

        Restituisce:
            (osservazione, reward, terminated, truncated, info)
        """
        if self._griglia is None:
            raise RuntimeError(
                "L'ambiente non è stato inizializzato. Chiamare reset() prima di step()."
            )

        griglia_precedente = self._griglia.copy()

        # Rileva adiacenza giocatore-cassa PRIMA della mossa (usato da AG-LLM-REW)
        adiacente_cassa = self._giocatore_adiacente_cassa(griglia_precedente)

        # Applica la mossa
        nuova_griglia, mossa_eseguita, cassa_spostata = applica_mossa(
            self._griglia, int(action)
        )
        self._griglia = nuova_griglia
        self._step_corrente += 1

        # Condizioni di terminazione
        terminated = controlla_vittoria(self._griglia)
        truncated = (self._step_corrente >= self.max_step) and not terminated

        # Calcola reward (con shaping Manhattan v9 + player->box shaping v10)
        reward = calcola_reward(
            griglia_precedente, self._griglia, terminated,
            scala_manhattan=self.scala_manhattan,
            adiacente_cassa=adiacente_cassa,
            scala_player_box=self.scala_player_box,
        )

        # Rendering automatico in modalita' 'human'
        if self.render_mode == "human":
            self.render()

        info = self._costruisci_info(
            mossa_eseguita=mossa_eseguita,
            cassa_spostata=cassa_spostata,
            adiacente_cassa=adiacente_cassa,
        )
        return self._osservazione(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renderizza lo stato corrente dell'ambiente.

        Restituisce:
            Array NumPy (H, W, 3) in modalità 'rgb_array', None altrimenti.
        """
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = RendererSokoban(
                modalita=self.render_mode,
                fps=self.metadata["render_fps"],
            )

        if self._griglia is None:
            return None

        return self._renderer.renderizza(self._griglia)

    def close(self) -> None:
        """Chiude il renderer e libera le risorse."""
        if self._renderer is not None:
            self._renderer.chiudi()
            self._renderer = None

    # ------------------------------------------------------------------
    # Metodi di supporto
    # ------------------------------------------------------------------

    def _osservazione(self) -> np.ndarray:
        """Restituisce l'osservazione corrente come array float32 10x10.

        Per griglie piu' piccole di 10x10 applica zero-padding centrato.
        L'observation space e' sempre (10,10) per compatibilita' SB3.
        """
        if self._griglia is None:
            return np.zeros((10, 10), dtype=np.float32)
        if self._usa_generatore:
            return padding_a_10x10(self._griglia).astype(np.float32)
        return self._griglia.astype(np.float32)

    def _giocatore_adiacente_cassa(self, griglia: np.ndarray) -> bool:
        """Controlla se il giocatore e' adiacente a una cassa nella griglia fornita.

        Usato da AG-LLM-REW per decidere se chiamare il LLM (~20% degli step).
        Controlla le 4 celle ortogonali attorno al giocatore.

        Parametri:
            griglia: griglia da controllare (tipicamente griglia pre-mossa).

        Restituisce:
            True se almeno una cella adiacente contiene CASSA o CASSA_SU_TARGET.
        """
        posizioni = np.argwhere(
            (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
        )
        if len(posizioni) == 0:
            return False
        riga_g, col_g = posizioni[0]
        righe, colonne = griglia.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = riga_g + dr, col_g + dc
            if 0 <= r < righe and 0 <= c < colonne:
                if griglia[r, c] in (CASSA, CASSA_SU_TARGET):
                    return True
        return False

    def _costruisci_info(
        self,
        mossa_eseguita: bool,
        cassa_spostata: bool,
        adiacente_cassa: bool = False,
    ) -> Dict[str, Any]:
        """Costruisce il dizionario info restituito da reset() e step()."""
        if self._griglia is None:
            return {}
        return {
            "step_corrente":              self._step_corrente,
            "casse_su_target":            conta_casse_su_target(self._griglia),
            "mossa_eseguita":             mossa_eseguita,
            "cassa_spostata":             cassa_spostata,
            "giocatore_adiacente_cassa":  adiacente_cassa,
        }

    # ------------------------------------------------------------------
    # Proprietà di convenienza
    # ------------------------------------------------------------------

    @property
    def griglia(self) -> Optional[np.ndarray]:
        """Vista sola lettura della griglia corrente."""
        if self._griglia is None:
            return None
        return self._griglia.copy()

    def __repr__(self) -> str:
        return (
            f"SokobanEnv("
            f"griglia={self.griglia_size}, "
            f"step={self._step_corrente}/{self.max_step}, "
            f"manhattan={self.scala_manhattan}, "
            f"player_box={self.scala_player_box}, "
            f"render_mode={self.render_mode!r})"
        )
