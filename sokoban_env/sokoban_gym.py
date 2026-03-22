"""Ambiente Sokoban compatibile con l'API Gymnasium.

Implementa l'interfaccia standard Gymnasium:
    reset()  -> (obs, info)
    step()   -> (obs, reward, terminated, truncated, info)
    render() -> None oppure np.ndarray
    close()  -> None

Spazio di osservazione:
    Box(0.0, 7.0, shape=(10, 10), dtype=float32)
    Sempre 10x10 per compatibilita' con SB3. Per griglia piu' piccole
    il valore 7 (PADDING) riempie le celle di bordo artificiale.

Spazio delle azioni:
    Discrete(4)  -- 0=su, 1=giu, 2=sinistra, 3=destra
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

# Limite di step per episodio: coerente con gym-sokoban e il dataset Boxoban
MAX_STEP_PER_EPISODIO = 120


class SokobanEnv(gymnasium.Env):
    """Ambiente Sokoban custom compatibile con Gymnasium e Stable Baselines 3.

    Puo' caricare livelli dal dataset Boxoban (fasi C4-C5) oppure generarli
    proceduralmente (fasi C0-C3 del curriculum). Il rendering e' lazy:
    il RendererSokoban viene creato solo se render_mode non e' None.

    Parametri:
        directory_livelli: percorso a data/boxoban/. None = livelli builtin o generati.
        difficolta:        'unfiltered' | 'medium' | 'hard'.
        split:             'train' | 'valid' | 'test'.
        render_mode:       None | 'human' | 'rgb_array'.
        max_step:          step massimi per episodio.
        seme:              seed per la riproducibilita'.
        griglia_size:      (righe, colonne). Default (10, 10).
        n_casse:           numero di casse (rilevante per il generatore).
        scala_manhattan:   fattore di scala reward shaping Manhattan (0.0 = off).
        scala_player_box:  fattore di scala reward shaping giocatore->cassa (0.0 = off).
        usa_generatore:    True per forzare GeneratoreLivelli anche su griglia 10x10.
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
        super().__init__()

        # Verifica che il render_mode sia uno di quelli supportati
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode non valido: '{render_mode}'. "
                f"Opzioni: {self.metadata['render_modes']}"
            )

        self.render_mode     = render_mode
        self.max_step        = max_step
        self.griglia_size    = griglia_size
        self.scala_manhattan = scala_manhattan
        self.scala_player_box = scala_player_box

        # L'observation space e' sempre 10x10 float32, indipendentemente dalla griglia.
        # high=7.0 perche' il valore di padding e' 7 (distinto da MURO=0 e celle 1-6).
        # Questo mantiene il modello compatibile tra le diverse fasi del curriculum
        # senza dover reinizializzare i pesi della rete
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=7.0, shape=(10, 10), dtype=np.float32
        )

        # 4 azioni discrete: su, giu, sinistra, destra
        self.action_space = gymnasium.spaces.Discrete(4)

        # Scelta sorgente livelli: generatore procedurale o dataset Boxoban.
        # usa_generatore=True e' necessario per le fasi C0-C3 su griglia 10x10.
        # Qualsiasi griglia != 10x10 usa sempre il generatore
        self._usa_generatore = usa_generatore or (griglia_size != (10, 10))
        if self._usa_generatore:
            self._generatore = GeneratoreLivelli(seme=seme)
            self._n_casse    = n_casse
            self._caricatore = None
        else:
            self._generatore = None
            self._n_casse    = n_casse
            self._caricatore = CaricatoreLivelli(
                directory_base=directory_livelli,
                difficolta=difficolta,
                split=split,
                seme=seme,
            )

        # Il renderer viene creato solo quando serve (lazy loading)
        self._renderer: Optional[RendererSokoban] = None

        # Stato interno dell'episodio
        self._griglia: Optional[np.ndarray] = None
        self._step_corrente: int = 0
        self._seme = seme

        # RNG richiesto da Gymnasium per azioni esplorative
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
        """Resetta l'ambiente caricando un nuovo livello.

        Se options contiene 'indice_livello', carica quel livello specifico
        dal dataset. Altrimenti sceglie un livello casuale (o generato).

        Parametri:
            seed:    seed opzionale per il reset (passa a Gymnasium base).
            options: dizionario con chiave opzionale 'indice_livello' (int).

        Restituisce:
            (osservazione 10x10 float32, info dict)
        """
        super().reset(seed=seed)

        if self._usa_generatore:
            # Genera un nuovo livello procedurale per questa fase del curriculum
            self._griglia = self._generatore.genera(
                righe=self.griglia_size[0],
                colonne=self.griglia_size[1],
                n_casse=self._n_casse,
            )
        elif options is not None and "indice_livello" in options:
            # Livello specifico richiesto (usato dalla valutazione deterministica)
            indice = int(options["indice_livello"])
            self._griglia = self._caricatore.ottieni(indice)
        else:
            # Livello casuale dal dataset
            self._griglia = self._caricatore.casuale()

        self._step_corrente = 0

        info = self._costruisci_info(mossa_eseguita=False, cassa_spostata=False)
        return self._osservazione(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Esegue un'azione e restituisce la transizione completa.

        Calcola la reward combinando la reward base con il reward shaping
        Manhattan e giocatore->cassa se attivati in configurazione.

        Parametri:
            action: intero in {0, 1, 2, 3}.

        Restituisce:
            (osservazione, reward, terminated, truncated, info)
        """
        if self._griglia is None:
            raise RuntimeError(
                "L'ambiente non e' stato inizializzato. Chiamare reset() prima di step()."
            )

        griglia_precedente = self._griglia.copy()

        # Controlla adiacenza giocatore-cassa PRIMA della mossa (usato da AG-LLM-REW)
        adiacente_cassa = self._giocatore_adiacente_cassa(griglia_precedente)

        # Applica la mossa e aggiorna lo stato interno
        nuova_griglia, mossa_eseguita, cassa_spostata = applica_mossa(
            self._griglia, int(action)
        )
        self._griglia = nuova_griglia
        self._step_corrente += 1

        # Verifica condizioni di terminazione
        terminated = controlla_vittoria(self._griglia)
        truncated  = (self._step_corrente >= self.max_step) and not terminated

        # Calcola reward con shaping attivo
        reward = calcola_reward(
            griglia_precedente, self._griglia, terminated,
            scala_manhattan=self.scala_manhattan,
            adiacente_cassa=adiacente_cassa,
            scala_player_box=self.scala_player_box,
        )

        # Rendering in tempo reale se modalita' 'human' attiva
        if self.render_mode == "human":
            self.render()

        info = self._costruisci_info(
            mossa_eseguita=mossa_eseguita,
            cassa_spostata=cassa_spostata,
            adiacente_cassa=adiacente_cassa,
        )
        return self._osservazione(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renderizza lo stato corrente tramite RendererSokoban.

        Crea il renderer al primo utilizzo (lazy). Restituisce un array
        RGB in modalita' 'rgb_array', None in tutte le altre modalita'.
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
        """Chiude il renderer e libera le risorse grafiche."""
        if self._renderer is not None:
            self._renderer.chiudi()
            self._renderer = None

    # ------------------------------------------------------------------
    # Metodi interni di supporto
    # ------------------------------------------------------------------

    def _osservazione(self) -> np.ndarray:
        """Restituisce lo stato corrente come array float32 (10, 10).

        Per griglie piu' piccole applica il padding centrato con valore 7.
        L'observation space e' sempre (10, 10) per compatibilita' con SB3.
        """
        if self._griglia is None:
            return np.zeros((10, 10), dtype=np.float32)
        if self._usa_generatore:
            return padding_a_10x10(self._griglia).astype(np.float32)
        return self._griglia.astype(np.float32)

    def _giocatore_adiacente_cassa(self, griglia: np.ndarray) -> bool:
        """Controlla se il giocatore e' adiacente a una cassa (4 direzioni).

        Usato da AG-LLM-REW per decidere se chiamare il LLM: il segnale
        LLM ha senso solo quando il giocatore sta per spingere una cassa.

        Parametri:
            griglia: griglia da ispezionare (tipicamente stato pre-mossa).
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
        """Costruisce il dizionario info restituito da reset() e step().

        Il dizionario e' usato da AG-LLM-REW (cassa_spostata) e dai callback
        di logging (step_corrente, casse_su_target).
        """
        if self._griglia is None:
            return {}
        return {
            "step_corrente":             self._step_corrente,
            "casse_su_target":           conta_casse_su_target(self._griglia),
            "mossa_eseguita":            mossa_eseguita,
            "cassa_spostata":            cassa_spostata,
            "giocatore_adiacente_cassa": adiacente_cassa,
        }

    # ------------------------------------------------------------------
    # Proprieta' di convenienza
    # ------------------------------------------------------------------

    @property
    def griglia(self) -> Optional[np.ndarray]:
        """Restituisce una copia della griglia corrente (sola lettura)."""
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
