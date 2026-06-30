# Ambiente Sokoban compatibile con l'API standard di Gymnasium.
#
# Implementa l'interfaccia attesa da Stable Baselines 3:
#   reset()  -> (obs, info)
#   step()   -> (obs, reward, terminated, truncated, info)
#   render() -> None oppure np.ndarray
#   close()  -> None
#
# Observation space: Box(0.0, 7.0, shape=(10, 10), dtype=float32). È sempre 10x10 per
# restare compatibile con SB3 lungo tutte le fasi del curriculum; per le griglie più
# piccole il valore 7 (PADDING) riempie le celle di bordo artificiale.
# Action space: Discrete(4) -- 0=su, 1=giù, 2=sinistra, 3=destra.

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

from core.ambiente.game_logic import (
    applica_mossa,
    controlla_vittoria,
    conta_casse_su_target,
    GIOCATORE, GIOCATORE_SU_TARGET,
    CASSA, CASSA_SU_TARGET,
)
from core.ambiente.reward import calcola_reward
from core.ambiente.level_loader import CaricatoreLivelli
from core.ambiente.level_generator import GeneratoreLivelli, padding_a_10x10
from core.ambiente.renderer import RendererSokoban

# Limite di step per episodio di default, coerente con gym-sokoban e con Boxoban.
MAX_STEP_PER_EPISODIO = 120


class SokobanEnv(gymnasium.Env):
    """
    Ambiente Sokoban custom, compatibile con Gymnasium e Stable Baselines 3.

    Può prendere i livelli dal dataset Boxoban (fasi C4-C5) oppure generarli al volo
    (fasi C0-C3 del curriculum). I parametri principali:
      - directory_livelli / difficolta / split: sorgente Boxoban (None -> builtin o generati)
      - usa_generatore: forza il generatore procedurale anche su griglia 10x10
      - scala_manhattan / scala_player_box: pesi delle due componenti di reward shaping (0 = off)
      - render_mode: None | 'human' | 'rgb_array' (il renderer è creato in modo lazy)
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

        # Accetta solo i render_mode dichiarati nei metadata
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

        # L'observation space è sempre 10x10 float32, qualunque sia la griglia di gioco.
        # high=7.0 perché il padding vale 7 (distinto da MURO=0 e dalle celle 1-6): così i
        # pesi della rete restano validi passando da una fase del curriculum alla successiva
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=7.0, shape=(10, 10), dtype=np.float32
        )

        # 4 azioni discrete: su, giù, sinistra, destra
        self.action_space = gymnasium.spaces.Discrete(4)

        # Sorgente dei livelli: generatore procedurale o dataset Boxoban.
        # usa_generatore=True serve per le fasi C0-C3 su 10x10; ogni griglia diversa da
        # 10x10 usa comunque sempre il generatore
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

        # Renderer creato solo al primo render() (lazy)
        self._renderer: Optional[RendererSokoban] = None

        # Stato interno dell'episodio
        self._griglia: Optional[np.ndarray] = None
        self._step_corrente: int = 0
        self._seme = seme

        # RNG di Gymnasium, usato per le azioni esplorative
        self.np_random: np.random.Generator = np.random.default_rng(seme)

    # API GYMNASIUM OBBLIGATORIA

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resetta l'ambiente caricando un nuovo livello; restituisce (osservazione, info).

        Se usa_generatore è attivo genera un livello procedurale per la fase corrente;
        altrimenti, se options contiene 'indice_livello' carica quel livello specifico
        (usato dalla valutazione deterministica), se no ne pesca uno casuale dal dataset.
        """
        super().reset(seed=seed)

        if self._usa_generatore:
            # Nuovo livello procedurale per questa fase del curriculum
            self._griglia = self._generatore.genera(
                righe=self.griglia_size[0],
                colonne=self.griglia_size[1],
                n_casse=self._n_casse,
            )
        elif options is not None and "indice_livello" in options:
            # Livello specifico richiesto esplicitamente (valutazione deterministica)
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
        """
        Esegue un'azione e restituisce (obs, reward, terminated, truncated, info).

        Applica la mossa, aggiorna lo stato e valuta le condizioni di fine episodio:
        terminated = livello risolto, truncated = limite di step raggiunto senza vittoria.
        La reward include lo shaping Manhattan e giocatore->cassa se sono attivi.
        """
        if self._griglia is None:
            raise RuntimeError(
                "L'ambiente non e' stato inizializzato. Chiamare reset() prima di step()."
            )

        griglia_precedente = self._griglia.copy()

        # Adiacenza giocatore-cassa calcolata PRIMA della mossa (la usa AG-LLM-REW)
        adiacente_cassa = self._giocatore_adiacente_cassa(griglia_precedente)

        # Applica la mossa e aggiorna lo stato interno
        nuova_griglia, mossa_eseguita, cassa_spostata = applica_mossa(
            self._griglia, int(action)
        )
        self._griglia = nuova_griglia
        self._step_corrente += 1

        # Condizioni di terminazione: vittoria (terminated) o limite step (truncated)
        terminated = controlla_vittoria(self._griglia)
        truncated  = (self._step_corrente >= self.max_step) and not terminated

        # Reward con shaping eventualmente attivo
        reward = calcola_reward(
            griglia_precedente, self._griglia, terminated,
            scala_manhattan=self.scala_manhattan,
            adiacente_cassa=adiacente_cassa,
            scala_player_box=self.scala_player_box,
        )

        # In modalità 'human' aggiorna la finestra a ogni step
        if self.render_mode == "human":
            self.render()

        info = self._costruisci_info(
            mossa_eseguita=mossa_eseguita,
            cassa_spostata=cassa_spostata,
            adiacente_cassa=adiacente_cassa,
        )
        return self._osservazione(), reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Renderizza lo stato corrente con RendererSokoban (creato in modo lazy al primo uso).
        Restituisce un array RGB in modalità 'rgb_array', None in tutte le altre modalità.
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
        """Chiude il renderer e libera le risorse grafiche di Pygame."""
        if self._renderer is not None:
            self._renderer.chiudi()
            self._renderer = None

    # METODI INTERNI DI SUPPORTO

    def _osservazione(self) -> np.ndarray:
        """
        Restituisce lo stato corrente come array float32 (10, 10).

        Per le griglie più piccole applica il padding centrato (valore 7); l'observation
        space resta sempre (10, 10) per compatibilità con SB3.
        """
        if self._griglia is None:
            return np.zeros((10, 10), dtype=np.float32)
        if self._usa_generatore:
            return padding_a_10x10(self._griglia).astype(np.float32)
        return self._griglia.astype(np.float32)

    def _giocatore_adiacente_cassa(self, griglia: np.ndarray) -> bool:
        """
        Indica se il giocatore ha una cassa in una delle 4 celle adiacenti.

        Lo usa AG-LLM-REW per interrogare il LLM solo quando una spinta è imminente: il
        segnale ha senso unicamente quando il giocatore sta per spostare una cassa.
        """
        posizioni = np.argwhere(
            (griglia == GIOCATORE) | (griglia == GIOCATORE_SU_TARGET)
        )
        if len(posizioni) == 0:
            return False
        riga_g, col_g = posizioni[0]
        righe, colonne = griglia.shape
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:   # le 4 celle adiacenti
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
        """
        Costruisce il dizionario info restituito da reset() e step().

        Contiene i campi letti da AG-LLM-REW (cassa_spostata) e dai callback di logging
        (step_corrente, casse_su_target).
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

    def __repr__(self) -> str:
        return (
            f"SokobanEnv("
            f"griglia={self.griglia_size}, "
            f"step={self._step_corrente}/{self.max_step}, "
            f"manhattan={self.scala_manhattan}, "
            f"player_box={self.scala_player_box}, "
            f"render_mode={self.render_mode!r})"
        )
