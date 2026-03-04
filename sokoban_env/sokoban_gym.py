"""Ambiente Sokoban compatibile con Gymnasium.

Implementa l'API completa Gymnasium:
    reset()  → (obs, info)
    step()   → (obs, reward, terminated, truncated, info)
    render() → None oppure np.ndarray
    close()  → None

Spazio di osservazione:
    Box(low=0, high=6, shape=(10, 10), dtype=np.int8)

Spazio delle azioni:
    Discrete(4)  — 0=su, 1=giù, 2=sinistra, 3=destra

Parametri costruttore:
    directory_livelli  percorso alla dir contenente i dati Boxoban.
                       Se None, usa i livelli integrati.
    difficolta         'unfiltered', 'medium', 'hard'.
    split              'train', 'valid', 'test'.
    render_mode        None, 'human', 'rgb_array'.
    max_step           numero massimo di step per episodio (default 120).
    seme               seme per la randomizzazione dei livelli.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium

from sokoban_env.game_logic import (
    applica_mossa,
    controlla_vittoria,
    conta_casse_su_target,
    GIOCATORE, GIOCATORE_SU_TARGET,
    NOMI_AZIONI,
)
from sokoban_env.reward import calcola_reward
from sokoban_env.level_loader import CaricatoreLivelli
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
    ):
        super().__init__()

        # Validazione render_mode
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode non valido: '{render_mode}'. "
                f"Opzioni: {self.metadata['render_modes']}"
            )

        self.render_mode = render_mode
        self.max_step = max_step

        # Spazio di osservazione: griglia 10×10, valori 0.0-6.0
        # dtype float32 per compatibilità con SB3/PyTorch (i layer lineari
        # richiedono tensori float). La griglia interna rimane int8.
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=6.0, shape=(10, 10), dtype=np.float32
        )

        # Spazio delle azioni: 4 direzioni discrete
        self.action_space = gymnasium.spaces.Discrete(4)

        # Caricatore livelli
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

        # Selezione livello: specifico per indice, oppure casuale
        if options is not None and "indice_livello" in options:
            indice = int(options["indice_livello"])
            self._griglia = self._caricatore.ottieni(indice)
        else:
            self._griglia = self._caricatore.casuale()

        self._step_corrente = 0

        info = self._costruisci_info(mossa_eseguita=False, cassa_spostata=False)
        return self._griglia.astype(np.float32), info

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

        # Applica la mossa
        nuova_griglia, mossa_eseguita, cassa_spostata = applica_mossa(
            self._griglia, int(action)
        )
        self._griglia = nuova_griglia
        self._step_corrente += 1

        # Condizioni di terminazione
        terminated = controlla_vittoria(self._griglia)
        truncated = (self._step_corrente >= self.max_step) and not terminated

        # Calcola reward
        reward = calcola_reward(griglia_precedente, self._griglia, terminated)

        # Rendering automatico in modalità 'human'
        if self.render_mode == "human":
            self.render()

        info = self._costruisci_info(
            mossa_eseguita=mossa_eseguita,
            cassa_spostata=cassa_spostata,
        )
        return self._griglia.astype(np.float32), reward, terminated, truncated, info

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

    def _costruisci_info(
        self, mossa_eseguita: bool, cassa_spostata: bool
    ) -> Dict[str, Any]:
        """Costruisce il dizionario info restituito da reset() e step()."""
        if self._griglia is None:
            return {}
        return {
            "step_corrente":     self._step_corrente,
            "casse_su_target":   conta_casse_su_target(self._griglia),
            "mossa_eseguita":    mossa_eseguita,
            "cassa_spostata":    cassa_spostata,
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
            f"step={self._step_corrente}/{self.max_step}, "
            f"render_mode={self.render_mode!r})"
        )
