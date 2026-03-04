"""Rendering grafico dell'ambiente Sokoban tramite Pygame.

Supporta due modalità:
    'human'     — finestra interattiva a schermo.
    'rgb_array' — restituisce un array NumPy (H, W, 3) per VecEnv/recording.

Il modulo gestisce l'inizializzazione lazy di Pygame e può funzionare in
ambienti headless impostando SDL_VIDEODRIVER=dummy prima dell'import.
"""

import os
from typing import Optional

import numpy as np

from sokoban_env.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA,
    CASSA_SU_TARGET, GIOCATORE, GIOCATORE_SU_TARGET,
)

# ---------------------------------------------------------------------------
# Verifica disponibilità Pygame
# ---------------------------------------------------------------------------

try:
    import pygame
    PYGAME_DISPONIBILE = True
except ImportError:
    PYGAME_DISPONIBILE = False

# ---------------------------------------------------------------------------
# Costanti grafiche
# ---------------------------------------------------------------------------

DIMENSIONE_CELLA = 60         # pixel per cella
N_RIGHE, N_COL = 10, 10
LARGHEZZA = DIMENSIONE_CELLA * N_COL    # 600 px
ALTEZZA = DIMENSIONE_CELLA * N_RIGHE   # 600 px

TITOLO_FINESTRA = "Sokoban — RL + LLM"

# Palette colori (R, G, B)
COLORI: dict[int, tuple[int, int, int]] = {
    MURO:                  (70,  70,  70),
    PAVIMENTO:             (210, 190, 150),
    TARGET:                (210,  70,  70),
    CASSA:                 (185, 125,  40),
    CASSA_SU_TARGET:       (50,  170,  50),
    GIOCATORE:             (40,   90, 210),
    GIOCATORE_SU_TARGET:   (40,   90, 210),
}
COLORE_SFONDO = (45, 45, 45)
COLORE_BORDO  = (25, 25, 25)


# ---------------------------------------------------------------------------
# Classe renderer
# ---------------------------------------------------------------------------

class RendererSokoban:
    """Gestisce il rendering grafico dell'ambiente Sokoban.

    Parametri:
        modalita: 'human' o 'rgb_array'.
        fps:      frame al secondo per la modalità 'human'.
    """

    def __init__(self, modalita: str = "human", fps: int = 10):
        if modalita not in ("human", "rgb_array"):
            raise ValueError(
                f"Modalità non valida: '{modalita}'. "
                "Usare 'human' o 'rgb_array'."
            )
        self.modalita = modalita
        self.fps = fps
        self._schermo: Optional[object] = None
        self._orologio: Optional[object] = None
        self._inizializzato = False

    # ------------------------------------------------------------------
    # Inizializzazione lazy
    # ------------------------------------------------------------------

    def _inizializza(self) -> None:
        """Inizializza Pygame al primo utilizzo."""
        if self._inizializzato:
            return

        if not PYGAME_DISPONIBILE:
            raise ImportError(
                "pygame non è installato. Esegui: pip install pygame>=2.5"
            )

        if self.modalita == "human":
            pygame.init()
            pygame.display.set_caption(TITOLO_FINESTRA)
            self._schermo = pygame.display.set_mode((LARGHEZZA, ALTEZZA))
            self._orologio = pygame.time.Clock()
        else:
            # rgb_array: surface offscreen, non serve display
            if not pygame.get_init():
                pygame.init()
            self._schermo = pygame.Surface((LARGHEZZA, ALTEZZA))

        self._inizializzato = True

    # ------------------------------------------------------------------
    # Rendering principale
    # ------------------------------------------------------------------

    def renderizza(self, griglia: np.ndarray) -> Optional[np.ndarray]:
        """Renderizza la griglia dello stato corrente.

        Parametri:
            griglia: matrice NumPy (10, 10) con i valori delle celle.

        Restituisce:
            Array NumPy (H, W, 3) uint8 in modalità 'rgb_array',
            None in modalità 'human'.
        """
        self._inizializza()

        schermo = self._schermo
        schermo.fill(COLORE_SFONDO)

        n_righe, n_col = griglia.shape
        for riga in range(n_righe):
            for col in range(n_col):
                valore = int(griglia[riga, col])
                colore = COLORI.get(valore, (255, 0, 255))  # magenta = errore
                rect = pygame.Rect(
                    col * DIMENSIONE_CELLA,
                    riga * DIMENSIONE_CELLA,
                    DIMENSIONE_CELLA,
                    DIMENSIONE_CELLA,
                )
                pygame.draw.rect(schermo, colore, rect)
                pygame.draw.rect(schermo, COLORE_BORDO, rect, 1)
                self._disegna_dettaglio(riga, col, valore)

        if self.modalita == "human":
            pygame.display.flip()
            if self._orologio is not None:
                self._orologio.tick(self.fps)
            # Gestione eventi (evita freeze della finestra)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.chiudi()
            return None
        else:
            # Trasponi da (W, H, 3) a (H, W, 3)
            return np.transpose(
                pygame.surfarray.array3d(schermo), (1, 0, 2)
            )

    def _disegna_dettaglio(self, riga: int, col: int, valore: int) -> None:
        """Disegna simboli aggiuntivi sopra le celle per maggiore leggibilità."""
        schermo = self._schermo
        cx = col * DIMENSIONE_CELLA + DIMENSIONE_CELLA // 2
        cy = riga * DIMENSIONE_CELLA + DIMENSIONE_CELLA // 2
        r = DIMENSIONE_CELLA // 4

        if valore == TARGET:
            # Cerchio rosso vuoto
            pygame.draw.circle(schermo, (180, 40, 40), (cx, cy), r, 3)

        elif valore == CASSA:
            # Quadrato interno più scuro
            m = DIMENSIONE_CELLA // 4
            rect_i = pygame.Rect(
                col * DIMENSIONE_CELLA + m,
                riga * DIMENSIONE_CELLA + m,
                DIMENSIONE_CELLA - 2 * m,
                DIMENSIONE_CELLA - 2 * m,
            )
            pygame.draw.rect(schermo, (140, 85, 20), rect_i)

        elif valore == CASSA_SU_TARGET:
            # Cassa verde + segno checkmark semplificato (cerchio vuoto)
            pygame.draw.circle(schermo, (20, 120, 20), (cx, cy), r, 3)

        elif valore == GIOCATORE:
            # Cerchio pieno blu
            pygame.draw.circle(schermo, (20, 55, 170), (cx, cy), r)

        elif valore == GIOCATORE_SU_TARGET:
            # Cerchio blu + cerchio target rosso
            pygame.draw.circle(schermo, (20, 55, 170), (cx, cy), r)
            pygame.draw.circle(schermo, (180, 40, 40), (cx, cy), r + 4, 3)

    # ------------------------------------------------------------------
    # Chiusura
    # ------------------------------------------------------------------

    def chiudi(self) -> None:
        """Chiude la finestra e termina Pygame."""
        if self._inizializzato and PYGAME_DISPONIBILE:
            pygame.quit()
            self._inizializzato = False
            self._schermo = None
