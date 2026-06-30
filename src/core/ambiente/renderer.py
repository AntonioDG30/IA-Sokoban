# Rendering di Sokoban con Pygame, inizializzato in modo lazy al primo frame.
#
# Modalità 'human': apre una finestra a schermo, utile per le demo e il debug manuale.
# Modalità 'rgb_array': restituisce un array (H, W, 3) senza finestra, comodo per
# registrare video o lavorare in ambienti headless (SDL_VIDEODRIVER=dummy).

from typing import Optional

import numpy as np

from core.ambiente.game_logic import (
    MURO, PAVIMENTO, TARGET, CASSA,
    CASSA_SU_TARGET, GIOCATORE, GIOCATORE_SU_TARGET,
)

# DISPONIBILITÀ DI PYGAME

# Pygame è una dipendenza opzionale: se manca, il rendering è disabilitato ma il resto
# dell'ambiente (logica, training, valutazione) continua a funzionare.
try:
    import pygame
    PYGAME_DISPONIBILE = True
except ImportError:
    PYGAME_DISPONIBILE = False

# COSTANTI GRAFICHE

DIMENSIONE_CELLA = 60         # lato in pixel di ogni cella
N_RIGHE, N_COL = 10, 10
LARGHEZZA = DIMENSIONE_CELLA * N_COL    # 600 px
ALTEZZA = DIMENSIONE_CELLA * N_RIGHE   # 600 px

TITOLO_FINESTRA = "Sokoban - RL + LLM"

# Colore (R, G, B) associato a ciascun tipo di cella
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


# CLASSE RENDERER

class RendererSokoban:
    """
    Disegna lo stato dell'ambiente Sokoban con Pygame.

    modalita è 'human' (finestra a schermo) o 'rgb_array' (surface offscreen restituita
    come array); fps regola la velocità di aggiornamento della finestra in modalità 'human'.
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

    # INIZIALIZZAZIONE LAZY

    def _inizializza(self) -> None:
        """Inizializza Pygame al primo frame da disegnare (una volta sola)."""
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
            # rgb_array: surface offscreen, non serve aprire un display
            if not pygame.get_init():
                pygame.init()
            self._schermo = pygame.Surface((LARGHEZZA, ALTEZZA))

        self._inizializzato = True

    # RENDERING PRINCIPALE

    def renderizza(self, griglia: np.ndarray) -> Optional[np.ndarray]:
        """
        Disegna la griglia dello stato corrente, una cella alla volta.

        Riceve la matrice (10, 10) dei valori di cella. Restituisce un array (H, W, 3) uint8
        in modalità 'rgb_array', None in modalità 'human' (dove aggiorna direttamente la finestra).
        """
        self._inizializza()

        schermo = self._schermo
        schermo.fill(COLORE_SFONDO)

        n_righe, n_col = griglia.shape
        for riga in range(n_righe):
            for col in range(n_col):
                valore = int(griglia[riga, col])
                colore = COLORI.get(valore, (255, 0, 255))  # magenta = valore inatteso
                rect = pygame.Rect(
                    col * DIMENSIONE_CELLA,
                    riga * DIMENSIONE_CELLA,
                    DIMENSIONE_CELLA,
                    DIMENSIONE_CELLA,
                )
                pygame.draw.rect(schermo, colore, rect)            # riempimento cella
                pygame.draw.rect(schermo, COLORE_BORDO, rect, 1)   # bordo della cella
                self._disegna_dettaglio(riga, col, valore)

        if self.modalita == "human":
            pygame.display.flip()
            if self._orologio is not None:
                self._orologio.tick(self.fps)
            # Svuota la coda eventi così la finestra non si "freeza"
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.chiudi()
            return None
        else:
            # surfarray dà (W, H, 3): si traspone in (H, W, 3) per la convenzione immagine
            return np.transpose(
                pygame.surfarray.array3d(schermo), (1, 0, 2)
            )

    def _disegna_dettaglio(self, riga: int, col: int, valore: int) -> None:
        """Sovradisegna un simbolo dentro la cella per distinguere meglio i tipi a colpo d'occhio."""
        schermo = self._schermo
        cx = col * DIMENSIONE_CELLA + DIMENSIONE_CELLA // 2   # centro cella (x)
        cy = riga * DIMENSIONE_CELLA + DIMENSIONE_CELLA // 2   # centro cella (y)
        r = DIMENSIONE_CELLA // 4

        if valore == TARGET:
            # Bersaglio vuoto: cerchio rosso non riempito
            pygame.draw.circle(schermo, (180, 40, 40), (cx, cy), r, 3)

        elif valore == CASSA:
            # Cassa libera: quadrato interno più scuro
            m = DIMENSIONE_CELLA // 4
            rect_i = pygame.Rect(
                col * DIMENSIONE_CELLA + m,
                riga * DIMENSIONE_CELLA + m,
                DIMENSIONE_CELLA - 2 * m,
                DIMENSIONE_CELLA - 2 * m,
            )
            pygame.draw.rect(schermo, (140, 85, 20), rect_i)

        elif valore == CASSA_SU_TARGET:
            # Cassa a posto: cerchio verde a conferma
            pygame.draw.circle(schermo, (20, 120, 20), (cx, cy), r, 3)

        elif valore == GIOCATORE:
            # Giocatore: cerchio blu pieno
            pygame.draw.circle(schermo, (20, 55, 170), (cx, cy), r)

        elif valore == GIOCATORE_SU_TARGET:
            # Giocatore fermo su un bersaglio: cerchio blu + anello rosso del target
            pygame.draw.circle(schermo, (20, 55, 170), (cx, cy), r)
            pygame.draw.circle(schermo, (180, 40, 40), (cx, cy), r + 4, 3)

    # CHIUSURA

    def chiudi(self) -> None:
        """Chiude la finestra e termina Pygame, liberandone le risorse."""
        if self._inizializzato and PYGAME_DISPONIBILE:
            pygame.quit()
            self._inizializzato = False
            self._schermo = None
