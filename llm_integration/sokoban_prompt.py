"""Conversione griglia Sokoban <-> testo per LLM + parsing risposta.

Funzioni:
    griglia_a_testo      obs float32 (10,10) -> stringa ASCII multiriga
    crea_prompt          griglia + stato + step -> prompt AG-LLM (~130 token)
    parsifica_azione     risposta testuale LLM -> int azione {0,1,2,3}
    conta_casse          conta casse su target e totale dall'obs
    crea_prompt_reward   griglia pre+post + azione -> prompt AG-LLM-REW (~260 token)
    parsifica_reward     risposta LLM -> float reward normalizzato [-0.5, +0.5]

Codifica celle (da game_logic.py):
    0=#(muro) 1= (pavim) 2=.(target) 3=$(cassa) 4=*(cassa/tgt) 5=@(gioc) 6=+(g/t)
"""

import random
import re

import numpy as np


# Simboli ASCII per ogni valore cella (0-7, dove 7=PADDING trattato come muro)
SIMBOLI = {0: "#", 1: " ", 2: ".", 3: "$", 4: "*", 5: "@", 6: "+", 7: "#"}

# Mappa parole -> azione int {0=su, 1=giu, 2=sinistra, 3=destra}
MAPPA_AZIONI = {
    "su":        0,
    "giu":       1,
    "sinistra":  2,
    "destra":    3,
}

NOMI_AZIONI = ["su", "giu", "sinistra", "destra"]


def griglia_a_testo(obs: np.ndarray) -> str:
    """Converte obs float32 (10, 10) in stringa ASCII multiriga."""
    righe = []
    for riga in obs:
        line = "".join(SIMBOLI.get(int(v), "?") for v in riga)
        righe.append(line)
    return "\n".join(righe)


def crea_prompt(
    grid_text: str,
    casse_su_target: int,
    n_casse: int,
    step_corrente: int = 0,
    max_step: int = 120,
    posizioni: str = "",
) -> str:
    """Crea il prompt per AG-LLM (direct policy, ~150 token).

    Parametri:
        grid_text:       griglia ASCII corrente (da griglia_a_testo).
        casse_su_target: numero di casse attualmente su target.
        n_casse:         numero totale di casse nel livello.
        step_corrente:   step gia' eseguiti nell'episodio.
        max_step:        limite massimo di step per episodio.
        posizioni:       stringa opzionale con coordinate esplicite di giocatore,
                         casse e target (es. 'Giocatore: riga 3, colonna 5').
                         Aiuta il LLM a ragionare spazialmente senza dover
                         interpretare la griglia ASCII pura.
    """
    step_rimasti = max_step - step_corrente
    prompt = (
        "Sokoban: spingi le casse sui target.\n"
        "# muro  @ tu  $ cassa  . target  * cassa su target\n"
        "\n"
        + grid_text + "\n"
        "\n"
        "Casse su target: " + str(casse_su_target) + "/" + str(n_casse)
        + "  |  Step rimasti: " + str(step_rimasti) + "/" + str(max_step) + "\n"
    )
    if posizioni:
        prompt += posizioni + "\n"
    prompt += (
        "Rispondi con UNA SOLA parola, senza spiegazioni: su giu sinistra destra\n"
        "Mossa:"
    )
    return prompt


def parsifica_azione(risposta: str) -> int:
    """Converte risposta testuale LLM in int azione in {0, 1, 2, 3}.

    Normalizza (lowercase, strip, rimuove punteggiatura),
    cerca match esatto in MAPPA_AZIONI.
    Fallback: azione casuale uniforme.
    """
    norm = re.sub(r"[^\w\s]", " ", risposta.lower().strip())

    for parola in norm.split():
        if parola in MAPPA_AZIONI:
            return MAPPA_AZIONI[parola]

    if risposta.strip():
        preview = risposta[:40].replace("\n", " ")
        print("[sokoban_prompt] Non parsificabile: " + repr(preview) + " -> random")
    return random.randint(0, 3)


def conta_casse(obs: np.ndarray) -> tuple:
    """Conta casse su target e totale dall'obs float32 (10, 10).

    Restituisce (casse_su_target, n_casse_totali).
    """
    su_target = int(np.sum(np.round(obs) == 4))   # CASSA_SU_TARGET
    libere    = int(np.sum(np.round(obs) == 3))   # CASSA
    return su_target, su_target + libere


def crea_prompt_reward(
    grid_text_pre: str,
    grid_text_post: str,
    azione_nome: str,
    casse_su_target: int,
    n_casse: int,
) -> str:
    """Crea il prompt per la valutazione dell'azione eseguita (AG-LLM-REW).

    Il LLM confronta lo stato prima e dopo la mossa, restituendo un punteggio
    0-3 che scala il segnale di reward PPO (via LAMBDA_LLM).

    Parametri:
        grid_text_pre:   griglia ASCII prima della mossa (da griglia_a_testo).
        grid_text_post:  griglia ASCII dopo la mossa (da griglia_a_testo).
        azione_nome:     nome italiano dell'azione ('su'|'giu'|'sinistra'|'destra').
        casse_su_target: numero di casse su target dopo la mossa.
        n_casse:         numero totale di casse nel livello.
    """
    return (
        "Sokoban: valuta l'azione eseguita confrontando prima e dopo.\n"
        "# muro  @ tu  $ cassa  . target  * cassa su target\n"
        "\n"
        "Prima:\n"
        + grid_text_pre + "\n"
        "\n"
        "Dopo:\n"
        + grid_text_post + "\n"
        "\n"
        "Azione eseguita: " + azione_nome.upper() + "\n"
        "Casse su target: " + str(casse_su_target) + "/" + str(n_casse) + "\n"
        "Rispondi con UN SOLO numero intero, nessuna spiegazione:\n"
        "0=azione sbagliata/bloccante  1=neutro  2=buono  3=ottimo\n"
        "Punteggio:"
    )


def parsifica_reward(risposta: str) -> float:
    """Converte la risposta reward LLM in float normalizzato in [-0.5, +0.5].

    Mappa:
        0 -> -0.5  (peggiorato/bloccato)
        1 ->  0.0  (neutro)
        2 -> +0.25 (buono)
        3 -> +0.5  (ottimo)
    Fallback: 0.0 (neutro) se risposta non parsificabile.
    """
    _MAPPA_SCORE = {0: -0.5, 1: 0.0, 2: 0.25, 3: 0.5}
    match = re.search(r"\b([0-3])\b", risposta.strip())
    if match:
        return _MAPPA_SCORE[int(match.group(1))]
    if risposta.strip():
        preview = risposta[:30].replace("\n", " ")
        print("[sokoban_prompt] Reward non parsificabile: " + repr(preview) + " -> 0.0")
    return 0.0

