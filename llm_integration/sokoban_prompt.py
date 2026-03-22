"""Generazione dei prompt per il LLM e parsing delle risposte.

Gestisce due tipi di prompt:
    - Prompt azione (AG-LLM, AG-LLM-GUIDE): chiede al LLM quale mossa eseguire.
    - Prompt reward (AG-LLM-REW): chiede al LLM di valutare una mossa appena eseguita.

Codifica celle della griglia (da game_logic.py):
    0 = muro (#)
    1 = pavimento ( )
    2 = target (.)
    3 = cassa ($)
    4 = cassa su target (*)
    5 = giocatore (@)
    6 = giocatore su target (+)
    7 = padding (trattato come muro nella visualizzazione ASCII)
"""

import random
import re

import numpy as np


# Simbolo ASCII per ciascun valore di cella; 7 e' il padding (bordo artificiale)
SIMBOLI = {0: "#", 1: " ", 2: ".", 3: "$", 4: "*", 5: "@", 6: "+", 7: "#"}

# Mappa parola italiana -> indice azione intero (0=su, 1=giu, 2=sinistra, 3=destra)
MAPPA_AZIONI = {
    "su":       0,
    "giu":      1,
    "sinistra": 2,
    "destra":   3,
}

# Lista ordinata dei nomi azione, usata per ricostruire il nome dall'indice
NOMI_AZIONI = ["su", "giu", "sinistra", "destra"]


def griglia_a_testo(obs: np.ndarray) -> str:
    """Converte l'osservazione float32 (10, 10) in una stringa ASCII multiriga.

    Ogni cella e' tradotta nel simbolo corrispondente tramite SIMBOLI.
    Le righe sono separate da newline, pronte per essere inserite nel prompt.

    Parametri:
        obs: array float32 di forma (10, 10) con valori interi 0-7.
    """
    righe = []
    for riga in obs:
        # Converte ogni cella nel simbolo ASCII corrispondente
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
    """Costruisce il prompt da inviare al LLM per scegliere la prossima mossa.

    Il prompt include la griglia ASCII, il contatore casse/target, gli step
    rimasti e (opzionalmente) le coordinate esplicite di giocatore, casse e
    target. Le coordinate esplicite aiutano il LLM a ragionare spazialmente
    senza dover interpretare la griglia ASCII pura.

    Parametri:
        grid_text:       griglia ASCII corrente (output di griglia_a_testo).
        casse_su_target: numero di casse attualmente posizionate sui target.
        n_casse:         numero totale di casse nel livello.
        step_corrente:   step gia' eseguiti nell'episodio corrente.
        max_step:        limite massimo di step per episodio.
        posizioni:       stringa opzionale con coordinate esplicite, ad es.
                         'Giocatore: riga 3, colonna 5 | Cassa 1: riga 4, colonna 5'.
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
    """Converte la risposta testuale del LLM nell'indice azione corrispondente.

    Normalizza la risposta (lowercase, rimozione punteggiatura) e cerca la
    prima parola valida tra 'su', 'giu', 'sinistra', 'destra'. Se nessuna
    parola valida viene trovata, restituisce un'azione casuale uniforme.

    Parametri:
        risposta: stringa restituita dal LLM.

    Restituisce:
        Intero in {0, 1, 2, 3}: indice dell'azione.
    """
    # Rimuove punteggiatura e normalizza in lowercase
    norm = re.sub(r"[^\w\s]", " ", risposta.lower().strip())

    for parola in norm.split():
        if parola in MAPPA_AZIONI:
            return MAPPA_AZIONI[parola]

    # Nessuna parola valida trovata: azione casuale (fallback)
    if risposta.strip():
        preview = risposta[:40].replace("\n", " ")
        print("[sokoban_prompt] Non parsificabile: " + repr(preview) + " -> random")
    return random.randint(0, 3)


def conta_casse(obs: np.ndarray) -> tuple:
    """Conta le casse su target e il totale delle casse dall'osservazione.

    Usata per costruire il prompt con le informazioni aggiornate sullo stato
    del livello senza dover accedere all'ambiente direttamente.

    Parametri:
        obs: array float32 di forma (10, 10).

    Restituisce:
        Tupla (casse_su_target, n_casse_totali).
    """
    su_target = int(np.sum(np.round(obs) == 4))   # celle con valore CASSA_SU_TARGET
    libere    = int(np.sum(np.round(obs) == 3))   # celle con valore CASSA
    return su_target, su_target + libere


def crea_prompt_reward(
    grid_text_pre: str,
    grid_text_post: str,
    azione_nome: str,
    casse_su_target: int,
    n_casse: int,
) -> str:
    """Costruisce il prompt per la valutazione della mossa (usato da AG-LLM-REW).

    Mostra al LLM la griglia prima e dopo la mossa, il nome dell'azione e
    lo stato attuale delle casse. Il LLM deve rispondere con un punteggio
    intero 0-3 che scala il contributo LLM alla reward dell'agente PPO.

    Parametri:
        grid_text_pre:   griglia ASCII prima della mossa.
        grid_text_post:  griglia ASCII dopo la mossa.
        azione_nome:     nome italiano dell'azione ('su'|'giu'|'sinistra'|'destra').
        casse_su_target: casse su target dopo la mossa.
        n_casse:         totale casse nel livello.
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
    """Converte la risposta reward del LLM in un float normalizzato in [-0.5, +0.5].

    Cerca il primo numero intero 0-3 nella risposta e lo mappa al valore
    di reward corrispondente. Fallback a 0.0 (neutro) se nessun numero valido.

    Mappa punteggi:
        0 -> -0.5  (mossa dannosa o bloccante)
        1 ->  0.0  (mossa neutra, nessun progresso)
        2 -> +0.25 (mossa buona, avanzamento verso il target)
        3 -> +0.5  (mossa ottima, cassa posizionata sul target)

    Parametri:
        risposta: stringa restituita dal LLM.
    """
    _MAPPA_SCORE = {0: -0.5, 1: 0.0, 2: 0.25, 3: 0.5}
    match = re.search(r"\b([0-3])\b", risposta.strip())
    if match:
        return _MAPPA_SCORE[int(match.group(1))]

    # Risposta non parsificabile: reward neutra come fallback sicuro
    if risposta.strip():
        preview = risposta[:30].replace("\n", " ")
        print("[sokoban_prompt] Reward non parsificabile: " + repr(preview) + " -> 0.0")
    return 0.0
