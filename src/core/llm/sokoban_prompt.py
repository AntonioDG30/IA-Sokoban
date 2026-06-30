# Costruzione dei prompt per il LLM e parsing delle sue risposte.
#
# Gestisce due tipi di prompt:
#   - Prompt azione (AG-LLM-ACT, AG-LLM-GUIDE): chiede al LLM quale mossa eseguire.
#   - Prompt reward (AG-LLM-REW): chiede al LLM di valutare una mossa appena eseguita.
#
# Codifica delle celle nella griglia (da game_logic.py):
#   0 = muro (#)          3 = cassa ($)             6 = giocatore su target (+)
#   1 = pavimento ( )     4 = cassa su target (*)   7 = padding (reso come muro nell'ASCII)
#   2 = target (.)        5 = giocatore (@)

import random
import re

import numpy as np


# Simbolo ASCII per ogni valore di cella; il 7 (padding) viene reso come muro '#'
SIMBOLI = {0: "#", 1: " ", 2: ".", 3: "$", 4: "*", 5: "@", 6: "+", 7: "#"}

# Mappa parola italiana -> indice azione (0=su, 1=giu, 2=sinistra, 3=destra)
MAPPA_AZIONI = {
    "su":       0,
    "giu":      1,
    "sinistra": 2,
    "destra":   3,
}

# Nomi delle azioni in ordine di indice: serve per ricostruire il nome a partire dall'indice
NOMI_AZIONI = ["su", "giu", "sinistra", "destra"]


def griglia_a_testo(obs: np.ndarray) -> str:
    """
    Converte l'osservazione float32 (10, 10) in una stringa ASCII multiriga.

    Traduce ogni cella nel simbolo di SIMBOLI e unisce le righe con un a capo, pronta da
    inserire nel prompt. Riceve un array con valori interi 0-7.
    """
    righe = []
    for riga in obs:
        # Ogni cella diventa il suo simbolo ASCII ('?' se il valore è inatteso)
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
    """
    Costruisce il prompt con cui si chiede al LLM la prossima mossa.

    Include la griglia ASCII, il contatore casse/target, gli step rimasti e, se passate,
    le coordinate esplicite di giocatore/casse/target: queste ultime aiutano il LLM a
    ragionare spazialmente senza dover interpretare carattere per carattere l'ASCII.
    La stringa posizioni, se presente, ha forma del tipo 'Giocatore: riga 3, colonna 5 | ...'.
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
    """
    Converte la risposta testuale del LLM nell'indice dell'azione (intero in {0,1,2,3}).

    Normalizza la risposta (minuscolo, via la punteggiatura) e prende la prima parola
    valida tra 'su', 'giu', 'sinistra', 'destra'. Se non ne trova nessuna, ripiega su
    un'azione casuale uniforme (e logga la risposta non interpretabile).
    """
    # Sostituisce la punteggiatura con spazi e porta tutto in minuscolo
    norm = re.sub(r"[^\w\s]", " ", risposta.lower().strip())

    for parola in norm.split():
        if parola in MAPPA_AZIONI:
            return MAPPA_AZIONI[parola]

    # Nessuna parola valida: fallback su azione casuale
    if risposta.strip():
        preview = risposta[:40].replace("\n", " ")
        print("[sokoban_prompt] Non parsificabile: " + repr(preview) + " -> random")
    return random.randint(0, 3)


def conta_casse(obs: np.ndarray) -> tuple:
    """
    Conta dalle osservazioni le casse su target e il totale delle casse.

    Serve a riempire il prompt con lo stato aggiornato del livello senza interrogare
    direttamente l'ambiente. Restituisce la tupla (casse_su_target, casse_totali).
    """
    su_target = int(np.sum(np.round(obs) == 4))   # celle CASSA_SU_TARGET (valore 4)
    libere    = int(np.sum(np.round(obs) == 3))   # celle CASSA libere (valore 3)
    return su_target, su_target + libere


def crea_prompt_reward(
    grid_text_pre: str,
    grid_text_post: str,
    azione_nome: str,
    casse_su_target: int,
    n_casse: int,
) -> str:
    """
    Costruisce il prompt con cui AG-LLM-REW fa valutare al LLM una mossa appena eseguita.

    Mostra la griglia prima e dopo la mossa, il nome dell'azione e lo stato delle casse, e
    chiede un punteggio intero 0-3 che verrà poi normalizzato e usato per scalare il
    contributo del LLM alla reward dell'agente PPO.
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
    """
    Converte la risposta-punteggio del LLM in un float normalizzato in [-0.5, +0.5].

    Cerca il primo intero 0-3 nella risposta e lo mappa secondo questa scala:
      0 -> -0.5   (mossa dannosa o bloccante)
      1 ->  0.0   (mossa neutra, nessun progresso)
      2 -> +0.25  (mossa buona, avvicina la cassa al target)
      3 -> +0.5   (mossa ottima, cassa posizionata sul target)
    Se non trova nessun numero valido ripiega su 0.0 (neutro), il fallback più sicuro.
    """
    _MAPPA_SCORE = {0: -0.5, 1: 0.0, 2: 0.25, 3: 0.5}
    match = re.search(r"\b([0-3])\b", risposta.strip())
    if match:
        return _MAPPA_SCORE[int(match.group(1))]

    # Risposta non interpretabile: reward neutra come fallback sicuro
    if risposta.strip():
        preview = risposta[:30].replace("\n", " ")
        print("[sokoban_prompt] Reward non parsificabile: " + repr(preview) + " -> 0.0")
    return 0.0
