"""Agente AG-LLM: policy diretta tramite LLM senza training RL.

Il LLM genera l'azione da eseguire ad ogni turno osservando la griglia ASCII,
le coordinate esplicite di giocatore/casse/target e le ultime 5 mosse eseguite.
Non c'e' nessun modello RL da addestrare: il LLM e' la policy a tutti gli effetti.

Questo agente serve come baseline di riferimento: mostra quanto riesce a fare
un LLM di buona qualita' (qwen3:14b) su Sokoban senza alcun training,
confrontandosi con gli agenti RL addestrati.

Metriche raccolte per episodio:
    solved (bool):            True se tutte le casse sono state piazzate.
    mosse (int):              step eseguiti nell'episodio.
    reward_cumulativa (float): reward totale accumulata.
    n_fallback (int):         azioni casuali per risposta LLM non parsificabile.
    casse_finali (int):       casse su target a fine episodio.
"""

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

_RADICE = Path(__file__).resolve().parent.parent
if str(_RADICE) not in sys.path:
    sys.path.insert(0, str(_RADICE))

from llm_integration.llm_client import ClienteLLM
from llm_integration.sokoban_prompt import (
    griglia_a_testo,
    crea_prompt,
    parsifica_azione,
    conta_casse,
    NOMI_AZIONI,
)


class AgenteAgLLM:
    """Agente AG-LLM: il LLM e' la policy diretta, nessun training RL.

    Parametri:
        seme: seed per la riproducibilita' (passato al reset dell'ambiente).
    """

    def __init__(self, provider: str = "ollama", seme: int = 42) -> None:
        self.seme   = seme
        self.client = ClienteLLM(provider=provider)

    # ------------------------------------------------------------------
    # Estrazione coordinate esplicite dall'osservazione
    # ------------------------------------------------------------------

    @staticmethod
    def _estrai_posizioni(obs: np.ndarray) -> str:
        """Converte l'obs float32 (10,10) in una stringa con coordinate 1-indexed.

        Le coordinate esplicite (riga, colonna) di giocatore, casse e target
        aiutano il LLM a ragionare spazialmente senza interpretare la griglia
        ASCII carattere per carattere. Le celle di padding (valore 7) vengono ignorate.

        Formato output: 'Giocatore: riga R, colonna C | Cassa 1: riga R, colonna C | ...'
        """
        grid = np.round(obs).astype(int)

        # 5=GIOCATORE, 6=GIOCATORE_SU_TARGET
        player_pos = np.argwhere((grid == 5) | (grid == 6))
        # 3=CASSA, 4=CASSA_SU_TARGET
        box_pos    = np.argwhere((grid == 3) | (grid == 4))
        # 2=TARGET, 4=CASSA_SU_TARGET, 6=GIOCATORE_SU_TARGET
        tgt_pos    = np.argwhere((grid == 2) | (grid == 4) | (grid == 6))

        parti = []
        if len(player_pos) > 0:
            r, c = int(player_pos[0, 0]), int(player_pos[0, 1])
            parti.append("Giocatore: riga " + str(r + 1) + ", colonna " + str(c + 1))
        for i, (r, c) in enumerate(box_pos):
            parti.append(
                "Cassa " + str(i + 1) + ": riga " + str(int(r) + 1)
                + ", colonna " + str(int(c) + 1)
            )
        for i, (r, c) in enumerate(tgt_pos):
            parti.append(
                "Target " + str(i + 1) + ": riga " + str(int(r) + 1)
                + ", colonna " + str(int(c) + 1)
            )
        return " | ".join(parti)

    # ------------------------------------------------------------------
    # Esecuzione di un singolo episodio
    # ------------------------------------------------------------------

    def valuta_episodio(self, env, max_step: int) -> Dict[str, Any]:
        """Esegue un singolo episodio usando il LLM come policy.

        Ad ogni step costruisce il prompt con stato corrente, coordinate
        esplicite e storico delle ultime 5 mosse, poi chiede al LLM la
        prossima azione da eseguire.

        Parametri:
            env:      SokobanEnv gia' costruito (non resettato).
            max_step: limite step dell'episodio (per il contatore nel prompt).

        Restituisce:
            Dizionario con le metriche dell'episodio.
        """
        obs, _ = env.reset()
        reward_ep  = 0.0
        n_fallback = 0
        casse_finali = 0
        step_ep    = 0

        # Storico delle ultime 5 mosse: il LLM lo vede nel prompt e lo usa
        # per evitare di ripetere la stessa mossa in loop (es. su->giu->su->giu)
        storico_mosse: Deque[str] = deque(maxlen=5)

        while True:
            grid_text     = griglia_a_testo(obs)
            casse_su_tgt, n_casse = conta_casse(obs)
            posizioni     = self._estrai_posizioni(obs)

            # Aggiunge lo storico mosse alle informazioni spaziali nel prompt
            if storico_mosse:
                posizioni += " | Ultime mosse: " + " ".join(storico_mosse)

            prompt = crea_prompt(
                grid_text=grid_text,
                casse_su_target=casse_su_tgt,
                n_casse=n_casse,
                step_corrente=step_ep,
                max_step=max_step,
                posizioni=posizioni,
            )

            # max_tokens=10 e' sufficiente per la parola piu' lunga ('sinistra')
            risposta = self.client.chiedi(prompt, max_tokens=10)
            azione   = parsifica_azione(risposta)

            # Registra l'azione scelta per il prossimo prompt
            storico_mosse.append(NOMI_AZIONI[azione])

            # Controlla se la risposta conteneva una parola valida o e' stato
            # usato il fallback casuale
            tokens = risposta.lower().strip().split()
            prima_parola_valida = any(t in ("su", "giu", "sinistra", "destra") for t in tokens)
            if not prima_parola_valida:
                n_fallback += 1

            obs, reward, terminated, truncated, info = env.step(azione)
            reward_ep   += float(reward)
            step_ep     += 1
            casse_finali = info.get("casse_su_target", 0)

            if terminated or truncated:
                break

        return {
            "solved":            terminated,
            "mosse":             step_ep,
            "reward_cumulativa": round(reward_ep, 4),
            "n_fallback":        n_fallback,
            "casse_finali":      casse_finali,
        }

    # ------------------------------------------------------------------
    # Valutazione su piu' episodi
    # ------------------------------------------------------------------

    def valuta(
        self,
        env,
        n_episodi: int,
        max_step: int,
        nome_fase: str = "",
    ) -> Dict[str, Any]:
        """Valuta l'agente su n_episodi e aggrega le metriche.

        Parametri:
            env:       SokobanEnv (verra' resettato n_episodi volte).
            n_episodi: numero di episodi da eseguire.
            max_step:  limite step per episodio.
            nome_fase: nome della fase per il logging (es. 'C0-1box-gen').

        Restituisce dizionario con:
            solve_rate:        % episodi risolti (0-100).
            mosse_medie:       media step per episodi risolti.
            reward_cumulativa: media reward totale su tutti gli episodi.
            fallback_rate:     % step con risposta LLM non parsificabile.
            casse_su_target:   media casse finali su target.
            n_episodi:         numero episodi eseguiti.
            n_risolti:         numero episodi risolti.
        """
        n_risolti = 0
        mosse_risolti: List[int]   = []
        rewards:       List[float] = []
        fallbacks_totali = 0
        step_totali      = 0
        casse_totali:  List[int]   = []

        t0 = time.time()
        for i in range(n_episodi):
            ep = self.valuta_episodio(env, max_step)
            if ep["solved"]:
                n_risolti += 1
                mosse_risolti.append(ep["mosse"])
            rewards.append(ep["reward_cumulativa"])
            fallbacks_totali += ep["n_fallback"]
            step_totali      += ep["mosse"]
            casse_totali.append(ep["casse_finali"])

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(
                    "[AG-LLM] " + (nome_fase + " " if nome_fase else "")
                    + str(i + 1) + "/" + str(n_episodi)
                    + " episodi | risolti=" + str(n_risolti)
                    + " | elapsed=" + str(round(elapsed, 1)) + "s"
                )

        solve_rate   = n_risolti / n_episodi * 100
        mosse_medie  = float(np.mean(mosse_risolti)) if mosse_risolti else 0.0
        reward_media = float(np.mean(rewards))
        fallback_rate = (fallbacks_totali / step_totali * 100) if step_totali > 0 else 0.0
        casse_medie  = float(np.mean(casse_totali))

        metriche: Dict[str, Any] = {
            "solve_rate":        round(solve_rate, 2),
            "mosse_medie":       round(mosse_medie, 2),
            "reward_cumulativa": round(reward_media, 4),
            "fallback_rate":     round(fallback_rate, 2),
            "casse_su_target":   round(casse_medie, 3),
            "n_episodi":         n_episodi,
            "n_risolti":         n_risolti,
        }

        print(
            "\n[AG-LLM] " + (nome_fase + " — " if nome_fase else "")
            + "Risultati (" + str(n_episodi) + " episodi):\n"
            + "  Solve rate:        " + str(round(solve_rate, 1)) + "%\n"
            + "  Mosse medie:       " + str(round(mosse_medie, 1)) + " (solo risolti)\n"
            + "  Reward cumulativa: " + str(round(reward_media, 3)) + "\n"
            + "  Fallback rate:     " + str(round(fallback_rate, 1)) + "% azioni random\n"
            + "  Casse su target:   " + str(round(casse_medie, 2)) + "/" + str(4)
        )

        return metriche

    def salva_risultati(self, risultati: Dict[str, Any], percorso: Path) -> None:
        """Salva le metriche di valutazione in formato JSON leggibile.

        Crea automaticamente le directory intermedie se non esistono.

        Parametri:
            risultati: dizionario restituito da valuta().
            percorso:  Path del file JSON di destinazione.
        """
        percorso.parent.mkdir(parents=True, exist_ok=True)
        with open(percorso, "w", encoding="utf-8") as f:
            json.dump(risultati, f, indent=2, ensure_ascii=False)
        print("[AG-LLM] Risultati salvati: " + str(percorso))
