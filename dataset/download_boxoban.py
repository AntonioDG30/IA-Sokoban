# Script per scaricare il dataset DeepMind Boxoban.
#
# Uso:
#   python dataset/download_boxoban.py           # scarica solo se assente
#   python dataset/download_boxoban.py --forza   # riscarica sempre
#
# Scarica il repository github.com/google-deepmind/boxoban-levels in dataset/boxoban/ tramite
# git clone (shallow) oppure, se git non è disponibile, scaricando lo ZIP.
#
# Struttura finale:
#   dataset/boxoban/
#       unfiltered/   (train/ valid/ test/)
#       medium/       (train/ valid/)
#       hard/         (file .txt diretti)

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_URL  = "https://github.com/google-deepmind/boxoban-levels"
ZIP_URL   = "https://github.com/google-deepmind/boxoban-levels/archive/refs/heads/master.zip"

# Radice del repo (la cartella che contiene dataset/): due livelli sopra questo file
RADICE = Path(__file__).resolve().parent.parent
DIR_DESTINAZIONE = RADICE / "dataset" / "boxoban"


def _git_disponibile() -> bool:
    """
    Indica se git è nel PATH prima di tentare il clone.
    Si preferisce git allo ZIP perché lo shallow clone è più veloce e non richiede di
    estrarre un archivio da centinaia di MB.
    """
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _rmtree(percorso: Path) -> None:
    """
    Rimuove ricorsivamente una directory, gestendo i file read-only di Windows.
    Su Windows git marca alcuni file come read-only (es. .git/objects) e shutil.rmtree
    fallirebbe senza il callback che prima toglie il flag.
    """
    import stat

    def _on_error(func, path, exc):
        # Toglie il flag read-only e riprova l'operazione che era fallita
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(percorso, onerror=_on_error)


def _scarica_con_git(destinazione: Path) -> None:
    """
    Clona il repo Boxoban con git shallow (--depth=1) e copia solo i dati utili.
    Il clone avviene in una directory temporanea _boxoban_tmp, così un errore a metà
    scaricamento non sporca la destinazione finale.
    """
    print(f"[Boxoban] git clone --depth=1 {REPO_URL}")
    tmp = destinazione.parent / "_boxoban_tmp"
    if tmp.exists():
        _rmtree(tmp)
    subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(tmp)],
        check=True,
    )
    # Copia solo unfiltered/medium/hard: il resto del repo (README, codice, ...) non serve
    for cartella in ("unfiltered", "medium", "hard"):
        sorgente = tmp / cartella
        if sorgente.exists():
            dest_cartella = destinazione / cartella
            if dest_cartella.exists():
                _rmtree(dest_cartella)
            shutil.copytree(sorgente, dest_cartella)
            print(f"[Boxoban] Copiato: {cartella}/")
    _rmtree(tmp)


def _scarica_con_zip(destinazione: Path) -> None:
    """
    Scarica il repo come ZIP da GitHub e lo estrae nella destinazione.
    È il fallback per gli ambienti senza git (alcuni CI o macchine Windows minimali). Lo ZIP
    di GitHub contiene tutto il repo sotto boxoban-levels-master/, quindi si entra in quella
    sottocartella prima di copiare i dati.
    """
    zip_path = destinazione.parent / "_boxoban.zip"
    print(f"[Boxoban] Scaricamento ZIP da {ZIP_URL} ...")

    def _progresso(blocchi_scaricati, dim_blocco, dim_totale):
        """Callback di urlretrieve: stampa la percentuale scaricata."""
        if dim_totale > 0:
            perc = blocchi_scaricati * dim_blocco / dim_totale * 100
            print(f"\r  {min(perc, 100):.1f}%", end="", flush=True)

    urllib.request.urlretrieve(ZIP_URL, zip_path, reporthook=_progresso)
    print()

    print("[Boxoban] Estrazione...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destinazione.parent / "_boxoban_zip")

    estratto = destinazione.parent / "_boxoban_zip" / "boxoban-levels-master"
    for cartella in ("unfiltered", "medium", "hard"):
        sorgente = estratto / cartella
        if sorgente.exists():
            dest_cartella = destinazione / cartella
            if dest_cartella.exists():
                shutil.rmtree(dest_cartella)
            shutil.copytree(sorgente, dest_cartella)
            print(f"[Boxoban] Copiato: {cartella}/")

    # Pulizia: ZIP e directory temporanea non servono più
    zip_path.unlink(missing_ok=True)
    shutil.rmtree(destinazione.parent / "_boxoban_zip", ignore_errors=True)


def _conta_file(directory: Path) -> int:
    """Conta ricorsivamente i file .txt in una directory (0 se la directory non esiste)."""
    return sum(1 for _ in directory.rglob("*.txt"))


def scarica_boxoban(forza: bool = False) -> None:
    """
    Scarica il dataset Boxoban nella directory dataset/boxoban/.

    Se i dati sono già presenti (almeno un .txt) lo script si ferma, a meno che forza=True
    non imponga il riscaricamento. Prova prima con git; se non c'è, ripiega sullo ZIP.
    """
    DIR_DESTINAZIONE.mkdir(parents=True, exist_ok=True)

    # Evita download inutili se il dataset è già lì
    if not forza and _conta_file(DIR_DESTINAZIONE) > 0:
        n = _conta_file(DIR_DESTINAZIONE)
        print(f"[Boxoban] Dati gia' presenti ({n} file). Usa forza=True per riscaricre.")
        return

    print(f"[Boxoban] Destinazione: {DIR_DESTINAZIONE}")

    if _git_disponibile():
        _scarica_con_git(DIR_DESTINAZIONE)
    else:
        print("[Boxoban] git non trovato, uso download ZIP.")
        _scarica_con_zip(DIR_DESTINAZIONE)

    n_file = _conta_file(DIR_DESTINAZIONE)
    print(f"\n[Boxoban] Download completato. File totali: {n_file}")
    print(f"[Boxoban] Directory: {DIR_DESTINAZIONE}")


if __name__ == "__main__":
    forza = "--forza" in sys.argv or "-f" in sys.argv
    scarica_boxoban(forza=forza)
