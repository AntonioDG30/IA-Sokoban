# Aggiunge src/ al sys.path cosi' i test importano i package del progetto
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
