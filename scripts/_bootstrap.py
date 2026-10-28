import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = Path(os.environ.get("VARA_OUTPUTS_DIR", "~/running/VARA/outputs")).expanduser()
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"
SIMULATIONS_DIR = OUTPUTS_DIR / "simulations"
TMP_DIR = OUTPUTS_DIR / "tmp"
MPLCONFIG_DIR = TMP_DIR / "matplotlib"

for path in (OUTPUTS_DIR, FIGURES_DIR, RESULTS_DIR, SIMULATIONS_DIR, TMP_DIR, MPLCONFIG_DIR):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
