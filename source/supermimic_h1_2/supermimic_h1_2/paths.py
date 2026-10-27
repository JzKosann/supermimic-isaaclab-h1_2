"""Default paths inside the repository. All of them can be overridden by environment variables (e.g. data or models stored elsewhere, or mounted somewhere else in Docker).

Requires an editable install (pip install -e source/supermimic_h1_2): the defaults are derived from this file's location in the repository.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# H1-2 USD from unitreerobotics/unitree_model (HuggingFace), downloaded by scripts/download_assets.py
UNITREE_MODEL_DIR = Path(os.environ.get("SUPERMIMIC_UNITREE_MODEL_DIR", REPO_ROOT / "assets" / "unitree_model"))
H1_2_USD = str(UNITREE_MODEL_DIR / "H1-2" / "h1_2_handless" / "h1_2_handless.usd")

# URDF parsed by the differentiable FK: the source model of the USD above (unitree_ros, BSD-3, shipped with this repository)
H1_2_URDF = str(REPO_ROOT / "assets" / "h1_2_description" / "h1_2_handless.urdf")

# LAFAN1 H1-2 retargeted CSVs (h1_2/ of lvhaidong/LAFAN1_Retargeting_Dataset) and the 50 Hz npz produced by scripts/csv_to_npz.py
LAFAN1_CSV_DIR = Path(os.environ.get("SUPERMIMIC_CSV_DIR", REPO_ROOT / "datasets" / "lafan1_csv" / "h1_2"))
MOTION_DIR = str(Path(os.environ.get("SUPERMIMIC_MOTION_DIR", REPO_ROOT / "datasets" / "lafan1_h1_2_npz")))

LOG_ROOT = Path(os.environ.get("SUPERMIMIC_LOG_ROOT", REPO_ROOT / "logs" / "supermimic"))
