"""Downloads the robot model and the motion data (no Isaac needed, only pip install huggingface_hub):

    assets/unitree_model/H1-2/h1_2_handless/   H1-2 USD (unitreerobotics/unitree_model, ~38 MB)
    datasets/lafan1_csv/h1_2/*.csv             LAFAN1 retargeted to H1-2 (lvhaidong/LAFAN1_Retargeting_Dataset, 40 clips)

Usage (from the repository root): python scripts/download_assets.py [--only model|motion]
Then run scripts/csv_to_npz.py to convert the CSVs into the npz used for training.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "model": dict(
        repo_id="unitreerobotics/unitree_model",
        allow_patterns=["H1-2/h1_2_handless/*", "H1-2/h1_2_handless/configuration/*"],
        local_dir=REPO_ROOT / "assets" / "unitree_model",
    ),
    "motion": dict(
        repo_id="lvhaidong/LAFAN1_Retargeting_Dataset",
        allow_patterns=["h1_2/*.csv", "LICENSE", "README.md"],
        local_dir=REPO_ROOT / "datasets" / "lafan1_csv",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(SOURCES), default=None)
    args = parser.parse_args()
    for name, src in SOURCES.items():
        if args.only and name != args.only:
            continue
        print(f"[download] {src['repo_id']} → {src['local_dir']}", flush=True)
        snapshot_download(repo_id=src["repo_id"], repo_type="dataset", allow_patterns=src["allow_patterns"], local_dir=src["local_dir"])
    usd = REPO_ROOT / "assets" / "unitree_model" / "H1-2" / "h1_2_handless" / "h1_2_handless.usd"
    csvs = sorted((REPO_ROOT / "datasets" / "lafan1_csv" / "h1_2").glob("*.csv"))
    print(f"[download] USD {'OK' if usd.exists() else 'MISSING'}: {usd}")
    print(f"[download] {len(csvs)} CSV in datasets/lafan1_csv/h1_2")


if __name__ == "__main__":
    main()
