"""Generate iqr_bounds.json and feature_names.json from the training CSV.

This recomputes the artifacts that the new Step 15 cell in Notebook 02 would
produce, so the serving code can work today without re-running the full
notebook. Safe to run multiple times (overwrites files with identical content).

Run from project root:
    python scripts/bootstrap_inference_artifacts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

X_TRAIN_RAW = PROCESSED_DIR / "X_train_raw.csv"
CANDIDATE_LISTS = ARTIFACTS_DIR / "candidate_lists.json"


def main() -> None:
    if not X_TRAIN_RAW.exists():
        raise FileNotFoundError(
            f"{X_TRAIN_RAW} not found — run Notebook 02 first to produce the processed splits."
        )
    if not CANDIDATE_LISTS.exists():
        raise FileNotFoundError(
            f"{CANDIDATE_LISTS} not found — run Notebook 02 first."
        )

    candidates = json.loads(CANDIDATE_LISTS.read_text())
    clip_candidates = list(candidates["clip_candidates"])

    X_train = pd.read_csv(X_TRAIN_RAW)

    Q1 = X_train[clip_candidates].quantile(0.25)
    Q3 = X_train[clip_candidates].quantile(0.75)
    IQR = Q3 - Q1
    lower_bounds = Q1 - 1.5 * IQR
    upper_bounds = Q3 + 1.5 * IQR

    iqr_bounds = {
        "clip_candidates": clip_candidates,
        "lower": {c: float(lower_bounds[c]) for c in clip_candidates},
        "upper": {c: float(upper_bounds[c]) for c in clip_candidates},
    }

    feature_names = list(X_train.columns)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "iqr_bounds.json").write_text(json.dumps(iqr_bounds, indent=2))
    (ARTIFACTS_DIR / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    print(f"Saved iqr_bounds.json   ({len(clip_candidates)} clipped features)")
    print(f"Saved feature_names.json ({len(feature_names)} features)")


if __name__ == "__main__":
    main()
