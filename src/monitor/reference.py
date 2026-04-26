"""Load and cache training-time reference distributions for drift detection.

Two reference distributions are needed:
  - class_reference(): class proportions from y_train_encoded.csv
  - feature_reference(): a random sub-sample from X_train_raw.csv

Both functions are expensive on first call (CSV I/O) and then cached for the
lifetime of the process.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ..serve.config import ARTIFACTS_DIR, PROCESSED_DIR


@lru_cache(maxsize=1)
def class_reference() -> dict:
    """Return training class proportions.

    Returns a dict with keys:
      available (bool), n_classes (int), probs (list[float]), counts (list[int])
    """
    y_path = PROCESSED_DIR / "y_train_encoded.csv"
    label_path = ARTIFACTS_DIR / "label_mapping.json"
    if not y_path.exists() or not label_path.exists():
        return {"available": False}

    n_classes = len(json.loads(label_path.read_text()))
    y = pd.read_csv(y_path).iloc[:, 0].to_numpy(dtype=int)
    counts = np.bincount(y, minlength=n_classes)
    probs = counts / counts.sum()

    return {
        "available": True,
        "n_classes": int(n_classes),
        "probs": probs.tolist(),
        "counts": counts.tolist(),
    }


@lru_cache(maxsize=1)
def feature_reference(n_sample: int = 5_000) -> dict:
    """Return a random sub-sample of the training features for PSI computation.

    Returns a dict with keys:
      available (bool), feature_names (list[str]), data (np.ndarray shape (n,p))
    """
    x_path = PROCESSED_DIR / "X_train_raw.csv"
    if not x_path.exists():
        return {"available": False}

    df = pd.read_csv(x_path, nrows=n_sample)
    return {
        "available": True,
        "feature_names": list(df.columns),
        "data": df.to_numpy(dtype=np.float64),
    }


@lru_cache(maxsize=1)
def test_feature_sample(n_sample: int = 5_000) -> dict:
    """Return a random sub-sample of X_test_raw for PSI comparison."""
    x_path = PROCESSED_DIR / "X_test_raw.csv"
    if not x_path.exists():
        return {"available": False}

    df = pd.read_csv(x_path, nrows=n_sample)
    return {
        "available": True,
        "feature_names": list(df.columns),
        "data": df.to_numpy(dtype=np.float64),
    }
