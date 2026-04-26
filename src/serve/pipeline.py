"""Preprocessing + prediction pipeline, reproducing Notebook 02's transforms.

The Preprocessor applies exactly the training-time steps of a variant:
  - raw         → no transform
  - out         → IQR clip on clip_candidates
  - scaled      → StandardScaler (scaler_raw)
  - out_scaled  → IQR clip, then StandardScaler (scaler_out)

ModelBundle wraps a loaded model + its preprocessor and exposes predict().
Models are discovered at import time from config.MODEL_REGISTRY; anything whose
files are absent is silently skipped so the service runs on partial datasets.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import (
    ARTIFACTS_DIR,
    FEATURE_NAMES_PATH,
    IQR_BOUNDS_PATH,
    LABEL_MAPPING_PATH,
    MODEL_REGISTRY,
    SCALER_OUT_PATH,
    SCALER_RAW_PATH,
    VARIANT_REQUIREMENTS,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


FEATURE_NAMES: list[str] = _load_json(FEATURE_NAMES_PATH)
# label_mapping.json is {class_name: id}. Invert for id→name.
_RAW_LABEL_MAP: dict[str, int] = _load_json(LABEL_MAPPING_PATH)
ID_TO_CLASS: dict[int, str] = {v: k for k, v in _RAW_LABEL_MAP.items()}
CLASS_NAMES: list[str] = [ID_TO_CLASS[i] for i in range(len(ID_TO_CLASS))]


class Preprocessor:
    """Applies the Notebook-02 transform for one variant."""

    def __init__(self, variant: str):
        if variant not in VARIANT_REQUIREMENTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self._iqr = None
        self._scaler = None

        if variant in ("out", "out_scaled"):
            self._iqr = _load_json(IQR_BOUNDS_PATH)

        if variant == "scaled":
            self._scaler = _load_pickle(SCALER_RAW_PATH)
        elif variant == "out_scaled":
            self._scaler = _load_pickle(SCALER_OUT_PATH)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Enforce exact column order the model expects.
        X = X[FEATURE_NAMES].copy()

        if self._iqr is not None:
            for col in self._iqr["clip_candidates"]:
                X[col] = X[col].clip(
                    lower=self._iqr["lower"][col],
                    upper=self._iqr["upper"][col],
                )

        if self._scaler is not None:
            return self._scaler.transform(X[FEATURE_NAMES])

        return X[FEATURE_NAMES].values.astype(np.float64)


@dataclass
class ModelBundle:
    key: str
    display_name: str
    variant: str
    loader_type: str
    model: Any  # sklearn estimator or torch.nn.Module
    preprocessor: Preprocessor

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        X = pd.DataFrame(rows)
        missing = set(FEATURE_NAMES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        Xt = self.preprocessor.transform(X)

        if self.loader_type == "joblib":
            if hasattr(self.model, "predict_proba"):
                return np.asarray(self.model.predict_proba(Xt))
            # Perceptron has no predict_proba — fall back to decision_function softmax.
            # Raw decision scores are unbounded, so we temperature-scale by the
            # per-sample std before softmax; without this, softmax collapses to
            # 100% on the winner even when the prediction is wrong.
            scores = self.model.decision_function(Xt)
            if scores.ndim == 1:
                scores = np.vstack([-scores, scores]).T
            std = scores.std(axis=1, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            scores = (scores - scores.max(axis=1, keepdims=True)) / std
            exp = np.exp(scores)
            return exp / exp.sum(axis=1, keepdims=True)

        if self.loader_type == "torch_dnn":
            import torch

            self.model.eval()
            device = next(self.model.parameters()).device
            with torch.no_grad():
                t = torch.as_tensor(Xt, dtype=torch.float32, device=device)
                logits = self.model(t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        raise RuntimeError(f"Unsupported loader_type: {self.loader_type}")


def _load_joblib_model(path: Path):
    return joblib.load(path)


def _load_dnn_model(path: Path):
    import torch
    from .dnn_model import DNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DNN(
        n_features=ckpt["n_features"],
        n_classes=ckpt["n_classes"],
        dropout_rate=ckpt["dropout_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


_LOADERS = {
    "joblib": _load_joblib_model,
    "torch_dnn": _load_dnn_model,
}


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def discover_available_models() -> list[str]:
    """Return keys of registry entries whose on-disk artifacts are all present
    AND whose loader's runtime dependencies are importable on this machine."""
    available: list[str] = []
    torch_ok = _torch_available()
    for key, spec in MODEL_REGISTRY.items():
        if spec["loader_type"] == "torch_dnn" and not torch_ok:
            continue
        model_path = ARTIFACTS_DIR / spec["filename"]
        required = [model_path, *VARIANT_REQUIREMENTS[spec["variant"]]]
        if all(p.exists() for p in required):
            available.append(key)
    return available


def load_model_bundle(key: str) -> ModelBundle:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key: {key}")
    spec = MODEL_REGISTRY[key]
    model_path = ARTIFACTS_DIR / spec["filename"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loader = _LOADERS[spec["loader_type"]]
    model = loader(model_path)
    preproc = Preprocessor(spec["variant"])
    return ModelBundle(
        key=key,
        display_name=spec["display_name"],
        variant=spec["variant"],
        loader_type=spec["loader_type"],
        model=model,
        preprocessor=preproc,
    )


def load_metrics(key: str) -> dict:
    spec = MODEL_REGISTRY[key]
    return _load_json(ARTIFACTS_DIR / spec["metrics_file"])


def load_confusion_matrix(key: str) -> np.ndarray:
    spec = MODEL_REGISTRY[key]
    return np.load(ARTIFACTS_DIR / spec["cm_file"])
