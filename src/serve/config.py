"""Static configuration for the serving layer.

The MODEL_REGISTRY fixes the best preprocessing variant per model (derived from
the *_comparison.csv files in data/artifacts/). At startup the service filters
this registry down to models whose artifact files actually exist on disk, so
the same code runs on machines where only a subset of the models is available.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

def _artifact_path(filename: str) -> Path:
    return ARTIFACTS_DIR / filename


LABEL_MAPPING_PATH = _artifact_path("label_mapping.json")
FEATURE_NAMES_PATH = _artifact_path("feature_names.json")
IQR_BOUNDS_PATH = _artifact_path("iqr_bounds.json")
SCALER_RAW_PATH = _artifact_path("scaler_raw.pkl")
SCALER_OUT_PATH = _artifact_path("scaler_out.pkl")

# Best variant per model from the respective *_comparison.csv files.
# loader_type ∈ {"joblib", "torch_dnn"}.
MODEL_REGISTRY: dict[str, dict] = {
    "random_forest": {
        "display_name": "Random Forest",
        "variant": "scaled",
        "filename": "random_forest_scaled.joblib",
        "loader_type": "joblib",
        "metrics_file": "rf_metrics_scaled.json",
        "cm_file": "rf_cm_scaled.npy",
    },
    "deep_neural_network": {
        "display_name": "Deep Neural Network",
        "variant": "out_scaled",
        "filename": "dnn_out_scaled.pt",
        "loader_type": "torch_dnn",
        "metrics_file": "dnn_metrics_out_scaled.json",
        "cm_file": "dnn_cm_out_scaled.npy",
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "variant": "out_scaled",
        "filename": "logistic_regression_out_scaled.joblib",
        "loader_type": "joblib",
        "metrics_file": "lr_metrics_out_scaled.json",
        "cm_file": "lr_cm_out_scaled.npy",
    },
    "perceptron": {
        "display_name": "Perceptron",
        "variant": "out_scaled",
        "filename": "perceptron_out_scaled.joblib",
        "loader_type": "joblib",
        "metrics_file": "perceptron_metrics_out_scaled.json",
        "cm_file": "perceptron_cm_out_scaled.npy",
    },
}

# Preprocessing pipelines each variant requires. The service will refuse to
# serve a model whose variant's required artifacts are missing.
VARIANT_REQUIREMENTS: dict[str, list[Path]] = {
    "raw": [],
    "out": [IQR_BOUNDS_PATH],
    "scaled": [SCALER_RAW_PATH],
    "out_scaled": [IQR_BOUNDS_PATH, SCALER_OUT_PATH],
}
