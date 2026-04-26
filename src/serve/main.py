"""FastAPI service exposing every locally-available model from MODEL_REGISTRY.

Run from project root:
    uvicorn src.serve.main:app --reload --host 127.0.0.1 --port 8000

Then open http://127.0.0.1:8000/docs for the interactive OpenAPI UI.
"""
from __future__ import annotations

import io

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import Response

from .config import MODEL_REGISTRY
from .pipeline import (
    CLASS_NAMES,
    FEATURE_NAMES,
    ID_TO_CLASS,
    ModelBundle,
    discover_available_models,
    load_confusion_matrix,
    load_metrics,
    load_model_bundle,
)
from .schemas import (
    ModelInfo,
    ModelsResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)
from ..monitor import store as pred_store
from ..monitor import detector, reference

app = FastAPI(
    title="CICIoT2023 IoT-IDS Inference Service",
    description=(
        "Serves the best variant of every trained model "
        "(Random Forest / DNN / Logistic Regression / Perceptron) over the "
        "CICIoT2023 feature schema. Available models are discovered at startup "
        "based on which artifacts exist in data/artifacts/."
    ),
    version="1.0.0",
)

# Lazy cache: models are loaded on first request, not at startup, so boot stays fast.
_bundles: dict[str, ModelBundle] = {}


def _get_bundle(key: str) -> ModelBundle:
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {key}")
    if key not in discover_available_models():
        spec = MODEL_REGISTRY[key]
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{key}' is registered but its artifact "
                f"'{spec['filename']}' (or a required preprocessing file) is not "
                f"present in data/artifacts/. Copy the file from the training "
                f"machine and retry."
            ),
        )
    if key not in _bundles:
        _bundles[key] = load_model_bundle(key)
    return _bundles[key]


@app.get("/", summary="Service info")
def root():
    available = discover_available_models()
    return {
        "service": "CICIoT2023 IoT-IDS Inference Service",
        "n_classes": len(CLASS_NAMES),
        "n_features": len(FEATURE_NAMES),
        "available_models": available,
        "docs": "/docs",
    }


@app.get("/health", summary="Liveness check")
def health():
    return {"status": "ok"}


@app.get("/features", summary="Ordered list of expected feature names")
def features():
    return {"n_features": len(FEATURE_NAMES), "feature_names": FEATURE_NAMES}


@app.get("/classes", summary="Class id â†’ name mapping")
def classes():
    return {"n_classes": len(CLASS_NAMES), "classes": ID_TO_CLASS}


@app.get("/models", response_model=ModelsResponse, summary="List registered models")
def models():
    available = set(discover_available_models())
    avail_list: list[ModelInfo] = []
    unavail_list: list[ModelInfo] = []
    for key, spec in MODEL_REGISTRY.items():
        info = ModelInfo(
            key=key,
            display_name=spec["display_name"],
            variant=spec["variant"],
            loader_type=spec["loader_type"],
            available=key in available,
        )
        (avail_list if info.available else unavail_list).append(info)
    return ModelsResponse(available=avail_list, unavailable=unavail_list)


@app.get("/models/{key}/metrics", summary="Training-time metrics for a model")
def model_metrics(key: str):
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {key}")
    try:
        return load_metrics(key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get(
    "/models/{key}/confusion_matrix",
    summary="Confusion matrix as a JSON 2-D array (rows=true, cols=pred)",
)
def model_confusion_matrix(key: str):
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {key}")
    try:
        cm = load_confusion_matrix(key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "class_names": CLASS_NAMES,
        "matrix": cm.astype(int).tolist(),
    }


@app.get(
    "/models/{key}/confusion_matrix.png",
    summary="Confusion matrix rendered as a PNG image",
)
def model_confusion_matrix_png(key: str, normalize: bool = False):
    if key not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {key}")
    try:
        cm = load_confusion_matrix(key).astype(np.float64)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    # Lazy import â€” matplotlib is heavy.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(CLASS_NAMES, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title_suffix = "(row-normalised)" if normalize else "(counts)"
    ax.set_title(f"Confusion Matrix â€” {MODEL_REGISTRY[key]['display_name']} {title_suffix}")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict on a single feature dict or a micro-batch",
)
def predict(
    request: PredictRequest,
    background_tasks: BackgroundTasks,
    model: str = Query(..., description="Model key from /models"),
):
    bundle = _get_bundle(model)

    raw_rows = request.rows
    batch = [raw_rows] if isinstance(raw_rows, dict) else list(raw_rows)
    if not batch:
        raise HTTPException(status_code=422, detail="Request contains no rows.")

    try:
        proba = bundle.predict_proba(batch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    predictions: list[PredictionItem] = []
    pred_ids: list[int] = []
    confidences: list[float] = []
    for row_probs in proba:
        pred_id = int(np.argmax(row_probs))
        conf = float(row_probs[pred_id])
        pred_ids.append(pred_id)
        confidences.append(conf)
        predictions.append(
            PredictionItem(
                predicted_class=ID_TO_CLASS[pred_id],
                predicted_class_id=pred_id,
                confidence=conf,
                probabilities={
                    ID_TO_CLASS[i]: float(p) for i, p in enumerate(row_probs)
                },
            )
        )

    # Log predictions asynchronously â€” does not block the response.
    background_tasks.add_task(
        pred_store.log_predictions, model, pred_ids, confidences, batch
    )

    return PredictResponse(
        model=bundle.display_name,
        variant=bundle.variant,
        n_predictions=len(predictions),
        predictions=predictions,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ Drift endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/drift/status", summary="Real-time drift status for a model")
def drift_status(
    model: str = Query(..., description="Model key"),
    window: int = Query(500, ge=10, description="Number of recent predictions to analyse"),
):
    """Returns class-distribution drift and confidence drift for the most
    recent *window* predictions logged for *model*.

    Requires at least some predictions to have been sent via /predict first.
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

    total = pred_store.get_total_logged(model)
    rows = pred_store.get_window(model, n=window)

    _NO_DATA = {"available": False, "message": "No predictions logged yet.", "alert": "none"}
    if not rows:
        return {
            "model": model,
            "total_logged": total,
            "window_size": 0,
            "alerts": [],
            "class_drift": _NO_DATA,
            "confidence_drift": _NO_DATA,
            "alert_rate": _NO_DATA,
        }

    class_ids = [r["predicted_class_id"] for r in rows]
    confs     = [r["confidence"] for r in rows]

    ref = reference.class_reference()

    # 1. Class-distribution drift (chi-squared).
    if ref["available"]:
        cls_report = detector.class_drift_report(
            reference_probs=ref["probs"],
            window_class_ids=class_ids,
            n_classes=ref["n_classes"],
        )
    else:
        cls_report = {"available": False, "message": "y_train_encoded.csv not found.", "alert": "none"}

    # 2. Confidence drift â€” baseline from all logged rows once enough data exists.
    if total >= 200:
        all_rows = pred_store.get_window(model, n=total)
        baseline = float(sum(r["confidence"] for r in all_rows) / len(all_rows))
    else:
        baseline = float(sum(confs) / len(confs))

    conf_report = detector.confidence_drift_report(
        baseline_mean=baseline,
        window_confidences=confs,
    )

    # 3. Alert-rate spike (sudden increase in attack traffic).
    # BENIGN = class id 1 in the CICIoT2023 label mapping.
    BENIGN_ID = 1
    if ref["available"]:
        alert_rate_report = detector.alert_rate_report(
            benign_class_id=BENIGN_ID,
            window_class_ids=class_ids,
            reference_benign_rate=float(ref["probs"][BENIGN_ID]),
        )
    else:
        alert_rate_report = {"available": False, "message": "y_train_encoded.csv not found.", "alert": "none"}

    # Aggregate active alerts from all three signals.
    alerts = []
    for src, report in [
        ("class_distribution", cls_report),
        ("confidence",         conf_report),
        ("alert_rate_spike",   alert_rate_report),
    ]:
        lvl = report.get("alert", "none")
        if lvl != "none":
            alerts.append({"source": src, "level": lvl, "message": report["message"]})

    return {
        "model": model,
        "total_logged": total,
        "window_size": len(rows),
        "alerts": alerts,
        "class_drift": cls_report,
        "confidence_drift": conf_report,
        "alert_rate": alert_rate_report,
    }


@app.get("/drift/confidence_history", summary="Rolling mean confidence over time")
def drift_confidence_history(
    model: str = Query(..., description="Model key"),
    bucket_size: int = Query(100, ge=10, description="Predictions per time-series bucket"),
):
    """Returns mean/min/max confidence in consecutive *bucket_size* buckets,
    oldest first â€” suitable for plotting a confidence trend chart.
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

    history = pred_store.get_confidence_history(model, bucket_size=bucket_size)
    return {
        "model": model,
        "bucket_size": bucket_size,
        "n_buckets": len(history),
        "history": history,
    }


@app.get("/drift/feature_analysis", summary="Live PSI feature drift: training vs recent traffic")
def drift_feature_analysis(
    model: str = Query(..., description="Model key"),
    window: int = Query(500, ge=10, description="Number of recent logged feature rows to analyse"),
    bins: int = Query(10, ge=4, le=50),
):
    """Computes Population Stability Index (PSI) for every feature by
    comparing training features against recent live traffic logged for *model*.

    PSI < 0.10 = stable, 0.10-0.25 = slight drift, > 0.25 = significant drift.
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")

    ref = reference.feature_reference()

    if not ref["available"]:
        raise HTTPException(status_code=503, detail="X_train_raw.csv not found in data/processed/.")
    obs_rows = pred_store.get_feature_window(model, ref["feature_names"], n=window)
    if not obs_rows:
        return {
            "available": False,
            "message": "No logged feature rows yet. Stream traffic through /predict first.",
            "model": model,
            "window": window,
            "n_observed": 0,
        }

    obs_data = np.asarray(obs_rows, dtype=np.float64)
    report = detector.feature_psi_report(
        ref_data=ref["data"],
        obs_data=obs_data,
        feature_names=ref["feature_names"],
        bins=bins,
    )
    report["note"] = (
        f"Comparing {len(ref['data'])} training rows vs {len(obs_data)} recent live rows "
        f"for model '{model}'."
    )
    report["model"] = model
    report["window"] = window
    report["n_observed"] = int(len(obs_data))
    return report


@app.post("/drift/reset", summary="Clear the prediction log for a model")
def drift_reset(model: str = Query(..., description="Model key")):
    """Deletes all logged predictions for *model* so the drift window starts
    fresh.  Useful when you want to re-run the stream_test client.
    """
    if model not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model}")
    n_deleted = pred_store.clear_log(model)
    return {"model": model, "rows_deleted": n_deleted}

