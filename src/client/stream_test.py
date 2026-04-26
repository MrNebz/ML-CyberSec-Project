"""Replay CSV rows as micro-batches against the /predict endpoint.

Useful for (a) smoke-testing the service, (b) producing a realistic traffic
stream that Step 7's drift-monitoring code will later consume, and (c) demoing
drift by replaying a custom CSV without replacing data/processed files.

Run from project root:
    python -m src.client.stream_test --model random_forest --batch-size 64 --limit 512
    python -m src.client.stream_test --model random_forest --x data/drift/step7/X_drift_test.csv --y data/drift/step7/y_drift_test.csv
    python -m src.client.stream_test --model random_forest --x data/drift/step7/Backdoor_Malware.pcap.csv --limit 1000

Flags:
    --model        Model key (see /models). Required.
    --x            Feature CSV to replay (default data/processed/X_test_raw.csv).
    --y            Optional encoded-label CSV for rolling accuracy.
    --batch-size   Rows per POST (default 64).
    --limit        Max rows to send (default 1000; 0 = entire test set).
    --sleep        Seconds between batches (default 0.0).
    --api          Base URL of the service (default http://127.0.0.1:8000).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"


def _project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_feature_names() -> list[str]:
    return json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())


def _load_class_names() -> list[str]:
    mapping = json.loads((ARTIFACTS_DIR / "label_mapping.json").read_text())
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def _validate_feature_csv(X: pd.DataFrame, X_path: Path) -> list[str]:
    feature_names = _load_feature_names()
    missing = [col for col in feature_names if col not in X.columns]
    extra = [col for col in X.columns if col not in feature_names]
    if missing:
        sys.exit(f"{X_path} is missing required features: {missing}")
    if extra:
        print(f"Warning: ignoring extra columns in {X_path}: {extra}", file=sys.stderr)
    return feature_names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Model key (see GET /models).")
    p.add_argument("--x", help="Feature CSV to replay. Defaults to data/processed/X_test_raw.csv.")
    p.add_argument("--y", help="Optional encoded-label CSV for rolling accuracy.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=1000, help="0 to send the entire test set.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    args = p.parse_args()

    api = args.api.rstrip("/")

    X_path = _project_path(args.x) if args.x else PROCESSED_DIR / "X_test_raw.csv"
    y_path = _project_path(args.y) if args.y else None
    if y_path is None and args.x is None:
        y_path = PROCESSED_DIR / "y_test_encoded.csv"
    if not X_path.exists():
        sys.exit(f"Feature CSV not found: {X_path}")
    if args.y and not y_path.exists():
        sys.exit(f"Label CSV not found: {y_path}")

    X = pd.read_csv(X_path)
    feature_names = _validate_feature_csv(X, X_path)
    X = X[feature_names]

    y = None
    if y_path is not None and y_path.exists():
        y = pd.read_csv(y_path).iloc[:, 0]
        if len(y) != len(X):
            sys.exit(f"Feature/label row-count mismatch: {len(X)} rows in X, {len(y)} rows in y.")

    if args.limit > 0:
        X = X.head(args.limit)
        if y is not None:
            y = y.head(args.limit)

    class_names = _load_class_names()

    total = len(X)
    correct = 0
    t0 = time.perf_counter()
    sent = 0
    pred_counts = {name: 0 for name in class_names}

    print(f"Streaming {total} rows from {X_path}")
    if y is None:
        print("No label CSV found/provided; reporting prediction distribution only.")
    else:
        print(f"Using labels from {y_path}")

    for start in range(0, total, args.batch_size):
        batch_X = X.iloc[start:start + args.batch_size]
        batch_y = y.iloc[start:start + args.batch_size].to_numpy() if y is not None else None
        payload = {"rows": batch_X.to_dict(orient="records")}

        r = requests.post(f"{api}/predict?model={args.model}", json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()

        for i, pred in enumerate(resp["predictions"]):
            pred_counts[pred["predicted_class"]] = pred_counts.get(pred["predicted_class"], 0) + 1
            if batch_y is not None and pred["predicted_class_id"] == int(batch_y[i]):
                correct += 1
        sent += len(batch_X)

        if y is not None:
            pct = correct / sent * 100 if sent else 0.0
            print(f"[{sent:>6d}/{total}] rolling accuracy = {pct:5.2f}%", flush=True)
        else:
            top = sorted(pred_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            summary = ", ".join(f"{name}={count}" for name, count in top if count)
            print(f"[{sent:>6d}/{total}] top predictions: {summary}", flush=True)

        if args.sleep > 0:
            time.sleep(args.sleep)

    dt = time.perf_counter() - t0
    rate = sent / dt if dt > 0 else 0.0
    print(f"\nDone - sent {sent} rows in {dt:.2f}s ({rate:.0f} rows/s).")
    if y is not None and sent:
        print(f"Final accuracy: {correct/sent*100:.2f}%")

    print("Prediction distribution:")
    for name, count in sorted(pred_counts.items(), key=lambda kv: kv[1], reverse=True):
        if count:
            print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
