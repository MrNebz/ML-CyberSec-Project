"""Statistical drift-detection primitives.

Three separate drift signals are computed:

1. class_drift_report   — chi-squared goodness-of-fit test on the predicted
                          class distribution vs the training distribution.
2. confidence_drift_report — checks whether rolling mean confidence has dropped
                              significantly relative to a baseline.
3. feature_psi_report   — Population Stability Index (PSI) for every numeric
                          feature, comparing a reference sample to a window
                          sample.  PSI < 0.10 = stable, 0.10–0.25 = slight
                          drift, > 0.25 = significant drift.

All functions return plain dicts so the FastAPI layer can serialize them
directly as JSON.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as scipy_chi2


# ──────────────────────────────────────────────────────────────────────────────
# 1. Class-distribution drift
# ──────────────────────────────────────────────────────────────────────────────

def class_drift_report(
    reference_probs: list[float],
    window_class_ids: list[int],
    n_classes: int,
) -> dict:
    """Chi-squared goodness-of-fit: are predicted classes still distributed
    like the training set?

    Parameters
    ----------
    reference_probs:  training class proportions, length n_classes.
    window_class_ids: integer class ids from the recent prediction window.
    n_classes:        total number of classes (handles unseen-class edge case).
    """
    if not window_class_ids:
        return {
            "available": False,
            "message": "No predictions logged yet.",
            "alert": "none",
        }

    n_window = len(window_class_ids)
    observed = np.bincount(window_class_ids, minlength=n_classes).astype(float)

    ref = np.array(reference_probs, dtype=float)
    expected = ref * n_window
    # Add a small Laplace pseudocount to avoid division-by-zero in any bin.
    expected = np.maximum(expected, 0.5)

    chi2_stat = float(((observed - expected) ** 2 / expected).sum())
    df = n_classes - 1
    pvalue = float(1.0 - scipy_chi2.cdf(chi2_stat, df=df))

    # Observed proportions for inspection.
    obs_probs = (observed / n_window).tolist()

    # KL divergence (reference ‖ observed) as a supplementary scalar.
    eps = 1e-10
    ref_smooth = ref + eps
    obs_smooth = observed / n_window + eps
    kl_div = float(np.sum(ref_smooth * np.log(ref_smooth / obs_smooth)))

    if pvalue < 0.01:
        alert = "critical"
    elif pvalue < 0.05:
        alert = "warning"
    else:
        alert = "none"

    return {
        "available": True,
        "n_window": n_window,
        "chi2_stat": chi2_stat,
        "pvalue": pvalue,
        "kl_divergence": kl_div,
        "observed_probs": obs_probs,
        "reference_probs": reference_probs,
        "alert": alert,
        "message": (
            f"Class distribution drift detected (p={pvalue:.4f})."
            if alert != "none"
            else f"Class distribution stable (p={pvalue:.4f})."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Confidence drift
# ──────────────────────────────────────────────────────────────────────────────

def confidence_drift_report(
    baseline_mean: float,
    window_confidences: list[float],
    warning_relative_drop: float = 0.10,
    critical_relative_drop: float = 0.20,
) -> dict:
    """Check whether rolling mean confidence has dropped relative to a baseline.

    Parameters
    ----------
    baseline_mean:          mean confidence on the held-out test set (or an
                            empirical warm-up window).
    window_confidences:     confidence scores from the recent prediction window.
    warning_relative_drop:  fraction of baseline below which we raise WARNING.
    critical_relative_drop: fraction of baseline below which we raise CRITICAL.
    """
    if not window_confidences:
        return {
            "available": False,
            "message": "No predictions logged yet.",
            "alert": "none",
        }

    window_mean = float(np.mean(window_confidences))
    window_std = float(np.std(window_confidences))
    relative_drop = (baseline_mean - window_mean) / max(baseline_mean, 1e-10)

    if relative_drop >= critical_relative_drop:
        alert = "critical"
    elif relative_drop >= warning_relative_drop:
        alert = "warning"
    else:
        alert = "none"

    return {
        "available": True,
        "baseline_mean_confidence": baseline_mean,
        "window_mean_confidence": window_mean,
        "window_std_confidence": window_std,
        "relative_drop": float(relative_drop),
        "alert": alert,
        "message": (
            f"Confidence dropped {relative_drop * 100:.1f}% relative to baseline."
            if alert != "none"
            else f"Confidence stable (window mean {window_mean:.4f} vs baseline {baseline_mean:.4f})."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature PSI
# ──────────────────────────────────────────────────────────────────────────────

def _psi_one_feature(
    ref: np.ndarray,
    obs: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute PSI for a single feature vector pair."""
    if len(obs) == 0:
        return 0.0

    # Build bin edges from the reference distribution (equal-frequency bins).
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    obs_counts, _ = np.histogram(obs, bins=edges)

    # Laplace smoothing so we never take log(0).
    n_ref = len(ref)
    n_obs = len(obs)
    ref_pct = (ref_counts + 0.5) / (n_ref + 0.5 * bins)
    obs_pct = (obs_counts + 0.5) / (n_obs + 0.5 * bins)

    psi = float(np.sum((obs_pct - ref_pct) * np.log(obs_pct / ref_pct)))
    return abs(psi)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Alert-rate spike detection
# ──────────────────────────────────────────────────────────────────────────────

def alert_rate_report(
    benign_class_id: int,
    window_class_ids: list[int],
    reference_benign_rate: float,
    warning_absolute_rise: float = 0.10,
    critical_absolute_rise: float = 0.20,
) -> dict:
    """Detect a sudden spike in the fraction of non-benign (attack) predictions.

    Parameters
    ----------
    benign_class_id:       integer id of the BENIGN class.
    window_class_ids:      class ids from the recent prediction window.
    reference_benign_rate: proportion of benign traffic in training set.
    warning_absolute_rise: absolute rise in attack rate that triggers WARNING.
    critical_absolute_rise: absolute rise that triggers CRITICAL.
    """
    if not window_class_ids:
        return {
            "available": False,
            "message": "No predictions logged yet.",
            "alert": "none",
        }

    n = len(window_class_ids)
    n_benign  = sum(1 for cid in window_class_ids if cid == benign_class_id)
    n_attacks = n - n_benign

    window_attack_rate    = n_attacks / n
    reference_attack_rate = 1.0 - reference_benign_rate
    absolute_rise         = window_attack_rate - reference_attack_rate

    if absolute_rise >= critical_absolute_rise:
        alert = "critical"
    elif absolute_rise >= warning_absolute_rise:
        alert = "warning"
    else:
        alert = "none"

    return {
        "available": True,
        "n_window": n,
        "n_attacks": n_attacks,
        "n_benign": n_benign,
        "window_attack_rate": float(window_attack_rate),
        "reference_attack_rate": float(reference_attack_rate),
        "absolute_rise": float(absolute_rise),
        "alert": alert,
        "message": (
            f"Attack rate spiked to {window_attack_rate*100:.1f}% "
            f"(+{absolute_rise*100:.1f}% above baseline {reference_attack_rate*100:.1f}%)."
            if alert != "none"
            else f"Attack rate normal: {window_attack_rate*100:.1f}% "
                 f"(baseline {reference_attack_rate*100:.1f}%)."
        ),
    }


_PSI_STABLE = 0.10
_PSI_WARNING = 0.25


def feature_psi_report(
    ref_data: np.ndarray,
    obs_data: np.ndarray,
    feature_names: list[str],
    bins: int = 10,
) -> dict:
    """Compute PSI for every feature between a reference and observation sample.

    Parameters
    ----------
    ref_data:      shape (N_ref, n_features) — training/reference sample.
    obs_data:      shape (N_obs, n_features) — test/window sample.
    feature_names: column names, length n_features.
    bins:          number of equal-frequency bins for the PSI histogram.
    """
    if ref_data.shape[0] == 0 or obs_data.shape[0] == 0:
        return {"available": False, "message": "Insufficient data for PSI."}

    results: list[dict] = []
    for i, name in enumerate(feature_names):
        psi = _psi_one_feature(ref_data[:, i], obs_data[:, i], bins=bins)
        if psi >= _PSI_WARNING:
            level = "critical"
        elif psi >= _PSI_STABLE:
            level = "warning"
        else:
            level = "stable"
        results.append({"feature": name, "psi": round(psi, 5), "level": level})

    # Sort by PSI descending for easy scanning.
    results.sort(key=lambda r: r["psi"], reverse=True)

    n_critical = sum(1 for r in results if r["level"] == "critical")
    n_warning = sum(1 for r in results if r["level"] == "warning")

    return {
        "available": True,
        "n_features": len(feature_names),
        "n_critical": n_critical,
        "n_warning": n_warning,
        "n_stable": len(results) - n_critical - n_warning,
        "features": results,
        "psi_thresholds": {"stable": _PSI_STABLE, "warning": _PSI_WARNING},
    }
