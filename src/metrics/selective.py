"""Selective prediction / abstention metrics - the lead mitigation.

Policy: answer if confidence >= threshold, else abstain. Higher confidence = answer first.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # noqa: BLE001 - sklearn optional at import time
    roc_auc_score = None


def _arrays(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.shape != y.shape:
        raise ValueError(f"shape mismatch: {c.shape} vs {y.shape}")
    return c, y


def auroc_error_detection(confidences, correct) -> float:
    """AUROC of confidence predicting correctness. NaN if only one class present."""
    c, y = _arrays(confidences, correct)
    if c.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    if roc_auc_score is None:
        raise ImportError("scikit-learn is required for auroc_error_detection")
    return float(roc_auc_score(y, c))


def risk_coverage_curve(confidences, correct) -> dict[str, np.ndarray]:
    """Sort by confidence desc; coverage = k/n answered, risk = error rate among them."""
    c, y = _arrays(confidences, correct)
    n = c.size
    if n == 0:
        return {"coverage": np.array([]), "risk": np.array([]), "aurc": float("nan")}
    order = np.argsort(-c, kind="stable")  # most confident first
    y_sorted = y[order]
    cum_correct = np.cumsum(y_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - (cum_correct / k)  # error rate among the top-k answered
    # np.trapz was removed in numpy 2.0 (renamed trapezoid); support both
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(_trapz(risk, coverage))
    return {"coverage": coverage, "risk": risk, "aurc": aurc}


def threshold_for_coverage(confidences, target_coverage: float) -> float:
    """Confidence threshold that answers ~`target_coverage` of items (answer if conf>=t)."""
    c = np.asarray(confidences, dtype=float)
    n = c.size
    if n == 0:
        return float("nan")
    if target_coverage >= 1.0:
        return float("-inf")  # answer everything
    if target_coverage <= 0.0:
        return float("inf")  # answer nothing
    k_answer = max(1, int(np.floor(target_coverage * n)))
    sorted_desc = np.sort(c)[::-1]
    return float(sorted_desc[k_answer - 1])


def accuracy_at_coverage(confidences, correct, target_coverage: float) -> dict[str, float]:
    """Accuracy among the top-`coverage` most-confident items."""
    c, y = _arrays(confidences, correct)
    n = c.size
    if n == 0:
        return {"coverage": float("nan"), "accuracy": float("nan"), "n_answered": 0}
    k = max(1, int(np.floor(min(1.0, max(0.0, target_coverage)) * n)))
    order = np.argsort(-c, kind="stable")[:k]
    return {
        "coverage": k / n,
        "accuracy": float(y[order].mean()),
        "n_answered": int(k),
    }


def errors_removed_at_threshold(confidences, correct, threshold: float) -> dict[str, float]:
    """At an abstention threshold: coverage, answered accuracy, fraction of errors removed."""
    c, y = _arrays(confidences, correct)
    n = c.size
    answered = c >= threshold
    n_ans = int(answered.sum())
    total_errors = int((y == 0).sum())
    errors_answered = int(((y == 0) & answered).sum())
    errors_removed = total_errors - errors_answered
    return {
        "threshold": float(threshold),
        "coverage": (n_ans / n) if n else float("nan"),
        "answered_accuracy": (float(y[answered].mean()) if n_ans else float("nan")),
        "errors_removed": errors_removed,
        "frac_errors_removed": (errors_removed / total_errors) if total_errors else float("nan"),
    }


def build_deployment_card(
    per_language: dict[str, tuple],
    coverage_targets: list[float],
) -> list[dict]:
    """Per-language abstention thresholds for each target coverage.

    `per_language`: {lang: (confidences, correct)}. Returns rows for a table:
    one per (language, target_coverage) with threshold, realized coverage,
    answered accuracy, and errors removed.
    """
    rows: list[dict] = []
    for lang, (conf, corr) in per_language.items():
        base_acc = float(np.asarray(corr, dtype=float).mean()) if len(corr) else float("nan")
        for tc in coverage_targets:
            thr = threshold_for_coverage(conf, tc)
            stats = errors_removed_at_threshold(conf, corr, thr)
            rows.append(
                {
                    "language": lang,
                    "target_coverage": tc,
                    "threshold": thr,
                    "realized_coverage": stats["coverage"],
                    "base_accuracy": base_acc,
                    "answered_accuracy": stats["answered_accuracy"],
                    "frac_errors_removed": stats["frac_errors_removed"],
                }
            )
    return rows
