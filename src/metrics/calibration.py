"""Calibration metrics. `confidences` = model's prob it is correct (max option prob);
`correct` = 0/1 correctness. All accept array-likes and return floats / arrays.
"""

from __future__ import annotations

import numpy as np


def _arrays(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.shape != y.shape:
        raise ValueError(f"shape mismatch: confidences {c.shape} vs correct {y.shape}")
    return c, y


def accuracy(correct) -> float:
    y = np.asarray(correct, dtype=float)
    return float(y.mean()) if y.size else float("nan")


def overconfidence_gap(confidences, correct) -> float:
    """mean(confidence) - accuracy. Positive => overconfident."""
    c, y = _arrays(confidences, correct)
    if c.size == 0:
        return float("nan")
    return float(c.mean() - y.mean())


def brier_score(confidences, correct) -> float:
    """Mean squared error between confidence and correctness (binary Brier)."""
    c, y = _arrays(confidences, correct)
    if c.size == 0:
        return float("nan")
    return float(np.mean((c - y) ** 2))


def reliability_bins(confidences, correct, n_bins: int = 15) -> dict[str, np.ndarray]:
    """Equal-width bins over [0,1]. Returns per-bin centers, accuracy, confidence, counts."""
    c, y = _arrays(confidences, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    # bin index in [0, n_bins-1]; right-closed so 1.0 lands in the last bin
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_acc[b] = float(y[mask].mean())
            bin_conf[b] = float(c[mask].mean())
    return {
        "centers": centers,
        "bin_acc": bin_acc,
        "bin_conf": bin_conf,
        "bin_count": bin_count,
        "edges": edges,
    }


def expected_calibration_error(confidences, correct, n_bins: int = 15) -> float:
    """ECE: sum over bins of (count/N) * |accuracy - confidence|."""
    c, _ = _arrays(confidences, correct)
    if c.size == 0:
        return float("nan")
    b = reliability_bins(confidences, correct, n_bins=n_bins)
    n = c.size
    ece = 0.0
    for acc, conf, cnt in zip(b["bin_acc"], b["bin_conf"], b["bin_count"]):
        if cnt > 0:
            ece += (cnt / n) * abs(acc - conf)
    return float(ece)
