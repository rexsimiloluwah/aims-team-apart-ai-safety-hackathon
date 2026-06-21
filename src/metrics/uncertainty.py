"""Bootstrap confidence intervals for any calibration/selective metric.

Per-language sets are small (Uhura ~hundreds), so always report CIs. Seeds are
derived via sha256 (never Python `hash()`, which is salted per interpreter).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.utils.io import stable_seed


def bootstrap_ci(
    confidences,
    correct,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    key: str = "bootstrap",
) -> dict[str, float]:
    """Percentile bootstrap CI for `metric_fn(confidences, correct)`.

    Returns {point, lo, hi, se}. Resamples items with replacement.
    """
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = c.size
    point = float(metric_fn(c, y)) if n else float("nan")
    if n == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}

    rng = np.random.default_rng(stable_seed(key))
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            stats[b] = float(metric_fn(c[idx], y[idx]))
        except Exception:  # noqa: BLE001 - degenerate resample (e.g. one-class AUROC)
            stats[b] = np.nan
    valid = stats[~np.isnan(stats)]
    if valid.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": lo, "hi": hi, "se": float(valid.std(ddof=1))}
