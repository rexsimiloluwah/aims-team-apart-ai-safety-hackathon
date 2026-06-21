"""Post-hoc temperature scaling (Guo et al., 2017), fit per language.

A cheap, analysis-only mitigation: rescale the model's confidence to match its
accuracy, without changing any predictions. We only stored per-option
probabilities (a softmax over choices), not raw logits - but softmax is invariant
to an additive constant, so ``log(probs)`` serves as logits:

    softmax(log(p) / T) == softmax(logits / T).

For each language we split items into a seeded calibration / evaluation half, fit
one temperature on the calibration half by minimising NLL over a grid, then report
the ECE reduction on the held-out evaluation half. Temperature scaling never changes
the arg-max, so accuracy is unchanged; only confidence (hence ECE / overconfidence)
moves. Fitting per language matters: the English temperature does not fix Yoruba.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import expected_calibration_error, overconfidence_gap
from src.utils.io import stable_seed

# Mirrors configs/uq/temperature_scaling.yaml (fallback if that file is absent).
DEFAULT_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]


def _scaled_probs(probs: np.ndarray, t: float) -> np.ndarray:
    """softmax(log(probs) / T) == temperature-scaled softmax of the original logits."""
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _items(preds: list[dict], language: str) -> list[tuple[np.ndarray, int]]:
    """(probability vector, true label) for one language's items that carry `probs`."""
    out = []
    for p in preds:
        if p["language"] != language:
            continue
        probs = p.get("probs")
        if not probs:
            continue
        out.append((np.asarray(probs, dtype=float), int(p["answer_index"])))
    return out


def _conf_corr(items: list[tuple[np.ndarray, int]], t: float):
    """Confidence (max scaled prob) and correctness at temperature `t`."""
    conf, corr = [], []
    for probs, y in items:
        sp = _scaled_probs(probs, t)
        pred = int(sp.argmax())  # arg-max is invariant to T; correctness unchanged
        conf.append(float(sp.max()))
        corr.append(1.0 if (0 <= y < sp.size and pred == y) else 0.0)
    return np.asarray(conf, dtype=float), np.asarray(corr, dtype=float)


def fit_temperature(
    items: list[tuple[np.ndarray, int]], grid=DEFAULT_GRID, n_bins: int = 15
) -> float:
    """Temperature on `grid` that minimises calibration-split ECE.

    We optimise ECE rather than NLL: these models are badly miscalibrated and
    low-accuracy, so NLL-optimal temperatures over-flatten - driving already-calibrated
    languages (e.g. English) from over- to under-confident. ECE directly targets what we
    report and stays near T=1 where calibration is already good, correcting only where needed.
    """
    if not items:
        return 1.0

    def cal_ece(t: float) -> float:
        c, y = _conf_corr(items, t)
        return expected_calibration_error(c, y, n_bins)

    return float(min(grid, key=cal_ece))


def temperature_scaling_table(
    preds: list[dict],
    calibration_fraction: float = 0.5,
    grid=DEFAULT_GRID,
    n_bins: int = 15,
    seed: int = 1234,
) -> pd.DataFrame:
    """Per-language temperature scaling: fit T on a seeded calibration half, report
    ECE / overconfidence before vs after on the held-out half. Adds a pooled `ALL` row."""
    langs = sorted({p["language"] for p in preds})
    rows: list[dict] = []
    pooled_before, pooled_after, pooled_corr = [], [], []
    for lang in langs:
        items = _items(preds, lang)
        if len(items) < 4:  # too few to split/fit meaningfully
            continue
        rng = np.random.default_rng(stable_seed(f"tempscale:{lang}:{seed}"))
        order = rng.permutation(len(items))
        n_cal = max(1, int(round(len(items) * calibration_fraction)))
        cal = [items[i] for i in order[:n_cal]]
        evl = [items[i] for i in order[n_cal:]] or cal  # fall back if the tail is empty
        t = fit_temperature(cal, grid, n_bins)
        c0, y0 = _conf_corr(evl, 1.0)  # before (T = 1)
        c1, _ = _conf_corr(evl, t)     # after (correctness identical -> reuse y0)
        rows.append(
            {
                "language": lang,
                "n_cal": len(cal),
                "n_eval": len(evl),
                "temperature": t,
                "ece_before": expected_calibration_error(c0, y0, n_bins),
                "ece_after": expected_calibration_error(c1, y0, n_bins),
                "overconf_before": overconfidence_gap(c0, y0),
                "overconf_after": overconfidence_gap(c1, y0),
            }
        )
        pooled_before.append(c0)
        pooled_after.append(c1)
        pooled_corr.append(y0)

    if not rows:
        return pd.DataFrame()

    cb = np.concatenate(pooled_before)
    ca = np.concatenate(pooled_after)
    yy = np.concatenate(pooled_corr)
    rows.append(
        {
            "language": "ALL",
            "n_cal": sum(r["n_cal"] for r in rows),
            "n_eval": int(yy.size),
            "temperature": float("nan"),  # per-language temperatures differ
            "ece_before": expected_calibration_error(cb, yy, n_bins),
            "ece_after": expected_calibration_error(ca, yy, n_bins),
            "overconf_before": overconfidence_gap(cb, yy),
            "overconf_after": overconfidence_gap(ca, yy),
        }
    )
    df = pd.DataFrame(rows)
    df["ece_reduction"] = df["ece_before"] - df["ece_after"]
    return df
