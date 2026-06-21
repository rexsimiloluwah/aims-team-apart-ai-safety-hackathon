"""Turn prediction records (dicts from predictions.jsonl) into metric tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import (
    accuracy,
    auarc,
    auroc_error_detection,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    overconfidence_gap,
)


def confidence_correct(
    preds: list[dict],
    language: str | None = None,
    domain: str | None = None,
    use_verbalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    conf, corr = [], []
    for p in preds:
        if language is not None and p["language"] != language:
            continue
        if domain is not None and p.get("domain") != domain:
            continue
        c = p.get("verbalized_confidence") if use_verbalized else p.get("confidence")
        if c is None:
            continue
        conf.append(float(c))
        corr.append(1.0 if p["correct"] else 0.0)
    return np.asarray(conf, dtype=float), np.asarray(corr, dtype=float)


def per_language_metrics(
    preds: list[dict], n_bins: int = 15, bootstrap_n: int = 0
) -> pd.DataFrame:
    langs = sorted({p["language"] for p in preds})
    rows = []
    for lang in langs:
        conf, corr = confidence_correct(preds, language=lang)
        row = {
            "language": lang,
            "n": int(corr.size),
            "accuracy": accuracy(corr),
            "mean_conf": float(conf.mean()) if conf.size else float("nan"),
            "ece": expected_calibration_error(conf, corr, n_bins),
            "brier": brier_score(conf, corr),
            "overconfidence": overconfidence_gap(conf, corr),
            "auroc": auroc_error_detection(conf, corr),
            "auarc": auarc(conf, corr),
        }
        if bootstrap_n > 0 and corr.size > 0:
            ci_ece = bootstrap_ci(
                conf, corr,
                lambda c, y: expected_calibration_error(c, y, n_bins),
                n_boot=bootstrap_n, key=f"ece-{lang}",
            )
            ci_acc = bootstrap_ci(
                conf, corr, lambda c, y: accuracy(y), n_boot=bootstrap_n, key=f"acc-{lang}"
            )
            row.update(
                {
                    "ece_lo": ci_ece["lo"], "ece_hi": ci_ece["hi"],
                    "acc_lo": ci_acc["lo"], "acc_hi": ci_acc["hi"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def per_domain_accuracy(preds: list[dict]) -> pd.DataFrame:
    rows = []
    keys = sorted(
        {(p["language"], p.get("domain")) for p in preds if p.get("domain") is not None}
    )
    for lang, dom in keys:
        _, corr = confidence_correct(preds, language=lang, domain=dom)
        rows.append(
            {"language": lang, "domain": dom, "n": int(corr.size), "accuracy": accuracy(corr)}
        )
    return pd.DataFrame(rows)


def logit_vs_verbalized(preds: list[dict], n_bins: int = 15) -> pd.DataFrame:
    """Per-language ECE/overconfidence: internal (logit/self-consistency) vs verbalized."""
    have_verb = [p for p in preds if p.get("verbalized_confidence") is not None]
    if not have_verb:
        return pd.DataFrame()
    rows = []
    for lang in sorted({p["language"] for p in have_verb}):
        ci, yi = confidence_correct(have_verb, language=lang, use_verbalized=False)
        cv, yv = confidence_correct(have_verb, language=lang, use_verbalized=True)
        rows.append(
            {
                "language": lang,
                "n": int(yi.size),
                "ece_internal": expected_calibration_error(ci, yi, n_bins),
                "ece_verbalized": expected_calibration_error(cv, yv, n_bins),
                "overconf_internal": overconfidence_gap(ci, yi),
                "overconf_verbalized": overconfidence_gap(cv, yv),
            }
        )
    return pd.DataFrame(rows)


def _measure_arrays(preds: list[dict], language: str | None = None):
    """(confidence, entropy, margin, correct) arrays for one language (or all)."""
    conf, ent, marg, corr = [], [], [], []
    for p in preds:
        if language is not None and p["language"] != language:
            continue
        if p.get("confidence") is None:
            continue
        conf.append(float(p["confidence"]))
        ent.append(float(p.get("entropy", float("nan"))))
        marg.append(float(p.get("margin", float("nan"))))
        corr.append(1.0 if p["correct"] else 0.0)
    return (np.asarray(conf), np.asarray(ent), np.asarray(marg), np.asarray(corr))


def _safe_auroc(score: np.ndarray, corr: np.ndarray) -> float:
    """auroc_error_detection on finite entries only (entropy can be NaN for unparsed items)."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(corr, dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return float("nan")
    return auroc_error_detection(s[m], y[m])


def uncertainty_measure_comparison(preds: list[dict]) -> pd.DataFrame:
    """Error-detection AUROC of each uncertainty signal (max-prob, entropy, margin), per language
    plus a pooled ALL row. Answers 'which signal best ranks correct vs incorrect?' (cf. Tomani
    et al. 2024, Table 1). Entropy is negated so higher = more likely correct, matching the others.
    """
    langs = sorted({p["language"] for p in preds})
    rows = []
    for lang in [*langs, "ALL"]:
        conf, ent, marg, corr = _measure_arrays(preds, None if lang == "ALL" else lang)
        if corr.size == 0:
            continue
        rows.append(
            {
                "language": lang,
                "n": int(corr.size),
                "auroc_maxprob": _safe_auroc(conf, corr),
                "auroc_entropy": _safe_auroc(-ent, corr),
                "auroc_margin": _safe_auroc(marg, corr),
            }
        )
    return pd.DataFrame(rows)
