"""Standard figures for a single experiment (used by analyze.py).

Every figure is labelled with the dataset (AfriMMLU / Uhura-TruthfulQA) via `title_label`.
"""

from __future__ import annotations

from collections import Counter
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.metrics_tables import confidence_correct
from src.metrics import expected_calibration_error, reliability_bins, risk_coverage_curve
from src.utils.plotting import (
    BAR_COLOR,
    PALETTE,
    language_bars,
    radar,
    reliability_diagram,
    risk_coverage_plot,
    set_style,
)


def _suffix(title_label: str) -> str:
    return f"  -  {title_label}" if title_label else ""


def save_reliability_grid(preds: list[dict], out_dir: Path, n_bins: int = 15,
                          title_label: str = "") -> Path:
    """Per-language reliability curves, cleanly spaced (<=3 cols, shared axes,
    one set of outer labels, one shared legend)."""
    set_style()
    langs = sorted({p["language"] for p in preds})
    ncol = min(3, len(langs))
    nrow = ceil(len(langs) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.5 * ncol, 3.5 * nrow),
        squeeze=False, sharex=True, sharey=True, constrained_layout=True,
    )
    for k, lang in enumerate(langs):
        ax = axes[k // ncol][k % ncol]
        conf, corr = confidence_correct(preds, language=lang)
        bins = reliability_bins(conf, corr, n_bins=n_bins)
        ece = expected_calibration_error(conf, corr, n_bins)
        reliability_diagram(bins, ax=ax, title=lang, ece=ece)
        ax.label_outer()  # only outer subplots show tick labels
    for k in range(len(langs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.supxlabel("confidence")
    fig.supylabel("accuracy")
    fig.suptitle(f"Reliability by language{_suffix(title_label)}", fontweight="bold")
    handles = [
        Line2D([0], [0], ls="--", color="0.55", lw=1.2),
        Line2D([0], [0], color=PALETTE[0], lw=2, marker="o", ms=4.5),
        Patch(facecolor=BAR_COLOR, alpha=0.6),
    ]
    fig.legend(handles, ["perfect calibration", "reliability curve (accuracy)", "accuracy per bin"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    out = out_dir / "reliability_by_language.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_risk_coverage(preds: list[dict], out_dir: Path, max_langs: int = 8,
                       title_label: str = "") -> Path:
    set_style()
    langs = sorted({p["language"] for p in preds})[:max_langs]
    curves = {lang: risk_coverage_curve(*confidence_correct(preds, language=lang)) for lang in langs}
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    risk_coverage_plot(curves, ax=ax, title=f"Risk-coverage by language{_suffix(title_label)}")
    out = out_dir / "risk_coverage.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_language_bars(df, out_dir: Path, title_label: str = "") -> list[Path]:
    """Accuracy (vs English baseline) and ECE bar charts from per_language_metrics df."""
    set_style()
    outs = []
    df = df.set_index("language")
    eng = df.loc["eng", "accuracy"] if "eng" in df.index else None
    langs = list(df.index)

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(langs)), 4), constrained_layout=True)
    language_bars(langs, df["accuracy"].values, ax=ax,
                  title=f"Accuracy by language{_suffix(title_label)}", ylabel="accuracy", baseline=eng)
    p1 = out_dir / "accuracy_by_language.png"
    fig.savefig(p1)
    plt.close(fig)
    outs.append(p1)

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(langs)), 4), constrained_layout=True)
    language_bars(langs, df["ece"].values, ax=ax,
                  title=f"ECE by language (higher = worse){_suffix(title_label)}", ylabel="ECE")
    p2 = out_dir / "ece_by_language.png"
    fig.savefig(p2)
    plt.close(fig)
    outs.append(p2)
    return outs


def save_language_radar(df, out_dir: Path, title_label: str = "") -> Path:
    """Radar over languages: accuracy and calibration (1 - ECE) profile for this model."""
    set_style()
    df = df.set_index("language")
    langs = list(df.index)
    series = {
        "accuracy": [float(x) for x in df["accuracy"].tolist()],
        "calibration (1 - ECE)": [max(0.0, 1.0 - float(x)) for x in df["ece"].tolist()],
    }
    fig, ax = plt.subplots(figsize=(5.8, 5.8), subplot_kw={"polar": True}, constrained_layout=True)
    radar(langs, series, ax=ax, title=f"Accuracy vs calibration by language{_suffix(title_label)}")
    out = out_dir / "language_radar.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def save_topic_calibration_radar(preds: list[dict], out_dir: Path, title_label: str = "",
                                 max_topics: int = 8, min_n: int = 15, n_bins: int = 10):
    """Radar of calibration (1 - ECE) by topic, English vs African languages (pooled).

    Needs `domain` on the predictions (Uhura topic / AfriMMLU subject). Returns None if
    there isn't enough per-topic data to make a clean radar (e.g. tiny smoke runs).
    """
    have = [p for p in preds if p.get("domain")]
    if not have:
        return None
    set_style()

    groups: dict[str, list] = {}
    eng = [p for p in have if p["language"] == "eng"]
    afr = [p for p in have if p["language"] != "eng"]
    if eng:
        groups["English"] = eng
    if afr:
        groups["African (pooled)"] = afr
    if not groups:
        groups["all"] = have

    def calib(subset: list[dict]) -> float | None:
        conf, corr = confidence_correct(subset)
        if conf.size < min_n:
            return None
        return max(0.0, 1.0 - expected_calibration_error(conf, corr, n_bins=n_bins))

    topics, series_vals = [], {g: [] for g in groups}
    for topic, _ in Counter(p["domain"] for p in have).most_common():
        vals = {g: calib([p for p in gp if p["domain"] == topic]) for g, gp in groups.items()}
        if all(v is not None for v in vals.values()):
            topics.append(topic)
            for g in groups:
                series_vals[g].append(vals[g])
        if len(topics) >= max_topics:
            break

    if len(topics) < 3:
        return None  # too few well-populated topics for a meaningful radar
    fig, ax = plt.subplots(figsize=(6.6, 6.6), subplot_kw={"polar": True}, constrained_layout=True)
    radar(topics, series_vals, ax=ax, title=f"Calibration (1 - ECE) by topic{_suffix(title_label)}")
    out = out_dir / "topic_calibration_radar.png"
    fig.savefig(out)
    plt.close(fig)
    return out
