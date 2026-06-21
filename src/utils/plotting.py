"""Shared plot style + reusable figure primitives. Style is settled in the smoke
notebook, then frozen here so analyze/compare inherit it.

Palette: Okabe-Ito (colourblind-safe), warm tones first.
"""

from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, warm-first
PALETTE = ["#D55E00", "#E69F00", "#CC79A7", "#009E73", "#0072B2", "#56B4E9", "#F0E442", "#000000"]
BAR_COLOR = "#C9CCD1"  # calm neutral grey for per-bin accuracy bars


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
        }
    )


def reliability_diagram(bins: dict, ax=None, title: str = "Reliability",
                        ece: float | None = None, show_bars: bool = True):
    """A proper reliability *curve*: per-bin accuracy (line) over faint bars, vs the
    y=x diagonal. A curve sitting below the diagonal = overconfident. No axis labels
    or legend are set here; the grid adds shared ones."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    centers = bins["centers"]
    acc = bins["bin_acc"]
    counts = bins["bin_count"]
    valid = counts > 0

    ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1.2, zorder=1)  # perfect calibration
    if show_bars and valid.any():
        w = (centers[1] - centers[0]) * 0.92 if len(centers) > 1 else 0.06
        ax.bar(centers[valid], acc[valid], width=w, color=BAR_COLOR, alpha=0.6,
               edgecolor="none", zorder=2)
    if valid.any():
        ax.plot(centers[valid], acc[valid], "-o", color=PALETTE[0], lw=2.0, ms=4.5, zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title if ece is None else f"{title}  (ECE={ece:.2f})")
    return ax


def risk_coverage_plot(curves: dict[str, dict], ax=None, title: str = "Risk-coverage"):
    """curves: {label: risk_coverage_curve(...) dict}."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.2))
    for i, (label, rc) in enumerate(curves.items()):
        ax.plot(rc["coverage"], rc["risk"], color=PALETTE[i % len(PALETTE)], lw=1.9, label=label)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate among answered)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def language_bars(labels, values, ax=None, title="", ylabel="", baseline: float | None = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.55 * len(labels)), 4))
    ax.bar(range(len(labels)), values, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="0.4", lw=1, label="English")
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def radar(categories, series: dict[str, list], ax=None, title: str = "",
          value_range: tuple[float, float] = (0.0, 1.0)):
    """Clean radar/spider plot. `series` = {label: values aligned to `categories`},
    values within `value_range`. One polygon per series."""
    cats = list(categories)
    n = len(cats)
    angles = [k / float(n) * 2 * pi for k in range(n)]
    closed = angles + angles[:1]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.4), subplot_kw={"polar": True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(cats, fontsize=9)
    lo, hi = value_range
    ax.set_ylim(lo, hi)
    ax.set_rlabel_position(180.0 / n)
    ax.tick_params(axis="y", labelsize=7, colors="0.4")
    ax.grid(True, alpha=0.3)
    for i, (label, vals) in enumerate(series.items()):
        v = list(vals) + list(vals)[:1]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(closed, v, color=c, lw=2, label=label)
        ax.fill(closed, v, color=c, alpha=0.08)
    ax.set_title(title, pad=18, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    return ax


def heatmap(matrix, row_labels, col_labels, ax=None, title="", cbar_label="", fmt="{:.2f}"):
    matrix = np.asarray(matrix, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.6 * len(col_labels)), max(4, 0.5 * len(row_labels))))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7)
    ax.figure.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.grid(False)
    return ax
