"""Three publication-quality slide plots following the workshop design system.

Plot A: grouped bars, ECE English vs African (Uhura-TruthfulQA)  -> figs/plot_a_ece_gap.png
Plot B: dumbbell, accuracy collapse English->African (AfriMMLU)  -> figs/plot_b_accuracy_collapse.png
Plot C: data-sensitivity, ECE after temp scaling vs calib size (AfriMMLU) -> figs/plot_c_calibration_data.png
        (uses revision/sensitivity_results.csv, produced by sensitivity_experiment.py)
"""
from __future__ import annotations
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(REPO, "figs")
os.makedirs(FIGS, exist_ok=True)

# --- palette ---
GOLD, TEAL, CORAL, VIOLET = "#FDD633", "#35C6BE", "#FF7E9D", "#B38BE0"
INK, MUTED, GRID = "#1A1A1A", "#6B6B6B", "#E3E3E3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})

MODELS = ["Qwen3-4B", "Gemma-4-12B", "Aya-Expanse-8B", "AfroLlama-V1"]


def base_ax():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=10.5, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    return fig, ax


def title(ax, text):
    ax.set_title(text, fontsize=15, fontweight="bold", color=INK, loc="left", pad=14)


def save(fig, name):
    p = os.path.join(FIGS, name)
    fig.savefig(p, transparent=True, bbox_inches="tight", pad_inches=0.15, dpi=200)
    plt.close(fig)
    return os.path.abspath(p)


# --------------------------------------------------------------------------- #
# Plot A: grouped bars, ECE English vs African (Uhura-TruthfulQA)
# --------------------------------------------------------------------------- #
def plot_a():
    eng = [0.07, 0.21, 0.08, 0.18]
    afr = [0.17, 0.26, 0.20, 0.13]
    fig, ax = base_ax()
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    x = np.arange(len(MODELS))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, eng, w, color=TEAL, linewidth=0, label="English")
    b2 = ax.bar(x + w / 2 + 0.01, afr, w, color=CORAL, linewidth=0, label="African")
    for bars, vals in ((b1, eng), (b2, afr)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.006, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11.5, color=MUTED)
    title(ax, "Calibration degrades from English to African languages")
    # AfroLlama exception annotation (group index 3, African < English)
    ax.annotate("exception", xy=(3 + w / 2 + 0.01, 0.13), xytext=(3 - 0.05, 0.225),
                fontsize=9.5, color=MUTED, ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, alpha=0.7))
    ax.legend(frameon=False, fontsize=10.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.12))
    return save(fig, "plot_a_ece_gap.png")


# --------------------------------------------------------------------------- #
# Plot B: dumbbell, accuracy collapse English -> African (AfriMMLU)
# --------------------------------------------------------------------------- #
def plot_b():
    data = {"Qwen3-4B": (0.53, 0.30), "Gemma-4-12B": (0.30, 0.27),
            "Aya-Expanse-8B": (0.52, 0.29), "AfroLlama-V1": (0.36, 0.28)}
    fig, ax = base_ax()
    ax.grid(False)  # design system: no vertical gridlines; value labels carry the numbers
    ypos = np.arange(len(MODELS))
    for y, m in zip(ypos, MODELS):
        e, a = data[m]
        ax.plot([e, a], [y, y], color=GRID, linewidth=2.5, zorder=1, solid_capstyle="round")
        ax.scatter([e], [y], s=130, color=TEAL, zorder=3, edgecolors="none")
        ax.scatter([a], [y], s=130, color=CORAL, zorder=3, edgecolors="none")
        # English label left of its dot, African right of its dot
        lo, hi = min(e, a), max(e, a)
        ax.text(e - 0.015 if e <= a else e + 0.015, y, f"{e:.2f}",
                ha="right" if e <= a else "left", va="center", fontsize=10, color=INK)
        ax.text(a + 0.015 if a >= e else a - 0.015, y, f"{a:.2f}",
                ha="left" if a >= e else "right", va="center", fontsize=10, color=INK)
    ax.axvline(0.25, color=VIOLET, linewidth=1.3, linestyle=(0, (4, 3)), zorder=0)
    ax.text(0.25, -0.55, "chance", color=VIOLET, fontsize=9.5, ha="center", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels(MODELS)
    ax.invert_yaxis()  # first model (Qwen) on top
    ax.set_xlim(0, 0.60)
    ax.set_ylim(len(MODELS) - 0.3, -0.7)
    ax.set_xlabel("Accuracy (AfriMMLU)", fontsize=11.5, color=MUTED)
    title(ax, "Strong models fall toward chance in African languages")
    handles = [Line2D([0], [0], marker="o", ls="", mfc=TEAL, mec="none", ms=10, label="English"),
               Line2D([0], [0], marker="o", ls="", mfc=CORAL, mec="none", ms=10, label="African")]
    ax.legend(handles=handles, frameon=False, fontsize=10.5, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.12))
    return save(fig, "plot_b_accuracy_collapse.png")


# --------------------------------------------------------------------------- #
# Plot C: data-sensitivity line (AfriMMLU) from saved sensitivity results
# --------------------------------------------------------------------------- #
def plot_c():
    path = os.path.join(REPO, "revision", "sensitivity_results.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["dataset"] == "afrimmlu"]
    order = ["10", "25", "50", "100", "full"]
    colors = {"Qwen3-4B": TEAL, "Gemma-4-12B": GOLD, "Aya-Expanse-8B": CORAL, "AfroLlama-V1": VIOLET}
    # full calibration size (items/lang) inferred from pct: ncal=10 at 2% -> total 500 -> half 250
    full_size = 250.0
    xmap = {"10": 10, "25": 25, "50": 50, "100": 100, "full": full_size}

    fig, ax = base_ax()
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    # plateau highlight
    ax.axvspan(25, 50, color=MUTED, alpha=0.06, zorder=0)
    for m in MODELS:
        mr = {r["ncal"]: r for r in rows if r["model"] == m}
        xs = [xmap[k] for k in order]
        ys = np.array([float(mr[k]["mean_ece"]) for k in order])
        sd = np.array([float(mr[k]["std_ece"]) for k in order])
        ax.fill_between(xs, ys - sd, ys + sd, color=colors[m], alpha=0.25, linewidth=0, zorder=1)
        ax.plot(xs, ys, "-o", color=colors[m], lw=2.2, ms=5, label=m, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, full_size])
    ax.set_xticklabels(["10", "25", "50", "100", "full"])
    ax.minorticks_off()
    ax.set_xlabel("Calibration examples per language (log scale)", fontsize=11.5, color=MUTED)
    ax.set_ylabel("ECE after temperature scaling", fontsize=11.5, color=MUTED)
    ymax = ax.get_ylim()[1]
    ax.text(35, ymax * 0.97, "most of the benefit by 25 to 50", color=MUTED, fontsize=9,
            ha="center", va="top")
    title(ax, "A small calibration set recovers most of the benefit")
    ax.legend(frameon=False, fontsize=10.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    return save(fig, "plot_c_calibration_data.png")


if __name__ == "__main__":
    pa, pb, pc = plot_a(), plot_b(), plot_c()
    print("Plot A:", pa)
    print("Plot B:", pb)
    print("Plot C (data-sensitivity version):", pc)
