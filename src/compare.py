"""Cross-model aggregation: model x language matrices, heatmaps, decay lines.

    uv run python -m src.compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.metrics_tables import per_language_metrics
from src.utils.io import load_predictions
from src.utils.plotting import PALETTE, heatmap, radar, set_style

# rough resource ordering for the decay x-axis; unknown langs appended alphabetically
LANG_ORDER = ["eng", "fra", "swa", "yor", "hau", "zul", "xho", "ibo", "amh", "nso",
              "sot", "twi", "kin", "lug", "lin", "orm", "sna", "wol", "ewe"]


def discover_experiments(artifacts_dir: Path) -> list[str]:
    out = []
    for d in sorted(artifacts_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "results" / "predictions.jsonl").exists():
            out.append(d.name)
    return out


def _ordered(langs: list[str]) -> list[str]:
    known = [lang for lang in LANG_ORDER if lang in langs]
    rest = sorted(set(langs) - set(known))
    return known + rest


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate all experiments under artifacts/.")
    ap.add_argument("--artifacts-dir", default="./artifacts")
    ap.add_argument("--out", default="./artifacts/_comparison")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exps = discover_experiments(art)
    if not exps:
        print(f"No experiments found under {art}")
        return
    print("Experiments:", exps)

    frames = []
    for e in exps:
        preds = load_predictions(art / e / "results")
        df = per_language_metrics(preds, bootstrap_n=0)
        df["experiment"] = e
        frames.append(df)
    allm = pd.concat(frames, ignore_index=True)
    allm.to_csv(out / "all_metrics.csv", index=False)

    set_style()
    for metric, fmt in [("accuracy", "{:.2f}"), ("ece", "{:.3f}"), ("overconfidence", "{:.2f}")]:
        pivot = allm.pivot_table(index="experiment", columns="language", values=metric)
        cols = _ordered(list(pivot.columns))
        pivot = pivot[cols]
        pivot.to_csv(out / f"{metric}_matrix.csv")
        fig, ax = plt.subplots(
            figsize=(max(6, 0.6 * len(cols)), max(3, 0.6 * len(pivot.index)))
        )
        heatmap(pivot.values, list(pivot.index), cols, ax=ax,
                title=f"{metric} (model × language)", cbar_label=metric, fmt=fmt)
        fig.savefig(out / f"{metric}_heatmap.png")
        plt.close(fig)

    # accuracy decay lines (one per model), x ordered by rough resource level
    acc = allm.pivot_table(index="experiment", columns="language", values="accuracy")
    cols = _ordered(list(acc.columns))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols)), 4))
    for i, exp in enumerate(acc.index):
        ax.plot(cols, acc.loc[exp, cols].values, marker="o", lw=1.6,
                color=PALETTE[i % len(PALETTE)], label=exp)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("language (≈ high → low resource)")
    ax.set_title("Factuality decay across languages")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7)
    fig.savefig(out / "accuracy_decay.png")
    plt.close(fig)

    # radar: accuracy across languages, one polygon per model
    series = {exp: [float(x) for x in acc.loc[exp, cols].fillna(0.0).tolist()] for exp in acc.index}
    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={"polar": True}, constrained_layout=True)
    radar(cols, series, ax=ax, title="Accuracy across languages (by model)")
    fig.savefig(out / "accuracy_radar.png")
    plt.close(fig)

    print(f"Comparison written to {out}")


if __name__ == "__main__":
    main()
