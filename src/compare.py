"""Cross-model aggregation: model x language matrices, heatmaps, decay lines.

    uv run python -m src.compare
"""

from __future__ import annotations

import argparse
import re
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


def _fewshot_comparison(allm: pd.DataFrame, out: Path) -> None:
    """If any experiment has a paired `_Nshot` (few-shot) and its base (zero-shot) run, tabulate
    and plot the zero-shot vs few-shot effect on accuracy / ECE / overconfidence (macro-mean over
    languages). No-op if no few-shot runs are present."""
    agg = allm.groupby("experiment")[["accuracy", "ece", "overconfidence"]].mean()
    pat = re.compile(r"^(.*)_(\d+)shot$")
    pairs = [(m.group(1), e, int(m.group(2)))
             for e in agg.index for m in [pat.match(e)] if m and m.group(1) in agg.index]
    if not pairs:
        return
    rows = []
    for base, fs, k in sorted(pairs):
        b, f = agg.loc[base], agg.loc[fs]
        rows.append({
            "model": base, "n_shots": k,
            "acc_0shot": b.accuracy, "acc_kshot": f.accuracy, "acc_delta": f.accuracy - b.accuracy,
            "ece_0shot": b.ece, "ece_kshot": f.ece, "ece_delta": f.ece - b.ece,
            "overconf_0shot": b.overconfidence, "overconf_kshot": f.overconfidence,
            "overconf_delta": f.overconfidence - b.overconfidence,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "fewshot_vs_zeroshot.csv", index=False)

    labels = [r["model"].replace("_uhura_truthfulqa", "").replace("_afrimmlu", "") for r in rows]
    k_label = f"{df['n_shots'].iloc[0]}-shot"
    x = list(range(len(labels)))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 1.8 * len(labels)), 4), constrained_layout=True)
    for ax, (metric, c0, ck) in zip(axes, [("accuracy", "acc_0shot", "acc_kshot"),
                                           ("ECE", "ece_0shot", "ece_kshot")]):
        ax.bar([xi - w / 2 for xi in x], df[c0].values, w, label="0-shot", color=PALETTE[0])
        ax.bar([xi + w / 2 for xi in x], df[ck].values, w, label=k_label, color=PALETTE[1])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: zero-shot vs {k_label}")
        ax.legend(fontsize=8)
    fig.savefig(out / "fewshot_vs_zeroshot.png")
    plt.close(fig)
    print(f"Few-shot vs zero-shot ({len(rows)} pair(s)) -> {out / 'fewshot_vs_zeroshot.csv'}")


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

    # zero-shot vs few-shot panel (only if paired _Nshot experiments exist)
    _fewshot_comparison(allm, out)

    print(f"Comparison written to {out}")


if __name__ == "__main__":
    main()
