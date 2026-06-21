"""Cross-model aggregation, SPLIT BY DATASET.

Uhura-TruthfulQA (7 langs, truthfulness MC1) and AfriMMLU (18 langs, knowledge MCQA) have
different language sets and tasks, so per-language figures are produced separately per dataset.
Only a summary (English-vs-African) table spans both.

    uv run python -m src.compare

Writes under _comparison/:
  overview_table.csv                         model × dataset, English-vs-African summary + mitigations
  all_metrics.csv                            per-(experiment, language) raw metrics
  <dataset>/summary.csv                      per-model metrics on that dataset
  <dataset>/{accuracy,ece,overconfidence}_{matrix.csv,heatmap.png}
  <dataset>/accuracy_decay.png               accuracy English→African, per model
  <dataset>/temperature_scaling.png          pooled ECE before vs after, per model (mitigation 1)
  <dataset>/risk_coverage.png                accuracy vs coverage, per model (mitigation 2)
  fewshot_vs_zeroshot.{csv,png}              if any _Nshot runs exist
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.metrics_tables import confidence_correct, per_language_metrics
from src.metrics import (
    accuracy,
    expected_calibration_error,
    overconfidence_gap,
    risk_coverage_curve,
)
from src.utils.io import load_predictions
from src.utils.plotting import PALETTE, heatmap, radar, set_style

DATASETS = ["uhura_truthfulqa", "afrimmlu"]
NONAFR = {"eng", "fra"}  # reference (non-African) languages for the decay/gap summary
# rough resource ordering for the decay x-axis; unknown langs appended alphabetically
LANG_ORDER = ["eng", "fra", "swa", "yor", "hau", "zul", "xho", "ibo", "amh", "nso",
              "sot", "twi", "kin", "lug", "lin", "orm", "sna", "wol", "ewe"]


def discover_experiments(artifacts_dir: Path) -> list[str]:
    out = []
    for d in sorted(artifacts_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and (d / "results" / "predictions.jsonl").exists():
            out.append(d.name)
    return out


def parse_exp(name: str) -> tuple[str, str, int]:
    """(model, dataset, n_shots) from an experiment dir name like
    'qwen3_4b_instruct_afrimmlu' or 'gemma4_12b_uhura_truthfulqa_5shot'."""
    n = 0
    m = re.search(r"_(\d+)shot$", name)
    if m:
        n = int(m.group(1))
        name = name[: m.start()]
    for ds in DATASETS:
        if name.endswith("_" + ds):
            return name[: -(len(ds) + 1)], ds, n
    return name, "unknown", n


def _ordered(langs: list[str]) -> list[str]:
    known = [lang for lang in LANG_ORDER if lang in langs]
    return known + sorted(set(langs) - set(known))


def _temp_after(art: Path, exp: str) -> tuple[float, float] | None:
    """(pooled ECE before, after) from an experiment's temperature_scaling.csv, or None."""
    p = art / exp / "results" / "temperature_scaling.csv"
    if not p.exists():
        return None
    row = pd.read_csv(p).query("language == 'ALL'")
    if row.empty:
        return None
    return float(row.ece_before.iloc[0]), float(row.ece_after.iloc[0])


def _heatmaps_and_decay(m: pd.DataFrame, ds: str, dout: Path) -> None:
    for metric, fmt in [("accuracy", "{:.2f}"), ("ece", "{:.3f}"), ("overconfidence", "{:.2f}")]:
        piv = m.pivot_table(index="model", columns="language", values=metric)
        cols = _ordered(list(piv.columns))
        piv = piv[cols]
        piv.to_csv(dout / f"{metric}_matrix.csv")
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols)), max(3, 0.6 * len(piv.index))))
        heatmap(piv.values, list(piv.index), cols, ax=ax,
                title=f"{ds}: {metric} (model × language)", cbar_label=metric, fmt=fmt)
        fig.savefig(dout / f"{metric}_heatmap.png")
        plt.close(fig)

    acc = m.pivot_table(index="model", columns="language", values="accuracy")
    cols = _ordered(list(acc.columns))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols)), 4), constrained_layout=True)
    for i, mod in enumerate(acc.index):
        ax.plot(cols, acc.loc[mod, cols].values, marker="o", lw=1.6,
                color=PALETTE[i % len(PALETTE)], label=mod)
    ax.set_ylabel("accuracy")
    ax.set_xlabel("language (≈ high → low resource)")
    ax.set_title(f"{ds}: factuality decay across languages")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7)
    fig.savefig(dout / "accuracy_decay.png")
    plt.close(fig)


def _temperature_fig(ds_exps: list[str], models: dict, art: Path, ds: str, dout: Path) -> None:
    rows = [(models[e], *t) for e in ds_exps if (t := _temp_after(art, e))]
    if not rows:
        return
    rows.sort()
    labels = [r[0] for r in rows]
    before = [r[1] for r in rows]
    after = [r[2] for r in rows]
    x = list(range(len(labels)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(labels)), 4), constrained_layout=True)
    ax.bar([xi - w / 2 for xi in x], before, w, label="before (T=1)", color=PALETTE[0])
    ax.bar([xi + w / 2 for xi in x], after, w, label="after temp-scaling", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("pooled ECE")
    ax.set_title(f"{ds}: temperature scaling (ECE before vs after)")
    ax.legend(fontsize=8)
    fig.savefig(dout / "temperature_scaling.png")
    plt.close(fig)


def _risk_coverage_fig(ds_exps: list[str], models: dict, art: Path, ds: str, dout: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    plotted = False
    for i, e in enumerate(sorted(ds_exps, key=lambda x: models[x])):
        conf, corr = confidence_correct(load_predictions(art / e / "results"))
        rc = risk_coverage_curve(conf, corr)
        if rc["coverage"].size == 0:
            continue
        ax.plot(rc["coverage"], 1.0 - rc["risk"], lw=1.6,
                color=PALETTE[i % len(PALETTE)], label=models[e])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("coverage (fraction answered, most-confident first)")
    ax.set_ylabel("accuracy on answered")
    ax.set_title(f"{ds}: risk–coverage (abstention)")
    ax.legend(fontsize=8)
    fig.savefig(dout / "risk_coverage.png")
    plt.close(fig)


def _language_radar(m: pd.DataFrame, ds: str, dout: Path) -> None:
    """Headline radar: accuracy across languages, one polygon per model."""
    acc = m.pivot_table(index="model", columns="language", values="accuracy")
    cols = _ordered(list(acc.columns))
    series = {mod: [float(x) for x in acc.loc[mod, cols].fillna(0.0).tolist()] for mod in acc.index}
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"polar": True}, constrained_layout=True)
    radar(cols, series, ax=ax, title=f"{ds}: accuracy across languages (by model)")
    fig.savefig(dout / "accuracy_radar.png")
    plt.close(fig)


def _per_category_metrics(preds: list[dict]) -> pd.DataFrame:
    """Accuracy / ECE / overconfidence per question category (domain), pooled over languages."""
    cats = sorted({p.get("domain") for p in preds if p.get("domain")})
    rows = []
    for c in cats:
        conf, corr = confidence_correct(preds, domain=c)
        if corr.size == 0:
            continue
        rows.append({"category": c, "n": int(corr.size), "accuracy": accuracy(corr),
                     "ece": expected_calibration_error(conf, corr),
                     "overconfidence": overconfidence_gap(conf, corr)})
    return pd.DataFrame(rows)


def _category_figs(ds_exps: list[str], models: dict, art: Path, ds: str, dout: Path,
                   top_k: int = 12) -> None:
    """Cross-model comparison across question categories (Uhura topics / AfriMMLU subjects):
    full CSV + accuracy/ECE heatmaps + an accuracy radar over the top-K most-frequent categories."""
    frames = []
    for e in ds_exps:
        d = _per_category_metrics(load_predictions(art / e / "results"))
        if not d.empty:
            frames.append(d.assign(model=models[e]))
    if not frames:
        return
    cat = pd.concat(frames, ignore_index=True)
    cat.to_csv(dout / "category_metrics.csv", index=False)
    top = cat.groupby("category")["n"].sum().sort_values(ascending=False).head(top_k).index.tolist()
    sub = cat[cat.category.isin(top)]
    for metric, fmt in [("accuracy", "{:.2f}"), ("ece", "{:.3f}")]:
        piv = sub.pivot_table(index="model", columns="category", values=metric).reindex(columns=top)
        fig, ax = plt.subplots(figsize=(max(6, 0.85 * len(top)), max(3, 0.6 * len(piv.index))))
        heatmap(piv.values, list(piv.index), top, ax=ax,
                title=f"{ds}: {metric} by category (top {len(top)})", cbar_label=metric, fmt=fmt)
        fig.savefig(dout / f"category_{metric}_heatmap.png")
        plt.close(fig)
    accp = sub.pivot_table(index="model", columns="category", values="accuracy").reindex(columns=top)
    series = {mod: [float(x) for x in accp.loc[mod, top].fillna(0.0).tolist()] for mod in accp.index}
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, constrained_layout=True)
    radar(top, series, ax=ax, title=f"{ds}: accuracy by category (top {len(top)})")
    fig.savefig(dout / "category_accuracy_radar.png")
    plt.close(fig)


def _overview_table(recs: dict, meta: dict, art: Path, out: Path) -> None:
    rows = []
    for e, (model, ds, n) in meta.items():
        if n != 0 or ds == "unknown":
            continue
        df = recs[e]
        eng = df[df.language == "eng"]
        afr = df[~df.language.isin(NONAFR)]
        ts = _temp_after(art, e)
        rows.append({
            "model": model, "dataset": ds,
            "acc_eng": eng.accuracy.mean(), "acc_afr": afr.accuracy.mean(),
            "ece_eng": eng.ece.mean(), "ece_afr": afr.ece.mean(),
            "overconf_afr": afr.overconfidence.mean(),
            "ece_after_tempscale": ts[1] if ts else float("nan"),
            "auarc": df.auarc.mean(),
        })
    if rows:
        pd.DataFrame(rows).sort_values(["dataset", "model"]).round(4).to_csv(
            out / "overview_table.csv", index=False)


def _fewshot_comparison(recs: dict, meta: dict, out: Path) -> None:
    agg = {e: recs[e][["accuracy", "ece", "overconfidence"]].mean() for e in recs}
    rows = []
    for e, (model, ds, n) in meta.items():
        base = f"{model}_{ds}"
        if n and base in recs:
            b, f = agg[base], agg[e]
            rows.append({
                "model": model, "dataset": ds, "n_shots": n,
                "acc_0shot": b.accuracy, "acc_kshot": f.accuracy, "acc_delta": f.accuracy - b.accuracy,
                "ece_0shot": b.ece, "ece_kshot": f.ece, "ece_delta": f.ece - b.ece,
                "overconf_0shot": b.overconfidence, "overconf_kshot": f.overconfidence,
                "overconf_delta": f.overconfidence - b.overconfidence,
            })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["dataset", "model"])
    df.to_csv(out / "fewshot_vs_zeroshot.csv", index=False)
    labels = [f"{r['model']} ({r['dataset'].split('_')[0]})" for r in df.to_dict("records")]
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
    ap = argparse.ArgumentParser(description="Aggregate experiments under artifacts/, split by dataset.")
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
    meta = {e: parse_exp(e) for e in exps}
    recs = {e: per_language_metrics(load_predictions(art / e / "results"), bootstrap_n=0) for e in exps}
    print("Experiments:", {e: f"{meta[e][1]}{'+'+str(meta[e][2])+'shot' if meta[e][2] else ''}" for e in exps})

    # raw dump + cross-dataset overview
    allm = pd.concat([recs[e].assign(experiment=e, model=meta[e][0], dataset=meta[e][1],
                                     n_shots=meta[e][2]) for e in exps], ignore_index=True)
    allm.to_csv(out / "all_metrics.csv", index=False)
    _overview_table(recs, meta, art, out)

    set_style()
    for ds in DATASETS:
        ds_exps = [e for e in exps if meta[e][1] == ds and meta[e][2] == 0]  # base (0-shot) only
        if not ds_exps:
            continue
        dout = out / ds
        dout.mkdir(parents=True, exist_ok=True)
        models = {e: meta[e][0] for e in ds_exps}
        m = pd.concat([recs[e].assign(model=models[e]) for e in ds_exps], ignore_index=True)
        m.groupby("model")[["accuracy", "ece", "overconfidence", "brier", "auroc", "auarc"]] \
            .mean().round(4).to_csv(dout / "summary.csv")
        _heatmaps_and_decay(m, ds, dout)
        _language_radar(m, ds, dout)
        _temperature_fig(ds_exps, models, art, ds, dout)
        _risk_coverage_fig(ds_exps, models, art, ds, dout)
        _category_figs(ds_exps, models, art, ds, dout)
        print(f"  {ds}: {len(ds_exps)} models -> {dout}")

    _fewshot_comparison(recs, meta, out)
    print(f"Comparison written to {out}")


if __name__ == "__main__":
    main()
