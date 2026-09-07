"""Calibration-data sensitivity of per-language temperature scaling.

Reviewer point: temperature scaling is the best mitigation but needs a labelled
in-language calibration split, the scarce resource. Question: how little
calibration data does it need?

Method (pure post-hoc, reuses the paper's exact TS code): for each language we
reproduce the paper's seeded 50% calibration/eval split. On the SAME fixed eval
half we evaluate ECE after fitting T* on calibration subsets of size
ncal in {10,25,50,100,full}. For ncal<full we draw 30 without-replacement
subsamples from the calibration half, fit T* on each (same grid, ECE objective),
and evaluate on the fixed eval half. We macro-average per-language ECE over the
African languages (non-English) and report mean +/- std across the 30 subsamples.
The ncal=full point reuses the whole calibration half (single value, std 0) and
must reproduce the paper's numbers.

Outputs: report/figures/sensitivity.png and revision/sensitivity_results.csv
"""
from __future__ import annotations
import os, sys, json
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.metrics.temperature import _items, fit_temperature, _conf_corr, DEFAULT_GRID
from src.metrics import expected_calibration_error
from src.utils.io import stable_seed

MODELS = {"qwen3_4b_instruct": "Qwen3-4B", "gemma4_12b": "Gemma-4-12B",
          "aya_expanse_8b": "Aya-Expanse-8B", "afrollama_v1": "AfroLlama-V1"}
DATASETS = {"uhura_truthfulqa": "Uhura-TruthfulQA", "afrimmlu": "AfriMMLU"}
NCALS = [10, 25, 50, 100, "full"]
N_SUB = 30
N_BINS = 15


def load_preds(path):
    return [json.loads(l) for l in open(path)]


def split(items, lang):
    """Reproduce the paper's seeded 50% calibration/eval split (seed 1234)."""
    rng = np.random.default_rng(stable_seed(f"tempscale:{lang}:1234"))
    order = rng.permutation(len(items))
    n_cal = max(1, int(round(len(items) * 0.5)))
    cal = [items[i] for i in order[:n_cal]]
    evl = [items[i] for i in order[n_cal:]] or cal
    return cal, evl


def lang_ece(cal_subset, evl):
    t = fit_temperature(cal_subset, DEFAULT_GRID, N_BINS)
    c, y = _conf_corr(evl, t)
    return expected_calibration_error(c, y, N_BINS)


def main():
    rows = []  # (dataset, model, ncal, mean, std, pct)
    for dkey, dlabel in DATASETS.items():
        for mkey in MODELS:
            pf = os.path.join(REPO, "artifacts", f"{mkey}_{dkey}", "results", "predictions.jsonl")
            if not os.path.exists(pf):
                continue
            preds = load_preds(pf)
            langs = sorted({p["language"] for p in preds if p["language"] != "eng"})
            # per-language calibration/eval split and total item count (for the % axis)
            per = {}
            totals = []
            for L in langs:
                it = _items(preds, L)
                if len(it) < 4:
                    continue
                per[L] = split(it, L)
                totals.append(len(it))
            langs = list(per.keys())
            total_per_lang = float(np.mean(totals))  # ~500 afrimmlu, ~800 uhura
            for nc in NCALS:
                if nc == "full":
                    eces = [lang_ece(per[L][0], per[L][1]) for L in langs]
                    mean, std = float(np.mean(eces)), 0.0
                    pct = 50.0
                else:
                    macro = []
                    for s in range(N_SUB):
                        per_lang = []
                        for L in langs:
                            cal, evl = per[L]
                            if nc >= len(cal):
                                per_lang.append(lang_ece(cal, evl))
                                continue
                            r = np.random.default_rng(stable_seed(f"sens:{dkey}:{mkey}:{L}:{nc}:{s}"))
                            idx = r.choice(len(cal), size=nc, replace=False)
                            per_lang.append(lang_ece([cal[i] for i in idx], evl))
                        macro.append(float(np.mean(per_lang)))
                    mean, std = float(np.mean(macro)), float(np.std(macro))
                    pct = 100.0 * nc / total_per_lang
                rows.append({"dataset": dkey, "dlabel": dlabel, "model": MODELS[mkey],
                             "ncal": nc, "mean_ece": round(mean, 4), "std_ece": round(std, 4),
                             "pct": round(pct, 2)})
                print(f"{dlabel:16s} {MODELS[mkey]:16s} ncal={str(nc):5s} "
                      f"ECE={mean:.4f} +/-{std:.4f}  ({pct:.1f}%)")

    # write csv
    import csv
    with open(os.path.join(REPO, "revision", "sensitivity_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "dlabel", "model", "ncal", "mean_ece", "std_ece", "pct"])
        w.writeheader(); w.writerows(rows)

    # figure: two panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"Qwen3-4B": "#d1622b", "Gemma-4-12B": "#e2a412",
              "Aya-Expanse-8B": "#d873a8", "AfroLlama-V1": "#12a082"}
    xs_num = {10: 10, 25: 25, 50: 50, 100: 100}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, dkey, dlabel in [(axes[0], "uhura_truthfulqa", "Uhura-TruthfulQA"),
                             (axes[1], "afrimmlu", "AfriMMLU")]:
        drows = [r for r in rows if r["dataset"] == dkey]
        tick_x, tick_pct = None, None
        for m in MODELS.values():
            mr = [r for r in drows if r["model"] == m]
            if not mr:
                continue
            mr_num = [r for r in mr if r["ncal"] != "full"]
            full = [r for r in mr if r["ncal"] == "full"][0]
            total = mr_num[0]["ncal"] / (mr_num[0]["pct"] / 100.0)  # items/language
            full_size = total / 2.0                                 # full cal half = 50% of total
            xs = [r["ncal"] for r in mr_num] + [full_size]
            ys = [r["mean_ece"] for r in mr_num] + [full["mean_ece"]]
            es = [r["std_ece"] for r in mr_num] + [0.0]
            ax.plot(xs, ys, "-o", color=colors[m], label=m, lw=2, ms=5)
            ax.fill_between(xs, np.array(ys) - np.array(es), np.array(ys) + np.array(es),
                            color=colors[m], alpha=0.15)
            tick_x = xs
            tick_pct = [r["pct"] for r in mr_num] + [50.0]
        ax.set_xscale("log")
        ax.set_xlabel("calibration-set size per language (log scale)")
        ax.set_ylabel(r"ECE$_{\mathrm{afr}}$ after temperature scaling")
        ax.set_title(f"{dlabel}: calibration-data sensitivity", fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        # top axis: same points, labelled as % of the language's total labelled items
        sec = ax.secondary_xaxis("top")
        sec.set_xscale("log")
        sec.set_xticks(tick_x)
        sec.set_xticklabels([f"{p:.0f}%" if p >= 2 else f"{p:.1f}%" for p in tick_pct])
        sec.set_xlabel(r"% of language's total labelled items used to fit $T^\ast$")
    fig.tight_layout()
    out = os.path.join(REPO, "report", "figures", "sensitivity.png")
    fig.savefig(out, dpi=150)
    print("[out]", out)


if __name__ == "__main__":
    main()
