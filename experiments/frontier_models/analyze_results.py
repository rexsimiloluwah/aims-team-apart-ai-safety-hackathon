#!/usr/bin/env python3
"""Analyze the verbalized-confidence results.

Per model x benchmark x language: N, accuracy, mean confidence, overconfidence,
ECE (15 bins), Brier, failure rate.

Primary comparison: Delta overconfidence = African overconfidence - English
overconfidence, per African language and as a macro-average, with paired bootstrap
95% CIs over the matched question IDs.

Writes CSVs into outputs/. Run after run_inference.py.
"""
from __future__ import annotations
import os, sys, json, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from src.metrics import expected_calibration_error  # noqa: E402

RESULTS_PATH = os.path.join(HERE, "outputs", "results.jsonl")
OUT = os.path.join(HERE, "outputs")
B = 2000
SEED = 42


def load_latest() -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not os.path.exists(RESULTS_PATH):
        sys.exit(f"No results at {RESULTS_PATH}; run run_inference.py first.")
    for line in open(RESULTS_PATH):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec["key"]] = rec
    return latest


def overconf(cell: dict[int, tuple], idxs) -> float:
    confs = np.array([cell[i][0] for i in idxs], float)
    corrs = np.array([cell[i][1] for i in idxs], float)
    return float(confs.mean() - corrs.mean())


def boot_ci(fn, rng, b=B):
    vals = np.array([fn(rng) for _ in range(b)], float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    latest = load_latest()

    # attempts per cell (any status) and successes as idx -> (conf01, correct)
    attempts: dict[tuple, int] = {}
    succ: dict[tuple, dict[int, tuple]] = {}
    for rec in latest.values():
        cell = (rec["model"], rec["benchmark"], rec["language"])
        attempts[cell] = attempts.get(cell, 0) + 1
        if rec.get("status") == "success" and rec.get("confidence") is not None \
                and rec.get("correct") is not None:
            succ.setdefault(cell, {})[int(rec["question_id"])] = (
                float(rec["confidence"]) / 100.0, 1.0 if rec["correct"] else 0.0)

    # ---- per-cell metrics ----
    per_cell_rows = []
    for cell in sorted(attempts):
        model, bench, lang = cell
        d = succ.get(cell, {})
        n = len(d)
        fr = round((attempts[cell] - n) / attempts[cell], 4) if attempts[cell] else 0.0
        if n == 0:
            per_cell_rows.append({"model": model, "benchmark": bench, "language": lang,
                                  "n": 0, "accuracy": "", "mean_confidence": "",
                                  "overconfidence": "", "ece": "", "brier": "",
                                  "failure_rate": fr})
            continue
        conf = np.array([v[0] for v in d.values()], float)
        corr = np.array([v[1] for v in d.values()], float)
        per_cell_rows.append({
            "model": model, "benchmark": bench, "language": lang, "n": n,
            "accuracy": round(float(corr.mean()), 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "overconfidence": round(float(conf.mean() - corr.mean()), 4),
            "ece": round(float(expected_calibration_error(conf, corr, 15)), 4),
            "brier": round(float(np.mean((conf - corr) ** 2)), 4),
            "failure_rate": fr,
        })

    with open(os.path.join(OUT, "metrics_per_cell.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "language", "n", "accuracy",
                                          "mean_confidence", "overconfidence", "ece", "brier",
                                          "failure_rate"])
        w.writeheader(); w.writerows(per_cell_rows)

    # ---- overconfidence gap (paired bootstrap over matched question ids) ----
    models = sorted({m for m, _, _ in succ})
    benches = sorted({b for _, b, _ in succ})
    by_lang_rows, macro_rows = [], []
    for model in models:
        for bench in benches:
            eng = succ.get((model, bench, "eng"))
            if not eng:
                continue
            afr_langs = sorted(l for (m, b, l) in succ if m == model and b == bench and l != "eng")
            if not afr_langs:
                continue
            rng = np.random.default_rng(SEED)

            # per-language delta
            for lang in afr_langs:
                cell = succ[(model, bench, lang)]
                common = sorted(set(eng) & set(cell))
                if len(common) < 3:
                    continue
                delta = overconf(cell, common) - overconf(eng, common)
                idx = np.array(common)

                def draw(r, cell=cell, idx=idx):
                    s = r.choice(idx, size=idx.size, replace=True)
                    return overconf(cell, s) - overconf(eng, s)

                lo, hi = boot_ci(draw, rng)
                by_lang_rows.append({
                    "model": model, "benchmark": bench, "language": lang,
                    "n_pairs": len(common),
                    "delta_overconfidence": round(delta, 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

            # macro delta over African languages, paired on the shared index set
            common_all = set(eng)
            for lang in afr_langs:
                common_all &= set(succ[(model, bench, lang)])
            common_all = sorted(common_all)
            if len(common_all) < 3:
                continue
            idx_all = np.array(common_all)

            def macro_delta(r, idx_all=idx_all, afr_langs=afr_langs):
                s = r.choice(idx_all, size=idx_all.size, replace=True)
                eng_oc = overconf(eng, s)
                afr_oc = np.mean([overconf(succ[(model, bench, l)], s) for l in afr_langs])
                return afr_oc - eng_oc

            point = macro_delta(np.random.default_rng(0))  # deterministic point on full set
            eng_oc = overconf(eng, idx_all)
            afr_oc = float(np.mean([overconf(succ[(model, bench, l)], idx_all) for l in afr_langs]))
            point = afr_oc - eng_oc
            lo, hi = boot_ci(macro_delta, rng)
            macro_rows.append({
                "model": model, "benchmark": bench, "n_afr_langs": len(afr_langs),
                "n_pairs": len(common_all),
                "delta_overconfidence_macro": round(point, 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4)})

    with open(os.path.join(OUT, "overconfidence_gap_by_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "language", "n_pairs",
                                          "delta_overconfidence", "ci_low", "ci_high"])
        w.writeheader(); w.writerows(by_lang_rows)
    with open(os.path.join(OUT, "overconfidence_gap_macro.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "benchmark", "n_afr_langs", "n_pairs",
                                          "delta_overconfidence_macro", "ci_low", "ci_high"])
        w.writeheader(); w.writerows(macro_rows)

    # ---- console summary ----
    print("=== Per-cell metrics (English rows) ===")
    for r in per_cell_rows:
        if r["language"] == "eng":
            print(f"  {r['model']:26s} {r['benchmark']:16s} eng  "
                  f"acc={r['accuracy']} conf={r['mean_confidence']} "
                  f"overconf={r['overconfidence']} ece={r['ece']} fr={r['failure_rate']}")
    print("\n=== Overconfidence gap (African macro - English), paired bootstrap 95% CI ===")
    for r in macro_rows:
        sig = "" if (r["ci_low"] <= 0 <= r["ci_high"]) else "  *disjoint from 0*"
        print(f"  {r['model']:26s} {r['benchmark']:16s} "
              f"Delta={r['delta_overconfidence_macro']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"(langs={r['n_afr_langs']}, pairs={r['n_pairs']}){sig}")
    print(f"\nWrote CSVs to {OUT}")


if __name__ == "__main__":
    main()
