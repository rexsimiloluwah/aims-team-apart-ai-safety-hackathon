"""
compute_bootstrap_ci.py
Bootstrap 95% CIs for ECE and Accuracy reported in Table 1.
Outputs a JSON file that is then embedded in paper.tex.
"""
import json, pathlib, numpy as np
from collections import defaultdict

ROOT = pathlib.Path(__file__).parent.parent / "artifacts"
SEED = 42
B    = 2000   # bootstrap draws

MODELS = {
    "qwen3_4b_instruct":  "Qwen3-4B",
    "gemma4_12b":         "Gemma-4-12B",
    "aya_expanse_8b":     "Aya-Expanse-8B",
    "afrollama_v1":       "AfroLlama-V1",
}
DATASETS = {
    "uhura_truthfulqa": "Uhura-TQA",
    "afrimmlu":         "AfriMMLU",
}
NBINS = 15


def ece(corrects, confidences):
    corrects    = np.asarray(corrects,    dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    bins  = np.linspace(0, 1, NBINS + 1)
    total = 0.0
    n     = len(corrects)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        acc_b  = corrects[mask].mean()
        conf_b = confidences[mask].mean()
        total += mask.sum() / n * abs(acc_b - conf_b)
    return total


def load_predictions(path):
    """Return list of (language, correct, confidence)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records.append((r["language"], int(r["correct"]), float(r["confidence"])))
    return records


def bootstrap_stats(records, is_english=False):
    """
    Macro-average ECE and accuracy across languages (or single-language slice).
    Returns (point_ece, lo_ece, hi_ece, point_acc, lo_acc, hi_acc).
    """
    rng = np.random.default_rng(SEED)

    # group by language
    by_lang = defaultdict(lambda: ([], []))
    for lang, correct, conf in records:
        by_lang[lang][0].append(correct)
        by_lang[lang][1].append(conf)

    langs = sorted(by_lang)

    def one_draw(sampled_by_lang):
        lang_eces, lang_accs = [], []
        for lang in langs:
            c_arr, conf_arr = sampled_by_lang[lang]
            if len(c_arr) == 0:
                continue
            lang_eces.append(ece(c_arr, conf_arr))
            lang_accs.append(np.mean(c_arr))
        return np.mean(lang_eces), np.mean(lang_accs)

    # point estimate
    pt_ece, pt_acc = one_draw({l: (np.array(v[0]), np.array(v[1])) for l, v in by_lang.items()})

    # bootstrap
    boot_eces, boot_accs = [], []
    for _ in range(B):
        sampled = {}
        for lang in langs:
            c_arr  = np.array(by_lang[lang][0])
            cf_arr = np.array(by_lang[lang][1])
            idx    = rng.integers(0, len(c_arr), size=len(c_arr))
            sampled[lang] = (c_arr[idx], cf_arr[idx])
        be, ba = one_draw(sampled)
        boot_eces.append(be)
        boot_accs.append(ba)

    boot_eces = np.array(boot_eces)
    boot_accs = np.array(boot_accs)
    return (
        pt_ece,
        np.percentile(boot_eces, 2.5),  np.percentile(boot_eces, 97.5),
        pt_acc,
        np.percentile(boot_accs, 2.5),  np.percentile(boot_accs, 97.5),
    )


results = {}
for model_dir, model_label in MODELS.items():
    for ds_dir, ds_label in DATASETS.items():
        pred_path = ROOT / f"{model_dir}_{ds_dir}" / "results" / "predictions.jsonl"
        if not pred_path.exists():
            print(f"  MISSING: {pred_path}")
            continue

        all_records = load_predictions(pred_path)

        # English slice
        eng = [(l, c, cf) for l, c, cf in all_records if l == "eng"]
        afr = [(l, c, cf) for l, c, cf in all_records if l != "eng"]

        pe, le, he, pa_e, la_e, ha_e = bootstrap_stats(eng)
        pf, lf, hf, pa_f, la_f, ha_f = bootstrap_stats(afr)

        key = f"{model_label}|{ds_label}"
        results[key] = {
            "ece_en":  {"pt": pe, "lo": le, "hi": he},
            "acc_en":  {"pt": pa_e, "lo": la_e, "hi": ha_e},
            "ece_afr": {"pt": pf, "lo": lf, "hi": hf},
            "acc_afr": {"pt": pa_f, "lo": la_f, "hi": ha_f},
        }
        print(f"{key}")
        print(f"  ECE_en  = {pe:.3f}  [{le:.3f}, {he:.3f}]")
        print(f"  Acc_en  = {pa_e:.3f}  [{la_e:.3f}, {ha_e:.3f}]")
        print(f"  ECE_afr = {pf:.3f}  [{lf:.3f}, {hf:.3f}]")
        print(f"  Acc_afr = {pa_f:.3f}  [{la_f:.3f}, {ha_f:.3f}]")
        print()

out = pathlib.Path(__file__).parent / "bootstrap_ci.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved → {out}")
