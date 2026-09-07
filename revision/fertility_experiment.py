"""Fertility vs. calibration experiment (mechanism analysis).

Question: does tokenizer *fertility* (subword tokens per whitespace word) on a
language predict that model's calibration error (ECE) on that language?

For each model we use ITS OWN tokenizer, compute corpus-level fertility on the
benchmark question stems per language, and correlate (Spearman) against the
model's per-language ECE. CPU-only, uses locally cached tokenizers + datasets.

Outputs:
  revision/fertility_results.csv        tidy (model,dataset,language,fertility,ece)
  revision/fertility_correlations.csv   per-model + pooled Spearman rho / p
  report/figures/fertility_ece.png      scatter for the appendix
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Use only local cache; never hit the network / gated-model prompts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import csv
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from src.data.loader import load_samples

# model key (matches ECE-matrix rows) -> HF tokenizer id
MODELS = {
    "qwen3_4b_instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "gemma4_12b": "google/gemma-4-12B-it",
    "aya_expanse_8b": "CohereLabs/aya-expanse-8b",
    "afrollama_v1": "Jacaranda/AfroLlama_V1",
}

# minimal dataset cfgs (category maps dropped: we only need question text)
UHURA = SimpleNamespace(
    hf_id="masakhane/uhura-truthfulqa", split="test", scoring="mc1", has_domain=False,
    languages=["eng", "amh", "hau", "nso", "swa", "yor", "zul"],
    hf_config_map={"eng": "en_multiple_choice", "amh": "am_multiple_choice",
                   "hau": "ha_multiple_choice", "nso": "nso_multiple_choice",
                   "swa": "sw_multiple_choice", "yor": "yo_multiple_choice",
                   "zul": "zu_multiple_choice"},
)
AFRIMMLU = SimpleNamespace(
    hf_id="masakhane/afrimmlu", split="test", scoring="cloze", has_domain=False,
    num_choices=4, choice_labels=["A", "B", "C", "D"],
    languages=["eng", "fra", "amh", "ewe", "hau", "ibo", "kin", "lin", "lug",
               "orm", "sna", "sot", "swa", "twi", "wol", "xho", "yor", "zul"],
)
DATASETS = {"uhura_truthfulqa": UHURA, "afrimmlu": AFRIMMLU}


def load_questions(cfg) -> dict[str, list[str]]:
    """language code -> list of question stems. Skips languages whose config is
    not in the local cache (offline), so a partially-cached dataset still runs."""
    by_lang: dict[str, list[str]] = {}
    for lang in cfg.languages:
        try:
            samples = load_samples(cfg, languages=[lang])
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {cfg.hf_id} lang={lang}: {str(exc).splitlines()[0][:80]}")
            continue
        for s in samples:
            by_lang.setdefault(s.language, []).append(s.question)
    return by_lang


def corpus_fertility(tok, texts: list[str]) -> float:
    """Total subword tokens / total whitespace words across the corpus."""
    total_tokens = total_words = 0
    for t in texts:
        words = t.split()
        if not words:
            continue
        n_tok = len(tok(t, add_special_tokens=False)["input_ids"])
        total_tokens += n_tok
        total_words += len(words)
    return total_tokens / total_words if total_words else float("nan")


def load_matrix(dataset: str, kind: str) -> dict[str, dict[str, float]]:
    """model -> {lang: value} from a comparison matrix (kind in {ece, accuracy})."""
    path = os.path.join(REPO, "artifacts", "_comparison", dataset, f"{kind}_matrix.csv")
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        r = csv.DictReader(f)
        langs = [c for c in r.fieldnames if c != "model"]
        for row in r:
            out[row["model"]] = {L: float(row[L]) for L in langs if row[L] not in ("", None)}
    return out


def main() -> None:
    # 1. question stems per dataset/language
    questions = {name: load_questions(cfg) for name, cfg in DATASETS.items()}
    for name, ql in questions.items():
        print(f"[data] {name}: " + ", ".join(f"{L}={len(v)}" for L, v in ql.items()))

    # 2. per-model tokenizer fertility per dataset/language
    ece = {name: load_matrix(name, "ece") for name in DATASETS}
    acc = {name: load_matrix(name, "accuracy") for name in DATASETS}
    tidy: list[dict] = []
    for mkey, hf_id in MODELS.items():
        print(f"[tok] loading {hf_id}")
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        for dname, ql in questions.items():
            for lang, texts in ql.items():
                if lang not in ece[dname].get(mkey, {}):
                    continue
                fert = corpus_fertility(tok, texts)
                tidy.append({"model": mkey, "dataset": dname, "language": lang,
                             "fertility": round(fert, 4),
                             "ece": round(ece[dname][mkey][lang], 4),
                             "accuracy": round(acc[dname][mkey].get(lang, float("nan")), 4)})

    # 3. write tidy table
    tidy_path = os.path.join(REPO, "revision", "fertility_results.csv")
    with open(tidy_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "dataset", "language", "fertility", "ece", "accuracy"])
        w.writeheader()
        w.writerows(tidy)
    print(f"[out] {tidy_path}  ({len(tidy)} rows)")

    # 4. correlations: fertility vs {ece, accuracy}, per model + pooled (gen.-purpose)
    rows = tidy
    GP = {"qwen3_4b_instruct", "gemma4_12b", "aya_expanse_8b"}  # general-purpose (exclude AfroLlama)

    def corr(subset, target):
        x = np.array([r["fertility"] for r in subset], float)
        y = np.array([r[target] for r in subset], float)
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[m], y[m]
        if len(x) < 3:
            return len(x), float("nan"), float("nan")
        rho, p = spearmanr(x, y)
        return len(x), rho, p

    corr_rows = []
    for target in ("ece", "accuracy"):
        print(f"\n=== Spearman (fertility vs {target.upper()}) ===")
        for dname in DATASETS:
            for mkey in MODELS:
                sub = [r for r in rows if r["dataset"] == dname and r["model"] == mkey]
                n, rho, p = corr(sub, target)
                corr_rows.append({"target": target, "scope": mkey, "dataset": dname,
                                  "n": n, "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
                print(f"  {dname:16s} {mkey:20s} n={n:2d}  rho={rho:+.3f}  p={p:.4f}")
        # pooled across the general-purpose models, AfriMMLU (complete 18-lang set)
        sub = [r for r in rows if r["dataset"] == "afrimmlu" and r["model"] in GP]
        n, rho, p = corr(sub, target)
        corr_rows.append({"target": target, "scope": "POOLED gen-purpose", "dataset": "afrimmlu",
                          "n": n, "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
        print(f"  {'afrimmlu':16s} {'POOLED gen-purpose':20s} n={n:2d}  rho={rho:+.3f}  p={p:.4f}")

    corr_path = os.path.join(REPO, "revision", "fertility_correlations.csv")
    with open(corr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["target", "scope", "dataset", "n", "spearman_rho", "p_value"])
        w.writeheader(); w.writerows(corr_rows)
    print(f"[out] {corr_path}")

    # 5. two-panel figure for the appendix (AfriMMLU, 18 languages, all 4 models)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"qwen3_4b_instruct": "#1f77b4", "gemma4_12b": "#d62728",
              "aya_expanse_8b": "#2ca02c", "afrollama_v1": "#9467bd"}
    labels = {"qwen3_4b_instruct": "Qwen3-4B", "gemma4_12b": "Gemma-4-12B",
              "aya_expanse_8b": "Aya-Expanse-8B", "afrollama_v1": "AfroLlama-V1"}
    afr = [r for r in rows if r["dataset"] == "afrimmlu"]
    gp = [r for r in afr if r["model"] in GP]
    gx = np.array([r["fertility"] for r in gp], float)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, target, ylab in [(axes[0], "ece", "Expected calibration error (ECE)"),
                             (axes[1], "accuracy", "Accuracy")]:
        for mkey in MODELS:
            sub = [r for r in afr if r["model"] == mkey]
            ax.scatter([r["fertility"] for r in sub], [r[target] for r in sub],
                       c=colors[mkey], marker="o", s=36, alpha=0.85,
                       edgecolors="white", linewidths=0.5)
        gy = np.array([r[target] for r in gp], float)
        b, a = np.polyfit(gx, gy, 1)
        xs = np.linspace(gx.min(), gx.max(), 50)
        rho, p = spearmanr(gx, gy)
        ax.plot(xs, a + b * xs, color="black", ls="--", lw=1.4)
        ax.set_xlabel("Tokenizer fertility (subword tokens / word)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Fertility vs. {ylab.split(' (')[0].lower()}\n"
                     f"gen.-purpose $\\rho$={rho:+.2f}, p={p:.2g} (n={len(gx)})", fontsize=10)
        ax.grid(alpha=0.25)
    model_handles = [Line2D([0], [0], marker="o", ls="", mfc=colors[m], mec="white", ms=8,
                            label=labels[m]) for m in MODELS]
    axes[0].legend(handles=model_handles, fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig_path = os.path.join(REPO, "report", "figures", "fertility_ece.png")
    fig.savefig(fig_path, dpi=150)
    print(f"[out] {fig_path}")


if __name__ == "__main__":
    main()
