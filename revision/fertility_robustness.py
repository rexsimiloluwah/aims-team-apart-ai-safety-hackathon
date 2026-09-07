"""Robustness / sanity checks for the fertility <-> ECE result.

Concerns being tested:
  (a) tokens/word confounds tokenizer coverage with intrinsic word length/script;
  (b) the null is driven by Amharic (Ge'ez script) as a leverage outlier;
  (c) an alternative, less word-boundary-sensitive metric changes the picture.

AfriMMLU is parallel (same 500 questions translated), so we add:
  - relative_fertility = tokens(lang)/tokens(eng) on the aligned question set
    (Petrov et al. 2023 style; controls for content, isolates tokenizer efficiency)
  - chars_per_token = mean token length in characters (coverage proxy, not tied to words)

Reports Spearman of each metric vs ECE and vs accuracy, on the full 18 languages,
Latin-only (drop Amharic), and per model.
"""
from __future__ import annotations
import os, sys, csv
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from fertility_experiment import MODELS, AFRIMMLU, load_questions, load_matrix

GP = ["qwen3_4b_instruct", "gemma4_12b", "aya_expanse_8b"]


def main() -> None:
    q = load_questions(AFRIMMLU)  # lang -> [question stems], 18 languages
    langs = list(q.keys())
    ece = load_matrix("afrimmlu", "ece")
    acc = load_matrix("afrimmlu", "accuracy")

    # per (model, lang): tokens/word, chars/token, total tokens (for relative fertility)
    metrics: dict[tuple[str, str], dict] = {}
    for mkey, hf_id in MODELS.items():
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        for lang in langs:
            tt = tw = tc = 0
            for t in q[lang]:
                w = t.split()
                if not w:
                    continue
                ids = tok(t, add_special_tokens=False)["input_ids"]
                tt += len(ids); tw += len(w); tc += len(t)
            metrics[(mkey, lang)] = {"tok_per_word": tt / tw, "chars_per_tok": tc / tt,
                                     "tot_tok": tt}
        # relative fertility vs English (parallel content)
        eng_tok = metrics[(mkey, "eng")]["tot_tok"]
        for lang in langs:
            metrics[(mkey, lang)]["rel_fertility"] = metrics[(mkey, lang)]["tot_tok"] / eng_tok

    def rows(metric, target, models, drop=()):
        out = []
        for m in models:
            for L in langs:
                if L in drop or L not in ece.get(m, {}):
                    continue
                y = ece[m][L] if target == "ece" else acc[m].get(L, float("nan"))
                out.append((metrics[(m, L)][metric], y))
        return out

    def rho(pairs):
        if len(pairs) < 3:
            return len(pairs), float("nan"), float("nan")
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        r, p = spearmanr(x, y)
        return len(pairs), r, p

    print("AfriMMLU parallel corpus. GP = general-purpose (Qwen, Gemma, Aya).\n")
    for metric in ["tok_per_word", "rel_fertility", "chars_per_tok"]:
        print(f"### {metric}")
        for target in ["ece", "accuracy"]:
            n, r, p = rho(rows(metric, target, GP))
            n2, r2, p2 = rho(rows(metric, target, GP, drop={"amh"}))
            print(f"  vs {target:8s} | GP all18 n={n} rho={r:+.3f} p={p:.4f}"
                  f"   | GP no-Amharic n={n2} rho={r2:+.3f} p={p2:.4f}")
        # per-model vs ECE, all 18
        for m in MODELS:
            n, r, p = rho(rows(metric, "ece", [m]))
            print(f"      per-model {m:20s} vs ECE  rho={r:+.3f} p={p:.4f}")
        print()

    # show the parallel relative-fertility values (Qwen) to sanity-check
    print("Relative fertility vs English (Qwen), sorted:")
    for L in sorted(langs, key=lambda L: metrics[("qwen3_4b_instruct", L)]["rel_fertility"]):
        mm = metrics[("qwen3_4b_instruct", L)]
        print(f"  {L}: rel_fert={mm['rel_fertility']:.2f}  tok/word={mm['tok_per_word']:.2f}"
              f"  chars/tok={mm['chars_per_tok']:.2f}  ECE={ece['qwen3_4b_instruct'][L]:.3f}")


if __name__ == "__main__":
    main()
