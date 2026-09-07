"""Surface confidently-wrong examples for qualitative illustration.

Predictions carry no text, so we rejoin them to the datasets, reproduce the exact
per-item choice shuffle the eval used, and (because both benchmarks are parallel
translations) map each wrong pick and the gold answer to their English glosses via
the aligned English item. Prints the most confidently-wrong critical cases per
dataset/language for manual selection.
"""
from __future__ import annotations
import os, sys, json, hashlib, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from fertility_experiment import UHURA, AFRIMMLU, MODELS
from src.data.loader import load_samples

MODEL_LABEL = {"qwen3_4b_instruct": "Qwen3-4B", "gemma4_12b": "Gemma-4-12B",
               "aya_expanse_8b": "Aya-Expanse-8B", "afrollama_v1": "AfroLlama-V1"}
LANG_NAME = {"amh": "Amharic", "swa": "Swahili", "yor": "Yoruba", "hau": "Hausa",
             "zul": "Zulu", "nso": "N.Sotho", "ibo": "Igbo", "eng": "English",
             "fra": "French", "ewe": "Ewe", "kin": "Kinyarwanda", "lin": "Lingala",
             "lug": "Luganda", "orm": "Oromo", "sna": "Shona", "sot": "Sesotho",
             "twi": "Twi", "wol": "Wolof", "xho": "Xhosa"}

# critical / interpretable domains
UHURA_CRIT = {"Health", "Nutrition", "Misconceptions", "Misinformation", "Law",
              "Science", "Finance", "Economics", "Psychology"}
AFRI_CRIT = {"global_facts", "international_law", "high_school_geography",
             "elementary_mathematics"}


def reproduce_order(lang: str, question: str, n: int, mc1: bool) -> list[int]:
    """Recompute the loader's per-item shuffle permutation (identity for fixed-option)."""
    if not mc1:
        return list(range(n))
    seed = int.from_bytes(hashlib.sha256(f"{lang}|{question}".encode()).digest()[:8], "big")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def build(cfg, lang, mc1):
    """idx -> (sample, order)."""
    out = {}
    for s in load_samples(cfg, languages=[lang]):
        idx = int(s.id.split("-")[-1])
        out[idx] = (s, reproduce_order(s.language, s.question, len(s.choices), mc1))
    return out


def eng_domain_map(dname: str) -> dict[int, str]:
    """idx -> domain from the English prediction rows (reliable categories)."""
    out = {}
    for mkey in MODELS:
        pf = os.path.join(REPO, "artifacts", f"{mkey}_{dname}", "results", "predictions.jsonl")
        if not os.path.exists(pf):
            continue
        for line in open(pf):
            d = json.loads(line)
            if d["language"] == "eng" and d.get("domain"):
                out[int(d["id"].split("-")[-1])] = d["domain"]
        break
    return out


def main():
    for cfg, dname, mc1, crit, conf_thr in [
        (UHURA, "uhura_truthfulqa", True, UHURA_CRIT, 0.55),
        (AFRIMMLU, "afrimmlu", False, AFRI_CRIT, 0.85),
    ]:
        dom_by_idx = eng_domain_map(dname)
        # english reference (for gloss + reliable domain)
        eng = build(cfg, "eng", mc1)
        # african languages available in cache
        avail = []
        for L in cfg.languages:
            if L == "eng":
                continue
            try:
                avail.append((L, build(cfg, L, mc1)))
            except Exception:
                continue

        cands = []
        for mkey in MODELS:
            pf = os.path.join(REPO, "artifacts", f"{mkey}_{dname}", "results", "predictions.jsonl")
            if not os.path.exists(pf):
                continue
            for line in open(pf):
                d = json.loads(line)
                L = d["language"]
                if L == "eng" or d["correct"] or d["confidence"] < conf_thr:
                    continue
                by_idx = dict(avail).get(L)
                if by_idx is None:
                    continue
                idx = int(d["id"].split("-")[-1])
                if idx not in by_idx or idx not in eng:
                    continue
                dom = dom_by_idx.get(idx)  # english category is the reliable one
                if dom not in crit:
                    continue
                s, order = by_idx[idx]
                es, eorder = eng[idx]
                n = len(s.choices)
                if len(es.choices) != n:  # need matching option count for gloss
                    continue
                pidx, aidx = d["predicted_index"], d["answer_index"]
                # canonical indices -> english text
                c_pick, c_gold = order[pidx], order[aidx]
                # alignment guard: english gold must sit at english's own gold slot
                if eorder.index(c_gold) != es.answer_index:
                    continue
                model_en = es.choices[eorder.index(c_pick)]
                gold_en = es.choices[eorder.index(c_gold)]
                cands.append({
                    "model": MODEL_LABEL[mkey], "lang": LANG_NAME.get(L, L), "domain": dom,
                    "conf": d["confidence"], "n": n,
                    "q_native": s.question, "q_en": es.question,
                    "pick_native": s.choices[pidx], "pick_en": model_en,
                    "gold_native": s.choices[aidx], "gold_en": gold_en,
                })
        cands.sort(key=lambda c: -c["conf"])
        print(f"\n{'='*90}\n{dname}: {len(cands)} confidently-wrong critical cases (conf>={conf_thr})\n{'='*90}")
        shown = 0
        for c in cands:
            if shown >= 18:
                break
            print(f"\n[{c['model']} | {c['lang']} | {c['domain']} | conf={c['conf']:.2f} | {c['n']} opts]")
            print(f"  Q ({c['lang']}): {c['q_native'][:150]}")
            print(f"  Q (EN):      {c['q_en'][:150]}")
            print(f"  MODEL chose: {c['pick_native'][:90]}   [EN: {c['pick_en'][:90]}]")
            print(f"  CORRECT:     {c['gold_native'][:90]}   [EN: {c['gold_en'][:90]}]")
            shown += 1


if __name__ == "__main__":
    main()
