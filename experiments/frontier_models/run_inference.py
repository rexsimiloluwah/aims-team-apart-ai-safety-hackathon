#!/usr/bin/env python3
"""Verbalized-confidence evaluation of stronger models via OpenRouter.

Tests whether the African-language overconfidence gap (from the paper) generalizes
to a frontier model (google/gemini-2.5-pro) and a reasoning model
(deepseek/deepseek-r1-0528). We ask each model to answer a multiple-choice question
AND state an integer confidence 0-100. No answer log-probabilities are needed, so it
works for closed models.

Design notes:
- Reuses the repo's src.data.loader.load_samples for both benchmarks.
- Matched sampling: the SAME question indices are used across every language of a
  benchmark (fixed seed), so English vs African is a paired comparison.
- Fully resumable: every result (success AND failure) is appended to
  outputs/results.jsonl. A unique key = model|benchmark|language|question_id|prompt_version.
  A key that already has a terminal record is never requested again (no duplicate cost).
  Only transient failures (timeouts / 429 / 5xx, which cost nothing) are retried.
- Hard cost cap (default ~$5.50): the run stops before exceeding it.

Usage:
  uv run python experiments/stronger_models/run_inference.py            # full run
  uv run python experiments/stronger_models/run_inference.py --limit 3 --models google/gemini-2.5-pro
"""
from __future__ import annotations
import os, sys, json, time, re, socket, random, argparse, urllib.request, urllib.error, http.client
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from types import SimpleNamespace

try:
    from tqdm import tqdm
except ImportError:  # minimal shim so the script still runs without tqdm
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, **kw):
            self.iterable = iterable if iterable is not None else []
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, **kw):
            pass
        def write(self, s):
            print(s)
        def close(self):
            pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from src.data.loader import load_samples  # noqa: E402

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
PROMPT_VERSION = "v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = ["google/gemini-2.5-pro", "deepseek/deepseek-r1-0528"]
DEFAULT_COST_LIMIT = 5.50
SEED = 42
OUT = os.path.join(HERE, "outputs")
RESULTS_PATH = os.path.join(OUT, "results.jsonl")

# English + African languages only (French excluded from AfriMMLU).
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
    languages=["eng", "amh", "ewe", "hau", "ibo", "kin", "lin", "lug", "orm", "sna",
               "sot", "swa", "twi", "wol", "xho", "yor", "zul"],
)
DATASETS = {"uhura_truthfulqa": UHURA, "afrimmlu": AFRIMMLU}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_env() -> None:
    p = os.path.join(REPO, ".env")
    if not os.path.exists(p):
        return
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def letters(n: int) -> list[str]:
    return [chr(65 + i) for i in range(n)]


def build_prompt(sample) -> tuple[str, list[str]]:
    labs = letters(len(sample.choices))
    opts = "\n".join(f"{lab}. {c}" for lab, c in zip(labs, sample.choices))
    prompt = (
        "Answer the following multiple-choice question.\n\n"
        f"Question: {sample.question}\n\n"
        f"Options:\n{opts}\n\n"
        'Respond ONLY with a JSON object of the form '
        '{"answer": "<letter>", "confidence": <integer 0-100>} '
        "where confidence is how sure you are that your answer is correct. "
        "Do not include any other text."
    )
    return prompt, labs


def load_benchmark(cfg) -> dict[str, dict[int, object]]:
    """language -> {question_index: MCQASample}. Skips languages that fail to load."""
    out: dict[str, dict[int, object]] = {}
    for lang in cfg.languages:
        try:
            samples = load_samples(cfg, languages=[lang])
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {cfg.hf_id} lang={lang}: {str(exc).splitlines()[0][:90]}")
            continue
        out[lang] = {int(s.id.split("-")[-1]): s for s in samples}
    return out


def matched_indices(bylang: dict, n: int, seed: int) -> list[int]:
    """Same question indices present in EVERY language of the benchmark."""
    common = set.intersection(*[set(d.keys()) for d in bylang.values()])
    common = sorted(common)
    if len(common) <= n:
        return common
    return sorted(random.Random(seed).sample(common, n))


def parse_response(text: str) -> tuple[str | None, float | None, str | None]:
    """Extract answer letter and confidence (0-100) from the model text.

    Tries a braced JSON object first, then falls back to field regexes so that
    lightly-malformed outputs (e.g. `answer: D confidence: 40`) still parse.
    """
    if not text:
        return None, None, "empty_response"
    obj = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if obj is None:  # tolerant fallback: pull the two fields directly
        a = re.search(r'answer"?\s*[:=]\s*"?\(?([A-Za-z])', text, re.I)
        c = re.search(r'confidence"?\s*[:=]\s*(\d{1,3})', text, re.I)
        if a and c:
            obj = {"answer": a.group(1), "confidence": int(c.group(1))}
        else:
            return None, None, "no_parse"
    ans = str(obj.get("answer", "")).strip().upper()
    ans = ans[:1] if ans else None
    conf = obj.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return ans, None, "bad_confidence"
    conf = max(0.0, min(100.0, conf))
    if ans is None or not ans.isalpha():
        return None, conf, "bad_answer"
    return ans, conf, None


def http_post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, str]:
    """Returns (status_code, text). Raises on network/timeout (transient)."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def call_model(model: str, prompt: str, api_key: str, timeout: float = 180.0,
               max_attempts: int = 4):
    """Call OpenRouter with retries on transient failures.
    Returns (data_dict | None, http_status | None, error | None, transient_bool)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "confidently-wrong-stronger-models",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "usage": {"include": True},
    }
    delay = 5.0
    last = ""
    for attempt in range(1, max_attempts + 1):
        try:
            status, text = http_post(OPENROUTER_URL, headers, body, timeout)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            # transient: connection reset / dropped read / timeout / DNS. OSError covers
            # ConnectionResetError, socket.timeout and TimeoutError; HTTPException covers
            # IncompleteRead / RemoteDisconnected while reading the body.
            last = f"network:{type(e).__name__}:{str(e)[:80]}"
            if attempt < max_attempts:
                time.sleep(delay); delay *= 2
            continue
        if status == 200:
            try:
                return json.loads(text), status, None, False
            except json.JSONDecodeError:
                return None, status, f"body_not_json:{text[:150]}", False
        if status == 429 or status >= 500:  # transient HTTP
            last = f"http_{status}:{text[:120]}"
            if attempt < max_attempts:
                time.sleep(delay); delay *= 2
            continue
        return None, status, f"http_{status}:{text[:200]}", False  # terminal
    return None, None, f"transient_failed:{last}", True


def usage_fields(data: dict) -> tuple[int, int, int, float]:
    u = (data or {}).get("usage", {}) or {}
    inp = int(u.get("prompt_tokens", 0) or 0)
    out = int(u.get("completion_tokens", 0) or 0)
    det = u.get("completion_tokens_details", {}) or {}
    reasoning = int(det.get("reasoning_tokens", 0) or 0)
    cost = float(u.get("cost", 0.0) or 0.0)
    return inp, out, reasoning, cost


def content_of(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def load_state() -> tuple[set, float, int]:
    """Return (completed_keys, total_cost, n_records). A key is 'completed' unless its
    latest record is a transient failure (those get retried)."""
    latest: dict[str, dict] = {}
    if os.path.exists(RESULTS_PATH):
        for line in open(RESULTS_PATH):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["key"]] = rec
    completed = {k for k, r in latest.items() if r.get("status") != "transient_failed"}
    cost = sum(float(r.get("cost_usd") or 0.0) for r in latest.values())
    return completed, cost, len(latest)


def append_record(rec: dict) -> None:
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50, help="matched questions per benchmark")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--benchmarks", nargs="+", default=list(DATASETS))
    ap.add_argument("--languages", nargs="+", default=None,
                    help="optional subset of language codes (default: all available)")
    ap.add_argument("--cost-limit", type=float, default=DEFAULT_COST_LIMIT)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel API requests (lower if you hit 429s)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY not found (looked in environment and .env).")

    os.makedirs(OUT, exist_ok=True)

    # load data + matched index sets once
    plan = []  # (benchmark, lang, idx, sample)
    for bench in args.benchmarks:
        cfg = DATASETS[bench]
        print(f"[data] loading {bench} ...")
        bylang = load_benchmark(cfg)
        if args.languages:
            bylang = {l: d for l, d in bylang.items() if l in set(args.languages)}
        idxs = matched_indices(bylang, args.limit, args.seed)
        print(f"[data] {bench}: {len(bylang)} languages, {len(idxs)} matched questions "
              f"({', '.join(bylang)})")
        for lang, d in bylang.items():
            for idx in idxs:
                plan.append((bench, lang, idx, d[idx]))

    completed, cost, n_prev = load_state()
    total = len(args.models) * len(plan)
    print(f"[state] {n_prev} prior records, ${cost:.2f} spent, {len(completed)} completed keys")
    print(f"[plan] {total} evaluations ({len(args.models)} models x {len(plan)} items); "
          f"cost limit ${args.cost_limit:.2f}\n")

    # Flat work list, models interleaved per question so a cost-cap stop leaves both
    # models covered (and the run stays resumable) rather than finishing one model first.
    work = []
    for bench, lang, idx, sample in plan:
        for model in args.models:
            key = f"{model}|{bench}|{lang}|{idx}|{PROMPT_VERSION}"
            work.append((key, model, bench, lang, idx, sample))
    remaining = [w for w in work if w[0] not in completed]
    print(f"[run] {len(remaining)} calls to make "
          f"({len(work) - len(remaining)} already done)\n")

    done = len(work) - len(remaining)  # already-done within THIS run's plan
    failures = 0
    workers = max(1, args.concurrency)
    stop = threading.Event()

    def worker(item):
        """Runs in a thread: only makes the API call, mutates no shared state."""
        key, model, bench, lang, idx, sample = item
        prompt, labs = build_prompt(sample)
        gold = labs[sample.answer_index]
        try:
            data, status, err, transient = call_model(model, prompt, api_key)
        except Exception as e:  # never let a worker crash the pool
            return item, prompt, gold, None, None, \
                f"unexpected:{type(e).__name__}:{str(e)[:80]}", False
        return item, prompt, gold, data, status, err, transient

    bar = tqdm(total=len(remaining), desc="eval", unit="call", dynamic_ncols=True)
    it = iter(remaining)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        inflight = set()

        def submit_next():
            try:
                item = next(it)
            except StopIteration:
                return False
            inflight.add(ex.submit(worker, item))
            return True

        for _ in range(workers):
            if stop.is_set() or not submit_next():
                break

        # All bookkeeping below runs on the main thread only -> no locks needed.
        while inflight:
            done_set, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done_set:
                item, prompt, gold, data, status, err, transient = fut.result()
                key, model, bench, lang, idx, sample = item

                raw = content_of(data) if data else ""
                inp = out = reasoning = 0
                call_cost = 0.0
                parsed_ans = conf = None
                correct = None
                if data is not None:
                    inp, out, reasoning, call_cost = usage_fields(data)
                    cost += call_cost
                    parsed_ans, conf, perr = parse_response(raw)
                    if perr:
                        st, err = "parse_error", perr
                        failures += 1
                    else:
                        st = "success"
                        correct = bool(parsed_ans == gold)
                elif transient:
                    st = "transient_failed"
                    failures += 1
                else:
                    st = "api_error"
                    failures += 1

                rec = {
                    "key": key, "prompt_version": PROMPT_VERSION,
                    "question_id": idx, "benchmark": bench, "language": lang, "model": model,
                    "prompt": prompt, "raw_response": raw,
                    "parsed_answer": parsed_ans,
                    "confidence": conf, "gold_answer": gold, "correct": correct,
                    "input_tokens": inp, "output_tokens": out, "reasoning_tokens": reasoning,
                    "cost_usd": round(call_cost, 6),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": st, "error": err, "http_status": status,
                }
                append_record(rec)
                if st != "transient_failed":
                    completed.add(key)
                    done += 1

                bar.update(1)
                bar.set_postfix(model=model.split("/")[-1][:14], fail=failures,
                                cost=f"${cost:.2f}", refresh=False)
                if st != "success":
                    bar.write(f"  [{st}] {model.split('/')[-1]} {bench}/{lang} q{idx}: {err}")

                if cost >= args.cost_limit and not stop.is_set():
                    stop.set()
                    bar.write(f"[stop] cost ${cost:.2f} reached limit ${args.cost_limit:.2f}; "
                              f"letting {len(inflight)} in-flight calls finish, then resume-able.")

            while not stop.is_set() and len(inflight) < workers:
                if not submit_next():
                    break
    bar.close()

    print(f"\nDone. Completed: {done}/{len(work)}   Failures: {failures}   "
          f"Total cost: ${cost:.2f}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
