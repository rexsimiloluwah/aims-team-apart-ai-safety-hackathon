# Stronger-models overconfidence test (verbalized confidence)

Does the African-language **overconfidence gap** from *Confidently Wrong* also appear in
stronger models? We test a frontier model and a reasoning model through OpenRouter,
using **verbalized confidence** (the model states its answer *and* a 0–100 confidence),
because closed models do not expose answer log-probabilities.

This is a **generalization / robustness** check, not an apples-to-apples comparison with
the paper's four log-prob-scored models: verbalized confidence is a different estimator.

## Models
- `google/gemini-2.5-pro` (frontier)
- `deepseek/deepseek-r1-0528` (reasoning)

## Data
- **Uhura-TruthfulQA** and **AfriMMLU**, English + available African languages (French excluded).
- **50 matched questions per benchmark**: the same question indices are sampled across every
  language (fixed seed 42), so English vs African is a *paired* comparison.
- Caveat: AfriMMLU is parallel across languages by index. Uhura-TruthfulQA is **not** perfectly
  index-aligned across languages, so its "matched" set is a matched-index design, not identical
  questions. Read Uhura results with that in mind.

## Setup
Put `OPENROUTER_API_KEY=...` in the repo-root `.env` (already present). No other secrets needed.

## Run
```bash
# Pilot (cheap): a couple of questions, one benchmark
uv run python experiments/stronger_models/run_inference.py --limit 2 --benchmarks afrimmlu

# Full run (see cost note). Resumable — just run again to continue.
uv run python experiments/stronger_models/run_inference.py --limit 50 --cost-limit 10

# Analyze
uv run python experiments/stronger_models/analyze_results.py
```
`experiment.ipynb` walks through the same steps interactively (load, inspect, prompt test,
model test, parse check, cost, tiny pilot) before any full run.

### Cost
Gemini 2.5 Pro "thinks" on every call (~$0.007/call); DeepSeek R1 is ~$0.001/call. A full
`--limit 50` run over both benchmarks and both models is ~2,400 calls ≈ **$9–10**. The default
`--cost-limit` is **$5.50**: the run stops safely before crossing it and can be resumed later
with a higher cap. Models are interleaved per question, so a cost-cap stop leaves both models
partially covered rather than one finished and one empty.

## Resumability
Every result — success *and* failure — is appended to `outputs/results.jsonl`, keyed by
`model|benchmark|language|question_id|prompt_version`. A key with a terminal record is never
requested again (no duplicate cost); only transient failures (timeout / 429 / 5xx, which cost
nothing) are retried. Stop with Ctrl-C any time and rerun to continue.

## Outputs
- `outputs/results.jsonl` — one row per evaluation (raw response, parsed answer, confidence,
  gold, correctness, tokens, reasoning tokens, cost, status, error, timestamp).
- `outputs/metrics_per_cell.csv` — per model × benchmark × language: N, accuracy, mean
  confidence, overconfidence, ECE (15 bins), Brier, failure rate.
- `outputs/overconfidence_gap_by_language.csv` — Δ overconfidence (African − English) per language.
- `outputs/overconfidence_gap_macro.csv` — Δ overconfidence macro-average, with paired bootstrap
  95% CI over the matched question IDs.

## Reading the result
The headline is **Δ overconfidence = African overconfidence − English overconfidence**. A
positive Δ whose 95% CI excludes 0 means the model is reliably more overconfident in African
languages — the paper's pattern, reproduced in a stronger model under a different confidence
estimator.
