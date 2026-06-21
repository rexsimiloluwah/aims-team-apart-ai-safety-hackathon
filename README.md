# Confidently Wrong: Evaluating Calibration Risks and Safe Abstention in African-Language Large Language Models

*Apart Global South AI Safety Hackathon 2026 - Africa Track*

**Factuality decay, miscalibration, and safe abstention in African-language MCQA.**

A judge-free, reproducible study: open-weight models answer multilingual multiple-choice
questions, confidence is read directly from the logits, and we show that in low-resource
African languages models stay confident as they lose accuracy ("confidently wrong"), then
mitigate it with selective abstention. Scored by exact match; everything runs offline on
open weights.

**Three-act spine**
1. **Decay** - accuracy drops English → low-resource.
2. **Overconfidence (the contribution)** - confidence does *not* fall as fast as accuracy, so
   calibration error grows as resource falls.
3. **Mitigation** - abstain below a confidence threshold; ship a per-language threshold
   deployment card.

---

## ⚠️ Step 0 - verify the data before any GPU run

A silent input-field drop invalidates an entire run. **Before** launching anything:

```bash
make setup
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/smoke_test.ipynb
```

The smoke notebook loads two languages per dataset, prints a full prompt verbatim, confirms
every field is present (Uhura's variable candidate list + correct index; AfriMMLU's 4 options +
answer letter), and checks English accuracy is plausible on one small model. Treat any
implausible number as a data defect until proven otherwise.

## Quickstart

```bash
cp .env.example .env          # fill in GCE_PROJECT_ID, GCS_BUCKET, HF_TOKEN
make setup                    # uv sync
make test                     # metrics unit tests (no GPU)
make smoke                    # end-to-end smoke (small, CPU-friendly)

# local run (needs a GPU for the real models)
make run MODEL=qwen3_4b_instruct DATASET=afrimmlu HARDWARE=a100

# GCE run (self-deleting instance; GCE_ZONE overrides .env)
make gce-submit MODEL=qwen3_4b_instruct DATASET=afrimmlu HARDWARE=a100
make download EXP=qwen3_4b_instruct_afrimmlu
make analyze MODEL=qwen3_4b_instruct DATASET=afrimmlu
make compare
```

## Datasets (both public, already in African languages, label-free)

| Dataset | HF id | Role | Format |
|---|---|---|---|
| Uhura-TruthfulQA | `masakhane/uhura-truthfulqa` | safety-domain truthfulness (primary) | MC1: variable candidates, one correct |
| AfriMMLU | `masakhane/afrimmlu` | clean fixed-option calibration + breadth | fixed 4-way A/B/C/D |

> Original English-only TruthfulQA (Lin et al. 2022) must **not** be used for African coverage -
> only the Uhura human-translated version.

## Models (open weights only - confidence lives in the logits)

| Config | HF id | Reasoning | Notes |
|---|---|---|---|
| `qwen3_4b_instruct` | `Qwen/Qwen3-4B-Instruct-2507` | no | non-thinking workhorse |
| `qwen3_4b_thinking` | `Qwen/Qwen3-4B-Thinking-2507` | yes | reasoning-vs-non-reasoning experiment (E6) |
| `gemma4_12b` | `google/gemma-4-12B-it` | no | newest, strong African coverage |
| `afrollama_v1` | `Jacaranda/AfroLlama_V1` | no | built-for-Africa anchor; supports Swahili, Zulu, Yoruba, Hausa only |
| `aya_expanse_8b` | `CohereLabs/aya-expanse-8b` | no | multilingual comparator (optional, time-permitting) |

## Prompts & confidence modes

Each dataset uses its canonical method:
- **AfriMMLU** -> `fixed_option`: IrokoBench Table 12 template t1 (English, zero-shot, `{subject}` +
  lettered options + `Answer:`); next-token logits restricted to the option-letter tokens (leading-space
  " A".." D"), softmax over options. One forward pass per item.
- **Uhura-TruthfulQA** -> `mc1`: canonical TruthfulQA MC1. A QA stem (`Q: .. / A:`); confidence is the
  softmax over each candidate answer's **length-normalized log-prob**. This avoids the confidence
  saturation that letter-scoring caused on Uhura's many-option questions (which flatlined abstention).
  Candidates are still deterministically shuffled (seeded by language+question), since the released
  `mc1_targets` list the correct answer first and the reasoning path letters them.
- **self-consistency** (Qwen3-4B-Thinking, either dataset): sample N traces over a lettered list,
  confidence = modal-answer agreement; also parse a verbalized "0-100%".

## Layout

See `claude-docs/PROJECT_PLAN_ConfidentlyWrong.md` for the full build plan (gitignored).
`src/` holds the package; `configs/` is Hydra; `artifacts/<model>_<dataset>/` holds outputs.
