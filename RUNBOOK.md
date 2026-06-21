# Team Runbook - Confidently Wrong (Africa Track)

Three people, each with their own GCP account, split the model roster. Every experiment runs
**end to end with one command** on a self-deleting GPU instance and uploads its results to a
**shared GCS bucket**. One person then pulls everything and builds the team-wide comparison.

```
clone repo -> setup (Part 1) -> verify (Part 3) -> submit your experiments (Part 4)
            -> results auto-upload to shared GCS -> combiner pulls all + compares (Part 6)
```

> Commands here call **`uv` and the scripts directly** - you do NOT need `make` installed.
> (The repo also ships a `Makefile` with the same targets as a shortcut, e.g.
> `make gce-submit MODEL=.. HARDWARE=.. DATASET=..`, if you happen to have `make`.)

---

## Roles & model split

| Person | Models (x both datasets unless noted) | Runs | Notes |
|---|---|---|---|
| **P1** (bucket owner + combiner) | `qwen3_4b_instruct` (uhura + afrimmlu), `qwen3_4b_thinking` (uhura) | 3 | owns the shared bucket; runs the final combine |
| **P2** | `gemma4_12b` (uhura + afrimmlu), `aya_expanse_8b` (uhura + afrimmlu) | 4 | runs the gated models (Gemma 4 + Aya) - covered by the shared token |
| **P3** | `afrollama_v1` (uhura + afrimmlu), `qwen3_4b_thinking` (afrimmlu) | 3 | owns the second (expensive) thinking run |

10 experiments total = 5 models x 2 datasets. The two `qwen3_4b_thinking` runs are the expensive
ones (self-consistency); they are split across P1 and P3.

---

## Part 1 - One-time setup (EVERY person)

**1. Install prerequisites** (local machine):
- `git`, `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`), Google Cloud SDK (`gcloud`).
- The team's **shared HuggingFace token** (one token for everyone - see step 3).

**2. Clone the repo and create the env**
```bash
git clone <repo-url> confidently-wrong && cd confidently-wrong
uv sync --extra notebook --extra dev    # installs from the committed uv.lock -> identical env for everyone
```

**3. HuggingFace access (one shared team token)**
- The whole team uses a **single shared HF token**. **One** account (e.g. P1's) accepts the gated
  model licenses **once**, on these pages:
  - `https://huggingface.co/google/gemma-4-12B-it`
  - `https://huggingface.co/CohereLabs/aya-expanse-8b`
  (Qwen3-4B and AfroLlama are ungated.)
- That same account creates a **read token** at `https://huggingface.co/settings/tokens` and shares
  it with the team. Everyone puts it in `.env` as `HF_TOKEN` (step 5) - no per-person license
  acceptance needed. The token is passed to the GPU instance by the submit script.

**4. Configure your GCP project**
```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud services enable compute.googleapis.com
gcloud auth application-default login          # SDK clients use ADC

# let instances self-delete when a job finishes (otherwise they bill forever)
PROJ_NUM=$(gcloud projects describe <YOUR_PROJECT_ID> --format='value(projectNumber)')
gcloud projects add-iam-policy-binding <YOUR_PROJECT_ID> \
  --member="serviceAccount:${PROJ_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1"
```
> Make sure you have **A100 (or L4) GPU quota** in your zone. If not, request it in the GCP console
> (Compute Engine -> Quotas -> "NVIDIA A100 GPUs") or pass a zone that has stock at submit time.

**5. Create `.env`**
```bash
cp .env.example .env
```
Edit `.env`:
```
GCE_PROJECT_ID=<YOUR_PROJECT_ID>        # your own project (you pay your own GPU)
GCS_BUCKET=<SHARED_BUCKET>              # the SAME bucket for all three (see Part 2)
GCE_ZONE=us-central1-a                  # any zone with GPU stock; override per-submit if needed
HF_TOKEN=hf_xxx                         # the SHARED team token (its account accepted Gemma 4 + Aya licenses)
WANDB_MODE=disabled
```

---

## Part 2 - Shared GCS bucket (P1 does this once, then grants P2 & P3)

**P1 creates the shared bucket** (name must be globally unique, no underscores):
```bash
gcloud storage buckets create gs://cw-<p1-project-id> --project=<p1-project-id> --location=US
```
Put `GCS_BUCKET=cw-<p1-project-id>` in **everyone's** `.env`.

**P2 and P3 send P1 their project NUMBER** (`gcloud projects describe <proj> --format='value(projectNumber)'`),
and **P1 grants each one write access** to the bucket:
```bash
# run once per teammate, by P1
gcloud storage buckets add-iam-policy-binding gs://cw-<p1-project-id> \
  --member="serviceAccount:<TEAMMATE_PROJECT_NUMBER>-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```
> Never share service-account key files. Project number only. (See the gcp-ml-experiments skill #8.)
>
> **Simpler fallback (no cross-project IAM):** each person sets `GCS_BUCKET` to *their own* bucket
> (the submit script auto-creates it), runs `./scripts/download_artifacts.sh <model>_<dataset>` for
> their own results, then sends their `artifacts/<exp>/` folders to P1 to combine. Use this if the
> IAM grant is fiddly.

---

## Part 3 - Verify your setup (EVERY person, before spending GPU)

Fast plumbing check (no big downloads - uses a tiny stand-in model):
```bash
uv run pytest -q          # metric unit tests
SMOKE_MODEL_ID=hf-internal-testing/tiny-random-LlamaForCausalLM \
  uv run jupyter nbconvert --to notebook --execute --inplace notebooks/smoke_test.ipynb
```
This confirms data loads, both scoring paths work, and the full `run -> analyze -> compare`
pipeline executes. Green = your code/data is good.

To confirm **your real models are accessible**, do one tiny real submit (it self-deletes).
The submit script is positional: `./scripts/gce_submit.sh <model> <hardware> <dataset>`.
```bash
./scripts/gce_submit.sh <one-of-your-models> a100 uhura_truthfulqa
```
Watch the first one: it should SSH in, run, upload to `gs://$GCS_BUCKET/<exp>/`, and delete the
instance. A gated-model `401/403` here means the shared token's account hasn't accepted the
license (step 3), or `HF_TOKEN` is wrong in `.env`.

---

## Part 4 - Run your experiments (one command each, end to end)

`./scripts/gce_submit.sh <model> <hardware> <dataset>` does the whole thing on a fresh self-deleting
instance: **provision GPU -> upload code -> `uv sync` -> run inference -> analyze (figures+tables+
deployment card) -> upload `artifacts/<exp>/` to GCS -> delete the instance.** One command = one
finished experiment in the shared bucket. Different experiments get different instance names, so you
can launch several at once if you have quota.

**P1:**
```bash
./scripts/gce_submit.sh qwen3_4b_instruct a100 uhura_truthfulqa
./scripts/gce_submit.sh qwen3_4b_instruct a100 afrimmlu
./scripts/gce_submit.sh qwen3_4b_thinking a100 uhura_truthfulqa   # expensive (~hrs)
```

**P2:**
```bash
./scripts/gce_submit.sh gemma4_12b     a100 uhura_truthfulqa
./scripts/gce_submit.sh gemma4_12b     a100 afrimmlu
./scripts/gce_submit.sh aya_expanse_8b a100 uhura_truthfulqa
./scripts/gce_submit.sh aya_expanse_8b a100 afrimmlu
```

**P3:**
```bash
./scripts/gce_submit.sh afrollama_v1      a100 uhura_truthfulqa
./scripts/gce_submit.sh afrollama_v1      a100 afrimmlu
./scripts/gce_submit.sh qwen3_4b_thinking a100 afrimmlu           # expensive (~hrs)
```

- GPU stockout? Prepend a different zone: `GCE_ZONE=us-central1-b ./scripts/gce_submit.sh ...`.
- `l4` (2nd arg) is a cheaper option for the <=8B non-thinking models; use `a100` for `gemma4_12b`
  and the `qwen3_4b_thinking` runs.

---

## Part 5 - Monitor & recover

```bash
# watch a running job
gcloud compute ssh cw-<model>-<dataset> --zone=$GCE_ZONE --tunnel-through-iap -- -t 'tmux attach -t cw'

# results land here as each job finishes
gsutil ls gs://$GCS_BUCKET/

# a job died before self-deleting? clean up any leftover instances:
./scripts/gce_reap_idle.sh            # dry run (lists cw-* instances)
./scripts/gce_reap_idle.sh --yes      # delete them
```
Re-running the same submit overwrites that experiment's outputs - safe to retry.

---

## Part 6 - Combine (P1, once everything has finished)

After all 10 experiments are in the shared bucket:
```bash
./scripts/download_all.sh             # pulls every gs://$GCS_BUCKET/<exp>/ into ./artifacts/
uv run python -m src.compare          # cross-model x dataset: heatmaps, decay lines, radar, all_metrics.csv
```
Outputs land in `artifacts/_comparison/` (the team-wide figures + `all_metrics.csv`), while each
experiment's own figures/tables stay in `artifacts/<model>_<dataset>/`. That's the material for the report.

> Everyone can also pull just their own results locally with
> `./scripts/download_artifacts.sh <model>_<dataset>`.

---

## Appendix - quick reference

**Datasets:** `uhura_truthfulqa` (7 langs, ~5.6k items, answer-text MC1, topic radar) ;
`afrimmlu` (18 langs, 9k items, 4-option letter scoring, subject breakdown).

**Per experiment you get:** `metrics_by_language.csv`, `deployment_card.csv`, reliability grid,
risk-coverage curve, accuracy/ECE bars, language radar, and (Uhura/AfriMMLU) a topic/subject
calibration radar.

**Per-experiment / local commands (no make):**
```bash
uv run python -m src.run model=<m> dataset=<d> hardware=<h>     # one experiment locally (needs a GPU)
uv run python -m src.analyze --experiment <m>_<d>              # figures + tables for one experiment
uv run python -m src.compare                                  # aggregate everything under artifacts/
```

**Troubleshooting:**
- `401/403` on model download -> the shared token's account hasn't accepted the gated license (step 3),
  or `HF_TOKEN` is missing/wrong in `.env`.
- `4003 failed to connect to backend` on SSH -> missing IAP firewall rule (the submit script creates
  `allow-iap-ssh`; ensure it ran, or create it manually per the gcp skill).
- bucket "AccessDenied" on upload -> P1 hasn't granted your compute SA `objectAdmin` (Part 2).
- GPU "ZONE_RESOURCE_POOL_EXHAUSTED" -> try another `GCE_ZONE`.
