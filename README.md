# Confidently Wrong: Measuring and Mitigating Calibration Risks in LLMs for African Languages

*Apart Global South AI Safety Hackathon 2026 — Africa Track*

📄 [Report](report/main.pdf)  ·  🖼️ [Slides](slides/main.pdf)

## Problem
- Large language models are increasingly used in African languages, but their reliability there is largely unmeasured.
- The safety failure we study is being **confidently wrong**: a wrong answer delivered with high confidence, which is most harmful where users cannot verify it.

## Methods
- We audit four open-weight LLMs on two African-language benchmarks, scoring each answer option by its length-normalised log-probability (robust to instruction-tuned models) and measuring calibration (ECE, overconfidence, AUROC, AUARC).
- We evaluate three post-hoc mitigations: per-language temperature scaling, selective abstention, and few-shot prompting.

**Models**

| Model | Params | Type | Hugging Face ID |
|---|---|---|---|
| Qwen3-4B-Instruct | 4B | Instruction-tuned | `Qwen/Qwen3-4B-Instruct` |
| Gemma-4-12B | 12B | Instruction-tuned | `google/gemma-4-12b-it` |
| Aya-Expanse-8B | 8B | Multilingual | `CohereLabs/aya-expanse-8b` |
| AfroLlama-V1 | 7B | Africa-specific (LLaMA-2 base) | `Jacaranda/AfroLlama_V1` |

**Datasets**

| Dataset | Hugging Face ID | Task | Coverage / format |
|---|---|---|---|
| Uhura-TruthfulQA | `masakhane/uhura-truthfulqa` | Truthfulness | English + 6 African languages; variable candidates |
| AfriMMLU | `masakhane/afrimmlu` | Knowledge | 18 languages; 4 options |

## What we found
- **Calibration degrades from English to African languages:** ECE and overconfidence rise markedly across models and tasks, and models are most overconfident where they are least accurate.
- **Accuracy collapses on knowledge tasks:** on AfriMMLU the strongest models fall from ~0.52 (English) to ~0.29 (African, near the 0.25 chance floor).
- **Temperature scaling is the best fix:** per-language recalibration cuts ECE to ~0.03 with no retraining; abstention helps less (weak uncertainty signal, AUROC ~0.55); few-shot helps three models but backfires on Gemma-4-12B.
- **Bigger is not safer:** the 12B model is the worst calibrated.

## Significance
- The models are least reliable in exactly the languages where users can least verify answers, so confident misinformation falls hardest on the populations current models serve least.
- A cheap, post-hoc fix already exists: per-language temperature scaling is a deployable safety lever today, best paired with confidence-thresholded abstention.

## Team
| Member | From | Email |
|---|---|---|
| Similoluwa Okunowo | 🇳🇬 Nigeria | similoluwa@aims.ac.za |
| Eliud Koto | 🇰🇪 Kenya | eliud@aims.ac.za |
| Dagmawi Misker | 🇪🇹 Ethiopia | dagmawi@aims.ac.za |

All at the African Institute for Mathematical Sciences (AIMS), South Africa.

## Reproduce

All experiments were run on GCP using VMs equipped with A100 GPUs. Kindly check the `configs/` folder for configuration files to re-run and reproduce these experiments.


All artifacts (predictions, per-language metrics, figures, and the cross-model comparison tables)
are committed under [`artifacts/`](artifacts/): per-experiment outputs in
`artifacts/<model>_<dataset>/`, and aggregated results in `artifacts/_comparison/`.

<details>
<summary><b>Scripts for reproducing our results</b></summary>

All experiments were run on Google Cloud Platform virtual machines with NVIDIA A100 GPUs.
`make gce-submit` launches a self-deleting GPU instance that runs
`uv run python -m src.run model=<model> dataset=<dataset> hardware=a100` and uploads the results.

```bash
# Zero-shot: 4 models x 2 datasets (8 runs)
for model in qwen3_4b_instruct gemma4_12b aya_expanse_8b afrollama_v1; do
  for ds in afrimmlu uhura_truthfulqa; do
    make gce-submit MODEL=$model DATASET=$ds HARDWARE=a100
  done
done

# 5-shot calibration runs, Uhura-TruthfulQA only (4 runs): same run command with n_shots=5
for model in qwen3_4b_instruct gemma4_12b aya_expanse_8b afrollama_v1; do
  uv run python -m src.run model=$model dataset=uhura_truthfulqa hardware=a100 n_shots=5
done

# Collect results, then build the cross-model tables and figures
make download-all
make compare
```
</details>


## LLM Usage Statement
Claude Code was used to assist with coding the experiments and evaluation pipelines. However, the design,
conceptualization, and review were done by the team.

---

Thanks to **Apart Research** for organising this Hackathon, which created a valuable learning
experience for solving pressing issues in AI safety for African languages like this.
