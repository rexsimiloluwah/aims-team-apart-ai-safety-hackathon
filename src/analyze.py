"""Per-experiment analysis: metric tables, figures, and the deployment card.

    uv run python -m src.analyze --experiment qwen3_4b_instruct_afrimmlu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.figures import (
    save_language_bars,
    save_language_radar,
    save_reliability_grid,
    save_risk_coverage,
    save_temperature_scaling,
    save_topic_calibration_radar,
    save_uncertainty_measures,
)
from src.analysis.metrics_tables import (
    confidence_correct,
    logit_vs_verbalized,
    per_domain_accuracy,
    per_language_metrics,
    uncertainty_measure_comparison,
)
from src.metrics.selective import build_deployment_card
from src.metrics.temperature import DEFAULT_GRID, temperature_scaling_table
from src.utils.io import experiment_paths, load_predictions

DATASET_PRETTY = {"afrimmlu": "AfriMMLU", "uhura_truthfulqa": "Uhura-TruthfulQA"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _temp_scaling_params() -> dict:
    """Read configs/uq/temperature_scaling.yaml if present; else sensible defaults."""
    params = {"calibration_fraction": 0.5, "grid": DEFAULT_GRID, "n_bins": 15, "seed": 1234}
    cfg_path = REPO_ROOT / "configs" / "uq" / "temperature_scaling.yaml"
    if cfg_path.exists():
        try:
            from omegaconf import OmegaConf

            c = OmegaConf.load(cfg_path)
            params["calibration_fraction"] = float(c.get("calibration_fraction", 0.5))
            params["grid"] = [float(x) for x in c.get("temperature_grid", DEFAULT_GRID)]
            params["n_bins"] = int(c.get("n_bins", 15))
        except Exception:  # noqa: BLE001
            pass
    return params


def _title_label(exp_dir: Path, experiment: str) -> str:
    """Human label like 'qwen3-4b-instruct on AfriMMLU' from the saved run config."""
    cfg_path = exp_dir / "config.yaml"
    if cfg_path.exists():
        try:
            from omegaconf import OmegaConf

            c = OmegaConf.load(cfg_path)
            ds = DATASET_PRETTY.get(str(c.dataset.name), str(c.dataset.name))
            return f"{c.model.name} on {ds}"
        except Exception:  # noqa: BLE001
            pass
    return experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze one experiment.")
    ap.add_argument("--experiment", required=True, help="<model>_<dataset>")
    ap.add_argument("--artifacts-dir", default="./artifacts")
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--coverage-targets", default="0.5,0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    exp_dir = Path(args.artifacts_dir) / args.experiment
    paths = experiment_paths(exp_dir)
    preds = load_predictions(paths["results"])

    lang_df = per_language_metrics(preds, n_bins=args.n_bins, bootstrap_n=args.bootstrap)
    lang_df.to_csv(paths["results"] / "metrics_by_language.csv", index=False)
    print(lang_df.to_string(index=False))

    dom_df = per_domain_accuracy(preds)
    if not dom_df.empty:
        dom_df.to_csv(paths["results"] / "accuracy_by_domain.csv", index=False)

    lv = logit_vs_verbalized(preds, n_bins=args.n_bins)
    if not lv.empty:
        lv.to_csv(paths["results"] / "logit_vs_verbalized.csv", index=False)
        print("\nLogit vs verbalized:\n", lv.to_string(index=False))

    # figures (all labelled with the dataset)
    label = _title_label(exp_dir, args.experiment)
    save_reliability_grid(preds, paths["figures"], n_bins=args.n_bins, title_label=label)
    save_risk_coverage(preds, paths["figures"], title_label=label)
    save_language_bars(lang_df, paths["figures"], title_label=label)
    save_language_radar(lang_df, paths["figures"], title_label=label)
    save_topic_calibration_radar(preds, paths["figures"], title_label=label)  # Uhura topic / AfriMMLU subject

    # deployment card (per-language abstention thresholds)
    langs = sorted({p["language"] for p in preds})
    per_lang = {lang: confidence_correct(preds, language=lang) for lang in langs}
    targets = [float(x) for x in args.coverage_targets.split(",")]
    card = pd.DataFrame(build_deployment_card(per_lang, targets))
    card.to_csv(paths["results"] / "deployment_card.csv", index=False)

    # which uncertainty signal best detects errors (max-prob vs entropy vs margin)?
    um = uncertainty_measure_comparison(preds)
    if not um.empty:
        um.to_csv(paths["results"] / "uncertainty_measures.csv", index=False)
        print("\nUncertainty-measure error-detection AUROC:\n", um.to_string(index=False))
        save_uncertainty_measures(um, paths["figures"], title_label=label)

    # temperature-scaling mitigation: post-hoc recalibration, analysis-only (no GPU).
    ts = temperature_scaling_table(preds, **_temp_scaling_params())
    if not ts.empty:
        ts.to_csv(paths["results"] / "temperature_scaling.csv", index=False)
        print("\nTemperature scaling (per-language; ECE before -> after):\n", ts.to_string(index=False))
        save_temperature_scaling(ts, paths["figures"], title_label=label)

    print(f"\nFigures + tables + deployment card written to {exp_dir}")


if __name__ == "__main__":
    main()
