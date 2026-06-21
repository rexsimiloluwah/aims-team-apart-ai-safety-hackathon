"""Hydra entrypoint: load data -> run inference -> compute per-language metrics -> save.

    uv run python -m src.run model=qwen3_4b_instruct dataset=afrimmlu hardware=a100
    uv run python -m src.run model=qwen3_4b_instruct dataset=afrimmlu +limit=20   # quick (per-language)
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.analysis.metrics_tables import per_domain_accuracy, per_language_metrics
from src.data.loader import load_samples
from src.models.inference import ModelRunner
from src.utils.io import (
    experiment_paths,
    load_dotenv,
    load_predictions,
    save_json,
    save_predictions,
)

log = logging.getLogger("src.run")

# cw/run.py -> parents[0]=cw, parents[1]=repo root
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(REPO_ROOT / ".env")

    import os

    token = os.environ.get("HF_TOKEN")
    languages = list(cfg.languages) if cfg.languages is not None else None
    limit = cfg.get("limit", None)

    log.info("Experiment: %s", cfg.experiment.name)
    log.info("Model=%s  Dataset=%s  langs=%s", cfg.model.hf_id, cfg.dataset.hf_id, languages)

    # 1. data
    samples = load_samples(cfg.dataset, languages=languages, token=token, limit=limit)
    log.info("Loaded %d samples", len(samples))

    # 2. model + prompts
    runner = ModelRunner(cfg.model, token=token, seed=int(cfg.experiment.seed))
    prompt_text = (REPO_ROOT / cfg.dataset.prompt_template).read_text()
    thinking_text = (REPO_ROOT / cfg.thinking_prompt_template).read_text()

    # 3. inference
    preds = runner.run(
        samples, prompt_text, thinking_text, batch_size=int(cfg.hardware.batch_size)
    )
    log.info("Produced %d predictions", len(preds))

    # 4. persist
    # anchor artifacts to the repo (not the CWD) so results land consistently
    paths = experiment_paths(REPO_ROOT / cfg.artifacts.experiment_dir)
    # save a fully-resolved snapshot (no ${hydra:...} interpolations to break on reload)
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.save(resolved, paths["base"] / "config.yaml")
    # always persist predictions: they are the core artifact and the metrics step reloads them
    save_predictions(preds, paths)

    # 5. metrics
    pred_dicts = load_predictions(paths["results"])
    n_bins = int(cfg.uq.get("n_bins", 15))
    boot = int(cfg.uq.get("bootstrap_n", 0))
    lang_df = per_language_metrics(pred_dicts, n_bins=n_bins, bootstrap_n=boot)
    lang_df.to_csv(paths["results"] / "metrics_by_language.csv", index=False)
    log.info("\n%s", lang_df.to_string(index=False))

    if bool(getattr(cfg.dataset, "has_domain", False)):
        per_domain_accuracy(pred_dicts).to_csv(
            paths["results"] / "accuracy_by_domain.csv", index=False
        )

    summary = {
        "experiment": cfg.experiment.name,
        "model": cfg.model.hf_id,
        "dataset": cfg.dataset.hf_id,
        "n_samples": len(samples),
        "overall_accuracy": float(lang_df["accuracy"].mean()),
        "mean_ece": float(lang_df["ece"].mean()),
        "mean_overconfidence": float(lang_df["overconfidence"].mean()),
        "languages": list(lang_df["language"]),
    }
    save_json(summary, paths["results"] / "summary.json")

    # 6. optional W&B (mode honored explicitly; skill #15)
    _maybe_wandb(cfg, summary, lang_df)
    log.info("Done -> %s", paths["base"])


def _maybe_wandb(cfg, summary, lang_df) -> None:
    mode = str(cfg.logging.wandb_mode)
    if mode == "disabled":
        return
    try:
        import wandb

        wandb.init(
            project=str(cfg.logging.wandb_project),
            name=str(cfg.experiment.name),
            mode=mode,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.log(summary)
        wandb.log({"metrics_by_language": wandb.Table(dataframe=lang_df)})
        wandb.finish()
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B logging skipped: %s", exc)


if __name__ == "__main__":
    main()
