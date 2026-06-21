#!/usr/bin/env bash
# Run ONE experiment end-to-end on the GPU box: inference (src.run) then analyze (src.analyze).
#
# Invoked by scripts/gce_submit.sh inside tmux as:
#   bash scripts/run_experiment.sh <model> <dataset> <hardware> <experiment> 2>&1 | tee experiment.log
#
# Kept as a committed file (not an inline SSH heredoc) so the run->analyze logic is readable and
# NOT subject to triple shell-escaping. Output flows to stdout so the tmux pane shows it live and
# `tee` also writes experiment.log. analyze is SKIPPED if inference fails, so a failed run does not
# bury the real error under a misleading "predictions.jsonl not found" traceback.
#
# NOTE: deliberately no `set -e` - we must capture src.run's exit code to decide whether to analyze.
set -uo pipefail

MODEL="${1:?usage: run_experiment.sh <model> <dataset> <hardware> <experiment>}"
DATASET="${2:?missing dataset}"
HARDWARE="${3:?missing hardware}"
EXPERIMENT="${4:?missing experiment}"

echo "[run_experiment] inference: model=${MODEL} dataset=${DATASET} hardware=${HARDWARE}"
uv run python -m src.run model="${MODEL}" dataset="${DATASET}" hardware="${HARDWARE}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "[run_experiment] src.run exited ${rc}; skipping analyze (no predictions to analyze)."
  exit "${rc}"
fi

echo "[run_experiment] analyze: ${EXPERIMENT}"
uv run python -m src.analyze --experiment "${EXPERIMENT}"
