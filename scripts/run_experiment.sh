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

# Strip stray carriage returns: a Windows (CRLF) checkout can leave trailing \r on the args,
# which corrupts the Hydra overrides (e.g. `model=afrollama_v1\r` -> LexerNoViableAltException).
MODEL="${MODEL//$'\r'/}"
DATASET="${DATASET//$'\r'/}"
HARDWARE="${HARDWARE//$'\r'/}"
EXPERIMENT="${EXPERIMENT//$'\r'/}"

# Optional per-language sample cap (LIMIT env). Used mainly to keep the self-consistency
# "thinking" runs feasible - the full Uhura/AfriMMLU set would take tens of hours otherwise.
LIMIT="${LIMIT:-}"; LIMIT="${LIMIT//$'\r'/}"
LIMIT_ARG=""
[ -n "${LIMIT}" ] && LIMIT_ARG="+limit=${LIMIT}"

# Optional few-shot (N_SHOTS env): in-context examples per language. run.py suffixes the
# experiment dir with _Nshot to match the EXPERIMENT id gce_submit derives.
N_SHOTS="${N_SHOTS:-}"; N_SHOTS="${N_SHOTS//$'\r'/}"
SHOTS_ARG=""
[ -n "${N_SHOTS}" ] && SHOTS_ARG="n_shots=${N_SHOTS}"

echo "[run_experiment] inference: model=${MODEL} dataset=${DATASET} hardware=${HARDWARE} ${LIMIT_ARG} ${SHOTS_ARG}"
uv run python -m src.run model="${MODEL}" dataset="${DATASET}" hardware="${HARDWARE}" ${LIMIT_ARG} ${SHOTS_ARG}
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "[run_experiment] src.run exited ${rc}; skipping analyze (no predictions to analyze)."
  exit "${rc}"
fi

echo "[run_experiment] analyze: ${EXPERIMENT}"
uv run python -m src.analyze --experiment "${EXPERIMENT}"
