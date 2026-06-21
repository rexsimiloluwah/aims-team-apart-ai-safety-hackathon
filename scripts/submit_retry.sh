#!/usr/bin/env bash
# Launch one experiment, retrying across zones until one has GPU stock (A100 stockouts
# are common). Usage:
#   scripts/submit_retry.sh <model> <hardware> <dataset> [zone1 zone2 ...]
# Defaults to the us-central1 zones (where this project has A100 quota).
set -uo pipefail

MODEL="${1:?usage: submit_retry.sh <model> <hardware> <dataset> [zones...]}"
HARDWARE="${2:?missing hardware}"
DATASET="${3:?missing dataset}"
shift 3
ZONES=("$@")
[ "${#ZONES[@]}" -eq 0 ] && ZONES=(us-central1-a us-central1-b us-central1-c us-central1-f)

for z in "${ZONES[@]}"; do
  echo "=== [${MODEL}/${DATASET}] trying zone ${z} ==="
  if GCE_ZONE="${z}" bash ./scripts/gce_submit.sh "${MODEL}" "${HARDWARE}" "${DATASET}"; then
    echo "=== [${MODEL}/${DATASET}] LAUNCHED in ${z} ==="
    exit 0
  fi
  echo "=== [${MODEL}/${DATASET}] zone ${z} failed (stockout/error); trying next ==="
done

echo "=== [${MODEL}/${DATASET}] ALL ZONES FAILED ==="
exit 1
