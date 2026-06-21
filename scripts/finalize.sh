#!/usr/bin/env bash
# Upload artifacts + log to GCS, then delete the instance.
#
# Invoked by scripts/gce_submit.sh inside tmux, AFTER run_experiment.sh, as:
#   bash scripts/finalize.sh <instance> <zone> <bucket> <experiment> <rc>
#
# On SUCCESS (rc==0): upload then delete immediately (normal self-deleting behaviour).
# On FAILURE (rc!=0): upload the log, then KEEP the instance alive for a grace window
#   (default 30 min, override with KEEPALIVE_SECONDS) so the error can be inspected over
#   SSH before the box is reaped. Self-deleting on error otherwise destroys the only log.
set -uo pipefail

INSTANCE="${1:?usage: finalize.sh <instance> <zone> <bucket> <experiment> <rc>}"
ZONE="${2:?missing zone}"
BUCKET="${3:?missing bucket}"
EXPERIMENT="${4:?missing experiment}"
RC="${5:-0}"
GRACE="${KEEPALIVE_SECONDS:-1800}"

# best-effort upload (do not let a missing artifacts dir block the delete)
gsutil -m cp -r "artifacts/${EXPERIMENT}/"* "gs://${BUCKET}/${EXPERIMENT}/" 2>/dev/null || true
gsutil cp experiment.log "gs://${BUCKET}/${EXPERIMENT}/experiment.log" 2>/dev/null || true

if [ "${RC}" -eq 0 ]; then
  echo "[finalize] success (rc=0); deleting ${INSTANCE}"
else
  echo "[finalize] FAILED rc=${RC}; keeping ${INSTANCE} alive ${GRACE}s for inspection, then deleting."
  echo "[finalize] inspect:  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --tunnel-through-iap -- 'cat ~/cw/experiment.log'"
  sleep "${GRACE}"
fi

gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" -q
