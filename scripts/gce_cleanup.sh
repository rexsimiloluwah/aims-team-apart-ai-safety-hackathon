#!/usr/bin/env bash
# Delete a specific instance (use if a run failed before self-delete).
# Usage: ./scripts/gce_cleanup.sh <instance-name> [zone]
set -euo pipefail

INSTANCE="${1:?usage: gce_cleanup.sh <instance-name> [zone]}"
[ -f .env ] && set -a && . ./.env && set +a
PROJECT_ID="${GCE_PROJECT_ID:?set GCE_PROJECT_ID in .env}"
ZONE="${2:-${GCE_ZONE:-us-central1-a}}"

gcloud compute instances delete "${INSTANCE}" --project="${PROJECT_ID}" --zone="${ZONE}" -q
echo "deleted ${INSTANCE} (${ZONE})"
