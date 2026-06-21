#!/usr/bin/env bash
# Safety net: find lingering cw-* instances (self-delete can miss on crash) and
# optionally delete them. Dry-run by default; pass --yes to actually delete.
# Usage: ./scripts/gce_reap_idle.sh [--yes]
set -euo pipefail

CONFIRM="${1:-}"
[ -f .env ] && set -a && . ./.env && set +a
PROJECT_ID="${GCE_PROJECT_ID:?set GCE_PROJECT_ID in .env}"

mapfile -t INSTANCES < <(
  gcloud compute instances list --project="${PROJECT_ID}" \
    --filter="name ~ ^cw-" --format="value(name,zone)"
)

if [ "${#INSTANCES[@]}" -eq 0 ]; then
  echo "no cw-* instances found."
  exit 0
fi

echo "cw-* instances:"
printf '  %s\n' "${INSTANCES[@]}"

if [ "${CONFIRM}" != "--yes" ]; then
  echo "(dry run) re-run with --yes to delete the above."
  exit 0
fi

for row in "${INSTANCES[@]}"; do
  name="$(echo "${row}" | awk '{print $1}')"
  zone="$(echo "${row}" | awk '{print $2}')"
  zone="${zone##*/}"  # zone may be a full URL
  echo "deleting ${name} (${zone})"
  gcloud compute instances delete "${name}" --project="${PROJECT_ID}" --zone="${zone}" -q || true
done
