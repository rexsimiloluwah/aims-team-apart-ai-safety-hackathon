#!/usr/bin/env bash
# Pull EVERY experiment from the (shared) GCS bucket into ./artifacts/ for the combiner,
# then `make compare` aggregates across all of them.
set -euo pipefail

[ -f .env ] && set -a && . ./.env && set +a
BUCKET="${GCS_BUCKET:?set GCS_BUCKET in .env}"

mkdir -p artifacts
# trailing star: copy each gs://BUCKET/<exp>/ as ./artifacts/<exp>/ (skill #6, no double-nest)
gsutil -m cp -r "gs://${BUCKET}/*" ./artifacts/
echo "downloaded all experiments from gs://${BUCKET} -> ./artifacts/"
find artifacts -maxdepth 1 -type d ! -name artifacts ! -name '_*' | sort
