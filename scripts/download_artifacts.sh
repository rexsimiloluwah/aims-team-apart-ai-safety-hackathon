#!/usr/bin/env bash
# Pull an experiment's artifacts from GCS into ./artifacts/<exp>/.
# Usage: ./scripts/download_artifacts.sh <model>_<dataset>
set -euo pipefail

EXP="${1:?usage: download_artifacts.sh <model>_<dataset>}"
[ -f .env ] && set -a && . ./.env && set +a
BUCKET="${GCS_BUCKET:?set GCS_BUCKET in .env}"

mkdir -p "artifacts/${EXP}"
# trailing star avoids the double-nesting bug (skill #6)
gsutil -m cp -r "gs://${BUCKET}/${EXP}/*" "artifacts/${EXP}/"
echo "downloaded -> artifacts/${EXP}/"
ls -R "artifacts/${EXP}" | head -40
