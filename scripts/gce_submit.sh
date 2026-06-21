#!/usr/bin/env bash
# Launch a self-deleting GCE GPU run for one (model, dataset).
# Usage: ./scripts/gce_submit.sh <model_config> <hardware: a100|l4> <dataset_config>
#   GCE_ZONE=... overrides .env zone (use for GPU stockouts).
set -euo pipefail

MODEL="${1:-qwen3_4b_instruct}"
HARDWARE="${2:-a100}"
DATASET="${3:-afrimmlu}"

# --- env ---
# capture command-line overrides BEFORE sourcing .env (sourcing would otherwise clobber them)
_CLI_ZONE="${GCE_ZONE:-}"
_CLI_IMAGE_FAMILY="${GCE_IMAGE_FAMILY:-}"
[ -f .env ] && set -a && . ./.env && set +a
PROJECT_ID="${GCE_PROJECT_ID:?set GCE_PROJECT_ID in .env}"
BUCKET="${GCS_BUCKET:?set GCS_BUCKET in .env}"
# precedence: command-line override -> .env -> default
ZONE="${_CLI_ZONE:-${GCE_ZONE:-us-central1-a}}"
IMAGE_FAMILY="${_CLI_IMAGE_FAMILY:-${GCE_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}}"
IMAGE_PROJECT="deeplearning-platform-release"
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_MODE="${WANDB_MODE:-disabled}"

EXPERIMENT="${MODEL}_${DATASET}"
# instance names: lowercase, alnum+hyphen only, no trailing hyphen, <=63 chars (skill #3)
INSTANCE_NAME="cw-$(echo "${MODEL}-${DATASET}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed -E 's/-+/-/g; s/-+$//' | cut -c1-50 | sed -E 's/-+$//')"
REMOTE_DIR="/home/\$USER/cw"
SESSION="cw"

# --- hardware mapping (shell needs machine/accelerator; Python reads batch_size from the YAML) ---
case "${HARDWARE}" in
  a100) MACHINE="a2-highgpu-1g"; ACCEL="type=nvidia-tesla-a100,count=1" ;;
  l4)   MACHINE="g2-standard-8"; ACCEL="type=nvidia-l4,count=1" ;;
  *) echo "Unknown hardware '${HARDWARE}' (a100|l4)"; exit 1 ;;
esac

echo "== submit ${EXPERIMENT} on ${HARDWARE} (${INSTANCE_NAME}) zone=${ZONE} =="

# --- preflight: image family current? (skill #1) ---
if ! gcloud compute images describe-from-family "${IMAGE_FAMILY}" \
      --project="${IMAGE_PROJECT}" >/dev/null 2>&1; then
  echo "WARNING: image family ${IMAGE_FAMILY} not found. Current families:"
  gcloud compute images list --project="${IMAGE_PROJECT}" \
    --filter="family ~ common-cu" --format="table(family,name,status)" | head -10
  exit 1
fi

# --- bucket (skill #3/#4) ---
# describe (own bucket) -> else create (own project) -> else assume it's a teammate's shared
# bucket we have objectAdmin on (we can write objects even without buckets.get).
gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://${BUCKET}" --project="${PROJECT_ID}" --location=US 2>/dev/null \
  || echo "NOTE: cannot describe/create gs://${BUCKET}; assuming it is a shared bucket you can write to."

# --- IAP firewall rule (skill #2) ---
gcloud compute firewall-rules describe allow-iap-ssh --project="${PROJECT_ID}" >/dev/null 2>&1 \
  || gcloud compute firewall-rules create allow-iap-ssh --project="${PROJECT_ID}" \
       --direction=INGRESS --action=ALLOW --rules=tcp:22 \
       --source-ranges=35.235.240.0/20 --description="Allow SSH via IAP tunnel"

# --- create instance (scopes incl storage-full so the final upload works; skill #9) ---
gcloud compute instances create "${INSTANCE_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" \
  --machine-type="${MACHINE}" \
  --accelerator="${ACCEL}" \
  --maintenance-policy=TERMINATE \
  --image-family="${IMAGE_FAMILY}" --image-project="${IMAGE_PROJECT}" \
  --boot-disk-size=100GB \
  --scopes=storage-full,cloud-platform \
  --metadata="install-nvidia-driver=True"

cleanup_on_fail() {
  echo "submit failed/aborted before launch; deleting ${INSTANCE_NAME}"
  gcloud compute instances delete "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" -q || true
}
# EXIT (not just ERR) so an SSH-wait `exit 1` or any pre-launch failure cannot orphan a billing instance
trap cleanup_on_fail EXIT

# --- wait for SSH (GPU drivers; skill #4) ---
echo "waiting for SSH..."
for i in $(seq 1 12); do
  if gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
       --tunnel-through-iap --command="echo ready" >/dev/null 2>&1; then
    echo "ssh ready"; break
  fi
  [ "${i}" -eq 12 ] && { echo "ERROR: SSH timeout"; exit 1; }
  sleep 20
done

# --- upload project as a single tarball (skill #5) ---
TARBALL="/tmp/cw_upload.tar.gz"
tar czf "${TARBALL}" \
  --exclude='.venv' --exclude='.git' --exclude='artifacts' --exclude='.env' \
  --exclude='__pycache__' --exclude='claude-docs' --exclude='.pytest_cache' \
  --exclude='wandb' .
gcloud compute scp --project="${PROJECT_ID}" --zone="${ZONE}" --tunnel-through-iap \
  "${TARBALL}" "${INSTANCE_NAME}:/tmp/cw_upload.tar.gz"
rm -f "${TARBALL}"

# --- run inside tmux, upload artifacts, self-delete (skills #7, #6, #14, #15) ---
gcloud compute ssh "${INSTANCE_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
  --tunnel-through-iap --command="
    set -e
    mkdir -p ${REMOTE_DIR}
    tar xzf /tmp/cw_upload.tar.gz -C ${REMOTE_DIR}
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=\$HOME/.local/bin:\$PATH
    cd ${REMOTE_DIR}
    uv sync
    tmux new-session -d -s ${SESSION} \"
      export PATH=\\\$HOME/.local/bin:/snap/bin:/usr/bin:\\\$PATH &&
      export HF_TOKEN='${HF_TOKEN}' &&
      export WANDB_API_KEY='${WANDB_API_KEY}' &&
      export WANDB_MODE='${WANDB_MODE}' &&
      export PYTHONUNBUFFERED=1 &&
      cd ${REMOTE_DIR} &&
      bash scripts/run_experiment.sh ${MODEL} ${DATASET} ${HARDWARE} ${EXPERIMENT} 2>&1 | tee experiment.log ;
      gsutil -m cp -r artifacts/${EXPERIMENT}/* gs://${BUCKET}/${EXPERIMENT}/ 2>/dev/null || true ;
      gsutil cp experiment.log gs://${BUCKET}/${EXPERIMENT}/experiment.log 2>/dev/null || true ;
      gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} -q
    \"
  "

trap - EXIT  # tmux launched successfully; the instance now self-deletes when the job finishes
echo "== launched in tmux on ${INSTANCE_NAME}. =="
echo "watch:   gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --tunnel-through-iap -- -t 'tmux attach -t ${SESSION}'"
echo "results: gs://${BUCKET}/${EXPERIMENT}/  (instance self-deletes when done)"
