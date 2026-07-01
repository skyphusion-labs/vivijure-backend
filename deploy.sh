#!/usr/bin/env bash
# vivijure-backend -- one-script deploy of the GPU render engine.
#
# Supply your keys in deploy.env (copy deploy.env.example), then run:  ./deploy.sh
# It stands up (or updates) a RunPod Serverless endpoint running the published, model-baked
# vivijure-backend image, wires in your R2 storage keys, and prints the ENDPOINT ID you paste
# into the Studio's deploy.env as RUNPOD_ENDPOINT_ID.
#
# It is idempotent and re-runnable: run it again after changing a value in deploy.env and it
# updates the same template + endpoint in place. It FAILS CLOSED: any error stops the whole run
# so you never end up with a half-built endpoint. Read docs/deploy.md for what each key is and why.
#
# You do NOT build anything. The image (ghcr.io/skyphusion-labs/vivijure-backend) is public and
# already carries every model weight, so the only thing this script needs from you is keys.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

say()  { printf "\n==> %s\n" "$*"; }
info() { printf "    %s\n" "$*"; }
die()  { printf "\nERROR: %s\n" "$*" >&2; exit 1; }

API="https://rest.runpod.io/v1"

# Strip an accidental "NAME=value" prefix and all surrounding whitespace/newlines, so a stray
# paste cannot poison a stored value.
strip_val() { printf "%s" "$1" | cut -d= -f2- | tr -d '[:space:]'; }

# ---- 0. load and check deploy.env -------------------------------------------
[ -f deploy.env ] || die "deploy.env not found. Run: cp deploy.env.example deploy.env  (then edit it)."
set -a; . ./deploy.env; set +a

# Accept the Studio's R2_S3_* names too, so you can reuse the same values you gave the Studio.
R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-${R2_S3_ACCESS_KEY_ID:-}}"
R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-${R2_S3_SECRET_ACCESS_KEY:-}}"
R2_BUCKET="${R2_BUCKET:-${R2_S3_BUCKET:-vivijure}}"
R2_ENDPOINT="${R2_ENDPOINT:-${R2_S3_ENDPOINT:-}}"

# Build R2_ENDPOINT from the account id when it was left blank.
if [ -z "${R2_ENDPOINT:-}" ] && [ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  R2_ENDPOINT="https://$(strip_val "$CLOUDFLARE_ACCOUNT_ID").r2.cloudflarestorage.com"
fi

need() { local v; eval "v=\${$1:-}"; [ -n "$v" ] || die "deploy.env: $1 is required but empty -- $2"; }
need RUNPOD_API_KEY        "your RunPod API key (runpod.io -> Settings -> API Keys)"
need R2_ACCESS_KEY_ID      "R2 access key id (Cloudflare dash -> R2 -> Manage R2 API Tokens)"
need R2_SECRET_ACCESS_KEY  "R2 secret access key"
need R2_ENDPOINT           "R2 S3 endpoint, or set CLOUDFLARE_ACCOUNT_ID so it can be built for you"

RUNPOD_API_KEY="$(strip_val "$RUNPOD_API_KEY")"
R2_ACCESS_KEY_ID="$(strip_val "$R2_ACCESS_KEY_ID")"
R2_SECRET_ACCESS_KEY="$(strip_val "$R2_SECRET_ACCESS_KEY")"
R2_BUCKET="$(strip_val "$R2_BUCKET")"

BACKEND_IMAGE="${BACKEND_IMAGE:-ghcr.io/skyphusion-labs/vivijure-backend:0.3.3}"
ENDPOINT_NAME="${ENDPOINT_NAME:-vivijure-backend}"
WORKERS_MAX="${WORKERS_MAX:-3}"
WORKERS_MIN="${WORKERS_MIN:-0}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-100}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-5}"
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H200,NVIDIA B200}"

command -v curl    >/dev/null || die "curl is required but not installed."
command -v python3 >/dev/null || die "python3 is required but not installed (used to read/write JSON safely)."

# ---- API helpers -------------------------------------------------------------
# Each call fails closed: a non-2xx status prints the RunPod error body and stops the run.
# The body is sent on stdin (never on the command line) so keys never land in the process list.
rp() {
  # rp <METHOD> <PATH> [json-on-stdin]
  local method="$1" path="$2" body="" code out
  if [ ! -t 0 ]; then body="$(cat)"; fi
  out="$(printf "%s" "$body" | curl -sS -w $'\n%{http_code}' -X "$method" \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        -H "Content-Type: application/json" \
        ${body:+--data @-} \
        "${API}${path}")" || die "network error calling $method $path"
  code="$(printf "%s" "$out" | tail -n1)"
  RP_BODY="$(printf "%s" "$out" | sed '$d')"
  case "$code" in
    2*) return 0 ;;
    401|403) die "RunPod refused the request ($code). Check RUNPOD_API_KEY. Body: $RP_BODY" ;;
    *) die "RunPod $method $path failed ($code). Body: $RP_BODY" ;;
  esac
}

# Pull a top-level string field out of RP_BODY without needing jq.
json_field() { printf "%s" "$RP_BODY" | python3 -c "import sys,json; print((json.load(sys.stdin) or {}).get('$1',''))" 2>/dev/null || true; }

# Find the id of the first item in a JSON array whose 'name' == $1. Empty if none.
find_id_by_name() {
  printf "%s" "$RP_BODY" | python3 -c "
import sys, json
want = sys.argv[1]
try:
    items = json.load(sys.stdin)
except Exception:
    items = []
if isinstance(items, dict):
    items = items.get('endpoints') or items.get('templates') or items.get('data') or []
for it in (items or []):
    if isinstance(it, dict) and it.get('name') == want:
        print(it.get('id','')); break
" "$1" 2>/dev/null || true
}

# Build a JSON object from environment values, safely quoted by python.
template_json() {
  python3 -c "
import json, os
env = {
  'R2_ENDPOINT': os.environ['R2_ENDPOINT'],
  'R2_ACCESS_KEY_ID': os.environ['R2_ACCESS_KEY_ID'],
  'R2_SECRET_ACCESS_KEY': os.environ['R2_SECRET_ACCESS_KEY'],
  'R2_BUCKET': os.environ['R2_BUCKET'],
}
obj = {
  'name': os.environ['ENDPOINT_NAME'] + '-template',
  'imageName': os.environ['BACKEND_IMAGE'],
  'isServerless': True,
  'containerDiskInGb': int(os.environ['CONTAINER_DISK_GB']),
  'env': env,
}
print(json.dumps(obj))
"
}

endpoint_json() {
  # $1 = templateId
  TEMPLATE_ID="$1" python3 -c "
import json, os
gpus = [g.strip() for g in os.environ['GPU_TYPE_IDS'].split(',') if g.strip()]
obj = {
  'name': os.environ['ENDPOINT_NAME'],
  'templateId': os.environ['TEMPLATE_ID'],
  'computeType': 'GPU',
  'gpuTypeIds': gpus,
  'gpuCount': 1,
  'workersMin': int(os.environ['WORKERS_MIN']),
  'workersMax': int(os.environ['WORKERS_MAX']),
  'idleTimeout': int(os.environ['IDLE_TIMEOUT']),
  'scalerType': 'QUEUE_DELAY',
  'scalerValue': 4,
  'flashboot': True,
}
ms = os.environ.get('EXECUTION_TIMEOUT_MS','').strip()
if ms:
    obj['executionTimeoutMs'] = int(ms)
print(json.dumps(obj))
"
}

export R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET ENDPOINT_NAME BACKEND_IMAGE \
       CONTAINER_DISK_GB GPU_TYPE_IDS WORKERS_MIN WORKERS_MAX IDLE_TIMEOUT EXECUTION_TIMEOUT_MS

say "vivijure-backend deploy -- endpoint: $ENDPOINT_NAME, image: $BACKEND_IMAGE"
info "GPU pool: $GPU_TYPE_IDS"
info "workers: min $WORKERS_MIN / max $WORKERS_MAX, idle ${IDLE_TIMEOUT}s, disk ${CONTAINER_DISK_GB}GB"

# ---- 1. template (image + R2 env) -------------------------------------------
say "Step 1/2: RunPod template ${ENDPOINT_NAME}-template"
rp GET /templates </dev/null
TPL_ID="$(find_id_by_name "${ENDPOINT_NAME}-template")"
if [ -n "$TPL_ID" ]; then
  info "found template $TPL_ID; updating image + R2 keys"
  template_json | rp PATCH "/templates/${TPL_ID}"
else
  info "creating a new template"
  template_json | rp POST /templates
  TPL_ID="$(json_field id)"
fi
[ -n "$TPL_ID" ] || die "could not determine the template id after create/update."
info "template id: $TPL_ID"

# ---- 2. serverless endpoint --------------------------------------------------
say "Step 2/2: RunPod Serverless endpoint ${ENDPOINT_NAME}"
rp GET /endpoints </dev/null
EP_ID="$(find_id_by_name "$ENDPOINT_NAME")"
if [ -n "$EP_ID" ]; then
  info "found endpoint $EP_ID; updating GPU + scaling"
  endpoint_json "$TPL_ID" | rp PATCH "/endpoints/${EP_ID}"
else
  info "creating a new endpoint"
  endpoint_json "$TPL_ID" | rp POST /endpoints
  EP_ID="$(json_field id)"
fi
[ -n "$EP_ID" ] || die "could not determine the endpoint id after create/update."

say "Done."
info "Your render engine endpoint id is:"
printf "\n    RUNPOD_ENDPOINT_ID=%s\n\n" "$EP_ID"
info "Next: paste that line into the Studio's deploy.env, then run the Studio's ./deploy.sh."
info "The endpoint scales to zero: it costs nothing until the Studio sends it a render."
