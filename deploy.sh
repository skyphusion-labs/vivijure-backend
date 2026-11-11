#!/usr/bin/env bash
# vivijure-backend -- one-script deploy of the GPU render engine.
#
# Supply your keys in deploy.env (copy deploy.env.example), then run:  ./deploy.sh
# It stands up (or updates) a RunPod Serverless endpoint running the published, model-baked
# vivijure-backend image, wires in your R2 storage keys, and prints the ENDPOINT ID you paste
# into the Studio's deploy.env as RUNPOD_ENDPOINT_ID.
#
# It is idempotent and re-runnable: run it again after changing a value in deploy.env and it
# updates the same endpoint in place. It FAILS CLOSED: any error stops the whole run
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

# RunPod REST v2 (REST v1 retired 2026-11-15; fleet-chezmoi#2085). On v2 an endpoint carries its own
# image, env and disk, so no template is made: a v2 template is applied once at create and keeps no
# link to the endpoint, so changing one would no longer change the endpoint. Paths and fields are
# from RunPod's v2 OpenAPI (https://api.runpod.io/v2/openapi.json).
API="${RUNPOD_API_BASE:-https://api.runpod.io/v2}"

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
# v2 has no "no limit": an endpoint created without a timeout gets RunPod's default, which is far
# shorter than a render. So a blank value means one hour, the hosted backend endpoint's setting.
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-3600000}"
GPU_TYPE_IDS="${GPU_TYPE_IDS:-NVIDIA H200,NVIDIA B200}"
# v2 schedules by GPU POOL. Set GPU_POOLS to name pools directly (comma-separated, e.g. HOPPER_141);
# left blank, it is derived from GPU_TYPE_IDS below.
GPU_POOLS="${GPU_POOLS:-}"

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

# From one v2 list page (RP_BODY), print "<id>\t<nextCursor>\t<hasNextPage>": the id of the endpoint
# named $1 on this page (empty if none) and where the next page is. v2 names the list `endpoints`;
# `items` is accepted too.
page_find() {
  printf "%s" "$RP_BODY" | python3 -c "
import sys, json
want = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception:
    d = {}
items = d if isinstance(d, list) else (d.get('endpoints') or d.get('items') or d.get('data') or [])
hit = ''
for it in items:
    if isinstance(it, dict) and it.get('name') == want:
        hit = it.get('id', ''); break
pg = d.get('pagination') if isinstance(d, dict) else None
nxt = pg.get('nextCursor') or '' if isinstance(pg, dict) else ''
more = 'yes' if isinstance(pg, dict) and pg.get('hasNextPage') is True else 'no'
print(hit + '\t' + nxt + '\t' + more)
" "$1" 2>/dev/null || true
}

# The id of the endpoint named $1, walking every v2 page. Empty if none. A page that claims more
# without a cursor stops the run: guessing "not found" would create a duplicate endpoint.
find_endpoint_id() {
  local cursor="" line hit nxt more page=0
  while :; do
    rp GET "/serverless?limit=1000${cursor:+&cursor=$cursor}" </dev/null
    line="$(page_find "$1")"
    hit="$(printf "%s" "$line" | cut -f1)"; nxt="$(printf "%s" "$line" | cut -f2)"; more="$(printf "%s" "$line" | cut -f3)"
    if [ -n "$hit" ]; then printf "%s" "$hit"; return 0; fi
    [ "$more" = "yes" ] || return 0
    [ -n "$nxt" ] || die "RunPod's endpoint list said there were more pages but gave no cursor; not guessing."
    cursor="$nxt"; page=$((page + 1))
    [ "$page" -lt 1000 ] || die "RunPod's endpoint list did not end; not guessing."
  done
}

# The v2 endpoint body: image, R2 env, disk, GPU pools, workers and scaling, all on the endpoint.
endpoint_json() {
  python3 -c "
import json, os, sys
# GPU type -> v2 pool, from RunPod's GPU catalog (list-gpu-types, read 2026-11-11). A pool holds
# several cards, so choosing a pool can admit a sibling card of the same class.
POOL = {
  'NVIDIA H200': 'HOPPER_141', 'NVIDIA B200': 'BLACKWELL_180', 'NVIDIA B300 SXM6 AC': 'BLACKWELL_288',
  'NVIDIA H100 80GB HBM3': 'ADA_80_PRO', 'NVIDIA H100 NVL': 'ADA_80_PRO', 'NVIDIA H100 PCIe': 'ADA_80_PRO',
  'NVIDIA A100 80GB PCIe': 'AMPERE_80', 'NVIDIA A100-SXM4-80GB': 'AMPERE_80',
  'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'BLACKWELL_96',
  'NVIDIA RTX PRO 6000 Blackwell Workstation Edition': 'BLACKWELL_96',
  'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition': 'BLACKWELL_96',
  'NVIDIA L40': 'ADA_48_PRO', 'NVIDIA L40S': 'ADA_48_PRO', 'NVIDIA RTX 6000 Ada Generation': 'ADA_48_PRO',
  'NVIDIA A40': 'AMPERE_48', 'NVIDIA RTX A6000': 'AMPERE_48',
  'NVIDIA GeForce RTX 5090': 'ADA_32_PRO', 'NVIDIA GeForce RTX 4090': 'ADA_24',
  'NVIDIA GeForce RTX 3090': 'AMPERE_24', 'NVIDIA RTX A5000': 'AMPERE_24', 'NVIDIA L4': 'AMPERE_24',
}
pools = [p.strip() for p in os.environ.get('GPU_POOLS', '').split(',') if p.strip()]
if not pools:
    for g in [g.strip() for g in os.environ['GPU_TYPE_IDS'].split(',') if g.strip()]:
        if g not in POOL:
            sys.exit('no RunPod v2 pool is known for GPU type ' + repr(g) + '; set GPU_POOLS (e.g. HOPPER_141) in deploy.env')
        if POOL[g] not in pools:
            pools.append(POOL[g])
obj = {
  'name': os.environ['ENDPOINT_NAME'],
  'image': os.environ['BACKEND_IMAGE'],
  'env': {
    'R2_ENDPOINT': os.environ['R2_ENDPOINT'],
    'R2_ACCESS_KEY_ID': os.environ['R2_ACCESS_KEY_ID'],
    'R2_SECRET_ACCESS_KEY': os.environ['R2_SECRET_ACCESS_KEY'],
    'R2_BUCKET': os.environ['R2_BUCKET'],
  },
  'disk': int(os.environ['CONTAINER_DISK_GB']),
  'gpu': {'pools': pools, 'count': 1},
  'workers': {'min': int(os.environ['WORKERS_MIN']), 'max': int(os.environ['WORKERS_MAX']),
              'idleTimeout': int(os.environ['IDLE_TIMEOUT'])},
  'scaling': {'type': 'QUEUE_DELAY', 'queueDelay': 4},
  'flashboot': 'FLASHBOOT',
  'timeout': int(os.environ['EXECUTION_TIMEOUT_MS']),
}
print(json.dumps(obj))
"
}

export R2_ENDPOINT R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET ENDPOINT_NAME BACKEND_IMAGE \
       CONTAINER_DISK_GB GPU_TYPE_IDS GPU_POOLS WORKERS_MIN WORKERS_MAX IDLE_TIMEOUT EXECUTION_TIMEOUT_MS

say "vivijure-backend deploy -- endpoint: $ENDPOINT_NAME, image: $BACKEND_IMAGE"
info "GPUs: ${GPU_POOLS:-$GPU_TYPE_IDS}"
info "workers: min $WORKERS_MIN / max $WORKERS_MAX, idle ${IDLE_TIMEOUT}s, disk ${CONTAINER_DISK_GB}GB"

# ---- 1. serverless endpoint (image + R2 env + GPU + scaling, all on the endpoint) -----------
say "Step 1/1: RunPod Serverless endpoint ${ENDPOINT_NAME}"
EP_BODY="$(endpoint_json)" || die "could not build the endpoint settings (see the message above)."
EP_ID="$(find_endpoint_id "$ENDPOINT_NAME")"
if [ -n "$EP_ID" ]; then
  info "found endpoint $EP_ID; updating image, R2 keys, GPU and scaling"
  rp PATCH "/serverless/${EP_ID}" <<<"$EP_BODY"
else
  info "creating a new endpoint"
  # NOT `endpoint_json | rp POST ...`: a pipeline runs rp in a subshell, so RP_BODY (read by
  # json_field below) would never reach this shell. Feed rp by here-string.
  # v2 requires the request-routing type on create; it is fixed then, and an update does not take it.
  CREATE_BODY="$(printf "%s" "$EP_BODY" | python3 -c "import sys,json; o=json.load(sys.stdin); o[\"type\"]=\"QUEUE\"; print(json.dumps(o))")" || die "could not build the create body"
  rp POST /serverless <<<"$CREATE_BODY"
  EP_ID="$(json_field id)"
fi
[ -n "$EP_ID" ] || die "could not determine the endpoint id after create/update."

say "Done."
info "Your render engine endpoint id is:"
printf "\n    RUNPOD_ENDPOINT_ID=%s\n\n" "$EP_ID"
info "Next: paste that line into the Studio's deploy.env, then run the Studio's ./deploy.sh."
info "The endpoint scales to zero: it costs nothing until the Studio sends it a render."
