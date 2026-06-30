#!/usr/bin/env bash
# Read-path OUTER for the #15 de-risk (and #9's harness): the boto3 -> R2 observability scaffold that
# wraps a base64'd INNER. It is AGNOSTIC of which inner it runs -- it just decodes DERISK_INNER_B64 and
# bashes it -- so the same wrapper carries any de-risk/regression inner. RunPod pod stdout is not
# API/CLI-readable (console-only) and this image ships no openssh-server, so the read path is: tee the
# inner's @event stream to /workspace/out/derisk.log and PUT that object to R2 every ~15s; the operator
# polls R2 to read it.
#
# CONTRACT:
#   - CREDS: R2_S3_* exfil key ONLY (R2_S3_ENDPOINT / R2_S3_ACCESS_KEY_ID / R2_S3_SECRET_ACCESS_KEY /
#     R2_S3_BUCKET), read-write scoped to the derisk/ prefix. NEVER inject the backend's R2_* model-pull
#     names -- then the baked backend has no model-pull creds at all.
#   - boto3 client MUST mirror src/vivijure_backend/harness/r2.py: signature_version="s3v4",
#     region_name="auto", no path-style addressing.
#   - boto3 ONLY, never rclone -- the inner arms an rclone tripwire to prove the baked image never pulls
#     from R2, so any rclone here would poison that proof.
#   - IDENTITY STAMP: the log is seeded with `@event derisk_meta {pod_id, fire_ts, boot_utc, label}` as
#     its FIRST line, so EVERY uploaded object carries the run identity. A reader confirms pod_id ==
#     the pod id returned at fire AND fire_ts == the recorded fire timestamp; a mismatch means a stale or
#     foreign object (this kills the stale-log misread class at the source, not just by pre-fire wipe).
#   - CLEAR-AT-BOOT: the uploader deletes any pre-existing object for this key before its first PUT, so a
#     stale object from a prior fire can never be read as live (belt to the operator pre-fire prefix-wipe).
#   - A derisk_boot marker is written to the log within ~15s of boot to self-prove the read path BEFORE
#     any GPU render spend (a blind pod is caught before a cent is spent on a card).
#
# ENV (injected via runpodctl --env at deploy time):
#   DERISK_INNER_B64  base64 (-w0) of the inner to run (e.g. deploy/derisk_pod_start.sh)
#   DERISK_LABEL      per-card label -> object key derisk/<label>/derisk.log (sm120 / sm90 / sm100)
#   DERISK_FIRE_TS    operator fire timestamp (UTC, ISO8601 Z); stamped into derisk_meta for identity
#                     (optional -- defaults to "unset"; the pod_id stamp alone still disambiguates)
#   R2_S3_*           the four exfil-key vars above
#   RUNPOD_POD_ID     auto-injected by RunPod; stamped into derisk_meta as the run identity
#
# TEARDOWN (operator, NOT this file -- documented here because it bit us): NEVER `runpodctl pod stop`/
# `get` (nor MCP get-pod/list-pods) on a secret-bearing pod -- they dump the FULL pod object, INCLUDING
# the env block (the R2 secret), to stdout. Tear down with `runpodctl remove pod <id>` and output
# suppressed/parsed. See docs/derisk-3arch.md.
#
# boto3 lives in the baked `vivijure` conda env, not base, so the uploader runs under `conda run`.
set -u
mkdir -p /workspace/out
LOG=/workspace/out/derisk.log
POD_ID="${RUNPOD_POD_ID:-unknown}"
FIRE_TS="${DERISK_FIRE_TS:-unset}"
BOOT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Seed the persistent log with the identity stamp + the read-path self-proof. Both are written INTO the
# log file (not PUT as a one-shot object body), so they survive every ~15s re-PUT instead of being
# clobbered. A reader matches pod_id + fire_ts against the fire they recorded.
{
  printf '@event derisk_meta {"pod_id": "%s", "fire_ts": "%s", "boot_utc": "%s", "label": "%s"}\n' "$POD_ID" "$FIRE_TS" "$BOOT_UTC" "$DERISK_LABEL"
  printf '@event derisk_boot {"label": "%s", "pod_id": "%s", "boot_utc": "%s"}\n' "$DERISK_LABEL" "$POD_ID" "$BOOT_UTC"
} > "$LOG"

# Background the boto3 -> R2 uploader. CLEAR-AT-BOOT: delete any stale object for this key first, then
# PUT the seeded log immediately (read-path self-proof), then re-PUT every ~15s. Keyed by DERISK_LABEL
# so parallel cards never collide. boto3 ONLY (never rclone, so the inner's no-pull tripwire stays clean).
conda run --no-capture-output -n vivijure python - "$DERISK_LABEL" <<'PY' &
import os, sys, time, boto3
from botocore.config import Config
label = sys.argv[1]
log_path = "/workspace/out/derisk.log"
c = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_S3_ENDPOINT"],
    aws_access_key_id=os.environ["R2_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_S3_SECRET_ACCESS_KEY"],
    region_name="auto",                       # R2 ignores region; boto3 insists on one
    config=Config(signature_version="s3v4"),  # no path-style addressing
)
bucket, key = os.environ["R2_S3_BUCKET"], f"derisk/{label}/derisk.log"
# clear-at-boot: a stale object from a prior fire can never be read as live.
try:
    c.delete_object(Bucket=bucket, Key=key)
except Exception:
    pass
def push():
    try:
        with open(log_path, "rb") as f:
            c.put_object(Bucket=bucket, Key=key, Body=f.read())
    except FileNotFoundError:
        pass
push()              # immediate: the seeded derisk_meta + derisk_boot self-prove the read path now
while True:
    time.sleep(15)
    push()
PY

# Run the committed inner from a FILE (NOT piped into bash's stdin) with stdin redirected from
# /dev/null. Piping the script into `bash` makes the SCRIPT bash's stdin, so a render subprocess that
# reads stdin (a finish-stage download / torch / hf) consumes the rest of the script off the pipe --
# the stdin-eats-the-script classic. That ate the landscape render line after portrait completed, so
# bash resumed mid-"$RUN" and parsed the fragment "UN" -> derisk_fail render_landscape, even though the
# committed inner was byte-clean. Running from a file + </dev/null closes it: bash reads the script from
# disk, and no child can consume it.
echo "$DERISK_INNER_B64" | base64 -d > /workspace/inner.sh
bash /workspace/inner.sh </dev/null 2>&1 | tee -a "$LOG"
