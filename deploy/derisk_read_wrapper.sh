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
#   - A derisk_boot marker is PUT within ~15s of boot to self-prove the read path BEFORE any GPU render
#     spend (a blind pod is caught before a cent is spent on a card).
#
# ENV (injected via runpodctl --env at deploy time):
#   DERISK_INNER_B64  base64 (-w0) of the inner to run (e.g. deploy/derisk_pod_start.sh)
#   DERISK_LABEL      per-card label -> object key derisk/<label>/derisk.log (sm120 / sm90 / sm100)
#   R2_S3_*           the four exfil-key vars above
#
# boto3 lives in the baked `vivijure` conda env, not base, so the uploader runs under `conda run`.
set -u
mkdir -p /workspace/out
LOG=/workspace/out/derisk.log

# Background the boto3 -> R2 uploader: PUT a derisk_boot marker immediately (read-path self-proof), then
# re-PUT the growing log every ~15s. Keyed by DERISK_LABEL so parallel cards never collide.
conda run --no-capture-output -n vivijure python - "$DERISK_LABEL" <<'PY' &
import os, sys, time, boto3
from botocore.config import Config
label = sys.argv[1]
c = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_S3_ENDPOINT"],
    aws_access_key_id=os.environ["R2_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_S3_SECRET_ACCESS_KEY"],
    region_name="auto",                      # R2 ignores region; boto3 insists on one
    config=Config(signature_version="s3v4"),  # no path-style addressing
)
bucket, key = os.environ["R2_S3_BUCKET"], f"derisk/{label}/derisk.log"
# derisk_boot self-proves the read path within ~15s, BEFORE any GPU render spend.
c.put_object(Bucket=bucket, Key=key, Body=("@event derisk_boot label=%s\n" % label).encode())
while True:
    time.sleep(15)
    try:
        with open("/workspace/out/derisk.log", "rb") as f:
            c.put_object(Bucket=bucket, Key=key, Body=f.read())
    except FileNotFoundError:
        pass
PY

# Run the committed inner, tee-ing its @event stream to the polled log.
echo "$DERISK_INNER_B64" | base64 -d | bash 2>&1 | tee -a "$LOG"
