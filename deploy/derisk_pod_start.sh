#!/usr/bin/env bash
# De-risk INNER for the #15 3-arch run (one pod per pooled DC card). This is the de-risk LOGIC only;
# the READ PATH is a separate observability scaffold applied at deploy time (see below).
#
# It replaces the image CMD (the serverless worker loop, which exits on a pod with no test_input.json):
# arms the rclone tripwire, materializes the INJECTED driver, then arch-gate -> probe -> render, emitting
# the @event channel to stdout.
#
# READ PATH (boto3 -> R2 poll, Strummer's deploy-time wrapper -- NOT in this file):
#   RunPod pod stdout is not API/CLI-readable (console-only) and this image has no sshd. So the deploy
#   wraps THIS inner in a scaffold that tees output to /workspace/out/derisk.log and, every ~15s, uploads
#   it to r2:vivijure/derisk/<label>/derisk.log via boto3 (the operator polls R2 to read the @event
#   stream). boto3 ONLY, never rclone, so the tripwire here stays meaningful. A derisk_boot marker is
#   uploaded within ~15s of boot to self-prove the read path BEFORE any GPU render spend.
#   CREDS: inject ONLY R2_S3_* (the exfil key) via --env; do NOT inject the backend's R2_* names -- then
#   the baked backend has no model-pull creds at all (can't pull from R2 even if is_baked() failed). The
#   boto3 client must match src/vivijure_backend/harness/r2.py: signature_version="s3v4",
#   region_name="auto", no path-style addressing.
#
# DEPLOY (the RunPod MCP cannot set a pod start command -- see the tooling-gap issue): base64 the full
# wrapped script into runpodctl --args. Image ghcr.io/skyphusion-labs/vivijure-backend:0.3.1 (PUBLIC, no
# registry auth), NO volume / NO mount at /opt/models (would shadow the baked weights), containerDisk
# ~220GB. runpodctl has no native TTL: watch R2 and `runpodctl remove pod` on a terminal @event or ~60min.
# Order: RTX PRO 6000 (sm_120) canary first; on @event derisk_pass, fan out H200 (sm_90) + B200 (sm_100).
#
# set -u correctness: VJ_RCLONE_TRIPWIRE is exported BEFORE the fake-rclone write, and the rclone body is
# single-quoted so the var is written UNexpanded (resolves when the fake rclone runs); otherwise set -u
# aborts on line 1 with the export pending (dead pod, zero @event).
set -u
export VJ_RCLONE_TRIPWIRE=/workspace/.rclone-fired
mkdir -p /workspace/bin /workspace/out

# rclone tripwire: shadow rclone with a fake earlier in PATH that touches the sentinel on spawn. A
# truthfully-baked image never pulls from R2, so the sentinel must never appear (@event rclone_tripwire
# fired=false is the no-pull proof).
printf '%s\n' '#!/bin/sh' 'touch "$VJ_RCLONE_TRIPWIRE"' 'exit 0' > /workspace/bin/rclone
chmod +x /workspace/bin/rclone
export PATH=/workspace/bin:$PATH

fail() { echo "@event derisk_fail stage=$1"; sleep 600; exit 1; }

# Driver INJECTED, not fetched. raw.githubusercontent returned STALE bytes for an immutable raw-by-SHA URL
# during the post-#151/#152-merge propagation window (2026-06-30): a curled pod ran the OLD driver and
# crashed at render despite the inner pinning the fixed commit. We now inject the reviewed driver bytes via
# DERISK_DRIVER_B64 (gzip + base64 -w0 of deploy/vj_derisk.py) and materialize them here: zero network/CDN
# dependency, so the pod runs EXACTLY the reviewed code. The fire builds DERISK_DRIVER_B64 from the COMMITTED
# driver on main; the materialized file (post-gunzip) is byte-identical to that driver.
[ -n "${DERISK_DRIVER_B64:-}" ] || fail no_driver
printf '%s' "$DERISK_DRIVER_B64" | base64 -d | gunzip > /workspace/vj_derisk.py 2>/dev/null || fail driver_decode

# Integrity gate (CPU, pre-GPU, $0): the materialized driver MUST match the reviewed bytes EXACTLY. Primary
# gate = sha256 (catches ANY transport corruption); secondary = the `def build_render_inputs` marker that
# exists ONLY in the #151 fix (absence == old/stale driver). Structured @event so the watcher asserts on the
# channel, not prose. On mismatch: emit @event driver_corrupt, then sleep so the boto3->R2 uploader PUTs the
# event (the read path needs the pod alive ~15s+ or the terminal event never reaches R2), then exit non-zero.
EXPECT_SHA=0a600a9594b70b56ddebd62b95a86b88b17f96c7ca2c02ca071955d7699660d0
GOT_SHA=$(sha256sum /workspace/vj_derisk.py | cut -d' ' -f1)
if [ "$GOT_SHA" != "$EXPECT_SHA" ] || ! grep -q 'def build_render_inputs' /workspace/vj_derisk.py; then
  echo "@event driver_corrupt {\"ok\": false, \"expected_sha256\": \"$EXPECT_SHA\", \"got_sha256\": \"$GOT_SHA\", \"marker\": \"build_render_inputs\"}"
  sleep 600   # keep the pod alive so the uploader PUTs the event; operator removes on read ($0-intent, pre-GPU)
  exit 1
fi
echo "@event driver_integrity {\"ok\": true, \"sha256\": \"$GOT_SHA\", \"marker\": \"build_render_inputs\"}"

RUN="conda run --no-capture-output -n vivijure python /workspace/vj_derisk.py"

# arch-gate FIRST: the build-time STOP-gate (get_arch_list covers {sm_90,sm_100,sm_120}). A missing arch
# halts here, before any probe/render -- @event derisk_fail stage=archgate is a hard do-not-promote.
$RUN arch-gate || fail archgate
$RUN probe     || fail probe
$RUN render --aspect portrait  --tier final --out /workspace/out || fail render_portrait
$RUN render --aspect landscape --tier final --out /workspace/out || fail render_landscape

echo "@event derisk_pass stages=archgate,probe,render_portrait,render_landscape"
sleep 600   # read-window only; the hard cap is the operator's `runpodctl remove pod`
