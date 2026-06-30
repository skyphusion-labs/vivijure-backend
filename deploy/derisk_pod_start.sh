#!/usr/bin/env bash
# Pod start-command for the #15 3-arch de-risk (one pod per pooled DC card).
#
# This is the container start command (CMD override) the de-risk pods run. Our baked image's CMD is
# the serverless worker loop, which exits immediately on a pod (no test_input.json); this script keeps
# the pod working: it self-runs the de-risk and emits the @event channel to the pod LOGS (no ssh).
#
# DEPLOY (the RunPod MCP cannot set a pod start command -- see the tooling-gap issue): deliver this
# verbatim via runpodctl, quoting-proof, as the container args:
#
#     B64=$(base64 -w0 deploy/derisk_pod_start.sh)
#     runpodctl create pod --imageName ghcr.io/skyphusion-labs/vivijure-backend:0.3.1 \
#       --gpuType '<NVIDIA H200 | NVIDIA B200 | NVIDIA RTX PRO 6000 Blackwell Server Edition>' \
#       --containerDiskSize 220 --args "bash -lc 'echo $B64 | base64 -d | bash'"
#
# base64 is [A-Za-z0-9+/=] only, so it survives the shell -> runpodctl -> RunPod -> container quoting
# layers byte-identical. NO network volume / NO mount at /opt/models (it would shadow the baked weights).
# GHCR package is public, so no containerRegistryAuthId. runpodctl has no native TTL: actively watch the
# logs and `runpodctl remove pod` on a terminal @event or at the ~60 min / cost cap.
#
# Order: RTX PRO 6000 (sm_120) canary first; on @event derisk_pass, fan out H200 (sm_90) + B200 (sm_100).
#
# set -u correctness: VJ_RCLONE_TRIPWIRE is exported BEFORE the fake-rclone write, and the rclone body is
# single-quoted so the var is written UNexpanded (it resolves when the fake rclone runs, not at write
# time) -- otherwise set -u would abort on line 1 with the export still pending (dead pod, zero @event).
set -u
export VJ_RCLONE_TRIPWIRE=/workspace/.rclone-fired
mkdir -p /workspace/bin /workspace/out

# rclone tripwire: shadow rclone with a fake earlier in PATH that touches the sentinel on spawn. On a
# truthfully-baked image the baked short-circuit never pulls from R2, so the sentinel must never appear
# (the driver re-checks it: @event rclone_tripwire fired=false is the no-pull proof).
printf '%s\n' '#!/bin/sh' 'touch "$VJ_RCLONE_TRIPWIRE"' 'exit 0' > /workspace/bin/rclone
chmod +x /workspace/bin/rclone
export PATH=/workspace/bin:$PATH

# Driver pinned to the merged #144 commit (integrity + reproducibility: the pod runs exactly the
# reviewed code, immune to a mid-run push to main).
SHA=481f8277eeafa0b045e97075bf5b2191e933b263
fail() { echo "@event derisk_fail stage=$1"; sleep 600; exit 1; }
curl -fsSL "https://raw.githubusercontent.com/skyphusion-labs/vivijure-backend/$SHA/deploy/vj_derisk.py" \
  -o /workspace/vj_derisk.py || fail fetch

RUN="conda run --no-capture-output -n vivijure python /workspace/vj_derisk.py"

# arch-gate FIRST: the build-time STOP-gate (get_arch_list covers {sm_90,sm_100,sm_120}). A missing arch
# halts here, before any probe/render -- @event derisk_fail stage=archgate is a hard do-not-promote.
$RUN arch-gate || fail archgate
$RUN probe     || fail probe
$RUN render --aspect portrait  --tier final --out /workspace/out || fail render_portrait
$RUN render --aspect landscape --tier final --out /workspace/out || fail render_landscape

echo "@event derisk_pass stages=archgate,probe,render_portrait,render_landscape"
sleep 600   # log-read window only; the hard cap is the operator's `runpodctl remove pod`
