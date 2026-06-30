#!/usr/bin/env bash
# Pod start-command for the #15 3-arch de-risk (one pod per pooled DC card).
#
# This is the container start command (CMD override) the de-risk pods run. Our baked image's CMD is the
# serverless worker loop, which exits immediately on a pod (no test_input.json); this script keeps the
# pod working: it self-runs the de-risk (arch-gate -> probe -> render) and emits the @event channel.
#
# OBSERVABILITY (why this looks the way it does): a RunPod pod's container stdout is NOT API/CLI-readable
# (console websocket only -- runpodctl has no `logs`, the MCP get-pod returns status only). AND this image
# ships no openssh-server. So this script provides its own read path:
#   - self-tees ALL output to /workspace/derisk.log (exec > >(tee ...)), and
#   - if PUBLIC_KEY is set, apt-installs + starts sshd with it BEFORE the de-risk, so the operator can
#     ssh in and `tail -f /workspace/derisk.log` live.
# PUBLIC_KEY is therefore MANDATORY -- without it the pod runs but is unreadable (blind). Set the pod env
# PUBLIC_KEY=<your ed25519 pubkey>.
#
# DEPLOY (the RunPod MCP cannot set a pod start command -- see the tooling-gap issue): deliver verbatim,
# quoting-proof, via runpodctl as base64:
#   B64=$(base64 -w0 deploy/derisk_pod_start.sh)
#   runpodctl create pod --imageName ghcr.io/skyphusion-labs/vivijure-backend:0.3.1 \
#     --gpuType '<NVIDIA H200 | NVIDIA B200 | NVIDIA RTX PRO 6000 Blackwell Server Edition>' \
#     --containerDiskSize 220 --ports '22/tcp' --env PUBLIC_KEY="$(cat ~/.ssh/derisk.pub)" \
#     --args "bash -lc 'echo $B64 | base64 -d | bash'"
# base64 survives the shell -> runpodctl -> RunPod -> container quoting layers byte-identical. NO network
# volume / NO mount at /opt/models (it would shadow the baked weights). GHCR is public (no registry auth).
# runpodctl has no native TTL: watch the log and `runpodctl remove pod` on a terminal @event or ~60 min.
#
# Order: RTX PRO 6000 (sm_120) canary first; on @event derisk_pass, fan out H200 (sm_90) + B200 (sm_100).
#
# set -u correctness: VJ_RCLONE_TRIPWIRE is exported BEFORE the fake-rclone write, and the rclone body is
# single-quoted so the var is written UNexpanded (resolves when the fake rclone runs, not at write time);
# otherwise set -u aborts on line 1 with the export pending (dead pod, zero @event). ${PUBLIC_KEY:-} is
# default-guarded so set -u tolerates it being unset.
set -u
exec > >(tee -a /workspace/derisk.log) 2>&1
export VJ_RCLONE_TRIPWIRE=/workspace/.rclone-fired
mkdir -p /workspace/bin /workspace/out

# Read path: install + start sshd from PUBLIC_KEY (this image has none), BEFORE the de-risk so progress
# is watchable live. No PUBLIC_KEY = blind pod; abort and set it.
if [ -n "${PUBLIC_KEY:-}" ]; then
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq openssh-server >/dev/null 2>&1
  mkdir -p /run/sshd ~/.ssh && chmod 700 ~/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  /usr/sbin/sshd && echo "### derisk: sshd up" || echo "### derisk: sshd_setup_failed"
else
  echo "### derisk: NO PUBLIC_KEY set -- pod is BLIND (no read path). Abort and redeploy with PUBLIC_KEY."
fi

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
