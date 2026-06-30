# 3-arch de-risk runbook (#15)

Prove the **real weighted bf16 baked image** renders correctly on **all three pooled DC arches** and
measure the per-arch first-call JIT cost, before any serverless promote or public 3-arch claim. This is
the manual, spend-bounded gate that backstops the build-time arch gate. Driver: `deploy/vj_derisk.py`.
Single-pod automated harness (the seam #9 fills): `deploy/runpod_verify.py`.

## The three arches (one pod each)

| Arch | RunPod gpuTypeId | VRAM | Pool | Card |
|------|------------------|------|------|------|
| sm_90  | `NVIDIA H200` | 141 GB | HOPPER_141 | H200 (Hopper) |
| sm_100 | `NVIDIA B200` | 180 GB | BLACKWELL_180 | B200 (Blackwell DC) |
| sm_120 | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | 96 GB | BLACKWELL_96 | RTX PRO 6000 (Blackwell) |

3-arch coverage rides on the prebuilt **cu128 wheels**: nothing in the image compiles CUDA from source,
so `TORCH_CUDA_ARCH_LIST` is a no-op and is absent by design. H100 (also Hopper sm_90) is excluded by
memory envelope (80 GB OOMs Wan2.2-A14B), not kernel target.

## Gate 0 -- build-time arch gate (CPU-only, run first, no GPU spend)

`torch.cuda.get_arch_list()` is the load-bearing proof that the wheel carries SASS/PTX for the three
base targets. Run it inside the image before spending a cent on a pod:

```
docker run --rm ghcr.io/skyphusion-labs/vivijure-backend:<ver> \
  conda run --no-capture-output -n vivijure python /opt/vivijure/vj_derisk.py arch-gate
```

PASS = `@event arch_gate {... "missing": [], "passed": true}` with `{sm_90, sm_100, sm_120}` all present
in the raw `arch_list`. A missing base target is a **STOP**: do not promote, do not make a 3-arch claim,
flag immediately -- the no-source-compile assumption silently dropped an arch. (The `arch_list` is
identical on any box; running it on one is sufficient for the build-time gate. The per-card runtime
backstop -- a kernel built only against an `sm_*a` variant -- is caught by the real render below.)

## Per-pod procedure (repeat for each of the three gpuTypeIds)

Every op runs in your own login shell: `sudo -u rollins bash -lc '<op>'`.

1. **Scoped exfil creds** -- the read path uses a dedicated `R2_S3_*` exfil key, read-write only to the
   `derisk/` prefix, NEVER the backend's `R2_*` model-pull names and never a reused identity key.
2. **Spin the pod** pinned to the gpuTypeId + the image (PUBLIC GHCR, no registry auth), the `R2_S3_*`
   exfil creds in env, a HARD TTL. Single pod, single concurrency.
3. **Arm the rclone tripwire** on the pod (the no-pull proof): shadow `rclone` with a fake earlier in
   PATH that touches `$VJ_RCLONE_TRIPWIRE` on spawn, then export `VJ_RCLONE_TRIPWIRE=/workspace/.rclone-fired`:
   ```
   mkdir -p /workspace/bin
   printf '#!/bin/sh\ntouch "$VJ_RCLONE_TRIPWIRE"\nexit 0\n' > /workspace/bin/rclone
   chmod +x /workspace/bin/rclone
   export PATH=/workspace/bin:$PATH VJ_RCLONE_TRIPWIRE=/workspace/.rclone-fired
   ```
4. **Run the driver** in the baked conda env, tee-ing the `@event` stream to a per-arch log:
   ```
   python /opt/vivijure/vj_derisk.py arch-gate
   python /opt/vivijure/vj_derisk.py probe
   python /opt/vivijure/vj_derisk.py render --aspect portrait  --tier final --out /workspace/out
   python /opt/vivijure/vj_derisk.py render --aspect landscape --tier final --out /workspace/out
   ```
5. **Assert** on the structured channel (NOT prose):
   - `arch_gate.passed == true`
   - `gpu_probe.kernel_ok == true` and `gpu_probe.capability_in_arch_list == true` (this card's sm is
     in the wheel) and a real `kernel_first_call_seconds`
   - `baked_probe.vj_baked == true`, `sentinel_meta.precision == "bf16"`, the Wan i2v repo baked
   - `rclone_tripwire.fired == false` in BOTH probe and render (baked short-circuit honest, no R2 pull)
   - `render_done` present with `film_bytes > 0`, sane `film_dims`, and the keyframe + film thumbnails
   - `i2v_jit[].first_call_jit_seconds` captured -- the per-arch first-call JIT cost (Triton/inductor),
     the number CPU CI cannot pre-warm. Record it; we decide accept-vs-warm-cache-bake from real data.
6. **Teardown** -- on PASS delete the pod; on FAIL `stop` (not delete) so the disk survives for a debug
   re-spin, and emit the resume handle.

## Spend discipline

Draft/final tier, single concurrency, a hard wall-clock TTL that auto-stops the pod regardless of
outcome, teardown-on-pass / stop-not-leak-on-fail, and a per-run cost estimate. Approximate on-demand
$/hr: H200 ~3.6-4.4, B200 ~5.9, RTX PRO 6000 ~1.7-2.1. Spend is authorized for this sprint; this never
leaves a pod billing silently.

## What STOPS the sweep

- arch gate missing a base target (real defect) -- flag, no promote.
- a render FAILS on any arch -- flag, no promote; leave that pod stopped-not-deleted for debug.
- prod **promote** of the serverless endpoint (separate task) needs Conrad's explicit word -- it touches
  prod / downtime. Everything up to the promote is GO on the authorized spend.

## Canonical start command + read path (committed)

The pod start command is committed at **`deploy/derisk_pod_start.sh`** -- deploy it byte-faithfully by
base64-ing that file into `runpodctl --args` (see its header). It self-runs the de-risk and provides its
own observability, because:

- RunPod pod container stdout is **not** API/CLI-readable (console websocket only; `runpodctl` has no
  `logs`, the MCP `get-pod` returns status only), and
- this image ships **no** openssh-server.

So the deploy wraps the committed inner in a **boto3 -> R2** observability scaffold: it tees output to
`/workspace/out/derisk.log` and uploads it to `r2:vivijure/derisk/<label>/derisk.log` every ~15s; the
operator polls R2 to read the @event stream. boto3 ONLY (never rclone, so the tripwire stays clean), and
a `derisk_boot` marker self-proves the read path within ~15s of boot, BEFORE any GPU render spend.
Inject ONLY `R2_S3_*` creds (the exfil key), never the backend's `R2_*` names -- then the baked backend
has no model-pull creds at all. The boto3 client must match `src/vivijure_backend/harness/r2.py`:
`signature_version="s3v4"`, `region_name="auto"`, no path-style. (An sshd read path was prototyped and
dropped: this image has no openssh-server and pod ssh was uncertain; boto3->R2 is guaranteed.)

The driver `deploy/vj_derisk.py` is curled at the pinned commit
`481f8277eeafa0b045e97075bf5b2191e933b263` (#144) -- the pod runs exactly the reviewed code.

### Open mechanism follow-ups (#146)

The RunPod MCP cannot deploy a custom-start-command pod: `create-pod` exposes no `dockerStartCmd`/
`templateId`; `create-template` silently drops `dockerStartCmd` (`args` stays empty); there is no MCP
template-to-pod deploy; and no pod-log API. Hence the runpodctl/console + boto3->R2 read path. #9 (the
automated regression engine) needs an MCP enhancement or a runpodctl/GraphQL shim in
`runpod_verify.py`'s PodClient seam, and a baked debug-entrypoint in the image.

## backend-v0.3.1 artifact evidence (registry-proven, #14)

The first REAL weighted bf16 image. Run 28411899140 completed/success; `:0.3.1` + `:latest` live in the
PUBLIC GHCR package. Verified against the artifact, not records:

- **assert-weights (gates the `.vj-baked` write via `&&`):** `assert-weights OK -- 104.7 GB across 155
  files, largest 4.78 GB, 40 shard(s) >= 1.0 GB (floor 60.0 GB)`. 40 multi-GB shards, ~75% over the
  60 GB bf16 floor -- a hollow image physically cannot reach this census (the #4/:0.3.0 trap was that
  size != baked; the shard census is the real proof).
- **Sentinel stamp:** `.vj-baked (baked_utc=2026-06-30T00:42:06Z precision=bf16 model_version=1)`.
- **Bin-pack:** 104.7 GB into 24 bins, largest bin 9.00 GB (< 9.0 ceiling). **verify-image:** all layers
  < 10 GB GHCR ceiling. **import smoke:** All 6 finish-stage imports OK.
- **GHCR OCI manifest (registry, no pull):** 51 layers, 96.72 GiB compressed; 15 layers >= 1 GB (weight
  layers), largest 7.64 GiB. NOT the 12.72 GB config-only stub that prod `:0.2.28` is.

Still pending to fully close #14: the `get_arch_list()` {sm_90, sm_100, sm_120} STOP-gate, which runs as
the first command (`arch-gate`) on the canary.

## Per-card deploy commands (Strummer, infra -- locked + reproducible)

Driven from a checkout of this repo (`~/vivijure-backend`) via `runpodctl` (the RunPod MCP cannot set a
pod start command). Every op runs in Strummer's own login shell (`sudo -u strummer bash -lc '<op>'`). The
start command is the committed `deploy/derisk_pod_start.sh` wrapped in the boto3->R2 read-path scaffold,
base64-transported so it survives the shell -> runpodctl -> RunPod -> container quoting layers
byte-identical.

**Reference template `reg2j3abgx`** (`vivijure-derisk-bf16-v0.3.1`) carries the right shape (image
`:0.3.1`, disk 220 GB, NO volume/mount, `registry=null`), but its `args` is EMPTY (the MCP
`create-template` silently dropped the start command). So we deploy with **explicit `--imageName`** and
inject the start command via `--args`; the template is a reference, not the deploy mechanism.

**Locked `--gpuType` strings + per-card `--cost` ceilings** (verified against the live RunPod catalog;
two of the three names are ambiguous, so the EXACT string matters -- a fuzzy match lands the wrong
silicon and de-risks the wrong arch):

| Order | Card (arch) | `--gpuType` (exact) | `--cost` | Cloud |
|---|---|---|---|---|
| 1 (canary) | RTX PRO 6000 Blackwell Server Edition (sm_120) | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | `2.50` | secure |
| 2 | H200 SXM 141 GB (sm_90, Hopper) | `NVIDIA H200` | `5.00` | secure |
| 3 | B200 180 GB (sm_100) | `NVIDIA B200` | `6.50` | secure |

NOT the RTX PRO 6000 `Workstation`/`Max-Q` editions; NOT `NVIDIA H200 NVL` (143 GB). B200 is
**secure-cloud only**, so all three deploy `--secureCloud` for a uniform, reliable pool. Ceilings sit
just above secure on-demand (~$2.09 / ~$4.39 / ~$5.89) as the runaway guard; at a 60-min cap the worst
case is ~$2 / ~$4.4 / ~$5.9 per card, all far under the #136 ~$50 cap (the 60-min terminate is the
binding limit, not `--cost`).

**R2 exfil creds (mandatory read path):** inject ONLY the `R2_S3_*` exfil key (`R2_S3_ENDPOINT` /
`R2_S3_ACCESS_KEY_ID` / `R2_S3_SECRET_ACCESS_KEY` / `R2_S3_BUCKET`), read-write scoped to the `derisk/`
prefix, NEVER the backend's `R2_*` model-pull names (so the baked backend has no model-pull creds at
all). The deploy wraps the committed inner in a boto3->R2 uploader that PUTs `/workspace/out/derisk.log`
to `r2:vivijure/derisk/<label>/derisk.log` every ~15s; no `R2_S3_*` creds = blind pod, do not fire. The
creds are sourced from your secret store and presence-checked with `${VAR:+SET}`, never echoed.

### Fire the canary (sm_120), then fan out

```
cd ~/vivijure-backend && git pull --ff-only

# The R2_S3_* exfil creds must be loaded in YOUR shell (presence-check only, NEVER echo a value):
for v in R2_S3_ENDPOINT R2_S3_ACCESS_KEY_ID R2_S3_SECRET_ACCESS_KEY R2_S3_BUCKET; do
  printf '%s=%s\n' "$v" "${!v:+SET}"
done

LABEL=sm120
INNER_B64=$(base64 -w0 deploy/derisk_pod_start.sh)

# Read-path wrapper: background a boto3->R2 uploader (matches src/vivijure_backend/harness/r2.py:
# s3v4 / region auto / no path-style), then run the committed inner tee'd to the polled log. boto3
# ONLY (never rclone) so the pod-side tripwire stays clean. derisk_boot self-proves the read path.
read -r -d '' WRAP <<'WRAPEOF'
set -u
mkdir -p /workspace/out
LOG=/workspace/out/derisk.log
conda run --no-capture-output -n vivijure python - "$DERISK_LABEL" <<'PY' &
import os, sys, time, boto3
from botocore.config import Config
label = sys.argv[1]
c = boto3.client("s3",
    endpoint_url=os.environ["R2_S3_ENDPOINT"],
    aws_access_key_id=os.environ["R2_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_S3_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=Config(signature_version="s3v4"))
bucket, key = os.environ["R2_S3_BUCKET"], f"derisk/{label}/derisk.log"
c.put_object(Bucket=bucket, Key=key, Body=("@event derisk_boot label=%s\n" % label).encode())
while True:
    time.sleep(15)
    try:
        with open("/workspace/out/derisk.log", "rb") as f:
            c.put_object(Bucket=bucket, Key=key, Body=f.read())
    except FileNotFoundError:
        pass
PY
echo "$DERISK_INNER_B64" | base64 -d | bash 2>&1 | tee -a "$LOG"
WRAPEOF
WRAP_B64=$(printf '%s' "$WRAP" | base64 -w0)

runpodctl create pod \
  --name vj-derisk-sm120-rtxpro6000 \
  --imageName ghcr.io/skyphusion-labs/vivijure-backend:0.3.1 \
  --gpuType 'NVIDIA RTX PRO 6000 Blackwell Server Edition' \
  --gpuCount 1 --secureCloud --cost 2.50 \
  --containerDiskSize 220 \
  --env DERISK_LABEL="$LABEL" \
  --env DERISK_INNER_B64="$INNER_B64" \
  --env R2_S3_ENDPOINT="$R2_S3_ENDPOINT" \
  --env R2_S3_ACCESS_KEY_ID="$R2_S3_ACCESS_KEY_ID" \
  --env R2_S3_SECRET_ACCESS_KEY="$R2_S3_SECRET_ACCESS_KEY" \
  --env R2_S3_BUCKET="$R2_S3_BUCKET" \
  --args "bash -lc 'echo $WRAP_B64 | base64 -d | bash'"
```

The `R2_S3_*` values are expanded from your already-loaded secret store; only the variable NAMES appear
here, never a plaintext value. H200 + B200 are identical except `LABEL` / `--name` / `--gpuType` /
`--cost`:
- H200 (sm_90):  `LABEL=sm90`   `--gpuType 'NVIDIA H200'`   `--cost 5.00`   `--name vj-derisk-sm90-h200`
- B200 (sm_100): `LABEL=sm100`  `--gpuType 'NVIDIA B200'`   `--cost 6.50`   `--name vj-derisk-sm100-b200`

Fan H200 + B200 out IN PARALLEL only AFTER the canary emits `@event arch_gate {... "passed": true}` (and
ideally `@event derisk_pass`). A canary `@event derisk_fail stage=archgate` is a hard STOP: do not fan
out, flag the lead + Rollins.

### Watch (poll R2) + send the pod ID

```
LABEL=sm120   # sm90 / sm100 for the fan-out pods
runpodctl get pod <podId>                 # status only (no pod-log API)
# Poll the @event stream the pod's boto3 uploader PUTs every ~15s. The operator-side rclone READ is
# fine -- the no-pull tripwire is on the POD, not here. derisk_boot appears within ~15s = read proven.
while sleep 12; do
  clear; rclone cat "r2:vivijure/derisk/$LABEL/derisk.log" 2>/dev/null || echo "(no object yet)"
done
```

Watch, in order: `@event derisk_boot` (read path proven); `@event arch_gate ... "passed": true` (record
the raw `arch_list` VERBATIM -- closes #14); `@event gpu_probe`/`baked_probe`/`rclone_tripwire
fired=false`; `@event render_done film_bytes>0`; `@event i2v_jit ... first_call_jit_seconds` (per-arch
JIT cost); terminal `@event derisk_pass` (green) or `@event derisk_fail stage=<...>`.

### TTL + cost cap + teardown (no native pod TTL)

`runpodctl` has NO native TTL, so the cap is enforced manually + by the lead's independent sweep:
- The instant each pod is up, send the **pod ID + fire timestamp** to the lead -- his RunPod-MCP sweep
  hard-terminates any straggler past ~75 min (independent backstop to a dropped watch).
- Primary cap: on a terminal `@event` (pass or fail) OR at 60 min, whichever first --
  `runpodctl remove pod <podId>` on PASS; `runpodctl pod stop <podId>` on FAIL (keep the disk for a
  debug re-spin, emit the resume handle).
- `--cost` is the $/hr price ceiling (runaway guard), not a spend cap; the 60-min terminate is the real
  bound. NEVER leave a pod billing past the watch.

## RESUME (post-checkpoint, step one)

Held at the clean pre-fire boundary (no GPU pods running; nothing to revert; prod `:0.2.28` untouched).
To resume the #15 3-arch de-risk:

1. Rollins pings Strummer to FIRE the **RTX PRO 6000 Blackwell Server Edition (sm_120) canary** from the
   committed `deploy/derisk_pod_start.sh` (wrapped in the boto3->R2 read-path scaffold, base64 via
   runpodctl), image `:0.3.1`, NO volume, `R2_S3_*` exfil creds set (boto3->R2 read path), #136 caps
   (~$50 / 60-min manual terminate). Strummer sends the pod ID to the lead.
2. Watch for `@event arch_gate {... "missing": [], "passed": true}` -- report the raw `arch_list`
   verbatim (closes #14). A missing base arch = hard STOP, do-not-promote, flag.
3. On `@event derisk_pass` (canary green), fan out **H200 (sm_90)** + **B200 (sm_100)** in parallel.
4. Per pod assert: `gpu_probe` kernel_ok + capability_in_arch_list; `baked_probe` vj_baked +
   precision=bf16 + a Wan i2v repo (image, not a mount); `rclone_tripwire fired=false` (no-pull);
   `render_done film_bytes>0`; `i2v_jit` first-call JIT captured per arch.
5. PRs awaiting the lead's merge: #145 (H200 conservative-high cost) and #147 (this wrapper + runbook).
   #5 serverless prod promote remains gated on Conrad.
