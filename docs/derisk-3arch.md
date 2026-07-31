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

## Canonical start command + read path

The pod start command is committed at **`deploy/derisk_pod_start.sh`**; it self-runs the de-risk. As of
2026-07-31 there is a default read path that needs no credentials in the pod, plus a fallback that does.

**Default path (verified against the live RunPod MCP tooling, not inferred):**

1. `create-template` with `dockerStartCmd` set to the base64-materialized inner script (see the
   script header). The MCP persists `dockerStartCmd` as the template `args` field verbatim; it does
   not drop it.
2. `create-pod` with `templateId` pointed at that template. The MCP deploys the pod using the
   template image, start command, ports, env, and disk as defaults.
3. Read progress with `stream-pod-logs`: it returns real container (and system) stdout over
   Server-Sent Events, tailed or live, timestamped -- no console session and no relay needed.

Nothing in this path touches R2, so the pod never holds `R2_S3_*` or any other credential it does not
need for the render itself. The driver `deploy/vj_derisk.py` stays INJECTED (gzip+base64 into the start
command, sha256-gated before any GPU work) for the reason it always was: the pod used to curl it by
pinned SHA, and raw.githubusercontent served STALE bytes in the post-#151/#152-merge propagation window
(2026-06-30), so injection removes the network/CDN dependency and guarantees the pod runs exactly the
reviewed code. That part of the mechanism is unchanged by this section.

**Fallback (boto3 -> R2, needs `R2_S3_*` credentials in the pod):** the observability scaffold this
section used to describe as the only option is still committed (`deploy/derisk_read_wrapper.sh`) and
still works: tee to `/workspace/out/derisk.log`, upload to `r2:vivijure/derisk/<label>/derisk.log`
every ~15s, a `derisk_boot` marker self-proving the read path, `derisk_meta` carrying run identity so a
stale object cannot be mistaken for a live one. Use it only when `stream-pod-logs` genuinely cannot
serve (for example a run that outlives an interactive session watching it). It is not the default,
because on the one axis that matters here it is strictly worse: it puts R2 write credentials into a GPU
pod that no longer needs them to be observable. Do not reach for it out of habit.

### Mechanism status (#146)

Re-verified 2026-07-31 with a read-back probe against the live RunPod MCP (create a template with a
trivial start command, read it back, confirm the start command survived), plus real templates already
in the account carrying persisted, non-trivial start commands, rather than by reasoning about it. All
three constraints this section used to state as blocking are now false:

- `create-template` persists `dockerStartCmd` as the template `args` field verbatim; it does not drop
  it.
- `create-pod` accepts `templateId` and deploys from that template start command.
- Pod container stdout is readable over the MCP via `stream-pod-logs` (Server-Sent Events), not
  console-websocket-only.

What is still true: `runpodctl` (the CLI the Per-card procedure below drives) has no `logs` subcommand,
confirmed against the installed CLI, and the MCP `get-pod` returns pod status and metadata only, no log
content; `stream-pod-logs` is a separate tool. So a pure-CLI fire through `runpodctl` still has no
built-in log read path of its own; only an MCP-driven fire gets one for free. #9 (the automated
regression engine) can drop the "needs an MCP enhancement" framing for its `PodClient` seam: the
enhancement it was waiting on now exists. Whether #9 should move onto it, versus keep its current
transport, is a separate call for whoever owns that issue.

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

**Status note (2026-07-31):** this procedure predates the `templateId` + `stream-pod-logs` path
described in "Canonical start command + read path" above and is kept here as a FALLBACK, not the
default. It is the tested, spend-authorized live procedure for an actual 3-arch sweep (the DC placement
empirics, the exact `--gpu-id` strings, and the TTL and disk-floor numbers below are all specific to
this `runpodctl` flow), and none of that has been re-verified against the newer MCP path end to end;
migrating it is separate work, not done here. If you are starting a NEW de-risk run and do not need
everything this section already proves out, start with the canonical section above instead.

Driven from a checkout of this repo (`~/vivijure-backend`) via `runpodctl`; this section, unlike the MCP
path above, still needs `runpodctl` because it drives DC-placement probing and TTL flags this doc has not
re-verified through the MCP (see the status note above). Every op runs in Strummer's own **login**
shell (`sudo -u strummer bash -l ...`) so
the per-identity `RUNPOD_API_KEY` loads -- `runpodctl pod create` errors `api key not found` from a
non-login shell. The start command is the committed `deploy/derisk_pod_start.sh` wrapped in the committed
boto3->R2 read-path outer `deploy/derisk_read_wrapper.sh`, base64-transported so it survives the shell ->
runpodctl -> RunPod -> container quoting layers byte-identical.

**Use `runpodctl pod create`, NOT the deprecated `runpodctl create pod`.** The deprecated form takes
repeated `--env KEY=VALUE` and splits each on `=`, which CORRUPTS a base64 value (base64 padding is `=`):
`--env DERISK_INNER_B64=<...=>` fails client-side with `wrong env value`. The modern `pod create` takes
`--env` as a single **JSON object** (carries `=` cleanly) and `--docker-args` for the start command.

**Reference template `reg2j3abgx`** (`vivijure-derisk-bf16-v0.3.1`) carries the right shape (image
`:0.3.1`, disk 220 GB, NO volume/mount, `registry=null`), but its `args` is EMPTY (the MCP
`create-template` silently dropped the start command). So we deploy with **explicit `--image`** and inject
the start command via `--docker-args`; the template is a reference, not the deploy mechanism.

**Locked `--gpu-id` strings + per-card expected secure $/hr** (verified against the live RunPod catalog +
`runpodctl gpu list`; two of the three names are ambiguous, so the EXACT string matters -- a fuzzy match
lands the wrong silicon and de-risks the wrong arch). The `--gpu-id` value is identical to the old
`--gpuType` string:

| Order | Card (arch) | `--gpu-id` (exact) | ~secure $/hr | Cloud |
|---|---|---|---|---|
| 1 (canary) | RTX PRO 6000 Blackwell Server Edition (sm_120) | `NVIDIA RTX PRO 6000 Blackwell Server Edition` | ~2.09 | secure |
| 2 | H200 SXM 141 GB (sm_90, Hopper) | `NVIDIA H200` | ~4.39 | secure |
| 3 | B200 180 GB (sm_100) | `NVIDIA B200` | ~5.89 | secure |

NOT the RTX PRO 6000 `Workstation`/`Max-Q` editions; NOT `NVIDIA H200 NVL` (143 GB). B200 is
**secure-cloud only**, so all three deploy `--cloud-type SECURE` for a uniform, reliable pool. `pod
create` has **no `--cost` flag**; the hard spend bound is the native `--terminate-after` (60-min cap
below), well under the #136 ~$50 ceiling. At a 60-min cap the worst case is ~$2 / ~$4.4 / ~$5.9 per card.

**Placement note (secure capacity for a 220 GB container disk is scarce).** An unconstrained `pod create`
on secure RTX PRO 6000 frequently returns `This machine does not have the resources to deploy your pod`
or `There are no longer any instances available with the requested specifications` -- the GPU stock is
"High" but the machines that can hold a 220 GB container disk for this card are not always placeable. Do
NOT lower the disk (an unpack failure would confound the de-risk) and do NOT fall back to community (the
exfil key is bucket-level Object R+W on `vivijure` -- R2 cannot prefix-scope -- so an untrusted host gets
R+W to the whole bucket). Instead **probe data centers**: retry `pod create` with `--data-center-ids`
targeted one DC at a time until one places (first hit wins; failed attempts cost $0). The sm_120 canary
placed in **US-KS-2** on 2026-06-30 after 8 secure DCs reported no stock.

**R2 exfil creds (mandatory read path):** inject ONLY the `R2_S3_*` exfil key (`R2_S3_ENDPOINT` /
`R2_S3_ACCESS_KEY_ID` / `R2_S3_SECRET_ACCESS_KEY` / `R2_S3_BUCKET`), a DEDICATED bucket-level Object R+W
token on `vivijure` (R2 tokens are bucket-scoped, NOT prefix-scoped -- "derisk/ only" is not enforceable
on the key, so the token is throwaway + revoked at teardown), NEVER the backend's `R2_*` model-pull names
(so the baked backend has no model-pull creds at all). The read-path outer PUTs `/workspace/out/derisk.log`
to `r2:vivijure/derisk/<label>/derisk.log` every ~15s; no `R2_S3_*` creds = blind pod, do not fire. The
creds are sourced from a 600 drop file and presence-checked with `${VAR:+SET}`, never echoed. Shred the
drop file only AFTER the pod is confirmed created (a pre-fire shred strands you if the fire fails
downstream).

### Fire the canary (sm_120), then fan out

Both halves of what fires are committed + change-controlled: the de-risk inner (`deploy/derisk_pod_start.sh`)
and the boto3->R2 read-path outer (`deploy/derisk_read_wrapper.sh`). The deploy only base64-transports
them, so what fires is exactly the reviewed files (no hand-pasted heredoc).

```
cd ~/vivijure-backend && git checkout main && git pull --ff-only

# R2_S3_* exfil creds loaded in YOUR login shell (presence-check only, NEVER echo a value):
for v in R2_S3_ENDPOINT R2_S3_ACCESS_KEY_ID R2_S3_SECRET_ACCESS_KEY R2_S3_BUCKET; do
  printf '%s=%s\n' "$v" "${!v:+SET}"
done

export DERISK_LABEL=sm120
export DERISK_FIRE_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"               # stamped into derisk_meta (run identity)
export DERISK_INNER_B64="$(base64 -w0 deploy/derisk_pod_start.sh)"   # de-risk inner
export DERISK_DRIVER_B64="$(gzip -c deploy/vj_derisk.py | base64 -w0)"  # injected driver (gzip+b64, ~8.6KB, sha256-gated)
WRAP_B64="$(base64 -w0 deploy/derisk_read_wrapper.sh)"               # boto3->R2 read-path outer

# --env is a JSON OBJECT (the fix): build it without echoing the secret.
ENVJSON="$(python3 -c 'import os,json;K=["DERISK_LABEL","DERISK_FIRE_TS","DERISK_INNER_B64","DERISK_DRIVER_B64","R2_S3_ENDPOINT","R2_S3_ACCESS_KEY_ID","R2_S3_SECRET_ACCESS_KEY","R2_S3_BUCKET"];d={k:os.environ[k] for k in K};L=os.environ.get("DERISK_EGRESS_LOCK");d.update({"DERISK_EGRESS_LOCK":L}) if L else None;print(json.dumps(d))')"
TERM_AFTER="$(date -u -d '+60 minutes' +%Y-%m-%dT%H:%M:%SZ)"         # native hard TTL
DC=US-KS-2   # probe DCs until one places; see the placement note above

runpodctl pod create \
  --name "vj-derisk-sm120-rtxpro6000-$(date -u +%H%M%S)" \   # unique per fire (no --name reuse)
  --image ghcr.io/skyphusion-labs/vivijure-backend:0.3.1 \
  --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  --gpu-count 1 --cloud-type SECURE \
  --data-center-ids "$DC" \
  --container-disk-in-gb 220 --ssh=false \
  --terminate-after "$TERM_AFTER" \
  --env "$ENVJSON" \
  --docker-args "bash -lc 'echo $WRAP_B64 | base64 -d | bash'" \
  -o json
```

The secret rides in the JSON `--env` value on `runpodctl`'s argv: acceptable here (local operator shell,
argv-transient for one short-lived invocation, never in a tracked file -- only variable NAMES appear
above; values expand from the loaded creds). H200 + B200 are identical except `DERISK_LABEL` / `--name` /
`--gpu-id` (and pick a DC that places):
- H200 (sm_90):  `DERISK_LABEL=sm90`   `--gpu-id 'NVIDIA H200'`   `--name vj-derisk-sm90-h200`
- B200 (sm_100): `DERISK_LABEL=sm100`  `--gpu-id 'NVIDIA B200'`   `--name vj-derisk-sm100-b200`

Fan H200 + B200 out IN PARALLEL only AFTER the canary emits `@event arch_gate {... "passed": true}` (and
ideally `@event derisk_pass`). A canary `@event derisk_fail stage=archgate` is a hard STOP: do not fan
out, flag the lead + Rollins.

### Watch (poll R2) + send the pod ID

```
LABEL=sm120   # sm90 / sm100 for the fan-out pods
runpodctl pod list 2>/dev/null | grep vj-derisk        # status only (no pod-log API)
# Poll the @event stream the pod's boto3 uploader PUTs every ~15s. The operator-side rclone READ is
# fine -- the no-pull tripwire is on the POD, not here. NOTE: the :0.3.1 image is ~96.7 GiB, so the
# cold-start pull takes several minutes before the wrapper runs; derisk_boot lands after the pull.
while sleep 12; do
  clear; rclone cat "r2:vivijure/derisk/$LABEL/derisk.log" 2>/dev/null || echo "(no object yet)"
done
```

FIRST confirm `@event derisk_meta` carries YOUR `pod_id` (the id `pod create` returned) and `fire_ts`; a
mismatch == a stale/foreign object, do NOT read it as your run. Then watch, in order: `@event
derisk_boot` (read path proven); `@event arch_gate ... "passed": true` (record
the raw `arch_list` VERBATIM -- closes #14); `@event gpu_probe`/`baked_probe`/`rclone_tripwire
fired=false`; `@event render_done film_bytes>0`; `@event i2v_jit ... first_call_jit_seconds` (per-arch
JIT cost); terminal `@event derisk_pass` (green) or `@event derisk_fail stage=<...>`.

### TTL + teardown (native `--terminate-after`)

`runpodctl pod create --terminate-after <ISO8601 Z>` IS a native hard TTL -- set it to fire+60min so the
pod self-terminates regardless of outcome. The lead's independent ~75-min RunPod-MCP sweep stays as the
second layer (send the **pod ID + fire timestamp** the instant the pod is up).
- On a terminal `@event` (pass or fail) OR at the TTL, whichever first -- tear down with `runpodctl
  remove pod <podId>` and output SUPPRESSED/parsed. NEVER `runpodctl pod stop`/`get` (nor MCP
  get-pod/list-pods) on a secret-bearing pod: they print the FULL pod object, INCLUDING the env block
  (the R2 secret), to stdout -- that is how a throwaway exfil token reached an operator transcript. If a
  FAIL needs disk forensics, prefer `remove` + reproduce from the PUBLIC image (`docker run ... :<ver>`)
  over a `stop` that leaks env; the baked artifact lives in the image, not pod-specific state. NEVER
  leave a pod billing past the watch.
- Teardown also REVOKES the throwaway R2 exfil token (bucket-level R+W on `vivijure` -- revoke promptly).

## RESUME (post-checkpoint, step one)

Held at the clean pre-fire boundary (no GPU pods running; nothing to revert; prod `:0.2.28` untouched).
To resume the #15 3-arch de-risk:

1. The lead mints a DEDICATED bucket-level Object-R+W-on-`vivijure` token and drops the four `R2_S3_*`
   into Strummer's shell (600 file). Strummer FIRES the **RTX PRO 6000 Blackwell Server Edition (sm_120)
   canary** via `runpodctl pod create` (login shell; JSON `--env`; `--docker-args`; `--gpu-id`;
   `--cloud-type SECURE`; `--container-disk-in-gb 220`; native `--terminate-after`=fire+60min; probe
   `--data-center-ids` until one places), image `:0.3.1`, NO volume. Strummer sends the pod ID + fire
   timestamp to the lead, then shreds the creds file (post-confirm).
2. Watch for `@event arch_gate {... "missing": [], "passed": true}` -- report the raw `arch_list`
   verbatim (closes #14). A missing base arch = hard STOP, do-not-promote, flag.
3. On `@event derisk_pass` (canary green), fan out **H200 (sm_90)** + **B200 (sm_100)** in parallel.
4. Per pod assert: `gpu_probe` kernel_ok + capability_in_arch_list; `baked_probe` vj_baked +
   precision=bf16 + a Wan i2v repo (image, not a mount); `rclone_tripwire fired=false` (no-pull);
   `render_done film_bytes>0`; `i2v_jit` first-call JIT captured per arch.
5. Teardown: `runpodctl remove pod` each pod (output suppressed; NEVER `stop`/`get` -- they leak the
   env/secret to stdout) and REVOKE the throwaway exfil token.
   #5 serverless prod promote remains gated on Conrad.

## Egress-locked render (#245): prove the userspace guard end-to-end

The egress guard in `deploy/vj_derisk.py` is dormant unless `DERISK_EGRESS_LOCK` is set. To fire an
egress-locked render that PROVES the guard end-to-end, `export DERISK_EGRESS_LOCK=1` in the fire shell
BEFORE building `ENVJSON` (the builder above appends it to the pod `--env` ONLY when set, so a baseline
run stays byte-identical). Then assert on the structured channel, never prose:

- `@event derisk_meta ... "egress_lock": 1` -- the read-path outer stamps it, so the watcher confirms the
  lock was active on THIS run (a baseline run stamps `"egress_lock": 0`).
- `@event egress_guard_installed {"mode": "full_block", "allow": ["af_unix", "loopback"]}` at the top of
  the render, before any model import or load.
- `@event egress_guard_proven {"hf_blocked": true, "github_blocked": true, "ok": true}` (negative control)
  and `@event egress_guard_sane {"loopback_ok": true, "ok": true}` (positive control).
- a clean terminal `@event render_done` with `film_bytes > 0` AND `@event rclone_tripwire ... "fired":
  false` under the lock: the render completed doing ZERO phone-home, ENFORCED at the socket layer, not
  merely asserted from the baked architecture.

A `$0` CPU-only pre-check of the guard alone (no GPU, no render) is `vj_derisk.py guard-probe`: it installs
the guard and runs the same negative + positive controls, exit 0 iff proven, and emits
`@event guard_probe {"proven": true}`. Run it with REAL network reachable so a blocked
huggingface.co/github.com connect can only be the guard (with the guard OFF the same controls report the
hosts reachable, the discriminator). Under `docker --network=none` it still passes but proves less (every
connect fails regardless), so real-network is the meaningful mode.

### Container disk floor (>= 500GB for the :0.4.x image)

`--container-disk-in-gb` MUST be >= 500 for `ghcr.io/skyphusion-labs/vivijure-backend:0.4.x` (prod uses
500). During image extract RunPod holds the ~110GiB downloaded layers AND the ~300GiB extracted rootfs on
the container disk SIMULTANEOUSLY, so the peak is ~410GiB. A 400GB pod runs out of disk mid-extract and
NEVER starts the container: it sits at `uptimeSeconds`=0 with an empty read path indefinitely, and NO
error is surfaced (RunPod still shows `desiredStatus: RUNNING`, not a failure), so the only symptom is an
eternally-pending pod. That invisible failure cost one wasted ~$1 fire on #245. Do NOT lower the disk to
ease placement; raise it or do not fire. (The `--container-disk-in-gb 220` in the older per-card deploy
section above was sized for `:0.3.1` and is too small for `:0.4.x`.)
