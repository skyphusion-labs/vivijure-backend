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

1. **Throwaway key** -- generate a per-pod ed25519, NEVER reuse conrad's key:
   `ssh-keygen -t ed25519 -f ~/.ssh/derisk_<arch>_$$ -N '' -C derisk-<arch>`
2. **Spin the pod** via the RunPod MCP `create-pod`, pinned to the gpuTypeId + the image, registry
   auth attached by `containerRegistryAuthId` (MCP-managed, no dashboard), the throwaway pubkey in env,
   a HARD TTL. Single pod, single concurrency.
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
6. **Teardown** -- on PASS delete the pod; on FAIL `stop` (not delete) so the disk survives for an SSH
   debug session, and emit the resume handle. Either way `shred -u` the throwaway key.

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

So the wrapper **self-tees** all output to `/workspace/derisk.log` and, from a **mandatory** `PUBLIC_KEY`
pod-env, apt-installs + starts `sshd` BEFORE the de-risk so the operator can `ssh` in and `tail -f
/workspace/derisk.log` live. No `PUBLIC_KEY` = blind pod (do not fire).

The driver `deploy/vj_derisk.py` is curled at the pinned commit
`481f8277eeafa0b045e97075bf5b2191e933b263` (#144) -- the pod runs exactly the reviewed code.

### Open mechanism follow-ups (#146)

The RunPod MCP cannot deploy a custom-start-command pod: `create-pod` exposes no `dockerStartCmd`/
`templateId`; `create-template` silently drops `dockerStartCmd` (`args` stays empty); there is no MCP
template-to-pod deploy; and no pod-log API. Hence the runpodctl/console + ssh-tee path. #9 (the
automated regression engine) needs an MCP enhancement or a runpodctl/GraphQL shim in
`runpod_verify.py`'s PodClient seam, and a baked sshd/debug-entrypoint in the image.

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

## RESUME (post-checkpoint, step one)

Held at the clean pre-fire boundary (no GPU pods running; nothing to revert; prod `:0.2.28` untouched).
To resume the #15 3-arch de-risk:

1. Rollins pings Strummer to FIRE the **RTX PRO 6000 Blackwell Server Edition (sm_120) canary** from
   `deploy/derisk_pod_start.sh` (base64 via runpodctl), image `:0.3.1`, NO volume, `PUBLIC_KEY` set,
   #136 caps (~$50 / 60-min manual terminate). Strummer sends the pod ID to the lead.
2. Watch for `@event arch_gate {... "missing": [], "passed": true}` -- report the raw `arch_list`
   verbatim (closes #14). A missing base arch = hard STOP, do-not-promote, flag.
3. On `@event derisk_pass` (canary green), fan out **H200 (sm_90)** + **B200 (sm_100)** in parallel.
4. Per pod assert: `gpu_probe` kernel_ok + capability_in_arch_list; `baked_probe` vj_baked +
   precision=bf16 + a Wan i2v repo (image, not a mount); `rclone_tripwire fired=false` (no-pull);
   `render_done film_bytes>0`; `i2v_jit` first-call JIT captured per arch.
5. PRs awaiting the lead's merge: #145 (H200 conservative-high cost) and #147 (this wrapper + runbook).
   #5 serverless prod promote remains gated on Conrad.
