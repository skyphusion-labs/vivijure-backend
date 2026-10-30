# RunPod serverless endpoint configuration

How to size and pin the RunPod serverless endpoints that back the Vivijure render
and finish chain. This is the operator-facing reference: the GPU-tier rule, the
economics behind it, the per-endpoint actuals, and the two hard deploy
constraints (version-tag pinning, the account worker quota).

See also `docs/configuration.md` (per-endpoint env + the i2v tier table) and
`docs/operations.md` (build/deploy/pin mechanics).

**Do not pin `:1.0.1`** (t2 weight-digest rewrite; RunPod `unexpected EOF`). Incident:
[`runpod-1.0.1-weight-digest-eof.md`](runpod-1.0.1-weight-digest-eof.md).

## Current state (read from `GET /v2/serverless` on 2026-10-30)

| | `vivijure-backend-hosted` | `vivijure-upscale-hosted` |
|---|---|---|
| id | `xe5d1ivs40ohjo` | `0m7puc409hqw8a` |
| Image | `vivijure-backend:1.1.2@sha256:69f06ab95ab75530af7e0fa135c768548d339435a3714036b6f32e95a2b0d586` | `vivijure-upscale:1.0.4` |
| Pools | `BLACKWELL_96`, `ADA_80_PRO`, `HOPPER_141`, `BLACKWELL_180`, `BLACKWELL_288` | `BLACKWELL_96` |
| `workers` min / max | 0 / 30 | 0 / 10 |
| Idle timeout | 180 s | 10 s |
| Scaler | `QUEUE_DELAY` 4 | `QUEUE_DELAY` 4 |
| FlashBoot | on | on |
| Container disk | 80 GB | 40 GB |
| Mounts | none | none |
| Execution timeout | 3600 s | 900 s |
| Min CUDA | 12.8 | 12.0 |

Backend env: `HF_HOME=/opt/models/hf-cache`, `VJ_MODELS_ROOT=/opt/models`, `VJ_MODEL_VERSION=1`,
`HF_HUB_OFFLINE=1`. `VJ_VOLUME_ROOT` and `VJ_VOLUME_KIND` are not set on the endpoint. The thin image
still sets them, and nothing is mounted there, so the worker logs that the volume is not mounted and
takes the R2 mirror.

**Why 80 GB.** A stills job pulls 26.22 GB from R2 (`mirror_complete total_bytes`). Six jobs on two
80 GB workers ran with no ENOSPC. Below 80 GB is untested.

**Why 180 s idle.** Measured on the same image: a cold keyframe cost $0.13 to $0.23 of worker time and
a warm one $0.02, at $3.49 per hour. Idle costs $0.058 per minute, so holding a worker pays when the next
job arrives within about 2 to 3.5 minutes. A one-off job pays up to about $0.17 of extra idle.

**Previous values, for rollback:** image
`vivijure-backend:1.1.1@sha256:0ebdc0e19b48676e310a10c863c7dfc4e6a8c642b3fbd887e73e248a6780157f`, same
disk and idle timeout. Before that: image
`vivijure-backend:1.1.0@sha256:b42084bb1e295976d4ad555566aee1240c5ce86f1692a33560a127ecb155e6ac`,
disk 300 GB, idle timeout 10 s. That image starts a 114 GiB Wan pull on every keyframe job and needs
the larger disk, so restore all three together.

### Pool prices

Serverless prices per hour, from `list-gpu-types` on 2026-10-30:

| Pool | GPU | Price |
|---|---|---|
| `BLACKWELL_96` | RTX PRO 6000 | $3.49 |
| `ADA_80_PRO` | H100 | $5.39 |
| `HOPPER_141` | H200 | $6.99 |
| `BLACKWELL_180` | B200 | $10.79 |
| `BLACKWELL_288` | B300 | $11.99 |

Nothing has been measured on `ADA_80_PRO` or `HOPPER_141`. The times below are not evidence for those pools, and a job that lands on one costs more per hour than `BLACKWELL_96`.

### Measured keyframe times, image 1.1.1

80 GB, no volume, RTX PRO 6000, one shot, `final` tier, scene lock, trained LoRA supplied.

| Run | Queue | Execution | First keyframe from submit |
|---|---|---|---|
| Cold | 161 s | 233 s | 6 min 31 s |
| Cold | 6 s | 130 s | 2 min 12 s |
| Warm | 0.2 s | 21 s | 22 s |
| Warm | 0 s | 25 s | 26 s |

Mirror on the cold run: 136 s at 192.5 MB/s. Each model load after that took 1.5 to 7 s.

### Two cold submits ran serially

Tested twice at `QUEUE_DELAY` 4 with `workers.max` 2. The second job waited for the first worker, then
ran on it. On the cold pair, a second worker only began initializing about 2 min 12 s after submit, as
the first job finished. So a burst of cold jobs queues before it fans out. Tracked in #545.

## Recommendation

**Run every endpoint on a high-end Blackwell-line GPU. Never the budget
L4 / L40S tiers.**

Realized across the fleet as two tiers, both Blackwell-line:

- **Heavy render backend** (`vivijure-backend`: SDXL keyframes, i2v, LoRA
  training) -> **datacenter top tier, B200 / H200.**
- **Finish-chain endpoints** (upscale; audio-upscale and musetalk are both RETIRED, see below) ->
  **RTX PRO 6000 Blackwell (96 GB).**

This is the standing standard. Do not "save money" by dropping a finish endpoint
to L4 / L40S; as the economics below show, the cheaper card is the more
expensive choice for a GPU-bound job.

## Why (justification)

### 1. The render backend basically REQUIRES the high-end card

The SDXL keyframe gen + i2v + LoRA training are heavy on both VRAM and compute,
and the backend images are built `cu128` / torch 2.8 for the Blackwell line
(`sm_120`). It is not a preference; the workload needs the silicon. Running it on
anything lower either will not fit in VRAM or will not run the compiled kernels.

### 2. For every other endpoint, the high-end Blackwell tier SAVES money

Serverless billing is **per second of active execution**, not per hour. So the
unit cost of a job is:

```
$/job = ($/sec of the GPU tier) x (seconds the handler runs)
```

A faster card finishes a GPU-bound job in fewer seconds, and the shorter
wall-time more than offsets the higher per-second rate. The premium card comes
out **cheaper per job**, not just faster.

On top of that:

- **Supply, not just speed.** The RTX PRO 6000 Blackwell tier is high-supply, so
  jobs schedule immediately. The budget L4 / L40S tiers are low-stock and
  **throttle**: jobs queue waiting for a free worker, and a queued job holds its
  worker slot longer. That is effectively *more* cost and worse latency. (This is
  exactly why `vivijure-video-upscale` was moved onto Blackwell.)
- **Scale-to-zero means no idle penalty.** With `workersMin = 0` plus flashboot
  standby (see billing note below), an idle endpoint costs **$0** regardless of
  tier. There is no carrying cost to provisioning the premium card; you only pay
  premium seconds while a job is actually running, and there are fewer of them.

Net: fastest finish + no throttling + per-second billing = **lowest $/job AND
best latency.** The "expensive" card is the cheaper one when the work is
GPU-bound.

### 3. The caveat that makes or breaks the math: the handler must be GPU-bound

Everything above holds **only when the endpoint's handler is GPU-bound.** A
CPU-bound or IO-bound handler on a Blackwell card pays premium per-second rates
for idle silicon (the pre-fix `vivijure-video-upscale`, which was doing CPU
encode/lipsync work on a GPU card and bleeding cost). The fix in that case is to
**make the handler GPU-bound** (move the encode/restoration onto the GPU), NOT to
downgrade the card. Downgrading just trades one waste for slower waste.

Rule of thumb before trusting the economics: confirm the handler saturates the
GPU for the bulk of its wall-time. If it does not, fix the handler first.

## Per-endpoint reference (live config)

The two endpoints in "Current state" are the only ones on the account (2026-10-30). The endpoints this
page used to list (`t9wcvlxh8rc5la`, `uf4iwoen5r48zx`, `zqb7tougbqfkqa`, `4q8idwbk6tyqbq`) no longer
exist, and their IDs are history. Read `GET /v2/serverless` rather than this page before acting.

## Deploy constraint 1: pin by `:version`, never `:sha`

Pin each endpoint's template image to its **`:version` tag** (e.g.
`ghcr.io/skyphusion-labs/vivijure-upscale:0.1.0`). RunPod's immutable
`:sha-<digest>` pin does **not** work for these endpoints; the version tag is
what RunPod resolves on the next cold start. The image build CI must therefore
emit a `:version` tag (see the build-image workflow's "Compute tags" step); a
build that only pushes `:sha` / `:latest` cannot be pinned. The now-retired
`vivijure-musetalk` and `vivijure-audio-upscale` satellites followed the same
rule, and their build workflows are the reference if you add a satellite.

Pinning is a deliberate, separate step from building: a build does not touch the
live endpoint (`docs/operations.md`). You pin the template, and the endpoint pulls
the new image on its next cold start.

After an image change, jobs can land on stale workers until the rollout finishes. Read the worker's `version` and image before crediting a proof.

Keep a test endpoint until its logs are read. A worker's logs go when the endpoint is deleted: `stream-worker-logs` then returns 404.

## Deploy constraint 2: the account worker quota

**The quota is 50 `workers.max` across all endpoints** (measured 2026-10-30: `POST /v2/serverless` refused
with "Max workers across all endpoints must not exceed your workers quota (50)" when the sum was 50). The
sum now is **40** (30 + 10). The old figures on this page (30 cap, 25 plan, 19 sum) no longer hold.

Consequence: **adding an endpoint, or raising `workers.max`, can be refused at create or update.** Check
the sum first. Lower another endpoint deliberately if needed, and read the result back.

## Billing note (so nobody re-flags standby)

- **Per-second execution billing.** A worker bills only while it is *running a
  job*, charged per second.
- **`workersMin > 0` = "Always Active" = billed 24/7**, idle or not. Keep
  `workersMin = 0` on every endpoint. This is the only knob that creates idle
  spend; check it (not standby) when worried about a bleed.
- **`workersStandby` + flashboot = warm idle workers that bill $0 while idle.**
  Standby is free cold-start insurance: a standby worker bills only when it
  actually picks up a job. Do not mistake a non-zero `workersStandby` for idle
  spend; it is not.
- **Measured 2026-10-30 on `xe5d1ivs40ohjo`:** hourly billing for 18:00 to 20:00Z read $0.14 and
  $0.46 while 11 to 19 workers read `IDLE` in `list-endpoint-workers`, so those workers were not
  billing. This settles rows reading `IDLE`. RunPod's concept doc says REST `IDLE` is billed; the
  buckets disagree, and the buckets are what we were charged. It does not separately measure the idle
  timeout after a job: "Why 180 s idle" above assumes that window bills, and that is not yet read.
