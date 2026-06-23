# RunPod serverless endpoint configuration

How to size and pin the RunPod serverless endpoints that back the Vivijure render
and finish chain. This is the operator-facing reference: the GPU-tier rule, the
economics behind it, the per-endpoint actuals, and the two hard deploy
constraints (version-tag pinning, the 10-worker account cap).

See also `docs/configuration.md` (per-endpoint env + the i2v tier table) and
`docs/operations.md` (build/deploy/pin mechanics).

## Recommendation

**Run every endpoint on a high-end Blackwell-line GPU. Never the budget
L4 / L40S tiers.**

Realized across the fleet as two tiers, both Blackwell-line:

- **Heavy render backend** (`vivijure-backend`: SDXL keyframes, i2v, LoRA
  training) -> **datacenter top tier, B200 / H200.**
- **Finish-chain endpoints** (upscale, audio-upscale, lipsync/MuseTalk) ->
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

All endpoints: `workersMin = 0` (no always-active billing), flashboot on,
`scalerType = QUEUE_DELAY`, scalerValue 4.

| Endpoint                 | id               | GPU tier                          | workersMax | standby | Image source repo            |
|--------------------------|------------------|-----------------------------------|------------|---------|------------------------------|
| `vivijure-backend`       | `t9wcvlxh8rc5la` | **B200 / H200** (datacenter)      | 4          | 4       | `vivijure-backend`           |
| `vivijure-video-upscale` | `4q8idwbk6tyqbq` | **RTX PRO 6000 Blackwell** (96 GB)| 2          | 2       | `vivijure-upscale`           |
| `vivijure-audio-upscale` | `sj0btgpjdtswa7` | **RTX PRO 6000 Blackwell** (96 GB)| 2          | 2       | `vivijure-audio-upscale`     |
| `vivijure-musetalk`      | `zw6pt4lymf69pk` | **RTX PRO 6000 Blackwell** (Server)| 2         | 2       | `vivijure-musetalk`          |

`vivijure-backend` lists three Blackwell RTX PRO 6000 SKUs only where it is the
finish tier; its render tier is B200 / H200. The finish endpoints list all three
RTX PRO 6000 Blackwell editions (Server / Workstation / Max-Q) so the scheduler
can place on whichever has stock.

## Deploy constraint 1: pin by `:version`, never `:sha`

Pin each endpoint's template image to its **`:version` tag** (e.g.
`ghcr.io/skyphusion-labs/vivijure-musetalk:0.1.0`). RunPod's immutable
`:sha-<digest>` pin does **not** work for these endpoints; the version tag is
what RunPod resolves on the next cold start. The image build CI must therefore
emit a `:version` tag (see the build-image workflow's "Compute tags" step, mirrored
across `vivijure-upscale` / `vivijure-audio-upscale` / `vivijure-musetalk`); a
build that only pushes `:sha` / `:latest` cannot be pinned.

Pinning is a deliberate, separate step from building: a build does not touch the
live endpoint (`docs/operations.md`). You pin the template, and the endpoint pulls
the new image on its next cold start.

## Deploy constraint 2: the 10-worker account cap

The RunPod account has a **hard cap of 10 concurrent workers.** The sum of every
endpoint's `workersMax` must be `<= 10`.

Current allocation is **exactly at the cap**:

```
vivijure-backend        4
vivijure-video-upscale  2
vivijure-audio-upscale  2
vivijure-musetalk       2
                       --
total                  10   (= the cap)
```

Consequence: **adding a new endpoint, or raising any `workersMax`, requires
lowering another endpoint's `workersMax` first.** There is no headroom. A
`wrangler`/Cloudflare deploy will not catch this; only a RunPod scale event will,
and it fails by silently capping concurrency (jobs queue) rather than erroring.
Re-balance deliberately when the finish chain's load profile changes.

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
