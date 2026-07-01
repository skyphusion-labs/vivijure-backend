# Serverless :0.3.3 -- the first BAKED image, confirmed in production

`backend-v0.3.3` (`ghcr.io/skyphusion-labs/vivijure-backend:0.3.3`, source `b93cdbd`) is the
first vivijure-backend image with **the model weights baked into the image layers** rather than
pulled from R2 at cold start. It was deployed to the production serverless endpoint
(`t9wcvlxh8rc5la`) and confirmed, end to end, on **both production GPU architectures** through the
real serverless handler.

**Bottom line: 28/28 clean full-pipeline renders across H200 and B200, zero errors, zero R2
cold-pulls. The image works, the weights are baked, and the cu128 kernels load and run on every
card the production pool bounces to.**

This is the operator-facing record of that confirmation. For endpoint sizing / the GPU-tier rule
see `docs/runpod-endpoint-config.md`; for cold-start staging internals see `docs/cold-start-design.md`.

---

## 1. What "baked" means, and why it matters

Through `:0.2.28` the deployed image was **config-only**: it shipped code plus HuggingFace config
stubs, and every cold worker pulled ~100 GB of weights from R2 before it could render. That pull is
slow, it burns egress, and it puts a hard R2 dependency on the render path.

`:0.3.3` bakes the weights **into the image** (bf16 throughout):

- **Keyframe:** RealVisXL V5.0 (SDXL) plus the IP-Adapter / InstantID / ControlNet / Lightning
  adapters.
- **Motion (i2v):** Wan2.2-I2V-A14B-Diffusers (bf16) plus the lightx2v / Lightning distill weights.
- **Finish:** RIFE (flownet), GFPGAN, antelopev2, and the **facexlib** detection + parsing weights
  (`detection_Resnet50_Final.pth`, `parsing_parsenet.pth`) baked in `:0.3.3` specifically so GFPGAN
  face-restore runs fully offline with no phone-home.

A baked worker therefore **never pulls from R2 at boot or at render**. The trade is a one-time
image pull onto a cold worker (~97 GB, paid once per fresh worker), in exchange for zero
per-render R2 dependency and no egress contention on fan-out.

### How the bake is proven, not asserted

An empty or hollow bake is worse than no bake (a lying `.vj-baked` sentinel would short-circuit the
very R2 pull that would have fetched the missing weights). Three gates prevent that, and one runtime
signal confirms it live:

- **`#138` empty-bake gate** -- `assert-weights` gates the `.vj-baked` write in the Dockerfile via
  `&&`, so the sentinel is physically un-writable over a stub tree; a byte-floor per weight shard
  rejects zero-byte layers.
- **`baked_probe`** -- the de-risk driver asserts the **exact** repo the runtime will load is present
  in the cache before any GPU work ($0, pre-kernel), closing the "globbed a similar repo, called it
  baked" coverage gap.
- **`mirror_done { pulled: false }`** -- the runtime signal. Every one of the 28 confirmation renders
  emitted `pulled: false`, i.e. the baked short-circuit fired and no R2 GET happened. This is the
  load-bearing proof that the weights came from the image, not the network.

---

## 2. Kernel / architecture coverage

The production serverless pool bounces a job across datacenter Blackwell-line cards. The image must
run the compiled kernels on each. Coverage rides on the **prebuilt `cu128` wheels** (torch / vision /
audio), which ship SASS/PTX for `sm_90`, `sm_100`, and `sm_120`. Nothing in this image compiles CUDA
from source, so `TORCH_CUDA_ARCH_LIST` is a deliberate no-op here; the load-bearing build-time proof
is `torch.cuda.get_arch_list()` inside the image showing `{sm_90, sm_100, sm_120}` (a STOP-gate: any
arch missing means the wheels silently dropped it).

**Production serverless tier = H200 or B200 only.** (The RTX PRO 6000 / `sm_120` is a finish-chain /
de-risk card, not part of the render pool.)

| card | arch | role | :0.3.3 verdict |
|---|---|---|---|
| **H200** | Hopper `sm_90` | production render tier | **CONFIRMED** -- 17/17 clean serverless renders under load |
| **B200** | Blackwell DC `sm_100` | production render tier | **CONFIRMED** -- 11/11 clean serverless renders |
| RTX PRO 6000 | Blackwell `sm_120` | finish chain / canary | proven via de-risk canary (`derisk_pass` on `:0.3.2`); not a render-pool arch |

A render **completing** on a card is itself the kernel proof: if the `cu128` kernel image were
missing for that arch, i2v would crash with "no kernel image is available for execution on the
device". It did not, on either production card.

---

## 3. The confirmation run (2026-07-01)

Conrad deployed `:0.3.3` to the production endpoint `t9wcvlxh8rc5la` (endpoint version 186 -> 188 over
the run) and we confirmed both arches through the **real serverless handler** -- real `render` jobs
(a minimal single-shot, no-cast bundle, `quality_tier: final`), asserting on the R2 structured event
channel plus `workerId -> gpuTypeId` correlation. This exercises the actual production path, not a pod
proxy.

### H200 (Hopper `sm_90`) -- 17/17 under load

17 concurrent full-pipeline renders. The endpoint scaled workers to match, drained the queue, and the
`failed` counter never moved. Every job: `mirror pulled=false` -> SDXL keyframe -> Wan2.2 i2v x40 ->
RIFE + GFPGAN (facexlib offline) -> assemble -> a real ~227 KB `full.mp4` object in R2. Warm-worker
wall time ~240-370 s per render.

### B200 (Blackwell DC `sm_100`) -- 11/11

B200 capacity is scarce, so the endpoint pool was pinned to all-B200 to force placement, the batch was
run, and the pool was then returned to the normal mixed H200/B200 configuration. All 11 clean; every
job `mirror pulled=false`, full chain to a verified `full.mp4`, `tier_mismatch` absent (planner planned
`b200`, actual `b200`).

**True-cold `sm_100` i2v JIT** (first ever measured -- the H200 figures above were warm-cache from
prior jobs): keyframe at +9.4 s; i2v model-load + first-call gap 24.7 s; JIT tail 9.9 s; then **steady
~3.45 s/step -- faster than H200's ~5.0 s/step.** Warm B200 renders came in ~177-206 s. A true-cold
worker pays a one-time Triton/inductor compile; this is expected and amortized by warm workers. Do not
claim "no JIT".

### Scorecard

```
arch / pool                     verdict          evidence
H200 / sm_90                     CONFIRMED 17/17  clean render_done, pulled=false, GFPGAN offline, actual=h200
B200 / sm_100                    CONFIRMED 11/11  clean render_done, pulled=false, cu128 kernel loaded on sm_100, actual=b200
RTX PRO 6000 / sm_120            proven (canary)  :0.3.3-lineage derisk_pass on sm_120; not a render-pool arch

28/28 clean serverless full-pipeline renders across H200 + B200, zero errors, zero R2 cold-pulls.
```

---

## 4. Operational notes

- **`tier_mismatch` is benign** (`pipeline.py`). It is an informational warn-once emitted when the
  card the job landed on differs from the tier the planner targeted for i2v. The render proceeds using
  the **job's** `quality_tier`, never the device; there is zero output change and no fallback, so it is
  not a degrade. On this run it fired on the H200 jobs (planner prefers B200 for final tier) and was
  absent on the B200 jobs. Standing signal only: planner preference and scheduler placement diverge on
  this endpoint; a routing reconciliation is optional and is a planner decision, not a correctness bug.
- **Cold-worker image pull (~97 GB)** is the one cost the bake introduces, paid once per fresh worker.
  On the B200 run it took several minutes before the first worker went `running`. This is the flip side
  of never paying an R2 cold-pull at render time.
- **Correlation method:** `workerId -> gpuTypeId` (authoritative), the render's own `actual=<card>`
  self-report, and `tier_mismatch` presence/absence (present on H200, absent on B200) -- three
  independent signals, all agreeing per card.

---

## 5. Credit

This confirmation was a full-crew effort: the baked image (bake pipeline, empty-bake gate, bf16 seed,
the i2v offline-load fix, the facexlib offline bake, `baked_probe` hardening), the de-risk harness and
the structured event channel it asserts on, the endpoint wiring, and the load + arch-correlation run.
A clean 28/28 across both production arches, proven honestly through the real handler. Fantastic work,
all around.
