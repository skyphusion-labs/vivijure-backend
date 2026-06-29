# Cold-start design (issue #55, Phase B)

How a cold worker gets its model weights, why that dominates wall-clock under concurrent
fan-out, and which structural fix to take. This is the decision doc the 0.2.20 telemetry
(`@event mirror_complete` / `i2v_mirror_complete`, issue #55 Phase A) was built to inform; it
weighs **preloaded network volumes** vs **bake-into-image** vs **pre-warm pool** vs **staged
cold-starts** with numbers instead of vibes.

> **Status: DECIDED (revised 2026-06-29) -- BAKE THE WEIGHTS INTO THE IMAGE (Option A), at
> half precision.** The original recommendation (preloaded per-datacenter network volumes, Option D)
> is NOT taken. Two things moved that the first pass did not have: (1) **half-precision weights** --
> the i2v MoE quantizes to fp8 (~half size) and the curated set drops the dead-weight repos, so the
> baked set is ~90 GB (fp8) / ~117 GB (bf16), not the 223 GB that made baking look unwieldy; (2) a
> **1200 GB GitHub-hosted build runner** (`vivijure-bake`, 32-core/128 GB/1200 GB) that builds the
> bf16 set with disk headroom. A baked worker carries its weights, so it is **datacenter-agnostic**
> (no per-DC volume pinning it to a provisioned region -- that pinning throttled spill to free GPUs
> elsewhere) and pays **no R2 cold-pull tax and no R2 egress at all**. Sequencing: **fp8 ships
> first** (smaller, proven CPU-quantize); **bf16 is a precision swap** on the same runner once
> validated. The R2 mirror is **demoted to the fallback** for a non-baked/legacy image (the
> `.vj-baked` marker selects the path). See "Recommendation (sequenced)" below for the revised plan
> and "Option A" for why it is now the structural fix; Option D's analysis is kept for the record.

## The problem

A cold worker carries no weights -- the image is ~1-2 GB on purpose (a hundreds-of-GB layer is
rejected by GHCR and ingests slowly), so it mirrors the kept model set from R2 into the local HF
cache at startup, then renders offline (`harness/models_mirror.ensure_models`). A warm worker
sees the completion sentinel and skips. The mirror is the entire cold-start cost, and it recurs
**on every backend redeploy** (pinning a new image recycles the workers, so they come back cold).

Under concurrent fan-out it does not scale: N cold workers each run `rclone --transfers 16`
against R2 at once, contend on egress, and the GPU sits at 0% util for the whole staging phase.
The 2026-06-13 8-way load test ([load-test doc](load-test-2026-06-13.md)) is the smoking gun:
8/8 jobs succeeded, but most workers sat at **0% GPU / 100-179% CPU** staging weights, and the
lone weak node (an RTX PRO 6000 Blackwell) spent **~35 min** copying before its first frame,
gating the batch by ~18 min. In a heterogeneous serverless pool the tail is set by the weakest
node that lands, and RunPod assigns one at random.

## The cost model (from 0.2.20 telemetry)

A single cold worker, best case (no cross-worker contention), measured live on an H200 on
2026-06-15 (`smoke-own-gpu`, an `i2v_clip` job):

| Mirror leg | Bytes | Time | Notes |
|---|---|---|---|
| `hf-cache` (base set) | -- | **76.6 s** | SDXL/RealVis/Animagine + IP-Adapter + InstantID configs/weights |
| `antelopev2` | -- | 6.3 s | insightface face models |
| `GFPGANv1.4` | -- | 4.0 s | face restore |
| `rife` | -- | 1.5 s | flownet.pkl |
| **base total** (`@event mirror_complete`) | **55 GB** | **88.4 s** | 622.6 MB/s; the common keyframe/preview floor |
| `Wan2.2-I2V-A14B` | -- | 158.2 s | the i2v model |
| `Wan2.2-Lightning` | -- | 208.7 s | the 4-step distill LoRA |
| **lazy i2v total** (`@event i2v_mirror_complete`) | **168 GB** | **366.9 s** | 458.8 MB/s; paid only by i2v workers |

Two facts drive every decision below:

1. **The lazy split already pays off.** A keyframe/preview/LoRA-train worker pulls only the
   55 GB base (88 s) -- it never touches the 168 GB Wan set (it's in `DEFAULT_SKIP_REPOS`, pulled
   lazily on first `i2v_pipeline` use). Keep this; it makes the common cheap op cheap.
2. **The dominant cost is Wan, not the base.** An i2v cold worker pays **88 s + 367 s ≈ 7.6 min**
   of staging before the first denoise step. The draft i2v it then runs is **~45 s** (4 distill
   steps x ~11 s). Staging is ~10x the actual draft compute. Whatever we do, the 168 GB Wan pull
   is the thing to attack.

Throughput also varies by node and by contention: 622 MB/s (base) vs 459 MB/s (Wan) on the *same*
warm-network H200; the Blackwell's ~35 min under 8-way load implies a far weaker link plus egress
contention. The single-worker numbers above are the **floor**; fan-out makes them worse.

## Already shipped (the cheap wins -- not the structural fix)

- **rclone multi-thread** (`--multi-thread-streams 8 --multi-thread-cutoff 100M`): chunks the big
  single files so per-worker throughput isn't single-stream-bound. Helps the floor, not contention.
- **Lazy i2v split**: keyframe/preview workers skip the 168 GB Wan pull entirely.
- **Eager i2v prefetch**: on a full render the Wan pull overlaps LoRA training (GPU-bound, network
  idle). Note: a standalone `i2v_clip` has no training to hide behind, so it pays the Wan pull serially.
- **Sentinel + `VJ_MODEL_VERSION`**: a warm worker with a current cache skips the mirror (0 s).
- **0.2.20 telemetry**: the numbers in this doc. This is the measurement, not the fix.

## Options

### A. Bake weights into the image

Ship the weights in a content-addressed Docker layer instead of mirroring from R2.

- **Upside:** a baked layer is pulled once per physical host and cached; unchanged across
  code-only deploys, so the per-deploy re-mirror tax disappears on hosts that have seen the layer,
  and R2 egress + cross-worker contention go to zero for what's baked.
- **Downside:** size. Base is 55 GB; the full set with Wan is **~223 GB**. The Dockerfile is
  explicit that a layer that big is rejected by GHCR and ingests slowly -- baking just relocates
  the bottleneck from R2 to the registry pull on any fresh host, and adds build time + GHCR
  storage. RunPod serverless workers can land on hosts that have never pulled the image, so the
  worst case (fresh host, cold pull of a 223 GB image) can be *worse* than the R2 mirror.
- **Hybrid bake (base only):** bake the stable 55 GB base into its own layer (it changes rarely),
  keep the 168 GB Wan on lazy R2. But this saves the *smaller* half (88 s) and leaves the dominant
  Wan pull (367 s) untouched, while still shipping a 55 GB image layer. Low leverage for the cost.
- **Verdict (original pass): reject for now.** It attacks the cheap half and makes the image
  unwieldy; baking the expensive half (Wan) is exactly the layer GHCR won't take.
- **Verdict (REVISED 2026-06-29): TAKE THIS -- it is the structural fix.** Three facts overturn the
  original reject:
  - **The "~223 GB" was gross, not what we bake.** The bake ships the *curated, loaded* set, not the
    raw R2 mirror. Dropping the dead-weight repos (T2V 126 GB, SDXL-base 62 GB, sdxl-turbo 34 GB --
    all in `DEFAULT_SKIP_REPOS`, never loaded) and trimming to the spec leaves ~117 GB at bf16.
  - **Half precision halves the expensive half.** The i2v MoE experts quantize bf16->fp8 on CPU
    (free, no GPU; `deploy/quantize_i2v_fp8.py`), so the fp8 baked set is **~90 GB**. fp8 ships
    first; bf16 (**~117 GB**) is a one-line precision swap (re-stage the seed) once validated.
  - **The GHCR per-layer limit is handled, and the build runner now fits.** GHCR rejects a >=10 GB
    layer, so the bake bin-packs the curated set into many <10 GB layers (`deploy/bake_layers.py`:
    pre-build assert + post-build `docker history` gate). The peak BUILD disk is ~3x the payload
    (staged seed + buildkit snapshot + loaded image) + the base stack, i.e. ~290 GB (fp8) / ~370 GB
    (bf16) -- which does NOT fit the 300 GB hosted runner the thin image used, but builds with
    headroom on the **1200 GB `vivijure-bake` larger runner** (~3-4x headroom; see the disk-budget
    note below). On a serverless HOST the baked layers are content-addressed and cached after the
    first pull per physical host, and -- unlike the original worry -- a baked worker that lands on a
    fresh host pulls from GHCR (fast, regional, free egress) instead of mirroring 168 GB from R2.
  - **It kills the per-DC volume ops AND the R2 egress.** Datacenter-agnostic placement means no
    volume to provision/refresh per DC (Option D's whole operational burden) and zero R2 cold-pull
    egress on the common path. The R2 mirror stays only as the fallback for a non-baked image.

  **DISK-BUDGET NOTE (the number that sized the runner):** peak transient build disk is dominated by
  three coexisting copies of the weight payload -- the staged seed in the build context, the BuildKit
  snapshot the `COPY` layers write, and the image loaded into the docker daemon for the pre-push
  smoke -- plus the ~18 GB CUDA/torch/conda base. So peak ~= 3 x payload + 18 GB: **~290 GB at fp8,
  ~370 GB at bf16.** bf16 therefore does NOT build on a 300 GB runner; the 1200 GB `vivijure-bake`
  tier gives ~3.3x headroom for bf16 and ~4x for fp8 (enough to build multiple image variants in one
  job). A 600 GB tier would be only ~1.6x for bf16 -- workable but not comfortable -- so 1200 GB is
  the right-sized choice, not overkill.

### B. Pre-warm pool (RunPod active/min workers)

Keep a small number of workers always-on. A warm worker has the sentinel + on-disk cache, so it
skips the mirror entirely (**0 s staging**) and the first job lands instantly.

- **Upside:** zero staging on the kept-warm workers; simple endpoint setting, no code.
- **Downside:** **cost.** An always-on H200 is **~$5/hr ≈ $3,600/mo per worker** whether it renders
  or sits idle -- and it only covers the first N concurrent jobs; a larger cold fan-out still pays
  full staging. For a bursty BYO-GPU product this is the wrong place to spend.
- **Verdict:** **reject as the primary fix** -- it buys 0 s staging for one job at the price of a
  network volume that covers an entire datacenter (see D). A single warm worker could be kept as a
  latency cushion later, but it is not the structural answer.

### D. Preloaded per-datacenter network volumes  *(recommended)*

Mount a RunPod network volume, preloaded with the weight set, on the serverless endpoint. A worker
reads weights from the local volume at `/runpod-volume` instead of copying 223 GB from R2 -- so the
cold-mirror cost goes to ~0 with **no idle GPU** and **no R2 egress contention** (the fan-out
killer). This is the option Conrad raised; the mechanics below are from the RunPod docs
([storage S3 API](https://docs.runpod.io/storage/s3-api),
[serverless network volumes](https://docs.runpod.io/serverless/storage/network-volumes)).

- **The datacenter rule (why the earlier attempt failed):** a serverless endpoint attaches **one
  network volume per datacenter**, and **data does not sync between volumes**. With only two DCs
  preloaded, any worker RunPod placed in a third DC had no local volume and fell back to the slow
  R2 mirror -- defeating the purpose. The fix is **full coverage**: a preloaded volume in *every*
  datacenter the endpoint is allowed to scale into.
- **Cost:** network storage is **$0.07/GB/mo** (first 1 TB). A 300 GB volume (223 GB weights +
  headroom) is ~$21/mo per DC; covering ~10-15 H200+ datacenters is **~$210-315/mo total** --
  less than one *month* of a single warm H200, and it makes **every** worker in **every** covered
  DC start hot, not just N of them.
- **Preload without a pod:** the S3-compatible API writes straight to a volume
  (`s3://<VOLUME_ID>/...` via `https://s3api-<DC>.runpod.io/`, a dedicated S3 API key), so we can
  push the R2 weight set into each DC's volume from CI with no GPU spend. Caveat from the docs:
  `aws s3 sync` "struggles with 10,000+ files or 10 GB+ directories" and 500 MB/PutObject (multipart
  for bigger) -- our HF cache is thousands of files, so the robust preload is a **one-shot cheap pod
  per DC that rclones R2 -> `/workspace`**, or a chunked uploader; plain `s3 sync` will choke.
- **GPU-availability trade-off:** pinning to volume-backed DCs can narrow GPU supply, so cover
  enough H200+ DCs (determine the current set from the RunPod console / availability API) and treat
  the volume list as the endpoint's DC allow-list.
- **GPU floor -- H200+ (141 GB):** Wan2.2-I2V-A14B is a ~28B two-expert MoE; full-step
  (standard/final) needs H200+ -- even an H100-80GB OOMs. A 96 GB card (RTX 6000 PRO) runs it only
  by CPU-offloading (~2x slower, fine for 4-step draft); 24-32 GB cards can't run it at all. So the
  volume coverage targets H200-class datacenters.
- **Worker change (small):** point `HF_HOME` / `VJ_MODELS_ROOT` at the mounted volume path; the
  existing sentinel/presence logic then sees the weights already there and **skips the mirror**.
  The mount path is **configurable** (it does not have to be `/runpod-volume`), so we set it to wherever
  `HF_HOME`/`VJ_MODELS_ROOT` already point and the worker code needs no path change. If a worker
  ever lands without a populated volume, the **R2 mirror still runs as the universal fallback** --
  so this is a speed layer over today's correctness, never a new single point of failure. One
  writer per volume (the preloader); workers read-only (the docs warn concurrent writes can
  corrupt a volume).
- **Verdict:** **take this as the structural fix.** Cheapest per-DC, removes both the staging floor
  and the contention, and degrades gracefully to the R2 mirror.

### C. Staged / jittered cold-starts

Attack the *contention* specifically (the load-test killer: N workers hammering R2 at once).

- **True staggering** (worker k waits for k-1) needs cross-worker coordination there's no clean
  primitive for in independent serverless workers (an R2-object semaphore is possible but ugly),
  and it serializes the batch -- worker 8 waits for 1-7.
- **Startup jitter** (each worker sleeps a small random interval before the mirror) is a cheap,
  stateless approximation: it de-syncs the egress spike so each pull gets closer to the
  best-case floor instead of all colliding. ~5-15 s of jitter is negligible against a 7.6 min
  pull but meaningfully spreads the 16xN concurrent transfers.
- **Verdict:** **take the jitter** (small code change, low risk); skip true staggering.

## Recommendation (sequenced) -- REVISED 2026-06-29

The structural fix is now **A (bake), at half precision**, not D (volumes). The pivot: half-precision
+ the curated set make the image fit, the 1200 GB `vivijure-bake` runner builds it, and
datacenter-agnostic placement removes the per-DC volume ops AND the R2 egress entirely.

1. **Bake the curated model set into the image (A) -- the structural fix.** fp8 first
   (~90 GB baked), bf16 as a precision swap (~117 GB) once the fp8 image validates on a GPU pod.
   The bake stages a curated, precision-selected seed from R2, bin-packs it into <10 GB layers
   (`deploy/bake_layers.py` + the release workflow on `vivijure-bake`), writes the `.vj-baked`
   marker, and ships. `harness/models_mirror` short-circuits volume + R2 when the marker is present.
   Result: **datacenter-agnostic, ~0 s staging on a warm host, no R2 egress, no per-DC ops.**
2. **R2 mirror -- DEMOTED to fallback (kept, not deleted).** A non-baked / legacy image (no
   `.vj-baked`) still mirrors from R2 exactly as before, so correctness degrades gracefully. The
   lazy split + prefetch + symlink reconstruction all stay as-is for that path.
3. **Keep the jitter knob (C) as the fallback's safety net** -- *small code, low risk.* A bounded
   random pre-mirror sleep in `ensure_models` (env-tunable, default 0 = off) so an uncovered
   fallback fan-out still de-staggers its R2 egress. Cheap insurance; the bake is the main act.
4. **Do NOT pre-warm (B)** -- ~$3,600/mo per idle H200 buys one hot worker; the bake makes *every*
   worker on *every* host start hot for the price of GHCR storage + a build runner's minutes.
5. **Do NOT provision network volumes (D)** -- superseded. Its per-DC provisioning/refresh burden
   and R2-preload egress are exactly what the bake removes; volumes also re-pin placement to
   provisioned DCs (the throttle we are escaping). The analysis above is kept for the record.

This keeps the lazy split + prefetch + R2-mirror correctness intact as the fallback, and makes the
baked image the hot common path -- on every host, in every datacenter, with no R2 egress.

> **Build + ship pipeline (see `deploy/`, the release + verify workflows, and
> `docs/release-gate.md`):** `release.yml` builds + pushes the baked image on `vivijure-bake`
> (stage seed -> reconstruct symlinks -> bin-pack -> build -> per-layer gate -> CPU import smoke ->
> push). `runpod-verify.yml` is the **staging gate**: it spins a GPU **pod** on the image, runs the
> structured-`@event` verify, and **only a passing image is promoted onto the production serverless
> endpoint**. Pod = staging/debug; serverless = production.

## Open questions to settle before building

- **DC coverage -- RESOLVED (snapshot 2026-06-15, RunPod `gpuAvailability` API):** the
  network-volume-capable datacenters with H200+ (141 GB+) availability are **12**:
  `AP-JP-1, CA-MTL-3, CA-MTL-4, EU-FR-1, EU-NL-1, EU-RO-1, US-CA-2, US-GA-2, US-MD-1, US-NC-1,
  US-NC-2, US-NE-1` (H200 / H200 NVL / B200 / B300). At 300 GB/volume that is **~$252/mo** total.
  Four more network-volume DCs have only RTX 6000 PRO (96 GB, draft-only via offload) --
  `EU-CZ-1, EUR-IS-1, US-KS-2, US-MO-2` -- excluded unless we deliberately route draft there
  (adding them is ~$336/mo, over the ~$315 target, for slower workers).
  - **Day-one allow-list (phased per Mackaye, the 4 DCs we land on most for H200):**
    `US-NC-1, US-CA-2, US-GA-2, CA-MTL-3` (~$84/mo). The 12 above are the expansion target;
    widen the moment fallbacks or capacity pressure show up.
  - **Keep it current:** `deploy/dc_availability.py` re-runs the `dataCenters { storageSupport
    gpuAvailability }` query (read-only) and prints the candidate set -- availability fluctuates,
    so re-run before provisioning and periodically after.
- **Volume size:** 300 GB (full 223 GB set + headroom) per DC, or split (a 70 GB base-only volume
  for keyframe/preview endpoints, a full one for i2v)? Simplest is one full volume per DC.
- **Preload mechanism:** one-shot cheap pod per DC that rclones R2 -> `/workspace` (handles the
  thousands-of-files HF cache cleanly), vs the S3 API with a chunked uploader (no pod, but fights
  the 10k-file / `s3 sync` limit). Lean pod-rclone for the initial load, S3 API for incremental
  top-ups.
- **Refresh:** on a model-set change (`VJ_MODEL_VERSION` bump), the preloader must re-run per DC
  (volumes don't sync); workers stay read-only.

## Decision needed

- **Conrad:** confirm the network-volume direction + the DC coverage list (which H200+ DCs to
  provision), and whether to size one full volume per DC or split base/i2v.
- **Rollins:** on confirmation, write the preload tooling (`deploy/` script: provision + R2->volume
  load per DC), make the worker read from `/runpod-volume` with the R2-mirror fallback intact, and
  ship the jitter knob. Keep watching the telemetry to confirm `mirror_skipped` (volume-hot) climbs
  toward 100% in covered DCs.

## Forward note: BYO-GPU

The product thesis is BYO-GPU (the user runs their own endpoint, own keys). A network volume is the
natural BYO-GPU answer too: the user provisions one volume in their datacenter, preloads it once,
and every render starts hot at storage prices -- no idle GPU bill, no 223 GB image pull over their
link. The lazy-from-R2 fallback assumes weights live in a store near the GPU; for a true BYO-GPU
user that store may be their own volume rather than our R2 (a separate "per-user weight source"
design, out of scope here). The lazy-split + telemetry + jitter all carry over unchanged.
