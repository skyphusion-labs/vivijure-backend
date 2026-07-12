# Worker image + deploy

The RunPod serverless image for the vivijure-backend render worker, and the plumbing that builds
and deploys it. This is our clean package (`src/vivijure_backend`), never the fork.

## What the image is

A GPU runtime (CUDA 12.8 + torch cu128, Blackwell-safe) + the render stack + our package, with the
**curated model set BAKED IN** (the "one singular baked image"). A baked worker carries its weights,
so it is **datacenter-agnostic** (no RunPod network volume pinning it to a provisioned DC) and pays
**no R2 cold-pull tax**. `harness/models_mirror` sees the `.vj-baked` marker at `VJ_MODELS_ROOT` and
short-circuits the volume-resolve + R2 mirror entirely. The R2 mirror stays only as the **fallback**
for a non-baked / legacy image (no marker). See [../docs/cold-start-design.md](../docs/cold-start-design.md)
for why the bake replaced the network-volume plan.

Two baked variants: the **fp8 PARTIAL** prod image (~90 GB; final tier still R2-pulls bf16, so
prod-only) and the **self-contained bf16** public image (~87 GB). The Wan i2v experts ship **FP32**,
so the bf16 seed is a free CPU **fp32->bf16 re-cast** (zero quality cost; the loader casts to bf16 at
load anyway) -- point the bake at the RE-CAST seed, not raw fp32 (~140 GB) and not the fp8 seed. The
weights are baked as many <10 GB layers (GHCR rejects a >=10 GB layer): CI stages the curated seed
from R2, bin-packs it (`deploy/bake_layers.py`), and the Dockerfile COPYs one layer per bin.

Entry: `python -m vivijure_backend.worker` -> `worker.main` -> `runpod.serverless.start({"handler":
worker.handler})`. Per job the handler builds a `GpuPipeline` from the request's typed
`RenderConfig`, registers it on the harness seam, and delegates to `harness.handler` (baked-model
load, R2 job I/O, plan, GPU stages, off-GPU finish, results out).

## Build (GitHub Actions, on a git tag)

Build + push happen on a `backend-vX.Y.Z` tag (see `../.github/workflows/release.yml`); a plain
commit is a no-op. The build runs on the **`vivijure-bake` larger runner** (32-core / 128 GB /
**1200 GB SSD**), NOT the 300 GB `heavy-runner` the thin image used: peak build disk is ~280 GB for
the bf16 re-cast (~87 GB) / fp8 (~90 GB) image (staged seed + buildkit snapshot + loaded image + base
stack), and the raw-fp32 contingency (~440 GB) needs the 1200 GB headroom.

```bash
git push origin main
git tag backend-v0.4.9 && git push origin backend-v0.4.9
#   -> ghcr.io/skyphusion-labs/vivijure-backend:0.4.9 (+ :latest)
```

Precision: a tag build reads `vars.VJ_BAKE_PRECISION` (default `fp8`); set it to `bf16` to cut a
bf16 release, or use the workflow_dispatch `precision` input. The pipeline: stage seed -> reconstruct
symlinks -> bin-pack (<10 GB layers) -> build -> per-layer GHCR gate -> CPU import smoke -> push.

**Prerequisites (deploy ordering -- the build fails without them):**
- The curated, precision-selected seed exists in R2 at `r2:vivijure/bake-seed-<precision>/`, arranged
  at the exact relative paths the loaders read under `VJ_MODELS_ROOT` (HF-cache `hf-cache/hub/...`
  for `from_pretrained` repos; flat `antelopev2/ rife/ GFPGANv1.4/`). The fp8 Wan i2v seed already
  exists at `r2:vivijure/models-fp8/`; the full bake-seed prefix (base set + fp8 Wan + the trimmed
  Lightning LoRA file) still needs curating. Every baked file MUST be < 10 GB (RealVisXL ships a
  9.99 GB blob -- 0.01 GB under the ceiling; reshard it if it ever grows. The Lightning repo's
  28.58 GB blob is a different variant and MUST NOT be baked: bake only the spec's
  `Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors`).
- Encrypted Actions secrets: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT` (a read-only,
  per-CI R2 token scoped to the `vivijure` bucket).

Local build needs the seed staged + binned first (`deploy/bake_layers.py bin --src <seed> --out
deploy/seed-bins`); a build with no staged bins fails at the bake COPY by design.

## Ship to production (the pod-staging gate; separate + deliberate)

Building + pushing does NOT touch the live endpoint. An image reaches the **production serverless
endpoint** ONLY by passing the automated **pod-staging verify**
(`../.github/workflows/runpod-verify.yml`): spin a GPU pod on the image, run the structured-`@event`
verify, and promote the image onto the serverless endpoint only on PASS. **Pod = staging/debug;
serverless = production.** Full doctrine: [../docs/release-gate.md](../docs/release-gate.md).

## Env vars

Baked into the image (non-secret), already set in the Dockerfile:

| Var | Value | Why |
|---|---|---|
| `HF_HOME` | `/opt/models/hf-cache` | local HF cache the mirror fills, read offline |
| `VJ_MODELS_ROOT` | `/opt/models` | mirror root + completion sentinel |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE` | `1` | read weights from the local mirror, never the Hub |
| `PYTHONPATH` | `/opt/vivijure` | `import vivijure_backend` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | fragmentation headroom |

Set on the RunPod endpoint at runtime (the only credential; never baked):

```
R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET=vivijure
```

The R2 token does double duty: the cold-start model mirror (`r2:<bucket>/models`) and job I/O
(bundle in, render + state out). The worker holds no Cloudflare Access or skyphusion secret.

**Sovereignty note (R2-ours vs self-contained):** with `HF_HUB_OFFLINE=1` baked in, `from_pretrained`
never reaches the HF Hub, so the R2 token is the ONLY runtime weight source for any NON-baked path.
On the fp8-PARTIAL image, the FINAL tier still lazy-pulls bf16 from OUR R2 (`ensure_i2v_models`), so
that image is **prod-only** -- a BYO-RunPod renter without our R2 keys cannot load the final tier. The
**full bf16 bake removes that R2 dependency** (final tier loads from baked weights), making the public
datacenter image self-contained. `HF_TOKEN`, if present on the endpoint, is **build-time only**
(`bake_hf_configs.py` fetches CONFIGS with the offline flags flipped); at runtime it is inert. Full
discussion: [../docs/release-gate.md](../docs/release-gate.md#sovereignty-r2-ours-prod-only-vs-self-contained-public).

## Dependency pins

`deploy/requirements.txt` is the single source of the runtime version set (torch installs
separately from the cu128 index in the Dockerfile). The pins are the exact set validated on real
hardware (H200 fp8 + fused 4-step i2v, H100 bf16 + CPU-offload, A6000 SDXL keyframe); the header
of that file records the last validation date. Change a pin only with a fresh GPU-validation
pass, then rebuild.

## RunPod account gotchas

- **The serverless worker quota is ACCOUNT-WIDE (currently 30).** Every endpoint's `workersMax`
  sums against the one account cap across ALL endpoints (backend + every satellite). At cap,
  creating ANY new endpoint fails; free a slot first (temporarily lower an idle endpoint's
  `workersMax` in a zero-traffic window), create, then RESTORE the borrowed slot.
- **RunPod v2 create-endpoint masks a quota overage as an opaque `500 Internal Server Error`.**
  The v2 REST/MCP path returns a bare 500 with no cause; the **v1 REST error body names the real
  reason** (e.g. "Max workers across all endpoints must not exceed your workers quota (30)"). If an
  endpoint create 500s while `create-template` on the same image SUCCEEDS (so the API is up),
  suspect the account quota and read the v1 error body, not the v2 500. (Seen S32, 2026-07-11:
  the account sat at 30/30, so the musetalk verify-rig endpoint create 500d until a slot was freed.)

- **Direct `update-endpoint` mutations 500 on the v2 REST/MCP path (image AND scaling), and this is
  NOT quota-masked.** Seen S32 (2026-07-11): repointing the prod musetalk endpoint image, and
  separately raising its `workersMax` while inside quota (29 -> 30), BOTH returned bare 500s. Two
  working paths instead: (a) **repoint the IMAGE by updating the endpoint's bound TEMPLATE**
  (`update-template imageName`, digest-pin OK as `:<ver>@sha256:...`) -- it propagates to the running
  endpoint, and that is how the prod pin actually lands; (b) **worker-count / scaling changes have no
  template escape hatch** and must go through the **v1 REST API** (`RUNPOD_REST_VERSION=v1`), same as
  the create path. Diagnostic tell: `update-template` on the same image SUCCEEDS while `update-endpoint`
  500s -> it is the v2 endpoint-mutation fault, not your payload and not quota.

## See also

The operator's view of the running system -- the cold-start model mirror, the R2 object-key map,
the structured progress channel, and the failure modes -- is in
[../docs/operations.md](../docs/operations.md).
