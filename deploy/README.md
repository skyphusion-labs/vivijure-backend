# Worker image + deploy

The RunPod serverless image for the vivijure-backend render worker, and the plumbing that builds
and deploys it. This is our clean package (`src/vivijure_backend`), never the fork.

## What the image is

A thin GPU runtime: CUDA 12.8 + torch cu128 (Blackwell-safe), the render stack, and our package.
It carries **no model weights**. A cold worker mirrors the kept models from R2 into the local HF
cache at startup (`harness/models_mirror.ensure_models`, rclone `--links`), then renders offline;
a warm worker reuses the on-disk cache. The only runtime credential is an R2 token.

Entry: `python -m vivijure_backend.worker` -> `worker.main` -> `runpod.serverless.start({"handler":
worker.handler})`. Per job the handler builds a `GpuPipeline` from the request's typed
`RenderConfig`, registers it on the harness seam, and delegates to `harness.handler` (model
mirror, R2 in, plan, GPU stages, off-GPU finish, results out).

## Build (GitHub Actions, on a git tag)

Build + push happen on a `backend-vX.Y.Z` tag (see `../.github/workflows/release.yml`); a plain commit is a no-op.

```bash
git push origin main
git tag backend-v0.1.0 && git push origin backend-v0.1.0
#   -> ghcr.io/skyphusion-labs/vivijure-backend:0.1.0 (+ :latest)
```

Build context is the repo root; the Dockerfile is `deploy/Dockerfile`. Local build:

```bash
docker build -f deploy/Dockerfile -t ghcr.io/skyphusion-labs/vivijure-backend:dev .
```

## Deploy (pin the RunPod template; separate + deliberate)

Building does not touch the live endpoint. Pin it to a built image when ready:

```bash
RUNPOD_API_KEY=... RUNPOD_TEMPLATE_ID=... \
  python3 scripts/pin-runpod-template.py ghcr.io/skyphusion-labs/vivijure-backend:0.1.0
```

New endpoint workers pull the pinned image on their next cold start.

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

## Dependency pins

`deploy/requirements.txt` is the single source of the runtime version set (torch installs
separately from the cu128 index in the Dockerfile). The pins are the exact set validated on real
hardware (H200 fp8 + fused 4-step i2v, H100 bf16 + CPU-offload, A6000 SDXL keyframe); the header
of that file records the last validation date. Change a pin only with a fresh GPU-validation
pass, then rebuild.

## See also

The operator's view of the running system -- the cold-start model mirror, the R2 object-key map,
the structured progress channel, and the failure modes -- is in
[../docs/operations.md](../docs/operations.md).
