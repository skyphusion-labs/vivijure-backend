# Vivijure backend serverless worker image (#537, Shape Y) -- the CONSUMER of the runtime base.
#
# This image is the full validated runtime + baked weights (the RUNTIME BASE) plus OUR app code. It is
# deliberately tiny to build: `FROM runtime@digest` + COPY src + CMD. Everything heavy -- CUDA/torch,
# the render deps, the HF config bake, and the ~87 GB baked model WEIGHTS -- lives in the pinned
# runtime base (deploy/runtime.Dockerfile, built by runtime-build.yml). So a src-only backend release
# re-pushes ONLY the app layers below; the runtime + weight layers are inherited by blob identity and
# dedup on GHCR ("layer already exists"). That is the release-upload win #537 targets.
#
# The runtime base already sets the ENV (HF_HOME, VJ_MODELS_ROOT, PYTHONPATH=/opt/vivijure, the
# HF offline flags, PATH=/opt/conda/bin), installs conda + the vivijure env, bakes the weights, and
# stamps .vj-baked -- all INHERITED here. This image only adds the package + the entrypoint.
#
# RELEASE CONTRACT (see docs/deploy.md):
#   * src-only change      -> tag backend-vX.Y.Z (this build; fast).
#   * weight-set change     -> rebuild seed -> repin SEED_REF in runtime.Dockerfile -> rebuild runtime
#                              -> repin RUNTIME_REF below -> tag.
#   * toolchain/deps/CUDA   -> prefer overlay FROM prior runtime (bump -t<N>, runtime-build.yml
#                              overlay_from=…) for deps-only; full FROM-cuda rebuild only when
#                              CUDA/torch/apt/seed change -> repin RUNTIME_REF below -> tag.

# ------------------------------------------------------------------ RUNTIME BASE (pinned by digest)
# The full validated runtime + weights, pinned by TAG AND DIGEST (design law #2). Two precisions from
# one Dockerfile via `FROM runtime-${VJ_BAKE_PRECISION}` selection; BuildKit pulls only the selected
# stage. release.yml passes VJ_BAKE_PRECISION (bf16 = self-contained public/prod; fp8 = partial-prod).
# Repin BOTH tag and @sha256 (matching precision) after a runtime rebuild -- that is the ONE line a
# weight or toolchain bump changes here. Digests are placeholders until runtime-build.yml prints them.
ARG RUNTIME_REF_BF16=ghcr.io/skyphusion-labs/vivijure-backend:runtime-2-bf16-t1@sha256:c8528fbc7b0280ca5af3633dd4dbcfacb4dfb62d568744f7c44e6ec9b0452fb7
ARG RUNTIME_REF_FP8=ghcr.io/skyphusion-labs/vivijure-backend-runtime:1-fp8-t1@sha256:0000000000000000000000000000000000000000000000000000000000000000
ARG VJ_BAKE_PRECISION=bf16
FROM ${RUNTIME_REF_BF16} AS runtime-bf16
FROM ${RUNTIME_REF_FP8} AS runtime-fp8
FROM runtime-${VJ_BAKE_PRECISION}

# Our package. src/ layout -> /opt/vivijure/vivijure_backend, on the inherited PYTHONPATH. This is the
# ONLY layer that changes on a src-only release, so it is the only thing re-pushed.
WORKDIR /opt/vivijure
COPY src/vivijure_backend /opt/vivijure/vivijure_backend
# Import smoke: run inside the image before pushing to catch missing runtime deps in seconds rather
# than at the end of a 33-min GPU render. See release.yml's import-smoke step (it needs COPY src, so
# it lives here in the consumer, not the runtime base).
COPY deploy/smoke_imports.py /opt/vivijure/smoke_imports.py

# The container's main process: the RunPod serverless loop (worker.main ->
# runpod.serverless.start({"handler": worker.handler})). The handler registers the per-job
# GpuPipeline and delegates to the harness (baked-model load, R2 job I/O, plan, finish).
CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "-m", "vivijure_backend.worker"]
