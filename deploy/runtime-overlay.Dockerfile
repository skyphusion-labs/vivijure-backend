# Vivijure backend RUNTIME BASE -- DEPS OVERLAY (#537 follow-up).
#
# Use this instead of a full `deploy/runtime.Dockerfile` rebuild when ONLY pip/render deps
# change (safetensors, tokenizers, transformers, …) and the CUDA/torch/apt/seed pins are
# unchanged.
#
# Why: a full runtime rebuild re-runs `COPY --from=seed` and BuildKit re-emits those weight
# layers with NEW digests (observed 2026-07-15 on runtime-1-bf16-t2: ~101 GB of near-identical
# weight blobs, +6 B … +422 B vs t1). RunPod then cold-pulls ~100 GB and dies with
# "unexpected EOF" on image load, while :1.0.0 (t1 layers) still loads from host cache.
# `FROM` a known-good runtime inherits weight layers by blob identity; only the pip layer(s)
# are new. That is the release-upload AND the RunPod-pull win for a deps-only toolchain bump.
#
# Full FROM-cuda rebuilds stay on deploy/runtime.Dockerfile (CUDA/torch/apt/seed changes).
#
# Build (repo root):
#   docker build -f deploy/runtime-overlay.Dockerfile \
#     --build-arg BASE_RUNTIME_REF=ghcr.io/skyphusion-labs/vivijure-backend:runtime-1-bf16-t1@sha256:… \
#     -t ghcr.io/skyphusion-labs/vivijure-backend:runtime-1-bf16-t3 .

ARG BASE_RUNTIME_REF
FROM ${BASE_RUNTIME_REF}
# Re-declare: ARGs before FROM are not in the build stage scope.
ARG BASE_RUNTIME_REF
ARG VJ_BAKE_PRECISION=bf16
ARG VJ_MODEL_VERSION=1
ARG VJ_OVERLAY_NOTE=deps-overlay

SHELL ["conda", "run", "--no-capture-output", "-n", "vivijure", "/bin/bash", "-c"]

# Refresh the pinned render stack. Exact pins live in deploy/requirements.txt (single source).
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Force a CUDA-capable onnxruntime and PROVE it (backend#346). insightface hard-depends on the
# CPU `onnxruntime` wheel, which shares the same package directory as onnxruntime-gpu and, when
# it wins, silently strips CUDAExecutionProvider from the provider list -- so the InstantID face
# embedding runs on CPU while every build stays green. This step removes it, repairs
# onnxruntime-gpu, and HARD-FAILS the build unless a fresh interpreter reports
# CUDAExecutionProvider. Needs no GPU: both failure modes are visible on a CPU-only runner.
COPY deploy/ensure_onnxruntime_gpu.py /tmp/ensure_onnxruntime_gpu.py
RUN python /tmp/ensure_onnxruntime_gpu.py /tmp/requirements.txt

# Re-apply finish-dep compat patches: a pip refresh can restore upstream imports the sed
# patched (basicsr / facexlib / gfpgan vs torchvision + numpy-2). Same surgical sed as
# deploy/runtime.Dockerfile; no-op when already fixed.
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && for pkg in basicsr facexlib gfpgan; do \
         [ -d "$SITE/$pkg" ] || continue; \
         find "$SITE/$pkg" -name "*.py" -exec sed -i \
           -e 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
           -e 's/np\.bool\b/bool/g' \
           -e 's/np\.int\b/int/g' \
           -e 's/np\.float\b/float/g' \
           -e 's/np\.complex\b/complex/g' \
           -e 's/np\.object\b/object/g' \
           -e 's/np\.str\b/str/g' \
         {} +; \
       done

# Re-stamp .vj-baked so the overlay is auditable (weights unchanged; toolchain moved).
RUN test -f /opt/models/.vj-baked \
    && printf 'baked_utc=%s\nprecision=%s\nmodel_version=%s\noverlay=%s\nbase_runtime=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${VJ_BAKE_PRECISION}" "${VJ_MODEL_VERSION}" \
        "${VJ_OVERLAY_NOTE}" "${BASE_RUNTIME_REF}" \
        > /opt/models/.vj-baked \
    && echo "overlay: re-stamped /opt/models/.vj-baked ($(tr '\n' ' ' < /opt/models/.vj-baked))"
