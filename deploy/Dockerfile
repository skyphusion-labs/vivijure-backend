# Vivijure backend serverless worker image (clean-room rewrite).
#
# This is OUR package (src/vivijure_backend), not the fork. The image is a thin GPU runtime:
# it carries no model weights (the full set is hundreds of GB; a layer that size is rejected by
# GHCR and ingests slowly). Instead a cold worker mirrors the kept models from R2 into the local
# HF cache at startup via harness/models_mirror.ensure_models (rclone --links), then renders
# offline; a warm worker reuses the on-disk cache. The only runtime credential is the R2 token,
# delivered as endpoint env vars (never baked here).
#
# CUDA target: 12.8 + torch cu128. This is Blackwell-safe (RTX PRO 6000 / B200, sm_120/sm_100):
# older cu124 builds raise "no kernel image is available for execution on the device" on those
# cards, a failure deep into a job that looks like a model bug. NEVER downgrade the CUDA/torch
# line. Any sm_70+ card (incl. the H100/H200 validation boxes) also runs cu128.
#
# Build context is the REPO ROOT (so `COPY src/...` resolves):
#   docker build -f deploy/Dockerfile -t ghcr.io/skyphusion-labs/vivijure-backend:<ver> .

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Python + the torch line are pinned here so a rebuild is reproducible. The torch spec is a
# build ARG (cu128) so the exact patch can be bumped without editing the file body.
ARG VJ_PYTHON_VERSION=3.11
ARG VJ_TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG VJ_TORCH_SPEC="torch==2.7.* torchvision==0.22.* torchaudio==2.7.*"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/opt/vivijure \
    HF_HOME=/opt/models/hf-cache \
    VJ_MODELS_ROOT=/opt/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps: ffmpeg for the off-GPU finish + Wan video export, git/build basics, the libs
# OpenCV / insightface (InstantID) need at import.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates unzip ffmpeg libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# rclone for the R2 model mirror (ensure_models). The distro package is old; the --links
# round-trip of the HF cache wants a current rclone.
RUN curl -fsSL https://rclone.org/install.sh | bash

# Python via Miniconda + conda-forge (NOT Anaconda's `defaults`, whose ToS blocks non-interactive
# use and carries commercial terms). Only python + pip come from conda; the rest is pip.
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh \
    && conda create -y -n vivijure -c conda-forge --override-channels "python=${VJ_PYTHON_VERSION}" pip \
    && conda clean -afy

# Run everything in the vivijure env from here on.
SHELL ["conda", "run", "--no-capture-output", "-n", "vivijure", "/bin/bash", "-c"]

# PyTorch (cu128) first so it layer-caches independently of the app deps.
RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install --index-url "${VJ_TORCH_INDEX}" ${VJ_TORCH_SPEC}

# Runtime stack (render engines + serverless SDK + R2). Exact pins are finalized from the H100
# i2v validation and live in deploy/requirements.txt; keep that file the single source of the
# version set so this layer is reproducible.
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# Compat patches for finishing-stage deps that predate the cu128 / numpy-2.x stack.
# These are surgical sed passes over the three packages we know are affected; sed -i is a no-op
# when a future pip release has already fixed the import, so the layer stays reproducible.
#   (1) basicsr 0.4.2 imports torchvision.transforms.functional_tensor (removed in 0.17).
#       facexlib and gfpgan occasionally share the same import; patch all three.
#   (2) basicsr + facexlib use np.bool / np.int / np.float / np.complex aliases (removed in 1.24).
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

# RIFE (ECCV2022 HDv3 lineage): not on PyPI, and the upstream GitHub repos do NOT ship an
# importable `rife/` package (Practical-RIFE's tree is demo/ + model/, no rife/RIFE_HDv3.py). We
# vendor a complete, self-consistent `rife/` package so `from rife.RIFE_HDv3 import Model`
# (models.py) resolves on PYTHONPATH. Source: the svjack/CogVideoX-5B-Space HF space, whose rife/
# package is the ECCV2022-RIFE HDv3 code with clean intra-package relative imports
# (RIFE_HDv3 -> .IFNet_HDv3 / .warplayer / .loss). This is the SAME RIFE_HDv3/IFNet(c=90) lineage
# as the seeded varnish flownet.pkl (models mirror -> $VJ_MODELS_ROOT/rife/flownet.pkl), so the
# state_dict loads without a key mismatch (the loader passes rank=-1 to strip the `module.` prefix).
# No model weights are baked here; only the ~25KB of inference code. Pinned to a HF revision so the
# layer is reproducible.
ARG VJ_RIFE_REPO=https://huggingface.co/spaces/svjack/CogVideoX-5B-Space
ARG VJ_RIFE_REV=main
RUN mkdir -p /opt/vivijure/rife \
    && for f in __init__.py RIFE_HDv3.py IFNet_HDv3.py warplayer.py loss.py; do \
         curl -fsSL "${VJ_RIFE_REPO}/resolve/${VJ_RIFE_REV}/rife/${f}" -o "/opt/vivijure/rife/${f}"; \
       done \
    && touch /opt/vivijure/rife/__init__.py

# Bake HF repo config files (no weights) into the image and write .no_exist stubs for the 3
# offline-fatal from_pretrained probes (shard-index + adapter_config). Weights arrive at render
# time from R2; baking configs eliminates the probes that forced HF_HUB_OFFLINE to be overridden
# at the endpoint. Placed BEFORE COPY src/ so code changes do not bust this download layer.
# Override the offline ENV flags for this build step only; the global ENV remains offline.
COPY deploy/bake_hf_configs.py /tmp/bake_hf_configs.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    python /tmp/bake_hf_configs.py

# Our package. src/ layout -> /opt/vivijure/vivijure_backend, on PYTHONPATH.
WORKDIR /opt/vivijure
COPY src/vivijure_backend /opt/vivijure/vivijure_backend
# Import smoke: run inside the image before pushing to catch missing runtime deps in seconds
# rather than at the end of a 33-min GPU render. See Jenkinsfile stage 'import smoke'.
COPY deploy/smoke_imports.py /opt/vivijure/smoke_imports.py

# Model + sentinel root the cold-start mirror populates (HF_HOME lives under it).
RUN mkdir -p /opt/models/hf-cache

# The container's main process: the RunPod serverless loop (worker.main ->
# runpod.serverless.start({"handler": worker.handler})). The handler registers the per-job
# GpuPipeline and delegates to the harness (model mirror, R2, plan, finish).
CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "-m", "vivijure_backend.worker"]
