# Vivijure backend serverless worker image (clean-room rewrite) -- BAKED MODEL IMAGE.
#
# This is OUR package (src/vivijure_backend), not the fork. The image now carries the CURATED model
# set baked in (the "one singular baked image"): the worker is datacenter-agnostic (no RunPod network
# volume pinning it to a provisioned DC -- that pinning throttled spill to free GPUs in other regions)
# and pays NO R2 cold-pull tax. harness/models_mirror sees the .vj-baked marker and short-circuits the
# volume-resolve + R2 mirror entirely (the R2 mirror stays only as the fallback for a non-baked image).
#
# The weights are baked as many <10GB layers (GHCR rejects a >=10GB layer). A single COPY of the whole
# set would be one oversized layer; per-file COPY would blow the image layer-count limit. So CI stages
# the precision-selected seed from R2 and bin-packs it (deploy/bake_layers.py) into
# deploy/seed-bins/bin-NN/, each bin < a 9GB target; the fixed COPY list below makes ONE layer per bin,
# and every bin COPYs to VJ_MODELS_ROOT so their union reconstructs the cache + finish-model tree.
#   fp8 baked set  ~90 GB   (Wan i2v fp8 + bf16 base set)        -- shipping now
#   bf16 baked set ~117 GB  (Wan i2v bf16 + bf16 base set)       -- precision swap, builds on vivijure-bake
#
# CUDA target: 12.8 + torch cu128. Blackwell-safe (RTX PRO 6000 / B200, sm_120/sm_100): older cu124
# builds raise "no kernel image is available for execution on the device" on those cards, a failure
# deep into a job that looks like a model bug. NEVER downgrade the CUDA/torch line. Any sm_70+ card
# (incl. the H100/H200 validation boxes) also runs cu128.
#
# Build context is the REPO ROOT (so `COPY src/...` and `COPY deploy/seed-bins/...` resolve):
#   docker build -f deploy/Dockerfile -t ghcr.io/skyphusion-labs/vivijure-backend:<ver> .
# REQUIRES the staged seed: deploy/seed-bins/bin-00..bin-23 must exist (the release workflow creates
# them). A build without a staged seed fails at the bake COPY by design -- this is a baked image.

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

# rclone for the R2 model mirror (ensure_models, the non-baked fallback path). The distro package is
# old; the --links round-trip of the HF cache wants a current rclone.
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

# RIFE (ECCV2022 HDv3 lineage): vendored from the build context at deploy/rife/ -- no runtime
# fetch, immutably pinned. The inference package (RIFE_HDv3.py / IFNet_HDv3.py / warplayer.py /
# loss.py) implements IFNet with three IFBlock(c=90) blocks, the architecture the flownet.pkl
# weights expect (models mirror -> $VJ_MODELS_ROOT/rife/flownet.pkl); the loader passes rank=-1
# to strip the `module.` prefix so the state_dict loads without a key mismatch. No model weights
# are baked here; ~25KB of inference code only. Source: THUDM/CogVideo (Apache 2.0) commit
# 7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2, which carries forward the hzwer/Practical-RIFE HDv3
# lineage (MIT, Megvii Inc.; the HDv3 files were released alongside model weights and were not
# tracked in the hzwer git repo directly). Code license: deploy/rife/ ships Apache-2.0 (CogVideo's
# own modifications) plus the restored Megvii/hzwer MIT lineage -- see deploy/rife/NOTICE. The
# matching flownet.pkl WEIGHTS are a separate artifact at deploy/licenses/imaginairy/rife/LICENSE
# (MIT, Megvii Inc.).
# Supersedes the svjack/CogVideoX-5B-Space curl-fetch (unlicensed, moving ref at VJ_RIFE_REV=main).
COPY deploy/rife /opt/vivijure/rife

# Bake HF repo config files (no weights) into the image and write .no_exist stubs for the 3
# offline-fatal from_pretrained probes (shard-index + adapter_config). Placed BEFORE the weight bake
# so the bake COPY overlays at the same revisions. NOTE: once the curated seed itself carries these
# configs + .no_exist stubs (a seed-curation follow-up), this step becomes redundant and can be
# dropped. Override the offline ENV flags for this build step only; the global ENV remains offline.
COPY deploy/bake_hf_configs.py /tmp/bake_hf_configs.py
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    python /tmp/bake_hf_configs.py

# ------------------------------------------------------------------ BAKED MODEL WEIGHTS (the bake)
# The curated, precision-selected model seed, bin-packed by CI into deploy/seed-bins/bin-NN/ (each
# bin < 9GB, so every COPY layer is under the 10GB GHCR ceiling -- deploy/bake_layers.py asserts it
# pre-build and re-verifies the built image's layers post-build). Each bin mirrors paths relative to
# VJ_MODELS_ROOT, so all bins COPY to /opt/models and their union is the full HF-cache (hf-cache/hub/
# ...) + finish-model tree (antelopev2/, rife/, GFPGANv1.4/) the loaders read offline.
#
# 24 bins = ~216GB capacity at the 9GB target -- generous headroom over the bf16 set (~117GB). Empty
# bins are harmless zero-byte layers (bake_layers.py pre-creates all 24 so this fixed list never
# references a missing path). Placed BEFORE `COPY src/` so a code-only change never busts the weight
# layers. CHANGING THE MODEL SET / PRECISION = re-stage the seed + bump VJ_MODEL_VERSION (below).
COPY deploy/seed-bins/bin-00/ /opt/models/
COPY deploy/seed-bins/bin-01/ /opt/models/
COPY deploy/seed-bins/bin-02/ /opt/models/
COPY deploy/seed-bins/bin-03/ /opt/models/
COPY deploy/seed-bins/bin-04/ /opt/models/
COPY deploy/seed-bins/bin-05/ /opt/models/
COPY deploy/seed-bins/bin-06/ /opt/models/
COPY deploy/seed-bins/bin-07/ /opt/models/
COPY deploy/seed-bins/bin-08/ /opt/models/
COPY deploy/seed-bins/bin-09/ /opt/models/
COPY deploy/seed-bins/bin-10/ /opt/models/
COPY deploy/seed-bins/bin-11/ /opt/models/
COPY deploy/seed-bins/bin-12/ /opt/models/
COPY deploy/seed-bins/bin-13/ /opt/models/
COPY deploy/seed-bins/bin-14/ /opt/models/
COPY deploy/seed-bins/bin-15/ /opt/models/
COPY deploy/seed-bins/bin-16/ /opt/models/
COPY deploy/seed-bins/bin-17/ /opt/models/
COPY deploy/seed-bins/bin-18/ /opt/models/
COPY deploy/seed-bins/bin-19/ /opt/models/
COPY deploy/seed-bins/bin-20/ /opt/models/
COPY deploy/seed-bins/bin-21/ /opt/models/
COPY deploy/seed-bins/bin-22/ /opt/models/
COPY deploy/seed-bins/bin-23/ /opt/models/

# Baked-image marker (harness/models_mirror.BAKED_SENTINEL = .vj-baked). is_baked() checks its
# EXISTENCE to short-circuit volume + R2; the contents are operator-facing metadata. Bumping
# VJ_MODEL_VERSION re-stamps the marker (and, paired with a re-staged seed, signals a new model set).
ARG VJ_BAKE_PRECISION=fp8
ARG VJ_MODEL_VERSION=1
RUN printf 'baked_utc=%s\nprecision=%s\nmodel_version=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${VJ_BAKE_PRECISION}" "${VJ_MODEL_VERSION}" \
        > /opt/models/.vj-baked \
    && echo "bake: wrote $VJ_MODELS_ROOT/.vj-baked ($(tr '\n' ' ' < /opt/models/.vj-baked))"

# Our package. src/ layout -> /opt/vivijure/vivijure_backend, on PYTHONPATH.
WORKDIR /opt/vivijure
COPY src/vivijure_backend /opt/vivijure/vivijure_backend
# Import smoke: run inside the image before pushing to catch missing runtime deps in seconds
# rather than at the end of a 33-min GPU render. See the release workflow's import-smoke step.
COPY deploy/smoke_imports.py /opt/vivijure/smoke_imports.py

# Ensure the cache root exists even on a degenerate empty-seed build (idempotent; the bake populates it).
RUN mkdir -p /opt/models/hf-cache

# The container's main process: the RunPod serverless loop (worker.main ->
# runpod.serverless.start({"handler": worker.handler})). The handler registers the per-job
# GpuPipeline and delegates to the harness (baked-model load, R2 job I/O, plan, finish).
CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "-m", "vivijure_backend.worker"]
