# Vivijure Wan 2.2 A14B character-LoRA TRAINING image (vivijure-cf #29, Phase D1).
#
# TRAIN-ONLY sibling of the render image. It trains the two-expert Wan 2.2 A14B video character LoRA
# the cloud-i2v cost door needs: a `train_lora` + `model_family=wan` job routes worker -> orchestrator
# -> pipeline -> wan_lora_train, which runs ai-toolkit's run.py as a subprocess.
#
# BASE DECISION (cf#29 D1): a LEAN CUDA+torch base, NOT the 87GB render-runtime base. A training
# endpoint never loads the render WEIGHTS (SDXL/Wan-i2v), so carrying them was ~87GB of dead weight
# that made every cold pod a ~20min pull -- the root cause of the D1 bring-up pain. This image
# reproduces the worker's VALIDATED env (the runtime.Dockerfile recipe: cu128 torch 2.7.1 + the
# pinned deploy/requirements.txt + the basicsr/facexlib/gfpgan compat patches) MINUS the weight bake,
# so worker/harness/orchestrator/pipeline import exactly as prod, and the image is ~15-20GB -> fast,
# reliable pods. It is also the correct D2 train-endpoint artifact.
#
# THE ISOLATION (the #1 coupling, resolved): ai-toolkit's deps conflict irreconcilably with the
# worker "vivijure" env's render pins (diffusers 0.39 stable vs a diffusers git pin; transformers
# 5.13 vs 5.5; torchao 0.17 vs 0.10; av 13 vs 16). So ai-toolkit gets its OWN conda env ("aitoolkit")
# and wan_lora_train launches run.py under VIVIJURE_AITOOLKIT_PYTHON (set below). Both envs share the
# same Blackwell-safe cu128 torch 2.7.1 line so neither fights the toolchain.
#
# BASE MODEL WEIGHTS (Wan2.2-T2V-A14B-Diffusers-bf16, ~53GB) are BAKED (D2, cf#29): the build workflow
# snapshot_downloads the repo into the HF-cache layout and bin-packs it into per-layer <9.6GB bins
# (bake_layers.py; the 10GB GHCR ceiling forces the split -- a bf16 expert shard is ~9.3GB), then this
# Dockerfile COPYs the bins into HF_HOME and verifies a sha256 manifest. A cold worker never re-pulls.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# ai-toolkit pinned commit (reproducibility; never float HEAD). @6e158dd = HEAD 2026-07-20; the
# Phase-0 spike rev was unrecorded, and D1 is plumbing not the bind, so HEAD is the deliberate pin.
ARG AITOOLKIT_REF=6e158dd1f1552b73b7aca6d7ddaa46a783538052
ARG VJ_PYTHON_VERSION=3.11
# cu128 torch line. NOTE (cf#29 D1): the render env validated on torch 2.7.1+cu128, but PyTorch has
# since DELISTED 2.7.1 from the cu128 index (only 2.9.0+ remain), so a from-scratch build cannot
# reproduce it. We use the closest available, torch 2.9.0+cu128 (Blackwell-safe cu128 line preserved),
# and GATE it with a build-time import smoke below -- the train path exercises the vivijure-env torch
# only for imports (real training runs in the aitoolkit env), so a minor-version bump is low-risk here,
# but the smoke proves diffusers/transformers still import under it before the image is trusted.
ARG VJ_TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG VJ_TORCH_SPEC="torch==2.9.0 torchvision torchaudio"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/conda/bin:$PATH \
    PYTHONPATH=/opt/vivijure \
    HF_HOME=/opt/models/hf-cache \
    VJ_MODELS_ROOT=/opt/models \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1
# HF offline is ON (D2): the Wan base is BAKED below into HF_HOME, so a cold worker never phones home.
# The endpoint template env can override (HF_HUB_OFFLINE=0) as an escape hatch if ai-toolkit ever needs
# a file beyond the baked base -- but the correct fix then is to ADD it to the bake, not to ship online.

# System deps: the SAME set the runtime base installs (ffmpeg + the libs opencv/insightface need at
# import), plus openssh-server for reliable debug-pod SSH (baked, so no fragile runtime install).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates unzip ffmpeg libgl1 libglib2.0-0 build-essential openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + conda-forge (NOT Anaconda defaults, whose ToS blocks non-interactive use).
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh \
    && conda create -y -n vivijure -c conda-forge --override-channels "python=${VJ_PYTHON_VERSION}" pip \
    && conda clean -afy

# --- the WORKER "vivijure" env: reproduce the validated render/runtime deps (no weights) ---
SHELL ["conda", "run", "--no-capture-output", "-n", "vivijure", "/bin/bash", "-c"]
RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install --index-url "${VJ_TORCH_INDEX}" ${VJ_TORCH_SPEC}
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt
# The SAME basicsr/facexlib/gfpgan compat patches the runtime base applies (cu128 / numpy-2.x).
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && for pkg in basicsr facexlib gfpgan; do \
         [ -d "$SITE/$pkg" ] || continue; \
         find "$SITE/$pkg" -name "*.py" -exec sed -i \
           -e 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
           -e 's/np\.bool\b/bool/g' -e 's/np\.int\b/int/g' -e 's/np\.float\b/float/g' \
           -e 's/np\.complex\b/complex/g' -e 's/np\.object\b/object/g' -e 's/np\.str\b/str/g' \
         {} +; \
       done

# --- the ISOLATED "aitoolkit" env: ai-toolkit's own (conflicting) deps ---
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n aitoolkit -c conda-forge --override-channels "python=${VJ_PYTHON_VERSION}" pip && conda clean -afy
# Explicit trio (NOT ${VJ_TORCH_SPEC}): ai-toolkit imports torchaudio at import time, and unpinned it
# resolves to 2.11.0 which mismatches torch 2.9.0. Pin all three to the 2.9.0 line.
RUN conda run --no-capture-output -n aitoolkit python -m pip install --upgrade pip wheel setuptools \
    && conda run --no-capture-output -n aitoolkit python -m pip install --index-url "${VJ_TORCH_INDEX}" torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
RUN git clone https://github.com/ostris/ai-toolkit /opt/ai-toolkit \
    && git -C /opt/ai-toolkit checkout ${AITOOLKIT_REF}
COPY deploy/aitoolkit-constraints.txt /tmp/aitoolkit-constraints.txt
COPY deploy/aitoolkit-overrides.txt /tmp/aitoolkit-overrides.txt
RUN conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /opt/ai-toolkit/requirements.txt \
    && conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /tmp/aitoolkit-overrides.txt

# Point the worker's ai-toolkit seam at the isolated env. The worker (CMD) stays the vivijure env.
ENV VIVIJURE_AITOOLKIT_DIR=/opt/ai-toolkit \
    VIVIJURE_AITOOLKIT_PYTHON=/opt/conda/envs/aitoolkit/bin/python

# --- BAKED Wan 2.2 A14B base (D2, cf#29) ------------------------------------------------------------
# train-image-build.yml snapshot_downloads ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16 (~53GB) into the
# HF-cache layout and bin-packs it into per-layer <9.6GB bins (deploy/train-bins/, gitignored + CI-only).
# COPY each bin as ONE layer into HF_HOME so the union reconstructs the cache tree; then verify the
# union-keyed sha256 manifest (a byte mismatch fails the build LOUD). BEFORE COPY src so a code change
# never busts the ~53GB weight layers. Empty bins (bin_pack pre-creates all 12) are free zero-byte layers.
COPY deploy/train-bins/bin-00/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-01/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-02/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-03/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-04/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-05/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-06/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-07/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-08/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-09/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-10/ /opt/models/hf-cache/
COPY deploy/train-bins/bin-11/ /opt/models/hf-cache/
COPY deploy/train-bins/weights-manifest.sha256 /opt/models/hf-cache/weights-manifest.sha256
RUN cd /opt/models/hf-cache \
    && sha256sum -c weights-manifest.sha256 --quiet \
    && echo "wan base bake: sha256 verified ($(wc -l < weights-manifest.sha256) files)." \
    && rm -f weights-manifest.sha256

# Our package. src/ layout -> /opt/vivijure/vivijure_backend, on the inherited PYTHONPATH.
WORKDIR /opt/vivijure
COPY src/vivijure_backend /opt/vivijure/vivijure_backend
COPY deploy/smoke_imports.py /opt/vivijure/smoke_imports.py

# Build-time import smoke (the torch-2.9 divergence gate, cf#29 D1): FAIL the build here, in
# seconds, if the vivijure env cannot import the worker + wan seam, or the aitoolkit env cannot
# import torch/diffusers, rather than discovering it on a GPU pod.
RUN conda run --no-capture-output -n vivijure python -c "import vivijure_backend.worker, vivijure_backend.wan_lora_train as W; assert W.aitoolkit_python().endswith('/aitoolkit/bin/python'), W.aitoolkit_python(); print('vivijure import smoke OK; seam ->', W.aitoolkit_python())"
RUN conda run --no-capture-output -n aitoolkit python -c "import torch, torchaudio, diffusers, transformers, safetensors; print('aitoolkit import smoke OK: torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'diffusers', diffusers.__version__)"

# Main process: the RunPod serverless loop in the vivijure env (prod). The train_lora+wan path shells
# into the aitoolkit env; nothing else does.
CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "-m", "vivijure_backend.worker"]
