# Vivijure Wan 2.2 A14B character-LoRA TRAINING image (vivijure-cf #29, Phase D1).
#
# This image trains the two-expert Wan 2.2 A14B video character LoRA the cloud-i2v cost door needs.
# It is the TRAIN-ONLY sibling of the render image (deploy/Dockerfile): same worker entrypoint, same
# orchestrator/pipeline routing, but it carries the ai-toolkit trainer instead of exercising the
# render stack. A `train_lora` + `model_family=wan` job routes worker -> orchestrator -> pipeline ->
# wan_lora_train, which runs ai-toolkit's run.py as a subprocess.
#
# THE ISOLATION (the #1 coupling, resolved -- cf#29 D1):
#   wan_lora_train._run_aitoolkit launches run.py under the interpreter aitoolkit_python() resolves,
#   which is VIVIJURE_AITOOLKIT_PYTHON (set below) -- NOT the worker's own interpreter. This exists
#   because ai-toolkit's dependency set conflicts IRRECONCILABLY with the worker "vivijure" env's
#   validated render pins:
#       package       vivijure (render, deploy/requirements.txt)   ai-toolkit @AITOOLKIT_REF
#       diffusers      0.39.0 (stable)                              git @c943837 (dev; wan22 needs it)
#       transformers   5.13.1                                       5.5.3
#       torchao        0.17.0                                       0.10.0
#       av             13.1.0                                       16.0.1
#   They cannot share one Python env. So ai-toolkit gets its OWN conda env ("aitoolkit"); the worker
#   stays the "vivijure" env (the CMD). The vivijure env -- and the validated render stack in it --
#   is left byte-for-byte untouched, so worker/harness/orchestrator/pipeline import exactly as prod.
#
# BASE: the pinned, validated runtime base (deploy/runtime.Dockerfile output) -- reused whole so the
# Blackwell-safe cu128 torch 2.7.1 line and the vivijure env are already built + validated (speed to
# GO). It carries ~87GB of baked RENDER weights this image never loads; that is dead weight, but GHCR
# dedups the push (FROM-inheritance) and it is harmless on a temp training pod. A lean weights-free
# training base is a D2 follow-up ONLY if image size actually bites cold-start/disk -- not pre-opted.
#
# BASE MODEL WEIGHTS (Wan2.2-T2V-A14B-Diffusers-bf16, ~50GB+) are NOT baked here in D1: they stage
# into the HF cache on first use. For the D2 prod TRAINING endpoint, bake them per the bake doctrine
# (per-file layers under the 10GB GHCR ceiling; a bf16 expert can exceed 10GB and needs splitting).

# ------------------------------------------------------------------ RUNTIME BASE (pinned by digest)
# Same pin as deploy/Dockerfile's RUNTIME_REF_BF16 (design law #2: tag AND digest). Repin in lockstep
# with deploy/Dockerfile after a runtime rebuild.
ARG RUNTIME_REF_BF16=ghcr.io/skyphusion-labs/vivijure-backend:runtime-1-bf16-t3@sha256:a38ed28546142937313d8743cbdec70e4f5833b336b1478d6d082add925edbbb
FROM ${RUNTIME_REF_BF16}

# ai-toolkit at a PINNED commit (reproducibility; never float HEAD). @6e158dd = ai-toolkit HEAD on
# 2026-07-20 -- the Phase-0 spike's exact rev was not recorded anywhere (repo/memory/search), and D1
# is plumbing not the bind, so HEAD is the deliberate pin (cf#29 D1). Bump + re-validate on a pod.
ARG AITOOLKIT_REF=6e158dd1f1552b73b7aca6d7ddaa46a783538052
# cu128 torch line, identical to the render env's Blackwell-safe set (deploy/runtime.Dockerfile).
ARG VJ_TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG VJ_TORCH_SPEC="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"

# The isolated ai-toolkit env. conda-forge only (Anaconda defaults ToS blocks non-interactive use),
# same discipline as the runtime base's vivijure env.
RUN conda create -y -n aitoolkit -c conda-forge --override-channels python=3.11 pip && conda clean -afy

# cu128 torch FIRST so ai-toolkit's install resolves against it instead of pulling a default-CUDA torch.
RUN conda run --no-capture-output -n aitoolkit python -m pip install --upgrade pip wheel setuptools \
    && conda run --no-capture-output -n aitoolkit python -m pip install --index-url "${VJ_TORCH_INDEX}" ${VJ_TORCH_SPEC}

# ai-toolkit at the pinned commit.
RUN git clone https://github.com/ostris/ai-toolkit /opt/ai-toolkit \
    && git -C /opt/ai-toolkit checkout ${AITOOLKIT_REF}

# ai-toolkit's OWN pinned deps into the aitoolkit env, under a torch constraint so no transitive dep
# can silently swap the validated cu128 torch. deploy/aitoolkit-constraints.txt documents the pins we
# hold; deploy/aitoolkit-overrides.txt is any deliberate post-install correction (each with a why).
COPY deploy/aitoolkit-constraints.txt /tmp/aitoolkit-constraints.txt
COPY deploy/aitoolkit-overrides.txt /tmp/aitoolkit-overrides.txt
RUN conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /opt/ai-toolkit/requirements.txt \
    && conda run --no-capture-output -n aitoolkit python -m pip install \
        -c /tmp/aitoolkit-constraints.txt -r /tmp/aitoolkit-overrides.txt

# Point the worker's ai-toolkit seam at the isolated env. VIVIJURE_AITOOLKIT_DIR = the checkout,
# VIVIJURE_AITOOLKIT_PYTHON = the aitoolkit env's interpreter. The worker (CMD) still runs in vivijure.
ENV VIVIJURE_AITOOLKIT_DIR=/opt/ai-toolkit \
    VIVIJURE_AITOOLKIT_PYTHON=/opt/conda/envs/aitoolkit/bin/python

# Our package. src/ layout -> /opt/vivijure/vivijure_backend, on the inherited PYTHONPATH. Same as
# the render image; the only layer a src-only change re-pushes.
WORKDIR /opt/vivijure
COPY src/vivijure_backend /opt/vivijure/vivijure_backend
COPY deploy/smoke_imports.py /opt/vivijure/smoke_imports.py

# Main process: the RunPod serverless loop in the vivijure env, exactly as the render image. The
# train_lora+wan path shells into the aitoolkit env; nothing else does.
CMD ["conda", "run", "--no-capture-output", "-n", "vivijure", "python", "-u", "-m", "vivijure_backend.worker"]
