# Vivijure backend RUNTIME BASE image (#537, Shape Y) -- the full validated runtime + baked weights.
#
# This is the middle image of the seed -> runtime -> backend chain. It carries EVERYTHING that is
# stable across an app-only release: the CUDA/torch/conda stack, the render deps, the finish-dep
# patches, RIFE, the HF config bake, the baked model WEIGHTS, and the .vj-baked marker. The consuming
# image (deploy/Dockerfile) is just `FROM this@digest` + COPY src + CMD, so a src-only release
# re-pushes ONLY the app layers and every runtime + weight layer dedups on GHCR ("layer already
# exists"). That is the release-upload win #537 targets.
#
# WEIGHTS come from the immutable SEED image (ghcr.io/skyphusion-labs/vivijure-backend-seed, built by
# seed-build.yml from the R2 seed), COPY --from'd in as 24 <10GB bin layers -- so a toolchain/CUDA
# rebuild of THIS image pulls the seed (NO R2 re-stage; the seed is staged from R2 exactly once per
# weight version) and the deterministic COPY --from layers dedup across rebuilds. hf-configs runs
# BEFORE the weight COPY (the #206 overlay order) exactly as in the pre-split monolith.
#
# CUDA target: 12.8 + torch cu128. Blackwell-safe (RTX PRO 6000 / B200, sm_120/sm_100): older cu124
# builds raise "no kernel image is available for execution on the device" on those cards, a failure
# deep into a job that looks like a model bug. NEVER downgrade the CUDA/torch line. Any sm_70+ card
# (incl. the H100/H200 validation boxes) also runs cu128.
#
# Build context is the REPO ROOT (so the small committed COPYs -- requirements.txt, rife/, the bake
# scripts -- resolve):
#   docker build -f deploy/runtime.Dockerfile --build-arg VJ_BAKE_PRECISION=bf16 -t <runtime>:<ver> .
# Needs the pinned SEED image to be pullable (seed-build.yml published it); a bad pin fails at the
# seed FROM by design. Built by .github/workflows/runtime-build.yml (dispatch-gated).

# ------------------------------------------------------------------ SEED (weights source of truth)
# The curated model seed lives in the immutable per-precision SEED image; pinned by TAG AND DIGEST
# (design law #2). Two precisions from one Dockerfile via `FROM seed-${VJ_BAKE_PRECISION}` selection;
# BuildKit pulls only the selected stage. Repin BOTH tag and @sha256 (matching precision) after a
# seed rebuild. bf16 = self-contained public/prod; fp8 = partial-prod.
ARG SEED_REF_BF16=ghcr.io/skyphusion-labs/vivijure-backend-seed:1-bf16@sha256:6c96ac5163d679786946a7f034ab446314dbde0c9f1a7c4b0f65403425aae971
ARG SEED_REF_FP8=ghcr.io/skyphusion-labs/vivijure-backend-seed:1-fp8@sha256:0000000000000000000000000000000000000000000000000000000000000000
ARG VJ_BAKE_PRECISION=bf16
FROM ${SEED_REF_BF16} AS seed-bf16
FROM ${SEED_REF_FP8} AS seed-fp8
FROM seed-${VJ_BAKE_PRECISION} AS seed

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Python + the torch line are pinned here so a rebuild is reproducible. The torch spec is a
# build ARG (cu128) so the exact patch can be bumped without editing the file body.
ARG VJ_PYTHON_VERSION=3.11
ARG VJ_TORCH_INDEX=https://download.pytorch.org/whl/cu128
# EXACT-PINNED (was the wildcard `torch==2.7.* torchvision==0.22.* torchaudio==2.7.*`): the
# wildcard let a fresh bake pull a newer 2.7.x off the cu128 index, so two bakes 2h apart
# baked different torch wheels -- and a torch-import ABI/CUDA break crashes the worker before
# it can emit anything (cost us the :0.4.1 verify gate: empty channel + restart loop). This is
# the validated set (torch 2.7.1+cu128, the version requirements.txt records as pod-validated);
# the tested-with-pin doctrine from the local doors' ML deps, now applied to the last unpinned
# ML dep in the constellation. Bump deliberately + re-validate on the pod, never by drift.
ARG VJ_TORCH_SPEC="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"

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

# Force a CUDA-capable onnxruntime and PROVE it (backend#346). insightface hard-depends on the
# CPU `onnxruntime` wheel, which shares the same package directory as onnxruntime-gpu and, when
# it wins, silently strips CUDAExecutionProvider from the provider list -- so the InstantID face
# embedding runs on CPU while every build stays green. This step removes it, repairs
# onnxruntime-gpu, and HARD-FAILS the build unless a fresh interpreter reports
# CUDAExecutionProvider. Needs no GPU: both failure modes are visible on a CPU-only runner.
COPY deploy/ensure_onnxruntime_gpu.py /tmp/ensure_onnxruntime_gpu.py
RUN python /tmp/ensure_onnxruntime_gpu.py /tmp/requirements.txt

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
# The curated, precision-selected model seed, COPY --from'd out of the immutable SEED image as 24
# <10GB bin layers (the seed build bin-packed them; deploy/bake_layers.py asserted < the 10GB GHCR
# ceiling pre-push). Each bin mirrors paths relative to VJ_MODELS_ROOT, so all bins COPY to /opt/models
# and their union is the full HF-cache (hf-cache/hub/...) + finish-model tree (antelopev2/, rife/,
# GFPGANv1.4/) the loaders read offline.
#
# COPY --from=seed was documented as digest-stable across toolchain rebuilds. In practice BuildKit
# re-emits those layers under NEW digests for near-identical bytes (runtime-1-bf16-t2 vs t1,
# 2026-07-15: ~101 GB RunPod cold-pull miss + "unexpected EOF"). For deps-only bumps use
# deploy/runtime-overlay.Dockerfile (FROM a known-good runtime) via runtime-build.yml
# overlay_from=… so weight layers inherit by blob identity. Full rebuilds here remain required for
# CUDA/torch/apt/seed changes; treat weight-layer digests as NOT guaranteed across rebuilds.
COPY --from=seed /seed-bins/bin-00/ /opt/models/
COPY --from=seed /seed-bins/bin-01/ /opt/models/
COPY --from=seed /seed-bins/bin-02/ /opt/models/
COPY --from=seed /seed-bins/bin-03/ /opt/models/
COPY --from=seed /seed-bins/bin-04/ /opt/models/
COPY --from=seed /seed-bins/bin-05/ /opt/models/
COPY --from=seed /seed-bins/bin-06/ /opt/models/
COPY --from=seed /seed-bins/bin-07/ /opt/models/
COPY --from=seed /seed-bins/bin-08/ /opt/models/
COPY --from=seed /seed-bins/bin-09/ /opt/models/
COPY --from=seed /seed-bins/bin-10/ /opt/models/
COPY --from=seed /seed-bins/bin-11/ /opt/models/
COPY --from=seed /seed-bins/bin-12/ /opt/models/
COPY --from=seed /seed-bins/bin-13/ /opt/models/
COPY --from=seed /seed-bins/bin-14/ /opt/models/
COPY --from=seed /seed-bins/bin-15/ /opt/models/
COPY --from=seed /seed-bins/bin-16/ /opt/models/
COPY --from=seed /seed-bins/bin-17/ /opt/models/
COPY --from=seed /seed-bins/bin-18/ /opt/models/
COPY --from=seed /seed-bins/bin-19/ /opt/models/
COPY --from=seed /seed-bins/bin-20/ /opt/models/
COPY --from=seed /seed-bins/bin-21/ /opt/models/
COPY --from=seed /seed-bins/bin-22/ /opt/models/
COPY --from=seed /seed-bins/bin-23/ /opt/models/

# The union-keyed sha256 manifest travels with the seed. Verify EVERY baked weight file is
# byte-identical to what the seed build recorded, BEFORE the .vj-baked sentinel is trusted (design
# law #2). `sha256sum -c` exits non-zero on any mismatch/missing file -> the build fails LOUD here.
COPY --from=seed /seed-bins/weights-manifest.sha256 /opt/models/weights-manifest.sha256
RUN cd /opt/models && sha256sum -c weights-manifest.sha256 --quiet \
    && echo "weights: sha256 manifest verified ($(wc -l < weights-manifest.sha256) file(s) match the seed)."

# Baked-image marker (harness/models_mirror.BAKED_SENTINEL = .vj-baked). is_baked() checks its
# EXISTENCE to short-circuit volume + R2; the contents are operator-facing metadata. Bumping
# VJ_MODEL_VERSION re-stamps the marker (and, paired with a re-staged seed, signals a new model set).
# Precision defaults to bf16 (the self-contained public/prod track); the release workflow passes it
# explicitly. fp8 stays selectable for the partial-prod image.
ARG VJ_BAKE_PRECISION=bf16
ARG VJ_MODEL_VERSION=1
# SELF-DEFENDING GATE (the #4 empty-bake fix): never stamp .vj-baked over an empty/stub weight tree.
# assert-weights HARD-FAILS the build unless VJ_MODELS_ROOT really holds the curated set (byte floor
# for the precision + at least one real shard). is_baked() trusts this sentinel to skip the R2 pull,
# so the sentinel must NOT be able to lie: the AND below writes it only after the assert passes.
COPY deploy/bake_layers.py /tmp/bake_layers.py
COPY deploy/facexlib_pins.py /tmp/facexlib_pins.py
COPY deploy/bake-manifest.json /tmp/bake-manifest.json
# Bake the manifest into the image as the auditable pin record. is_baked() trusts .vj-baked; the
# manifest travels with the artifact so the de-risk runtime probe re-asserts the SAME facexlib sha256.
COPY deploy/bake-manifest.json /opt/models/bake-manifest.json
RUN python /tmp/bake_layers.py assert-weights --root "$VJ_MODELS_ROOT" --precision "${VJ_BAKE_PRECISION}" \
    && python /tmp/bake_layers.py assert-finish-shas --root "$VJ_MODELS_ROOT" --manifest /tmp/bake-manifest.json \
    && python /tmp/bake_layers.py assert-no-tree-cache --root "$VJ_MODELS_ROOT" \
    && printf 'baked_utc=%s\nprecision=%s\nmodel_version=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${VJ_BAKE_PRECISION}" "${VJ_MODEL_VERSION}" \
        > /opt/models/.vj-baked \
    && echo "bake: wrote $VJ_MODELS_ROOT/.vj-baked ($(tr '\n' ' ' < /opt/models/.vj-baked))"

# Third-party model licenses (Apache-2.0 Sec 4(b), MIT, CreativeML RAIL++-M). Baked alongside
# the weights so the image is redistribution-compliant by construction. Full license text at
# /opt/models/licenses/<repo_id>/LICENSE; manifest: THIRD_PARTY_MODELS.md in the repo root.
COPY --from=seed /seed-bins/licenses /opt/models/licenses

# Ensure the cache root exists (idempotent; the bake populated it). The APP (COPY src, CMD) is added
# by the consuming image, deploy/Dockerfile (FROM this runtime base) -- keeping it out of this base is
# what makes a src-only release re-push only the app layers.
RUN mkdir -p /opt/models/hf-cache
