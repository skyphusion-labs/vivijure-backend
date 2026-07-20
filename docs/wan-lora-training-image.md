# Wan 2.2 A14B LoRA training image (vivijure-cf #29, Phase D1)

The train-only backend image that fits the two-expert Wan 2.2 A14B video character LoRA the cloud-i2v
cost door needs. It shares the worker/orchestrator/pipeline with the render image but carries the
ai-toolkit trainer instead of the render stack. Built by `.github/workflows/train-image-build.yml`
from `deploy/train.Dockerfile`; the D2 training endpoint pins it by `:train-<version>` (bare version,
never `:sha`).

## The routing (already merged, Phase A -- no dispatch gap)

A job with `action="train_lora"` + `model_family="wan"` flows:

    worker.build_pipeline -> harness.handler.run_job -> orchestrator.plan (lora_family="wan")
      -> pipeline.execute -> pipeline._train_slot_wan -> wan_lora_train.train_slot_wan
        -> ai-toolkit run.py (subprocess) -> harvest two .safetensors
      -> handler._finish uploads BOTH experts to R2 at
         loras/<project>/<slot>/wan_high_noise.safetensors  and  wan_low_noise.safetensors

Both experts must land or the slot is not "trained" (`_restore_prior_state` checks both keys); a
half-train fails loud (`harvest_experts`).

## The #1 coupling, resolved: two isolated conda envs

`wan_lora_train._run_aitoolkit` runs ai-toolkit's `run.py` as a subprocess. In Phase A that used
`sys.executable` -- the worker's own interpreter. That is wrong for a shipped image, because
ai-toolkit's dependency set conflicts IRRECONCILABLY with the worker "vivijure" env's validated
render pins:

| package      | vivijure env (render)            | ai-toolkit @6e158dd            |
|--------------|----------------------------------|--------------------------------|
| diffusers    | `0.39.0` (stable)                | git `@c943837` (dev; wan22)    |
| transformers | `5.13.1`                         | `5.5.3`                        |
| torchao      | `0.17.0`                         | `0.10.0`                       |
| av           | `13.1.0`                         | `16.0.1`                       |

They cannot share one Python env.

**Resolved pin sets (verified on the A100 80GB build, cf#29 D1).** Both envs share the SAME cu128
`torch==2.9.0` line (torch 2.7.1, the render-validated version, was delisted from the PyTorch cu128
index -- see the base note below), pinned as the full trio `torch==2.9.0 / torchvision==0.24.0 /
torchaudio==2.9.0` (unpinned, torchaudio resolves to 2.11.0 and mismatches; ai-toolkit imports
torchaudio at import time). No ai-toolkit-specific dependency OVERRIDES were needed -- ai-toolkit's own
pinned set (`deploy/aitoolkit-overrides.txt` is empty) installed cleanly under the held torch trio, incl.
`torchcodec==0.9.1`, `torchao==0.10.0`, `optimum-quanto==0.2.4`, `bitsandbytes==0.49.2`.

    aitoolkit env:  diffusers 0.39.0.dev0 (git) · transformers 5.5.3 · torchao 0.10.0 · av 16.0.1 · peft 0.18.1
    vivijure env:   diffusers 0.39.0 (stable) · transformers 5.13.1 · peft 0.19.1

A build-time import smoke gates both envs (worker + seam import in vivijure; torch/torchaudio/diffusers
in aitoolkit) so a broken isolation or a torch-version regression fails the build in seconds. So the image builds a SECOND conda env, `aitoolkit`, with
ai-toolkit's deps, and `wan_lora_train.aitoolkit_python()` resolves the subprocess interpreter from
`VIVIJURE_AITOOLKIT_PYTHON` (set to `/opt/conda/envs/aitoolkit/bin/python` in the image), defaulting
to `sys.executable` when unset -- so single-env installs and the CPU tests keep the old behavior. The
worker itself still runs in the vivijure env (the image CMD); ONLY the ai-toolkit subprocess crosses
into the aitoolkit env. The validated render stack in the vivijure env is left byte-for-byte
untouched.

Both dependency sets share the SAME Blackwell-safe cu128 `torch==2.7.1` line (installed into the
aitoolkit env first, held by `deploy/aitoolkit-constraints.txt`) so neither env fights the toolchain.

## ai-toolkit pin

Pinned at `6e158dd1f1552b73b7aca6d7ddaa46a783538052` (HEAD on 2026-07-20). The Phase-0 spike's exact
ai-toolkit rev was not recorded anywhere recoverable (repo, memory, search); D1 is plumbing, not the
identity bind, so HEAD is the deliberate pin. Bump + re-validate on a pod; never float HEAD.

## Base image decision: LEAN, not the render-runtime base

`deploy/train.Dockerfile` builds `FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` and reproduces
the worker's validated env (the runtime.Dockerfile recipe: cu128 `torch==2.7.1` + the pinned
`deploy/requirements.txt` + the basicsr/facexlib/gfpgan compat patches) MINUS the ~87GB baked render
weight bake. Why lean:

- A training endpoint NEVER loads the render weights (SDXL / Wan-i2v); they were pure dead weight.
- Carrying 87GB made every cold RunPod pod a ~20min image pull -- the root cause of the D1 bring-up
  pain (6 pods, no observability, hours lost). Lean -> ~15-20GB image -> fast, reliable pods.
- It is the correct D2 train-endpoint artifact: a dedicated train endpoint shouldn't carry render
  weights.

The env is reproduced from the SAME pinned `deploy/requirements.txt` the runtime base uses, so
worker/harness/orchestrator/pipeline import and run exactly as prod; only the baked WEIGHTS (which the
train path never loads) are absent. (Earlier this image was FROM the runtime base; the 87GB pull cost
made that the wrong call for a train-only image -- superseded.)

## Base model weights

`ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16` (both experts, bf16, ~50GB+). NOT baked in the D1 image:
it stages into the HF cache (`HF_HOME=/opt/models/hf-cache`) on first use. For the D2 prod training
endpoint, BAKE it per the bake doctrine (per-file layers under the 10GB GHCR ceiling; a bf16 expert
can exceed 10GB and needs splitting) so a cold worker does not re-pull ~50GB every start.

## The recipe (spike-proven defaults, `WanLoraTrainConfig`)

rank 32, bf16 both experts resident (`quantize:false`, `low_vram:false`, 80GB card), `adamw8bit`,
`switch_boundary_every=10` (MoE high/low alternation), identity via a per-slot trigger token in the
captions. Inference binds identity at LoRA scale ~1.5 (that is a later phase; D1 only proves training
+ harvest).

## D1 proof

Trained one character end to end THROUGH `vivijure_backend.worker` (not a hand-run of ai-toolkit) on
a PROD SECURE A100 80GB pod: both experts trained and two `.safetensors` harvested to R2. See the PR
description for the harvested keys, cost, and any deltas from the spike recipe.
