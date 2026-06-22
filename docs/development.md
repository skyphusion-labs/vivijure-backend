# Development

How to work on the backend locally: the CPU-vs-GPU split that shapes everything, the test
suite, running a single render stage by hand, and the conventions that keep the code testable.
For the house style and the PR process see [CONTRIBUTING](../CONTRIBUTING.md); for the
security boundary see [SECURITY](../SECURITY.md).

## The CPU / GPU split

The single most important design rule: **heavy imports (torch, diffusers, transformers, peft,
imageio) are deferred into the functions that need them, never at module top.** Every module
imports and unit-tests on a CPU box with no GPU and no model weights present. The GPU only runs
inside the stage functions.

This is what lets the entire control path (contract parsing, config, routing, the planner, the
harness job flow, assembly planning, keyframe / pose geometry) be covered by a fast CPU test
suite, while the irreducible GPU work (SDXL keyframes, LoRA training, Wan i2v, RIFE / face
restore) is validated by the maintainer on real hardware, gated and tagged. CI does not have a
GPU and does not block on the render path; keep new logic CPU-testable where you can, and call
out anything that needs a GPU-validation pass in the PR.

## Local setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt   # PyYAML + pytest + Pillow (no torch)
pytest                                 # the full CPU suite
```

`requirements.txt` is intentionally minimal (`PyYAML`, for contract parsing).
`requirements-dev.txt` adds `pytest` and `Pillow` (the pose-skeleton geometry test). The GPU
stack is pinned separately in `deploy/requirements.txt` and only installed in the worker image;
you do not need it to run the suite. See [operations.md](operations.md) for the image.

## The test suite

`pytest.ini` sets `pythonpath=src`, `testpaths=tests`. The suite is CPU-only by design and
covers the pure logic and serialization; the GPU bodies are stubbed or skipped.

| Test | Covers |
|---|---|
| `test_device.py` | Card fingerprinting (arch / tier / quant / attention) across the fleet matrix. |
| `test_routing.py` | Stage-to-tier routing (i2v climbs draft -> standard -> final). |
| `test_models.py` | Quantization matrix (SDXL never reaches fp4; video stays fp8). |
| `test_config.py` | Tier baselines, forgiving + clamped parsing, invalid-combo guards. |
| `test_config_mapping.py` | Config -> engine-param mapping completeness (a meta-test that fails if a config field goes unmapped). |
| `test_contract.py` | Bundle extract, Cast / Scene / Storyboard, request / result serialization. |
| `test_orchestrator.py` | The planner: elimination paths per action, cost arithmetic. |
| `test_pipeline.py` | `GpuPipeline` config -> params mappers, `execute()` with GPU stages stubbed. |
| `test_harness.py` | `run_job` flow with a fake pipeline + fake R2 store; keys / mirror / config. |
| `test_progress.py` | The progress channel: R2 writes, snapshot, best-effort, throttled callbacks. |
| `test_models_mirror.py` | `.rclonelink` -> symlink reconstruction. |
| `test_deploy.py` | RunPod template-pin transform; per-job pipeline build over a shared `ModelServer`. |
| `test_keyframe.py` | Prompt building, region geometry, single-vs-regional path choice. |
| `test_i2v.py` | Frame-count math (temporal-VAE stride), duration, tier -> profile. |
| `test_finish.py` | Frame / fps math, restorer wrappers with fakes. |
| `test_instantid.py` | Face selection, keypoint drawing. |
| `test_assemble.py` | Manifest build, ffmpeg concat command (live merge skipped if ffmpeg absent). |
| `test_lora_train.py` | Caption building, default base repo (the training loop needs CUDA). |

The `test_config_mapping.py` completeness meta-test is worth knowing about: it cross-checks
`dataclasses.fields()` against what the pipeline actually forwards, so adding a config field
without wiring it through fails the suite rather than silently doing nothing.

A quick syntax check without running anything:

```bash
python -m py_compile src/vivijure_backend/*.py src/vivijure_backend/harness/*.py
```

## Running a single stage locally

Three standalone drivers run one render stage against a bundle, for validating the GPU path on a
CUDA pod (Hopper / Blackwell). They put `src/` on the path and use the real package, so they
need the `deploy/requirements.txt` stack installed.

```bash
# Train one (or all) character LoRA(s) from a bundle
python scripts/run_lora_train.py BUNDLE.tar.gz OUT_DIR [--slot A] [--steps 1000] [--rank 16]

# Render one shot's keyframe (point --lora at trained adapters)
python scripts/run_keyframe.py BUNDLE OUT_DIR --shot shot_02 \
    --lora A=/path/A.safetensors --lora B=/path/B.safetensors [--full-step]

# Animate one keyframe into a clip
python scripts/run_i2v.py BUNDLE OUT_DIR --shot shot_01 \
    --keyframe /path/shot_01.png [--quality draft|standard|final]
```

Each prints a JSON result (engine path taken, output path, frame counts, seed) so you can see
exactly which path a shot resolved to.

## Conventions

- **No em-dashes (U+2014) or en-dashes (U+2013)** anywhere in source, comments, docs, or commit
  messages. Use commas, semicolons, parentheses, or a double hyphen (`--`).
- **Conventional Commits**: `fix(scope): ...`, `feat(scope): ...`, `docs: ...`, `ci: ...`; the
  body explains the *why*.
- **Releases** are SemVer-style `backend-vX.Y.Z` tags (PATCH for fixes, MINOR for features,
  pre-1.0). The tag must be pushed to origin to build (see [RELEASES.md](../RELEASES.md)).
- **Original work + sign-off**: contribute only your own original or appropriately-licensed
  work, and sign your commits off (`git commit -s`, DCO; see [CONTRIBUTING](../CONTRIBUTING.md)).
- **Mirror every binding in a typed config / `Env`**: a generation knob belongs in
  `config.py` (typed, clamped, tier-aware), not read ad hoc from the raw overrides dict.

## See also

- The layers and data flow you are working within: [architecture.md](architecture.md).
- The contract your changes must keep: [contract.md](contract.md).
- Every config field and its range: [configuration.md](configuration.md).
