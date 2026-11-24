# Vivijure Backend

This file is the guide for any coding agent and any contributor. Follow it.

## What this project is

This repo is the GPU render backend for Vivijure. It runs on RunPod serverless.

It reads a project bundle from R2. The bundle is `storyboard.yaml` plus the cast. It plans the render on the CPU. It then runs that plan on the GPU. It trains per-character SDXL LoRAs and draws SDXL keyframes.

The code also holds Wan image-to-video, finish, and off-GPU assembly paths. The published thin image and the operated endpoint serve keyframes.

It writes artifacts and progress back to R2 in the shape the studio panels expect. The language is Python. The panels are `vivijure-cf` and `vivijure-local`. The hub repo `vivijure` is not the panel.

Image tags use the `backend-v*` line. See [CHANGELOG.md](CHANGELOG.md), [RELEASES.md](RELEASES.md), and the latest tag. Do not freeze a number here.

Wan cast training moved to `vivijure-wan-train`. That repo is archived. Hosted Wan training is retired. Self-host is the remaining path.

## Where it sits

Panels pin this image on a RunPod endpoint. Do not freeze an endpoint id in this file.

```
slate / vivijure-mcp
        |
        v
vivijure-cf / vivijure-local  (panels; orchestration: vivijure-core)
        |
        v
vivijure-backend (this repo: RunPod SDXL keyframes)
        |
        +-- upscale, cloud motion doors, local-12gb, local-16gb
```

## Docs

Update the matching doc when you change that area.

- [docs/architecture.md](docs/architecture.md) -- plan, then execute; CPU versus GPU
- [docs/contract.md](docs/contract.md) -- R2 bundle in, artifacts out
- [docs/configuration.md](docs/configuration.md) -- endpoint env and knobs
- [docs/cold-start-design.md](docs/cold-start-design.md) -- baked layers and warmup
- [docs/runpod-endpoint-config.md](docs/runpod-endpoint-config.md) -- serverless endpoint config
- [docs/operations.md](docs/operations.md), [docs/release-gate.md](docs/release-gate.md), [docs/regression-plan.md](docs/regression-plan.md)
- [docs/development.md](docs/development.md) -- local loop
- [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md) -- model licenses
- [CONTRIBUTING.md](CONTRIBUTING.md) -- pull request flow
- [changelog.d/README.md](changelog.d/README.md) -- changelog fragments

`THIRD_PARTY_MODELS.md` has no MuseTalk entry. Do not add one. That provider is retired.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
python scripts/run_keyframe.py
python scripts/run_i2v.py
python scripts/run_lora_train.py
docker build -f deploy/Dockerfile -t <ghcr-tag> .
```

`pytest` is the suite (`.github/workflows/tests.yml`). `scripts/pin-runpod-template.py` is not a production path. See [docs/operations.md](docs/operations.md).

`thin-image.yml` builds the worker image and pushes it to GHCR. See [RELEASES.md](RELEASES.md). The promote path is `runpod-verify`. An endpoint carries its own image.

## Verify

`.github/workflows/runpod-verify.yml` is `workflow_dispatch` only. With `live` false it dry-runs. It uses a mocked client and no GPU. With `live` true, `live-gate` starts one Secure GPU pod and tears it down. `promote` defaults off. A proof run does not repin production.

The only production pin path is `promote_image()` in `deploy/runpod_verify.py`. It runs only when dispatch passes with `live` true and `promote` true. Do not pin the template by hand.

`release.yml` is disabled. [docs/release-gate.md](docs/release-gate.md) is the gate contract. For a local stage, use `scripts/run_*.py`.

## Architecture

The handler plans the whole render on the CPU first. The plan is deterministic. It lists train, draw, animate, finish, and assemble steps, and what can be reused. The GPU runs only that plan. Keep planning off the GPU.

R2 is the contract surface. Shapes are [docs/contract.md](docs/contract.md). They must match `vivijure-cf/docs/CONTRACT.md`. A change here is a change on both sides.

A finish step that fails after bounded retry fails the shot with the real error. It does not ship a raw clip. A degrade is never silent.

Cold start is designed. Read [docs/cold-start-design.md](docs/cold-start-design.md) before you change the Dockerfile or the model load path.

## Prove a GPU image

A green build is not a proven render. Run a real job on the candidate image before you pin production.

Use one path. Spin a dedicated pod on the candidate image, run a short job, and check the outputs. Or pin the candidate on the serverless endpoint. Set min workers to 1. Run the same job. Check the outputs. A community pod is not the proof.

If you set min workers above 0 for the test, set them back to 0 when the test ends. Max workers can stay at the production pin. The floor that keeps a warm worker goes back to 0, unless production capacity already requires a warm worker. Record the before and after values. Do not leave the floor up because you might test again later.

Do not trust CI alone, a green bake alone, or image size alone. Verify the built image or the live endpoint. Pin a versioned tag, and a digest where Hub requires one. Do not treat `:latest` as the production contract.

## Release

A version moves only in a release pull request. A feature pull request does not bump. A fix pull request does not bump.

There is no manifest version. The git tag is the version of record. The tag shape is `backend-v*`. Read the latest tag. Do not copy a number from this file.

A pull request that touches `src/` adds `changelog.d/<issue>-<slug>.md`. See [changelog.d/README.md](changelog.d/README.md). The CI job is `changelog`. Do not edit [CHANGELOG.md](CHANGELOG.md) in that pull request. The release commit runs `scripts/changelog-assemble.py`.

A merge to `main` runs CI only. Publish the image by dispatching `thin-image.yml` from `main`. `release.yml` is disabled and fails closed. An org admin tags the release pull request merge commit.

Publishing the image is not the production pin. Production is the `runpod-verify.yml` promote path. See [docs/release-gate.md](docs/release-gate.md).

Before a release pull request, check tags. If the version you are about to cut already has a tag, stop. Read why before you tag again.

Use a Conventional Commit (`feat(i2v):`, `fix(keyframe):`, `docs:`). The body says why.

## Docs language and safety

Write Simplified Technical English. Use short sentences. Put one idea in each sentence. Use the active present tense. Use plain words. Use the same word every time.

Do not use an em dash (U+2014). Do not use an en dash (U+2013). Use a comma, a semicolon, parentheses, or two hyphens (`--`).

State the limit next to the capability. Do not turn a measured result into a promise.

Every third-party model gets an entry in [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md) in the same change. Justify a runtime dependency. Each image layer adds cold-start cost.

This tree is clean-room. It shares no code with wavevryn.

The CSAM ban includes synthetic images. Do not weaken that check.

Do not put a secret in a tracked file. Do not freeze an endpoint id in this file.
