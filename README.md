# vivijure-backend

A render backend for Vivijure: it consumes a project bundle (a standard
`storyboard.yaml` plus the cast), generates SDXL keyframes, trains per-character LoRAs,
turns the keyframes into motion with image-to-video, and returns the artifacts in the
shape the Vivijure control plane already expects.

![vivijure-backend architecture: the control plane writes a bundle to R2 and submits a render job; a RunPod serverless GPU worker plans the render on the CPU, then trains character LoRAs, draws SDXL keyframes, animates them with Wan image-to-video, finishes the clips, and assembles the final film off-GPU, writing all artifacts and progress back to R2.](assets/diagram-architecture.svg)

## The Vivijure ecosystem

Vivijure is an AI film studio built as a thin control plane plus opt-in GPU modules. These repos
form the constellation; this block is identical in each so the whole map is visible from any one of
them.

```
   friends + Slate (Discord)
            |
            v
        slate  -->  vivijure (studio control plane / JSON API)
                        |
                        v
                  vivijure-backend (GPU render: keyframes -> i2v -> assemble)
                        |
            +-----------+-----------------------------+
            |           |               |             |
            v           v               v             v
     vivijure-     vivijure-       vivijure-      (more finish
     musetalk      upscale         audio-upscale   modules over time)
   (lip-sync)    (video upscale)  (speech enhance)
```

| Repo | Role |
|---|---|
| [slate](https://github.com/skyphusion-labs/slate) | Collaborative AI screenwriter assistant for Discord. Friends and Slate co-author a film in-channel; Slate then submits it to the studio entirely through the vivijure JSON API. |
| [vivijure](https://github.com/skyphusion-labs/vivijure) | The studio control plane (a Cloudflare Worker): planner, cast, and render UI plus the JSON API. A thin module host that orchestrates render jobs behind a typed hook contract. |
| [vivijure-backend](https://github.com/skyphusion-labs/vivijure-backend) | The GPU render backend (RunPod serverless): SDXL keyframes, Wan image-to-video, and ffmpeg assembly. The half that turns a storyboard bundle into a film. |
| [vivijure-musetalk](https://github.com/skyphusion-labs/vivijure-musetalk) | MuseTalk audio-driven lip-sync GPU module (finish-class). Syncs a character's mouth to dialogue audio. |
| [vivijure-upscale](https://github.com/skyphusion-labs/vivijure-upscale) | Real-ESRGAN CUDA video-upscale GPU module (finish-class). Raises the assembled film's resolution. |
| [vivijure-audio-upscale](https://github.com/skyphusion-labs/vivijure-audio-upscale) | CUDA speech-audio enhancement (resemble-enhance) GPU module. The GPU half of the cost-aware audio finish path. |

## Team

Vivijure is built by Conrad (`skyphusion`) and his named AI crew. The crew are treated as
individuals, each working in their own lane with their own GitHub identity; this is the same
transparent framing used across the project.

| Member | Role | GitHub |
|---|---|---|
| Conrad | Creator / director | [@skyphusion](https://github.com/skyphusion) |
| Mackaye | PM / tech lead | [@skyphusion-mackaye](https://github.com/skyphusion-mackaye) |
| Strummer | Infrastructure | [@skyphusion-strummer](https://github.com/skyphusion-strummer) |
| Rollins | Backend / modules | [@skyphusion-rollins](https://github.com/skyphusion-rollins) |
| Joan | Frontend / extraction | [@skyphusion-joan](https://github.com/skyphusion-joan) |

## What it is

The control plane writes a project bundle to object storage (R2) and submits a render job. A
RunPod serverless GPU worker pulls the bundle, **plans** the whole render on the CPU (what must
train, draw, animate, and what can be reused), then **executes** only that plan on the GPU:
train a small identity LoRA per character, draw an SDXL keyframe per shot, animate each keyframe
into a clip with Wan 2.2 image-to-video, optionally finish each clip (frame interpolation + face
restore), and concatenate the clips off-GPU into the final film. Every artifact is written back
to R2 under a project-keyed layout the control plane polls for, and the project tree is
snapshotted so the next render reuses everything that did not change.

The LLM-free render path is proven end-to-end on RunPod serverless: it renders complete films.
GPU-validated features land tagged; the CPU-testable logic is covered by the suite in `tests/`.

## Why this exists

This is an independent, built-from-scratch implementation, written against the control-plane
API contract and the underlying models' own documentation. There is no inherited pipeline code
and no legacy cruft; the contract (the `storyboard.yaml` schema, the cast registry, the
render-job input/output) is the only thing carried over, and that contract is the control
plane's own. The payoff is a clean-sheet codebase that is easy to extend and to reason about:
contract, config, orchestrator, routing, models, stages, and harness are each cleanly
separated. See [CONTRIBUTING](CONTRIBUTING.md) for the house style and PR process.

## Architecture at a glance

```mermaid
flowchart LR
    CP["Control plane"] -->|submit job| W
    CP -->|write bundle| R2["R2 object store"]
    R2 -->|bundle, refs, state| W
    subgraph W["RunPod GPU worker"]
        direction LR
        PLAN["Plan (CPU)"] --> TRAIN["Train LoRA"] --> KF["Keyframe (SDXL)"] --> I2V["i2v (Wan)"] --> FIN["Finish"] --> ASM["Assemble (CPU)"]
    end
    W -->|keyframes, clips, full.mp4, state, progress| R2
    W -->|RenderResult| CP
```

| Module | Role |
|---|---|
| `contract.py` | storyboard + cast + job I/O types; the bundle reader |
| `config.py` | typed `RenderConfig`: every generation knob, clamped and tier-aware |
| `orchestrator.py` | the CPU planner: what to train / draw / animate / reuse |
| `routing.py` / `device.py` | stage-to-GPU-tier policy; card-to-precision fingerprinting |
| `models.py` | capability-aware model loading (the warm `ModelServer`) |
| `lora_train.py` / `keyframe.py` / `i2v.py` / `finish.py` | the GPU render stages |
| `assemble.py` | off-GPU ffmpeg concat into the final film |
| `pipeline.py` | `GpuPipeline.execute`: runs the plan over the stages |
| `harness/` | the serverless spine: RunPod handler, R2 I/O, key layout, progress, model mirror |

Full detail, with diagrams of the render-job sequence, the planner decision tree, and the
capability-aware precision model, is in [docs/architecture.md](docs/architecture.md).

## Quality tiers

One job parameter, `quality_tier`, sets every stage baseline (overridable per knob via
`render_overrides`):

| Tier | Keyframe | Image-to-video | Finish | i2v GPU tier |
|---|---|---|---|---|
| `draft` | Hyper-SD 4-step | Lightning 4-step distill | none (preview) | RTX PRO 6000 |
| `standard` | Hyper-SD 8-step | full 20-step + EasyCache | interpolate 2x | H200 |
| `final` | full 30-step | full 40-step + MixCache | interpolate 2x + face restore | B200 |

See [docs/configuration.md](docs/configuration.md) for every field, default, and range.

## Quickstart

```bash
# CPU dev: no GPU, no model weights needed
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                  # the full CPU suite

# quick syntax check
python -m py_compile src/vivijure_backend/*.py src/vivijure_backend/harness/*.py
```

Run a single render stage by hand against a bundle (needs a CUDA pod + the `deploy/` stack):

```bash
python scripts/run_lora_train.py BUNDLE.tar.gz OUT_DIR --slot A
python scripts/run_keyframe.py   BUNDLE OUT_DIR --shot shot_02 --lora A=/path/A.safetensors
python scripts/run_i2v.py        BUNDLE OUT_DIR --shot shot_01 --keyframe /path/shot_01.png
```

Build and deploy the worker image: [docs/operations.md](docs/operations.md) (and
[deploy/README.md](deploy/README.md) for the build mechanics).

## Documentation

- [docs/architecture.md](docs/architecture.md) -- how it works and how the pieces interface, with diagrams.
- [docs/contract.md](docs/contract.md) -- the bundle, the render job, the result; worked example.
- [docs/configuration.md](docs/configuration.md) -- `RenderConfig`: every knob, default, range, and quality-tier baseline.
- [docs/operations.md](docs/operations.md) -- build, deploy, the model mirror, the R2 key map, the progress channel, failure modes.
- [docs/development.md](docs/development.md) -- the CPU/GPU split, the test suite, running stages locally.
- [docs/cold-start-design.md](docs/cold-start-design.md) -- issue #55 Phase B: the cold-start cost model (from telemetry) and the bake/pre-warm/stage decision.
- [CONTRIBUTING](CONTRIBUTING.md) -- house style, PR process.
- [SECURITY](SECURITY.md) -- the security boundary (one R2 credential, control-plane-trusted input).
- [CHANGELOG](CHANGELOG.md) / [RELEASES](RELEASES.md) -- per-release notes and the release ledger.

## Acceptable use

This is the generative render engine behind Vivijure (text-to-image keyframes, image-to-video motion,
and LoRA training). Using it to generate sexual content involving minors, real or synthetic, or
non-consensual intimate imagery or deepfakes of real people, is absolutely prohibited; CSAM is also a
crime (18 U.S.C. 1466A / 2252A). That bright line is the project-wide spine. The full policy is the
[Vivijure Acceptable Use Policy](https://github.com/skyphusion-labs/vivijure/blob/main/docs/legal/ACCEPTABLE-USE.md).

## License

**AGPL-3.0-only.** A labor of love, given freely: use it, learn from it, self-host it, build your own creative visions on it. Run it as a network service and the AGPL has you share your changes back, so it stays a commons. It is not for sale, and not to be resold as a SaaS.
