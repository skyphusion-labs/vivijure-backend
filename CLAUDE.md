# CLAUDE.md

Guidance for Claude Code (and the crew) working in this repo.

## What this is

**The GPU render backend for Vivijure** (RunPod serverless). It consumes a project bundle (a standard
`storyboard.yaml` plus the cast) from R2, **plans** the whole render on the CPU (what must train, draw,
animate, what can be reused), then **executes** only that plan on the GPU: trains per-character LoRAs,
draws SDXL keyframes, animates them with Wan image-to-video, runs the finish chain, and assembles the
film off-GPU, writing all artifacts + progress back to R2 in the shape the control plane already
expects. Python; the Worker side is `vivijure`. This is the half that turns a storyboard into a film.

## The Vivijure constellation (the same map is in each repo's README)

```
   friends + Slate (Discord)
            |
            v
        slate  -->  vivijure (studio control plane / JSON API)
                        |
                        v
                  vivijure-backend (GPU render: keyframes -> i2v -> assemble)   <-- THIS REPO
                        |
            +-----------+-------------+-------------------+
            |           |             |                   |
     vivijure-     vivijure-     vivijure-audio-    (more finish
     musetalk      upscale       upscale            modules over time)
   (lip-sync)    (video upscale) (speech enhance)
```

## Documentation map

Deep docs live in `docs/`; this file is the working method. When a change touches one of these areas,
update the matching doc.

- `docs/architecture.md` -- the plan/execute model, the render stages, where each runs (CPU vs GPU).
- `docs/contract.md` -- the R2 bundle-in / artifacts-out contract with the `vivijure` control plane.
- `docs/configuration.md` -- the endpoint env + knobs.
- `docs/cold-start-design.md` -- the cold-start strategy (baked image layers, model warmup).
- `docs/runpod-endpoint-config.md` -- the RunPod serverless endpoint config.
- `docs/operations.md` + `docs/release-gate.md` + `docs/regression-plan.md` -- run, gate, and regress.
- `docs/development.md` -- local dev loop.
- `THIRD_PARTY_MODELS.md` -- model licenses/attributions (SDXL, Wan, MuseTalk, etc.). Keep it current
  when a model is added or swapped.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # runtime + dev deps
pytest                                                    # the test suite (pytest.ini: src on path, tests/)
python scripts/run_keyframe.py                            # local: SDXL keyframe stage
python scripts/run_i2v.py                                 # local: image-to-video stage
python scripts/run_lora_train.py                          # local: per-character LoRA training
docker build -f deploy/Dockerfile -t <ghcr-tag> .         # build the serverless GPU image
python scripts/pin-runpod-template.py                     # pin the RunPod template to the new image
```

The image is baked in `deploy/` (layer baking, HF config baking, fp8 i2v quantize) and pushed to
**GHCR**; the RunPod template is then pinned to the new image digest. **Registry-auth is MCP/API
managed** (RunPod MCP v1.4.0+): create the container-registry-auth and attach it via
`containerRegistryAuthId` on the template, no dashboard step.

## Verifying changes

`pytest` is the suite (CI: `.github/workflows/tests.yml`). The GPU path is verified against a LIVE
RunPod endpoint by `.github/workflows/runpod-verify.yml` (the real-artifact gate -- verify against the
deployed image/endpoint, never the tag or our records). `release.yml` cuts releases; `docs/release-gate.md`
is the checklist. For a local stage check, run the `scripts/run_*.py` entries.

## Architecture (load-bearing)

- **CPU plans, GPU executes.** The handler plans the entire render on the CPU first (a deterministic
  plan of train/draw/animate/finish/assemble steps + what is reusable), then runs only that on the GPU.
  Keep planning off the GPU; that is what keeps cost bounded.
- **R2 is the contract surface.** Bundle in, artifacts + progress out; the shapes are
  `docs/contract.md` and must match the `vivijure` side (`vivijure/docs/CONTRACT.md`). A change here is
  a change to both repos.
- **Honest failures.** A finish step that genuinely fails (after bounded retry + reclaim) fails the
  shot with the real error; it never silently ships a raw/unfinished clip. A degrade is never silent.
- **Cold start is engineered, not incidental.** Image layers + model configs are baked at build; see
  `docs/cold-start-design.md` before changing the Dockerfile or model load path.

## Conventions

- **No em-dashes (U+2014) or en-dashes (U+2013) anywhere.** Use commas, semicolons, parentheses, or `--`.
- Python; keep runtime deps justified (this is a GPU image -- every layer is cold-start cost).
- Every third-party model is attributed in `THIRD_PARTY_MODELS.md` with its license; add the entry in
  the same change that adds the model.
- Verify against the deployed ARTIFACT (built image contents / live endpoint), never the git tag, the
  image size, or our records -- they drift.

## Crew + identity + spend

- Crew work as their own identity: FIRST command in any op is `sudo -u <member> bash -lc '<ops>'` (own
  `$HOME`, own clone, own creds); commits/PRs land under `skyphusion-<member>`. This is the backend lane
  (Rollins owns the render contracts).
- **GPU / RunPod render spend is GATED.** Spinning a GPU endpoint or a render run costs money; ration it
  per the vivijure render-spend rule and confirm before a non-trivial GPU spend. (This is the one place
  the "execute autonomously" default yields to the spend gate.)
- Operating memory: this repo's per-project memory (`vivijure` project segments cover the backend);
  load it before acting.

## Commits & versioning

Conventional Commits (`feat(i2v):`, `fix(keyframe):`, `docs:`); body explains the why. SemVer-style
`0.MINOR.PATCH` while pre-1.0; a release commit bumps the version and updates `RELEASES.md` / `CHANGELOG.md`.
