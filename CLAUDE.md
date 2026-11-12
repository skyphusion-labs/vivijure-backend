# CLAUDE.md

Guidance for Claude Code (and the crew) working in this repo.

## What this is

**The GPU render backend for Vivijure** (RunPod serverless). It consumes a project bundle (a standard
`storyboard.yaml` plus the cast) from R2, **plans** the whole render on the CPU (what must train, draw,
animate, what can be reused), then **executes** only that plan on the GPU: trains per-character **SDXL**
LoRAs and draws SDXL keyframes. (The code also holds Wan image-to-video, finish and off-GPU
assembly paths; the published thin image and the operated endpoint serve keyframes.) It writes artifacts +
progress back to R2 in the shape the studio panels expect. Python; the panel side is `vivijure-cf` /
`vivijure-local` (not the hub `vivijure` repo). This is the half that turns a storyboard into a film.

**Image line:** GHCR tags / `backend-v*` release process (see `CHANGELOG.md` / `RELEASES.md` / latest
tags). Do not freeze a version number here as current forever. **Wan cast train** moved to
`vivijure-wan-train`, which is now archived: hosted Wan training is retired (self-host only).

## The Vivijure constellation

```
   slate / vivijure-mcp
            |
            v
   vivijure-cf / vivijure-local  (panels; orchestration: vivijure-core)
            |
            v
   vivijure-backend (THIS REPO: RunPod SDXL keyframes)
            |
     +------+----------------+------------------+
     |      |                |                  |
  upscale  cloud motion doors  local-12/16gb
```

Panels pin this image on RunPod endpoints. **Never freeze specific endpoint IDs** in this file.

## Documentation map

Deep docs live in `docs/`; this file is the working method. When a change touches one of these areas,
update the matching doc.

- `docs/architecture.md` -- the plan/execute model, the render stages, where each runs (CPU vs GPU).
- `docs/contract.md` -- the R2 bundle-in / artifacts-out contract with the studio panels.
- `docs/configuration.md` -- the endpoint env + knobs.
- `docs/cold-start-design.md` -- the cold-start strategy (baked image layers, model warmup).
- `docs/runpod-endpoint-config.md` -- the RunPod serverless endpoint config.
- `docs/operations.md` + `docs/release-gate.md` + `docs/regression-plan.md` -- run, gate, and regress.
- `docs/development.md` -- local dev loop.
- `THIRD_PARTY_MODELS.md` -- model licenses/attributions (SDXL, Wan, RIFE, etc.). Keep it current
  when a model is added or swapped. It has no MuseTalk entry and should not gain one: that
  provider is retired and ruled out permanently.

## Commands

```bash
pip install -r requirements.txt -r requirements-dev.txt   # runtime + dev deps
pytest                                                    # the test suite (pytest.ini: src on path, tests/)
python scripts/run_keyframe.py                            # local: SDXL keyframe stage
python scripts/run_i2v.py                                 # local: image-to-video stage
python scripts/run_lora_train.py                          # local: per-character LoRA training
docker build -f deploy/Dockerfile -t <ghcr-tag> .         # build the serverless GPU image
python scripts/pin-runpod-template.py                     # not a prod path (own or staging endpoint; docs/operations.md)
```

The worker image is built by `thin-image.yml` (see RELEASES.md "Current build path") and pushed
to **GHCR**; the `runpod-verify` promote path PATCHes the endpoint's image (RunPod REST v2: an
endpoint carries its own image; a template is applied once at create).

## Verifying changes

`pytest` is the suite (CI: `.github/workflows/tests.yml`). The pod-staging gate is
`.github/workflows/runpod-verify.yml` (`workflow_dispatch` only). With `live` false it runs the
verify harness dry-run (mocked client, no GPU). With `live` true, `live-gate` spins one Secure GPU
pod and tears it down. `promote` defaults off, so a proof run does not repin production. The only
prod path is `promote_image()` in `deploy/runpod_verify.py`, and it runs only on a dispatch that
passes with `live` true and `promote` true. Do not pin the template by hand. Never trust the tag
or our records in place of the artifact. `thin-image.yml` publishes the GHCR image (`release.yml` is
disabled); `docs/release-gate.md` is the gate contract. For a local stage check, run the `scripts/run_*.py` entries.

## Architecture (load-bearing)

- **CPU plans, GPU executes.** The handler plans the entire render on the CPU first (a deterministic
  plan of train/draw/animate/finish/assemble steps + what is reusable), then runs only that on the GPU.
  Keep planning off the GPU; that is what keeps cost bounded.
- **R2 is the contract surface.** Bundle in, artifacts + progress out; the shapes are
  `docs/contract.md` and must match the panel side (`vivijure-cf/docs/CONTRACT.md`). A change here is
  a change to both sides.
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
- **Clean room** (no wavevryn). **CSAM bright-line** absolute. **Ignore Cursor `AGENTS.md`.**
- **Never freeze open sprint boards or specific RunPod endpoint IDs** here.
- Image pin discipline: pin by versioned tag (and digest where Hub requires); never treat `:latest`
  as the production contract.

## Pre-prod GPU test spend (Conrad, 2026-08-06) -- do not thrift here

**Never hold back RunPod spend when proving an image before it is production-pinned.** A green
build is not a proven render. Pre-prod verification is **expected spend**, not an escalation.
Community pods are not the gate.

Pick **one** of these for the gate (either is fine; pick what fits the change):

| Path | How | When |
| --- | --- | --- |
| **SecurePod** | Spin a SecurePod (or equivalent dedicated pod) on the candidate image; run selftest / short real job; verify the **artifact** (outputs, logs, structured events). | Prefer for isolated image proof without touching prod traffic. |
| **Serverless endpoint** | Deploy/pin the candidate image on the relevant serverless endpoint, set **active workers (min idle / `workersMin`) to 1** so cold-start does not fake a hang or a false timeout, run the same selftest / short job, verify the artifact. | Prefer when the proof needs the real endpoint shape (env, network, bindings). |

### HARD: restore workers after serverless test

If you raised the serverless endpoint's active / min workers **above 0** for testing, **bring it
back to 0 when the test is done** (same session, before you walk away). Leaving `workersMin` /
active workers at 1+ burns idle GPU money with no proof benefit. Max workers can stay at the
prod pin; the restore rule is about the **floor that keeps a warm worker** (min/active = 0 in
steady state unless Conrad ruled otherwise for prod capacity).

Do **not** skip the restore because "we might test again tomorrow." Re-raise to 1 at the next
test if needed. Document the before/after values in the PR or runlog when you change them.

Never trust: CI green alone, bake green alone, or image size alone.

## Crew + identity + spend

- Crew work as their own identity: FIRST command in any op is `sudo -n -H -u <member> bash -lc '<ops>'` (own
  `$HOME`, own clone, own creds); commits/PRs land under `skyphusion-<member>`. This is the backend lane
  (Rollins owns the render contracts).
- **Pre-prod GPU proof spend is authorized** (SecurePod or serverless workersMin=1). Do not thrift out
  of the proof. **Serverless min/active workers back to 0 when the test ends.** Routine prod render
  volume still follows estate cost discipline; the thrift ban is about **not skipping** the pre-prod
  gate, not about unbounded open-ended farm spend.
- Operating memory: this repo's per-project memory (`vivijure` project segments cover the backend);
  load it before acting.

## Commits & versioning

Conventional Commits (`feat(i2v):`, `fix(keyframe):`, `docs:`); body explains the why. SemVer is the
`backend-v*` tag. The line is past 0.x; read the latest tag, do not copy a number from this file.
A release commit updates `RELEASES.md` / `CHANGELOG.md`.
A PR that touches `src/` adds a `changelog.d/<issue>-<slug>.md` fragment (see `changelog.d/README.md`; CI job
`changelog`) instead of editing `CHANGELOG.md`; the release commit runs `scripts/changelog-assemble.py`.

## Release / deploy

**Image publish is a dispatch, not a tag.** Merges to `main` run CI only. The image is built by
dispatching `thin-image.yml` from `main` (see RELEASES.md "Current build path"); `release.yml`,
which used to run on a `backend-v*` tag, is disabled and fails closed. A release is a PR; an org
admin tags its merge commit.
Publishing the image is not the prod pin. Prod is the `runpod-verify.yml` promote path; see `docs/release-gate.md`.

## Release convention (bump-at-release)

One convention for every skyphusion-labs repo (Conrad, ruled 2026-11-10, fleet-chezmoi#1766). A version moves ONLY in a release PR: bump the version, then assemble the changelog. Feature and fix PRs do not bump. Releases go through a PR (fleet-chezmoi#1622) and the tag goes on that PR's merge commit.

- Version of record here: no manifest version: the git tag is the version of record.
- Tag shape: `backend-v*`.
- Check before a release PR: does a tag already exist for the version currently in the manifest? Yes means the cycle is closed, so the release PR bumps. No means a bump already landed outside a release PR; stop and read why before bumping again.
