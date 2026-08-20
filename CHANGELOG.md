# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

## Unreleased

**Fix: verify pod pulls GHCR with the same RunPod registry auth as serverless.**

The GraphQL SDK cannot attach `containerRegistryAuthId`. The harness was rejecting it
and assuming GHCR was public. REST create now passes the serverless template's auth
id (`cmqbz5bba0018e11d6bpcnu4n`). The PAT stays in RunPod; we never mint a second one.
REST v1 takes `dockerStartCmd` (argv array), not GraphQL `dockerArgs`.

**Weights: seed `2-bf16` (canny ControlNet in the importable seed).**

`ghcr.io/skyphusion-labs/vivijure-backend-seed:2-bf16@sha256:45d036fd50c83dfa0347cf837c43bcccb07e4d2c52c3fb5c02d3deee962a7e66`.
Runtime imports this; `backend-v*` still only COPY src.

**Fix: seed-build and runtime-build register on Plane C before they bake.**

Group 9 is workflow-restricted. `release.yml` already ran `sync-gpu-allowlist` on
ubuntu-latest first. The seed and runtime workflows did not, so a dispatch sat
queued on idle bake runners. Same job, `--apply` (these are main-ref, not tags).

**Fix: tests.yml concurrency no longer cancels against the dummy coverage check.**

`tests.yml` and `coverage.yml` both used `group: coverage-${{ github.ref }}`. GitHub
concurrency is repository-wide, so the org-ruleset dummy `coverage` job cancelled the
real pytest + required `ci` job (PR 432). tests.yml now uses `tests-${{ github.ref }}`.

**Fix: keyframes stay in the scene (plate then canny face, not studio portraits).**

Live LoRA scale was 0.7 on every shot, including single-char, via `keyframe_params_from`
always reading `multi_char.lora_scale_per_slot`, plus a full-frame IP-Adapter of the
cast portrait. That reconstructed the LoRA training studio. There is no img2img pipe.

- Two-pass SDXL keyframe: Pass A is a scene plate (no LoRA, no IP-Adapter); Pass B is
  canny ControlNet of that plate plus a face-crop IP-Adapter. InstantID is not Pass B.
- New `ModelRole.CONTROLNET_CANNY` (`xinsir/controlnet-canny-sdxl-1.0`, rev
  `1271357eda52d54b857c650cacb5b51144643ccb`). Both ControlNets are cached; the pipe
  swaps `pipe.controlnet`. Missing canny weights raise `HarnessError` naming the role.
- Split LoRA mapper: single-char `lora_scale` 0.30; regional `lora_scale_per_slot` 0.35.
- `kf_hash` includes `scene_lock`, `canny_scale`, `scene_lock_v1`, and the split scales.
- `scene_lock` defaults true in this src. Do not tag `backend-v*` until seed-build and
  runtime-build contain the canny weights.

**Fix: job-authored bundle paths and job-supplied R2 endpoints stay in contract.**

- `start_image` and `refs_dir` must be relative paths inside the extracted bundle (no absolute
  paths, no `..`). The resolved realpath is checked against the bundle root, so a symlink that
  looks in-bundle but points out is refused.
- A job-supplied `r2.endpoint` must be `https` on a Cloudflare R2 host
  (`*.r2.cloudflarestorage.com` or `R2_ALLOWED_ENDPOINT_HOSTS` / `CLOUDFLARE_ACCOUNT_ID`).
  `http`, IP literals, `file:`, and metadata hosts are refused. Operator `R2_*` env (the
  dedicated-endpoint path) is unchanged.
- Dockerfile `USER nonroot` was not applied: the runtime base is a root-owned conda GPU image
  (`/opt/models`, `/opt/conda`) and RunPod serverless writes workdirs from the entrypoint. A
  non-root USER in the consumer Dockerfile would break that contract without a runtime-base
  rebuild.

## [1.0.12] -- 2026-07-31

**Fix: the InstantID face-embedding path runs on the GPU again.** PATCH.

The face-embedding path has been executing entirely on CPU on GPU workers. Renders succeeded, no
build failed, and nothing in the logs said so. This release puts it back on the CUDA execution
provider. Known affected: **1.0.9, 1.0.10 and 1.0.11**, which share one pip layer by digest, so
rolling back within that range is not a mitigation. No bisect was run to find where the provider
was first lost, so earlier releases may also be affected.

**No output was corrupted, and nothing needs re-rendering.** Verified rather than assumed: every
antelopev2 model was run on byte-identical seeded input under both providers and compared. The
recognition embedding that InstantID actually conditions on, `glintr100`, matched at
`cosine(CPU, CUDA) = 0.999998`, and `scrfd_10g_bnkps` (the source of the keypoints) at a max
relative delta of 1.8e-03. Those are ordinary CPU-versus-GPU kernel differences, not corruption.
Honest limit on that measurement: the inputs were seeded random tensors rather than real faces, so
relative deltas on models running outside their normal input distribution are inflated.

**The cost was time only, and it was small.** One face-analysis pass measured 483 ms on CPU versus
14 ms on CUDA, a 469 ms delta, and `analyze_face` runs once per single-character keyframe with the
analyzer cached per worker. Against renders measured in tens of minutes that is well under one
percent. This is a correctness fix, not a cost recovery, and it is not a plausible driver of any
timeout or failure-rate symptom.

- **fix(deploy): restore the CUDA execution provider (#347).** Two stacked defects, the first
  masking the second. `insightface` hard-depends on the bare CPU `onnxruntime` wheel, which shares
  the `onnxruntime/` package directory with `onnxruntime-gpu`, so whichever pip installs last owns
  the native module; the CPU build has no CUDA provider compiled in, so `CUDAExecutionProvider` was
  not failing to start, it was absent, and the CPU fallback in `face_analyzer()` swallowed it.
  Separately, `onnxruntime-gpu` moved its default wheel to a CUDA 13 build at 1.27.0, which cannot
  import against this deliberately CUDA 12.8 base. Capped to `>=1.21.0,<1.27.0`, added
  `deploy/ensure_onnxruntime_gpu.py` to purge the CPU wheel and repair the GPU one after
  requirements install, and added a build-time gate that fails the bake if
  `CUDAExecutionProvider` is missing.
- **fix(deploy): restore two requirements pins that cannot resolve on py3.11 (#351).** `numpy==2.5.1`
  requires Python >= 3.12 against a 3.11 conda env, and `tokenizers==0.23.1` is unsatisfiable
  against transformers 5.14.1, which caps it at `<=0.23.0` (a version PyPI never shipped stable).
  Both arrived in one dependabot group bump and both had an adjacent comment forbidding exactly
  that change, left standing above the violation. **`main` could not build at all for eight days**
  and no check noticed, because `deploy/requirements.txt` is resolved only when the runtime base is
  baked, never by CI. Process fix tracked as #355.
- **build(bake): repin `RUNTIME_REF_BF16` to `runtime-1-bf16-t5` (#357),**
  `@sha256:0f3c9bd6818f9dc7d5d1f21f6e0b7ebd59c9e88b7d49682f4a2964d28d57f5f2`, in both
  `deploy/Dockerfile` and `.runpod/Dockerfile`. Built as a deps-only overlay on t4, so the weight
  layers are inherited by blob identity: the dedup gate confirmed 48 of 48 RootFS layers shared,
  and no worker cold-pulls the ~100 GB weight set for this release.

**Verification.** The build gate asserts `get_available_providers()`, which cannot prove a session
because the docker build has no GPU, so attachment was checked on real hardware with a controlled
pair: same script, same `src`, same GPU, only the image differs.

| image | onnxruntime | attached providers |
|---|---|---|
| `1.0.11` (control) | 1.27.0 | 5 of 5 models `[CPUExecutionProvider]` |
| `runtime-1-bf16-t5` | 1.26.0 | 5 of 5 models `[CUDAExecutionProvider, CPUExecutionProvider]` |

Asserted on `session.get_providers()`, the providers actually attached, never the requested list;
`face_analyzer()` has requested `CUDAExecutionProvider` for the entire time it ran on CPU. Models
covered: detection, genderage, landmark_2d_106, landmark_3d_68, recognition. The control ran first
and reproduces the defect on the shipped production image.

Closes #346, which closes on this release rather than on the merges above, because
`deploy/requirements.txt` installs in the runtime base and the merges alone shipped nothing.

## [1.0.11] -- 2026-07-25

- **chore(deps):** `av` (PyAV) 13.1.0 -> 18.0.0, a deliberate major bump (backend#313; the prior
  dependabot PR was closed stale). finish.py only reads clips through imageio.v3 `pyav` plugin
  (immeta/imiter), never the av API directly; that exact path is smoke-tested identical across
  both versions (see deploy/requirements.txt). **Render-verified 2026-07-25:** the live pod
  verify gate (runpod-verify, run 30140653491) passed on the RC image `1.0.11-rc1` baked from
  this change -- keyframe + i2v + finish over the structured verify channel, teardown clean.
  Prod endpoints repin via the gate's promote leg on the release image.

## [1.0.10] -- 2026-07-23

**Fix: allow cast-registry keys in pretrained_loras (E2E cast_loras).** PATCH.

- `pretrained_loras` accepts cast-banked keys (`loras/cast-{id}/…`, `loras/lora-{slug}-…/…`) resolved
  by the control plane; project-scoped `loras/<slug>/` and `bundle_key` tenancy unchanged.

## [1.0.9] -- 2026-07-23

- **fix(security): K3 worker.py close-out (#327).** `@event` JSON escaping in `worker.py`, so a value
  carrying a quote or a newline cannot break the structured progress channel the studio parses.
- **docs(security): the medium and low false-positive disposition (#328)**, recorded rather than
  silently dismissed.
  (Backfilled 2026-07-28 from the backend-v1.0.9 GitHub release; the row was missing from this file.)

## [1.0.8] -- 2026-07-23

**Fix: scope harness audio_key reads (KF3 closeout).** PATCH.

- `renders/` audio beds must sit under `renders/<project>/`; flat `audio/<filename>` studio beds
  remain allowed (no nested `audio/` paths).

## [1.0.7] -- 2026-07-23

**Fix: bind harness R2 keys to project slug (KF3 audit, backend#324).** PATCH.

- `bundle_key`, explicit `keyframe_key`, `clip_key`, and `pretrained_loras` reads are scoped to the job project slug before any R2 I/O.
- Flat `audio/` staging keys unchanged (studio bed upload convention).

## [1.0.6] -- 2026-07-22

**Security PATCH: allowlist HF model repo_ids at cold start (#317).** PATCH.

- **fix(security):** allowlist HF `org/name` model `repo_id`s (DEFAULT_SPECS namespaces) at cold start.
- **fix(promote):** unpause job plane after flush restore (#315).
- **ci:** adversarial security audit workflow.

## [1.0.5] -- 2026-07-21

- **docs(hub): RunPod Hub listing probe files.** The Hub listing checklist needs to see real entry
  files, and this repo's root `handler.py` / `Dockerfile` are git **symlinks** for the package
  layout, which the probe cannot follow. So `.runpod/handler.py` and `.runpod/Dockerfile` are real
  files; the root entries stay symlinks.
- `.runpod/README.md` gained the Hub R2 env blurb (#307), which names the trap: this backend reads
  `R2_ENDPOINT` while the satellites read `R2_ENDPOINT_URL`.
- Also `docs(runpod)`: the Jul 30 render `workersMax` floor and quota-30 note (#306).
- **No runtime or weight change**; `release.yml` assembles the consumer image from the pinned
  runtime base. Hub re-indexes from the release, usually within an hour.
  (Backfilled 2026-07-28 from the backend-v1.0.5 GitHub release; the row was missing from this file.)

## [1.0.4] -- 2026-07-22

**Pillow 12.3.0 security overlay** (cf#178 / Dependabot CVE batch).

- **deps:** `Pillow==12.3.0` in `deploy/requirements.txt` (#295).
- **runtime:** assemble on `runtime-1-bf16-t4` (overlay from t3; Pillow + CVE floor without re-emitting weight digests).
- **Note:** GHCR `:1.0.3` exists as a tagless dispatch and still FROM t3 -- do **not** pin `:1.0.3` for the Pillow fix. Pin **`:1.0.4`**.

## [1.0.2] -- 2026-07-15

- **RunPod-safe successor to the broken `1.0.1` pin.** Dependencies ship via the overlay runtime
  `runtime-1-bf16-t3` (`FROM` t1), so weight layers stay shared with `1.0.0`: roughly 104 GB shared
  against roughly 96 MB new.
- **fix(bake): the deps-overlay runtime path (#273)** (`deploy/runtime-overlay.Dockerfile` plus
  `overlay_from`), so a toolchain or pip bump inherits weight layers **by blob identity** instead of
  rewriting them. This is the fix for the v1.0.1 class, not a workaround for one bad build.
- **Gate: `bake_layers.py assert-shared-diff-ids`**, so the sharing property is asserted rather than
  assumed on the next bump.
- Resolvable t2-era pins restored for the py3.11 bake (`safetensors==0.8.0`, `tokenizers==0.22.2`,
  numpy 2.4.6) without a full seed re-`COPY` (#274, #275).
- **Both backend endpoints were repinned** (image tag only; ids in the release notes). Incident
  write-up: `docs/runpod-1.0.1-weight-digest-eof.md`.
  (Backfilled 2026-07-28 from the backend-v1.0.2 GitHub release; the row was missing from this file.)

## [1.0.1] -- 2026-07-15

- **Do NOT pin `:1.0.1`.** Repinning `RUNTIME_REF_BF16` to `runtime-1-bf16-t2` (#271) rewrote
  roughly 101 GB of weight digests, so a cold worker pulling this image hits `unexpected EOF`.
  **v1.0.2 is the RunPod-safe successor** and carries the same dependency content over an overlay
  runtime that keeps the weight layers shared. Incident:
  `docs/runpod-1.0.1-weight-digest-eof.md`.
- The dependency content this tag intended, and which shipped safely in v1.0.2: av 13.1.0 -> 18.0.0
  (#268), diffusers 0.38.0 -> 0.39.0 paired with safetensors 0.8.0 (#258, #262), insightface
  `>=1.0.1` (#259), accelerate 1.14.0 (#260), transformers 5.13.1 (#261), onnxruntime-gpu `>=1.27.0`
  (#257), plus boto3 / requests / pyyaml floors and standardized Dependabot grouping (#264).
- **build(bake): warm the runtime rebuild by pre-pulling the seed into the snapshot (#263).**
- (Backfilled 2026-07-28 from the commit log and from the v1.0.2 release notes, which are where the
  reason this tag must not be pinned was recorded. This tag has no GitHub release of its own, and
  the row was missing from this file.)

## [1.0.0] -- 2026-07-13

**First stable release of the GPU render backend.** The clean-room RunPod serverless render engine
(SDXL keyframes -> i2v -> assemble, plus LoRA training) that powers Vivijure Studio v1.0.0. The core
render contract (`docs/CONTRACT.md`: bundle in, artifacts out) and the baked-image deploy path are
stable. Ships on top of 0.4.9:

- **fix(verify): promote smoke uses a TRAIN-FREE bundle (#243)** -- the promote self-check no longer
  reuses the verify render bundle, so a promote can't be gated on a training path.
- **docs(deploy):** RunPod v2 `update-endpoint` 500 + template-repoint trick and the account-quota /
  v2-500-masking gotchas are banked in the deploy notes (#246, #247).

Cut as part of the constellation-wide v1.0.0 milestone.

## [0.4.9] -- 2026-07-10

**fix(finish): mux the source audio back after the RIFE finish (#240)**

The finishing pass re-encodes each clip from a rawvideo rgb24 stdin stream, so its output was
always video-only: there was no audio input, mapping, or mux in the module. Before core v0.17.0
this was invisible (audio was laid on later), but since vivijure#595 dialogue shots lipsync FIRST
(lipsync -> rife -> upscale), so MuseTalk muxes the dialogue audio into the clip that reaches
finish and the RIFE re-encode silently discarded it. In prod the sound cut off at the musetalk
shot and stayed dead for the rest of the film, because the silent segment poisons the stream-copy
concat in assemble. Fix: after the interpolation encode, if the source clip carries an audio
stream, mux it back onto the finished clip with a stream copy (ffmpeg -map 0:v -map 1:a? -c copy),
the vivijure-upscale pattern; RIFE keeps wall-clock duration so the audio lines up 1:1. Honest
failure (#245): a mux that fails FAILS the shot with the real error, never silently shipping a
video-only clip when audio was present. An audio-less source is unchanged (no audio step, no
failure).

Version note: this is a src-only release (assemble + push from the pinned runtime base). GHCR is
published through 0.4.8 and a GHCR tag is immutable, so per the #222 drift rule this release takes
the next FREE GHCR semver, 0.4.9. Prod was on :0.4.8 before this promote.

## [0.4.8] -- 2026-07-06

**feat(finish): stamp the #583 param-hash sidecar after the finished clip (#224)**

Ships the producer-side leg of the #583 provenance contract on the GPU backend: after a finished clip
is written to R2, the worker writes an opaque `<outputKey>.hash` sidecar (artifact FIRST, sidecar LAST,
best-effort) carrying the param hash the core passes in `output_hash`. Inert until the core sends
`output_hash` (post the S25 core tag), so deploying this before the core is safe: no `output_hash` in the
input means the worker skips stamping. Rolls up two intervening main commits: docs of the GHCR-vs-git-tag
semver drift rule (#222) and the skyphusion-search corpus-sync notify on push (#223).

Version note: image tags `0.4.5`/`0.4.6`/`0.4.7` were consumed on GHCR by 2026-07-05 snapshot-runner test
dispatch builds (fc#377) and were never git-tagged or promoted; per the #222 drift rule this release takes
the next FREE GHCR semver, `0.4.8`. Prod was on `:0.4.4` before this promote.

## [0.4.5] -- 2026-07-05

**ci(release): point release.yml at the snapshot runner vivijure-bake-snap (fc#377)**

The release bake now runs on the enterprise SNAPSHOT runner (label `vivijure-bake-snap`, Vivijure Bake
enterprise group, org-scoped to skyphusion-labs, allow-public). Its custom image
(`vivijure-bake-snapshot`) ships the pinned runtime base pre-pulled into the docker daemon store, so the
`FROM` is a local cache hit and the bake is assemble+push only, no cold ~87 GB base pull. Accelerator
only; the built image is byte-for-byte what the prior runner produced.

- **Flip:** `runs-on: vivijure-bake` -> `runs-on: vivijure-bake-snap`; header + runs-on comments updated
  to reflect that the snapshot runner is live and that the access control is the label + the `backend-v*`
  tag ruleset (`backend-release-tags`, id 18524678) -- a fork PR cannot create a `backend-v*` tag, so no
  untrusted code reaches the runner.
- **No re-bake:** the runtime image is unchanged; this release consumes the existing snapshot. A stale or
  absent snapshot degrades to a one-time base pull -- SLOW, never WRONG.

## [0.4.4] -- 2026-07-04

**fix(bake): scrub the hf_hub tree cache so the offline SDXL keyframe load stops failing (#206)**

The first live SDXL keyframe job on `:0.4.3` failed loud with huggingface_hub `IncompleteSnapshotError` for `SG161222/RealVisXL_V5.0`: the cached snapshot was reported "missing" the two root single-file checkpoints (`RealVisXL_V5.0_fp16.safetensors` / `RealVisXL_V5.0_fp32.safetensors`), which the diffusers folder-layout `from_pretrained` never loads and which the curated bake deliberately omits (the fp32 one, 13.8 GB, is over the 10 GB GHCR per-layer ceiling and cannot be baked at all). An over-strict completeness gate, NOT a partial bake; every file the pipeline actually loads is present. Confirmed against the shipped `:0.4.3` artifact (manifest `sha256:ac43e7a8...`).

- **Root cause:** `huggingface_hub` (unpinned, transitive) drifted up to 1.x, which added an offline snapshot-completeness check that fires only when a repo tree listing is cached. The build-time online config bake (`bake_hf_configs.py`) writes `hub/<repo>/trees/<commit>.json` listing ALL siblings; baked into the image, it makes the render-time offline load hard-fail against the curated subset. Only the baked path breaks -- the rclone-staged R2 mirror and the local-gpu doors carry no tree cache.
- **Fix:** `bake_hf_configs.py` scrubs every `<repo>/trees/` after the online config bake, so the offline check reverts to its documented no-op path (folder-as-is, the pre-1.x behaviour). Protects the Wan I2V config bake too.
- **Fail-at-bake gate:** `bake_layers.py` gains `assert-no-tree-cache`, wired into the Dockerfile sentinel `RUN` -- the bake FAILS if any tree listing survives, so this class dies at build, never at the first prod job. (A "loaded-files-present" assertion would have PASSED this exact failure, since every loaded file was present.)
- **Pin:** `huggingface_hub==1.22.0` (was transitive/unpinned) stops silent behaviour drift; in-bounds for `transformers 5.10.2` (`<2.0,>=1.5.0`) and `diffusers 0.38.0` (`>=0.34.0`).

## [0.4.3] -- 2026-07-03

**feat(observability): mirror a job-done callback rejection into the run-scoped R2 channel (#90)**

The `job_done_error{status:400}` on a successful job was printed ONLY to serverless-worker stdout, which is not retrievable via GraphQL / runpodctl / MCP, so the actual 400 could never be inspected. This makes the rejection observable in-band before we root-cause it.

- **Run-scoped R2 mirror of the rejection (#90):** the SDK `_transmit` patch that catches a >=400 job-done post now also writes the record (`status`, `body`, `content_type`, `url`, `posted_status`) to a NDJSON object colocated with the render's other progress objects (`renders/<project>/progress/<job_id>.job-done-errors.ndjson`), a DISTINCT object that never clobbers the live progress snapshot the control plane polls. The patch has no run context, so the handler registers the live R2 store + project/job id at job start (`harness/job_done_diag`).
- **Purely additive:** best-effort, never retries the post, never changes the job outcome, and every write is swallowed so a diagnostic path cannot become a worker failure mode. The `url` query string is stripped before it lands in the channel. The existing stdout `@event job_done_error` is unchanged (it gains the `url` field).

## [0.4.2] -- 2026-07-03

**fix(release-gate): the live pod-staging gate is passable + correct end to end; verify/bake/harness hardening**

The batch that makes the automated verify gate render, promote, and reap reliably, plus the worker/harness
correctness fixes it surfaced. A clean re-bake off `main`; ships to prod through the gate itself.

Release gate:
- **Promote repins the endpoint TEMPLATE, not the endpoint (#194):** RunPod rejects an endpoint-level env PATCH (400); the promote now PATCHes the template (env lives on the template) and reads back the pinned image.
- **Verify pod reads R2 via `RUNPOD_SECRET` references, not plaintext env (#184, #195):** the pod-side R2 creds are secret references, never rendered into pod env or logs.
- **DC / cache affinity to skip the ~87GB cold pull (#187):** an ordered SECURE data-center affinity pins the verify pod to a machine warm on the image's weight layers, then falls back UNPINNED so a capacity miss never fails the gate.
- **Make the gate passable (#190, #192, #193):** preflight cleans every injected pod-env value + the R2 shape / key-names; the pod TTL/timeout and `max_polls` follow the measured cold-pull (#186, #188); the smoke bundle `Verify_Smoke` is the default and the verify sharpness baseline is calibrated to 75.0.

Verify + bake:
- **Authoritative first channel write -- bad R2 creds fail LOUD (#189):** the first structured `@event` write is authoritative, so a bad/missing R2 credential fails loudly at the start instead of a silent empty prefix.
- **Exact-pin torch / torchvision / torchaudio in the bake (#191):** kills wildcard drift so a re-bake is byte-reproducible.

Worker + harness:
- **Quiesce progress mirrors before the terminal result post (#90, #197):** progress mirrors are quiesced before the terminal COMPLETED/FAILED post, so a late `IN_PROGRESS` can never race and misattribute after the job is done.
- **Per-artifact R2 state replaces the shared state tarball (#112, #196):** each artifact carries its own R2 state, removing the shared-tarball contention.
- **Retire the false-alarm `tier_mismatch` warn (#163):** demoted to an informational `plan_tier` trace.
- **i2v_clip conformance guard vs a shared golden (#129):** the i2v_clip contract is guarded against a shared golden so a drift is caught in CI.

## [0.4.1] -- 2026-07-02

**fix(verify): the pod-staging gate renders end to end (pipeline registration + loud fatal + cold-pull TTL)**

Fixes for the gaps the first live gate runs on `:0.4.0` surfaced. The verify emitter and R2 channel were proven healthy on an H200 pod (`gpu_probe` with `torch_cuda`/`kernel_ok`/`vj_baked` true, `summary.json` + `events.ndjson` written in ~3s); these close what was left so the gate renders a draft clip end to end and promotes.

- **Register the GPU pipeline for the verify render (#183):** `python -m vivijure_backend.verify` is a DIFFERENT entrypoint than the serverless worker and never triggered the per-job pipeline registration the worker does, so `_pod_draft_render`'s handler call died in ~3s with "no GPU Pipeline registered". The verify render now routes through `worker.handler` (the same production serverless entrypoint), which registers the pipeline before delegating, so the verify path is byte-identical to a real render and cannot drift from it.
- **No silent pre-emitter death; loud `verify_fatal {stage, missing}` (#180, #182):** a missing/misnamed R2 config or an absent `VJ_VERIFY_RUN_ID` used to kill `main()` before the emitter existed -- a silent empty prefix. `main()` now emits a structured terminal `verify_fatal` naming the missing env vars to stdout before exiting non-zero, so a launch-side env mistake is a one-line diagnosis, never a 30-minute blind hang.
- **Cold-pull TTL headroom + pod-state timing (#181):** the ~87GB baked image's cold pull could eat the pod TTL before verify started; the TTL is raised to 3000s and pull/boot time is measured on the structured state channel, so a slow cold start is visible instead of a mystery empty prefix.

## [0.4.0] -- 2026-07-02

**feat(release-gate): the automated pod-staging verify gate goes LIVE end to end (verify @event channel + live SECURE RunPod pod client + promote)**

The release doctrine (pod = staging, serverless = production; `docs/release-gate.md`) is now ENFORCED, replacing the dry-run-only harness. A pushed `backend-v*` tag builds the baked image; the verify gate then spins a SECURE GPU pod on it, runs a draft render that emits a machine-readable structured `@event` channel, asserts on it, and promotes the image onto the production serverless endpoint ONLY on a green verify before tearing the pod down (spend proven to zero). This is the FIRST image to carry the verify emitter.

- **Pod-side verify `@event` emitter (#175):** `python -m vivijure_backend.verify`, armed only by `VJ_VERIFY` (a hard no-op otherwise, zero effect on a normal render). Emits `gpu_probe` / `first_frame` / `sharpness` / `complete` (+ `error`) to a run-scoped R2 channel (`verify/<run_id>/summary.json` + `events.ndjson`), mirrored byte-identically to stdout as a fallback transport. The emitted payloads are exactly what the harness reader asserts on, so `events_from_summary()` feeds straight into `runpod_verify.evaluate` with no prose parsing. 31 CPU tests including a cross-module contract test.
- **Live SECURE RunPod pod client (#173, #174):** the previously-stubbed pod-lifecycle seam is implemented, SECURE cloud only (never COMMUNITY), with SECURE resolved from the `get_gpu(id)` detail and the real GeForce GPU id (both found by live smoke). up / down / list, hard TTL auto-stop, teardown-confirmed-zero.
- **Live gate + promote wiring (#176):** `runpod_verify` grows the real up|verify|promote|down path (the old workflow advertised a phantom gate it could never run); promote pins the verified image onto the prod endpoint under maintainer authorization. The gate launches the verify entrypoint on the pod and paces its polls (#178).
- **RunPod key hygiene (#177):** the API key is normalised (strip whitespace + matched quotes) with a no-leak shape diagnostic, so a mis-shaped secret fails with a safe message, never an echoed value.

**perf(finish): NVENC-encode the finish stage + stream interpolation (#172)**

The finish pass hardware-encodes via NVENC and streams RIFE interpolation frame-by-frame instead of buffering the whole clip, bounding host RAM on long shots.

**fix(harness): never mirror a terminal snapshot through RunPod progress_update (F17, #171)**

F17 root cause: the SDK's `progress_update` posts `IN_PROGRESS` from a daemon thread, racing and clobbering the handler's terminal FAILED/COMPLETED result. The progress hook now drops terminal snapshots, so RunPod's terminal status comes only from the handler's own return/raise.

**fix(pipeline): silent degrades become real errors; draft default; one slug; job-key pinning (#170)**

A finish/polish failure now fails the render with the real per-shot error instead of silently shipping a raw clip with `applied=[]`; the default quality tier is `draft`; one canonical project slug; job-supplied R2 keys are pinned to the render key map before any store I/O.

**feat(derisk): userspace full-block egress guard for the offline-correctness proof (#17, #161)**

A userspace egress guard that fully blocks network egress, so the baked-image offline-correctness proof (weights come from the image, never R2/HF) is enforced, not merely asserted.

Plus docs/legal hardening: outsider-runnable deploy + mirrored constellation map (#166), `SUPPORT.md` + security@ routing (#168), canonical verbatim AGPL-3.0 `LICENSE` (#165), `NOTICE` copyright holder + uniform README license footer (#167), and the 0.3.x baked-image line documented (#164).

## [0.3.3] -- 2026-06-30

**feat(bake): facexlib baked offline + FIRST BAKED IMAGE CONFIRMED IN PRODUCTION (#158)**

The first vivijure-backend image with model weights baked into the image layers (no R2 cold-pull), deployed to the production serverless endpoint (`t9wcvlxh8rc5la`) and confirmed end-to-end on both production GPU arches through the real serverless handler. Bakes the facexlib detection + parsing weights so GFPGAN face-restore runs fully offline. Full confirmation record in `docs/serverless-0.3.3.md`.

- **Baked, proven:** all 28 confirmation renders emitted `mirror_done { pulled: false }` -- the weights came from the image, not R2.
- **Kernels:** the prebuilt `cu128` wheels carry `sm_90` / `sm_100` / `sm_120`; the `get_arch_list()` build-gate enforces all three.
- **28/28 clean serverless renders, ZERO errors:** H200 (`sm_90`) 17/17 under concurrent load; B200 (`sm_100`) 11/11. True-cold `sm_100` i2v JIT measured for the first time (steady ~3.45 s/step, faster than H200's ~5.0 s/step).
- Production serverless tier confirmed as **H200 | B200 only**.

## [0.3.2] -- 2026-06-30

**fix(models): the baked path never references an un-baked repo + baked_probe hardening (#155, #156)**

- **i2v offline-load fix** (`models.py`): the baked fast-path loaded the `-fp8` i2v repo, which the bf16 bake never staged, so `from_pretrained` failed offline with `LocalEntryNotFoundError`. `_select_i2v_weights()` now gates the fp8 fast-path on the fp8 repo actually being cached, else loads the baked bf16 repo offline (bf16 throughout; fp8 buys nothing on the 96-141 GB pool).
- **baked_probe hardening** (`vj_derisk.py`): asserts the exact runtime repo is cached before the CUDA kernel ($0, pre-GPU); CI guards the driver sha + marker to kill the drift class permanently.

## [0.3.1] -- 2026-06-29

**feat(bake): first REAL bf16 baked image (#14)**

First image built from the staged bf16 weight seed (~105 GB, 15 multi-GB layers): the #138 gates pass on real weights and `.vj-baked` is stamped `precision=bf16`. 3-arch coverage rides on the prebuilt `cu128` wheels; the build-time proof is `get_arch_list() == {sm_90, sm_100, sm_120}`. Supersedes the burned `:0.3.0`.

## [0.3.0] -- 2026-06-29

**feat(bake): bake pipeline (#127) -- BURNED, do not pin**

First bake attempt. The seed prefix was empty, so the bake produced a hollow image (config stubs, zero weight shards) while still writing `.vj-baked` -- caught by the #4 manual de-risk. The lying sentinel is fixed in `:0.3.1`+ by the #138 empty-bake gate (assert-weights gates the sentinel write via `&&`). Do not pin `:0.3.0`.

## [0.2.16] -- 2026-06-14

**fix(harness): worker hygiene -- Sprint 1 batch (#24, #25, #26, #46)**

- **Workdir disk leak** (`handler.py`): `mkdtemp` job workdir is now cleaned up in a `try/finally` on all exit paths; previously leaked GBs of keyframes/clips/bundles per job on warm workers.
- **Checkpoint orphans** (`lora_train.py`): `save_every` checkpoint subdirs are removed after the final adapter saves; no longer accumulate per training slot.
- **R2 truncation detection** (`r2.py`): `get_file` now calls `head_object` before download and verifies `stat().st_size == ContentLength`; a truncated `.safetensors` now raises immediately instead of loading silently.
- **Sentinel versioning** (`models_mirror.py`): both sentinels write `VJ_MODEL_VERSION` (default `"1"`) instead of `"ok"`; a version mismatch forces a re-mirror; no-creds skip warns loudly when the HF cache looks empty.
- **ffprobe error surfacing** (`assemble.py`): `probe_duration` and `probe_has_audio` now log ffprobe's stderr and returncode on failure instead of returning silent `None`/`False`.
- **Stray-clip warning** (`assemble.py`): `order_for_storyboard` logs dropped clip IDs when a shot_id is absent from the storyboard.
- **Additive i2v negative prompt** (`pipeline.py`): a config `negative_prompt` is now prepended to the engine's base anti-static guard rather than replacing it.
- **ip_adapter_scale forwarding** (`pipeline.py`): `keyframe_params_from` now uses `kc.ip_adapter_scale` (the top-level single-char field, default 0.65) instead of `mc.ip_adapter_scale_per_slot` (0.7 multi-char default). Closes the KEYFRAME_UNMAPPED note in the completeness guard.
- **Empty-prompt validation** (`orchestrator.py`): scenes with blank/whitespace prompts are flagged at validate time.
- **Cast membership validation** (`orchestrator.py` + `handler.py`): `validate()` accepts an optional `cast` and checks every `use_characters` slot exists in the cast registry; post-plan ref check in `run_job` ensures slots the plan will train have at least one reference image.
- **Cast-missing HarnessError** (`pipeline.py`): `execute()` raises `HarnessError` instead of silently skipping a slot whose character is absent from the cast.
- **i2v_tier mismatch warning** (`pipeline.py`): emits a `tier_mismatch` progress event when the running GPU card doesn't match the planned i2v tier.
- Test bundles updated to include fake ref images; 5 new tests. 324 tests total.

## [0.2.28] -- 2026-06-27

**security(identity): complete the anti-SaaS identity strip (#292) backend-side (#122).** The
control plane stripped the submitter-identity primitive (the studio is single-operator: it sends no
`user_email` and serves `/api/artifact` by key with no per-row ownership check), but this backend
still parsed `user_email` from the job input and stamped it as `customMetadata.user_email` on every
uploaded artifact, with docs describing an ownership-gated `/api/artifact` that does not exist.
Removed the `user_email` field + parsing (`contract.py`) and the artifact owner-stamping
(`harness/handler.py`); an injected `user_email` is now dropped, never persisted, so a stripped
identity cannot resurface as R2 object metadata. The generic `metadata` passthrough on the store is
retained (no caller) with a corrected, identity-free docstring. Aligned `SECURITY.md`,
`docs/contract.md`, `docs/operations.md`. The stamping tests are replaced by a regression test that
proves an injected identity is ignored. Full suite green.

**perf(mirror): lazy-load the heavy i2v weights to cut cold-start startup ~5x.** The cold-start
model mirror pulled the entire `r2:<bucket>/models/hf-cache` (~257 GiB after the prior skips) on
every worker, but a keyframe/preview worker (the common cheap op) loads none of the i2v stack. Now:

- `Wan2.2-I2V-A14B` (117.5 GiB) is in `DEFAULT_SKIP_REPOS` (out of the cold-start pull) and mirrored
  LAZILY by the new `ensure_i2v_models()` on the first `i2v_pipeline()` call (its own sentinel,
  idempotent). Keyframe/preview workers never pay for it.
- Two stray repos that nothing in the spec loads -- `stable-diffusion-xl-base-1.0` (57.6 GiB) and
  `sdxl-turbo` (32 GiB) -- are added to the cold-start skip (dead weight, ~90 GiB).
- Net: a keyframe/preview cold start drops from ~257 GiB to ~50 GiB (the SDXL stack); the first
  animation job pays the Wan pull once. R2 storage is unchanged (these are pull-time excludes only).

Also seeds the two needed-but-missing models into the R2 mirror so the worker no longer depends on a
live HF fetch: `Hyper-SDXL-8steps-lora.safetensors` (keyframe distill, cold-start) and
`lightx2v/Wan2.2-Lightning` (i2v distill, lazy).

Code: `harness/models_mirror.py` (expanded `DEFAULT_SKIP_REPOS`, `I2V_LAZY_REPOS`,
`ensure_i2v_models`), `models.py` (`i2v_pipeline` calls the lazy pull),
`tests/test_models_mirror.py` (cold-start skip + lazy early-returns). Full suite green.
**fix(instantid): make the InstantID path actually render (pod-validated 2026-06-10).** The 0.1.17
wiring loaded but produced noise; debugged live on an A6000 and validated end to end (a clean,
identity-matched anime keyframe). The fixes:

- The `ip_adapter` weights are keyed by each layer's index over ALL attn processors (self + cross),
  not the cross-only subset: load each cross-attention's `to_k_ip`/`to_v_ip` by its overall index.
  (The previous ModuleList-over-cross-only compressed the indices and mismatched the per-layer
  hidden sizes.)
- Identity tokens reach the UNet through a SIDE channel (`proc.id_embeds`), NOT concatenated onto the
  prompt embeds: concatenation corrupted the ControlNet, which never saw appended tokens.
- Dropped the InstantID IdentityNet (face-keypoints) ControlNet. It must receive the face embedding
  as its `encoder_hidden_states` (a custom unet+controlnet denoise loop the stock pipeline does not
  expose); fed the text embeds it produces noise, and the face-pose lock is undesirable for a
  scene-posed keyframe. `instantid_pipeline` is now a plain SDXL pipe with the face IP-Adapter only;
  the IdentityNet is documented future work (`draw_kps` kept for it).
- `face_analyzer` auto-flattens the antelopev2 pack (the zip extracts one level too deep) with a
  download-retry.
- `analyze_face` accepts a PIL Image (what `_ref_images` returns), not only a path.

Identity transfers best in-domain (anime ref + anime base, or a real-face ref); insightface
antelopev2 is real-face-trained, so an anime ref is its hardest case (it still works). In production
the cast LoRA stacks on top to lock hair/outfit while InstantID locks the face.

Code: `instantid.py` (index-based IP-attn load, side-channel `IPAttnProcessor`, PIL `analyze_face`),
`models.py` (`instantid_pipeline` -> plain SDXL, `face_analyzer` flatten+retry), `keyframe.py`
(`_render_instantid` side channel, no ControlNet). CPU suite green; validated on an A6000.

## backend-v0.1.17

**InstantID single-character face identity (the consistent-identity lever).** Wires the
scaffolded-but-dead InstantID path into the GPU keyframe stage: for a single-character shot with
`identity_method="instantid"` and a reference face, the keyframe now uses insightface (antelopev2)
to extract a face embedding + 5 keypoints, projects the embedding to identity tokens through a
Resampler image-projection, injects them via IP cross-attention processors, and conditions a
second (InstantID) ControlNet on the keypoints to pin face structure. Built as its own
`instantid_pipeline()` on a fresh SDXL base (no entanglement with the shared keyframe pipe's
per-scene LoRA / IP-Adapter state), sharing the few-step distill attach so single-char drafts stay
cheap. Multi-character shots and the default IP-Adapter single path are untouched. The face-keypoint
geometry and face selection are pure + unit-tested; the model construction, attn-processor wiring,
and per-render embed-concat defer their imports and are GPU-validation-pending (the parts to eyeball:
the ip-adapter.bin key mapping, the identity-token concat, and the insightface antelopev2 path).
Clean-room: built from the published InstantID architecture + diffusers interfaces, no prior pipeline.

Code: new `instantid.py` (Resampler image-proj, IP attn processor, kps drawing, face analysis),
`models.py` (`face_analyzer`, `instantid_pipeline`, `_attach_keyframe_distill` shared with
`keyframe_pipeline`), `keyframe.py` (`_render_instantid` + the single-char InstantID branch),
`pipeline.py` (thread `identity_method` + instantid scales), `KeyframeParams` fields,
`deploy/requirements.txt` (insightface + onnxruntime-gpu), `tests/test_instantid.py` +
`tests/test_pipeline.py`. CPU suite green.

## backend-v0.1.16

**Wire the few-step keyframe distill into the GPU path (the speed lever).** The tier config
already asked draft/standard for a 4/8-step, cfg=0, DDIM-trailing keyframe, but the pipeline
never loaded the Hyper-SD distill LoRA, never set the scheduler, and the `few_step` flag was
dead, so draft previews ran few-step *without* the LoRA that makes few-step work (degraded
stills). Now the ModelServer loads the Hyper-SDXL distill LoRA as a persistent base adapter
("distill"), `keyframe._bind_loras` gates its weight on the tier (1.0 on draft/standard, 0.0
on final, so one warm pipe serves every tier with no reload), and `keyframe._apply_scheduler`
pins DDIM-trailing for the few-step path and restores a full-step solver for final. The final
tier is untouched; this speeds up previews (the most-repeated op). Validated green on a pod
(2026-06-10): a draft keyframes-only render of all 10 neon_halflife shots came out sharp at
4-step, including the multi-character frames; the feared "8-step LoRA at 4-step draft = soft"
did not happen, so it ships as-is.

Code: `models.py` (ModelSpec.weight_name; load the distill LoRA in `keyframe_pipeline`),
`keyframe.py` (DISTILL_ADAPTER, `KeyframeParams.scheduler`, `_apply_scheduler`, few-step gating
in `_bind_loras`), `pipeline.py` (thread `scheduler` through `keyframe_params_from`),
`tests/test_keyframe.py` + `tests/test_pipeline.py` (distill-weight + scheduler mapping).
Full suite green.

Public-readiness housekeeping ahead of open-sourcing the repository: added
`CONTRIBUTING.md` (clean-room / independence-protective posture, house rules, DCO),
`SECURITY.md` (private vulnerability reporting + the render-backend security boundary),
`CODE_OF_CONDUCT.md`, and this `CHANGELOG.md`. Corrected the README architecture table to
reflect the shipped pipeline, and removed em-dashes from the README to match the house
style.

## backend-v0.1.15

Multi-character pose: wire the OpenPose ControlNet so a 2+ character shot plants two
distinct bodies instead of a blended one (`keyframe.py`).

## backend-v0.1.14

Stamp `user_email` on uploaded artifacts so the control plane's `/api/artifact` ownership
check can serve them back (`harness`).

## backend-v0.1.13

Fail loud when a staged LoRA registers no adapter, and make reused-LoRA injection
fail-fast too, so a silent no-op never ships a render without the intended character
(`keyframe.py`, `pipeline.py`).

## backend-v0.1.12

First-class `preview` action for keyframes-only renders: the orchestrator short-circuits
after the SDXL pass when only keyframes are requested (`orchestrator.py`).

## backend-v0.1.11

Cache both Wan 2.2 MoE experts to clear the step-12 hit-cliff in image-to-video
(`i2v.py`).

## backend-v0.1.10

Use the matched `enable_cache` / `disable_cache` pair for the i2v feature cache (`i2v.py`).

## backend-v0.1.2 -- backend-v0.1.9

Earlier pipeline build-out: the structured render progress channel (R2-backed), the
feature-cache denoise accelerator, reused-LoRA staging from R2 so warm workers skip
retraining, the R2-mirror model loader, and the AGPL-3.0-only license. See the git tags
for the per-release detail.
