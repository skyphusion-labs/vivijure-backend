# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

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
