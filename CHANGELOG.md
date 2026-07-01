# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

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
