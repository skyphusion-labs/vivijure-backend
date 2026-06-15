# Releases -- vivijure-backend

Render backend for RunPod serverless. A release is an annotated git tag
`backend-v<semver>` **pushed to origin**; the fleet Jenkins (tag discovery) builds and
pushes a Docker image to `ghcr.io/skyphusion-labs/vivijure-backend:<semver>` (the image
tag drops the `backend-v` prefix).

> **Lesson (2026-06-12):** the release step MUST push tags to origin. See the 0.2.1-0.2.3
> gap below -- those tags were cut on mindcrimes local clone, never pushed, and lost when
> the box was released.

| git tag | GHCR image | source commit | built | notes |
|---|---|---|---|---|
| backend-v0.2.22 | 0.2.22 | 2479397 | (pending) | feat(models-mirror): read weights from a preloaded per-datacenter RunPod network volume (#55 Phase C, PRs #91 design + #92 code). VJ_VOLUME_ROOT: if a mounted volume carries BOTH the base + i2v sentinels at the current VJ_MODEL_VERSION, repoint HF_HOME/VJ_MODELS_ROOT and skip the R2 mirror (~0s staging, no idle GPU, no egress contention); any miss (unset / partial preload / version mismatch / probe error) falls back to the R2 mirror wholesale (read-only volume, never a SPOF). VJ_MIRROR_JITTER_SEC (default 0): de-stagger R2 egress on fan-out fallback. deploy/dc_availability.py: read-only RunPod gpuAvailability query for the H200+ DC allow-list. Inert unless the new envs are set, so safe across the fleet. 354 passed, 3 skipped. US-NC-1 volume preloaded + verified; read-throughput validation next. |
|---|---|---|---|---|
| backend-v0.2.21 | 0.2.21 | e59ec7e | (pending) | feat(harness): standalone i2v_clip action (per-shot image-to-video) (issue #87, PR #88) -- backend half of studio #81. A sibling job type (like finish_clip) that animates one keyframe into one clip via Wan2.2-I2V: run_i2v_clip_job fetches the keyframe from R2, builds I2VParams from the typed I2VConfig over the quality-tier baseline (clamping + distill/feature-cache invariant enforced), snaps num_frames to 4k+1, threads per-shot overrides (seed/flow_shift/height/width/negative_prompt, additive negative), and uploads to renders/<project>/clips/<shot>_i2v.mp4. worker.py skips build_pipeline for i2v_clip but KEEPS the i2v prefetch (it needs the Wan models). Lets the studio motion module do keyframe -> i2v_clip per shot -> finish_clip -> assemble. Documented in docs/contract.md. 344 passed, 3 skipped. |
|---|---|---|---|---|
| backend-v0.2.20 | 0.2.20 | 4131e81 | (pending) | feat(models-mirror): structured cold-start staging telemetry (#55, Sprint 4 Phase A, PRs #85). ensure_models emits @event mirror_complete (per-leg + total seconds, mirrored bytes, derived throughput_mbps) and @event mirror_skipped {reason} on warm/no-creds returns; ensure_i2v_models emits @event i2v_mirror_complete for the lazy Wan pull. Makes cold-start staging cost (the open #55 problem) readable from logs instead of SSH snapshots -- the data Phase B needs to weigh bake vs pre-warm vs stage. Pure best-effort helpers (_timed/_dir_bytes/_mirror_event/_skip_event); telemetry wrapped so a _dir_bytes walk error never fails a good mirror (review hardening, f4994be). No render-path change. 334 passed, 3 skipped. |
|---|---|---|---|---|
| backend-v0.2.19 | 0.2.19 | fa61ec3 | (pending) | fix(instantid): remove instantid_controlnet_scale phantom knob + draw_kps dead code (#23, PR #82) -- IdentityNet ControlNet never wired; IP-Adapter face embedding path unchanged. fix(worker): capture job-done 4xx response body as @event job_done_error (#65, PR #83) -- surfaces WHY RunPod rejects the callback so the root cause can be fixed. 333 passed, 3 skipped. |
|---|---|---|---|---|
| backend-v0.2.18 | 0.2.18 | 277a718 | (pending) | fix(harness): gate i2v prefetch on action != finish_clip (issue #76, PR #80) -- prefetch was unconditional before the finish_clip dispatch; on a cold worker the Wan2.2 mirror thread ran long after the finish job returned, keeping the job stuck at IN_PROGRESS. One-line guard mirrors the existing action check. |
|---|---|---|---|---|
| backend-v0.2.17 | 0.2.17 | 01f18c9 | (pending) | fix: Sprint 2 batch -- LoRA training correctness (#27, PR #77): batch_size honored (real batching), steps scale by ref count (max(50, max_steps*n//5)), random_flip updates time_ids crop_left, lora_alpha knob added, save_every checkpoints returned in checkpoint_dirs not deleted; dead-code cleanup (#28, PR #75): params_for removed, RenderConfig.from_dict added, dit_quant rename, Scene.from_dict target_seconds derivation, Storyboard dedup, MultiCharConfig.regional removed; test coverage (#29, PR #77): _loss_target both prediction types, _time_ids shape, effective_steps formula, R2 arcname=. invariant; finish_clip job-done 400 (#76, PR #78): worker.handler skips build_pipeline for finish_clip (eliminates spurious i2v prefetch on finish pods), run_finish_job return trimmed to pointer dict. 333 passed, 3 skipped (torch-only). |
|---|---|---|---|---|
| backend-v0.2.16 | 0.2.16 | 0ea8679 | (pending) | fix: Sprint 1 batch -- worker hygiene (#26, PR #69): mkdtemp workdir disk leak, save_every checkpoint orphans, R2 download truncation detection, sentinel versioning; assemble robustness + ip_adapter_scale forwarding (#25/#46, PR #70): ffprobe stderr surfacing, stray-clip warning, additive i2v negative prompt, ip_adapter_scale forwarding fix; planner accuracy + validation (#24, PR #71): empty-prompt validate, cast membership + refs checks, cast-missing HarnessError, i2v_tier mismatch warning, cost model comment. 324 tests. |
|---|---|---|---|---|
| backend-v0.2.15 | 0.2.15 | 58bf277 | (pending) | feat+fix: standalone finish_clip action for the finish-rife module (aaa2e51); fix reused/inject keyframes now reported in render result -- callers no longer get keyframes:[] on re-runs (Closes #67, 632f2ca). 25 pipeline tests. |
|---|---|---|---|---|
| backend-v0.2.14 | 0.2.14 | 82874fa | (pending) | fix(worker): rewrite job-done content-type patch -- drop aiohttp_retry dep, use client_session.post(json=...) so aiohttp sets Content-Type: application/json natively; replace silent except-pass with @event patch_applied / patch_failed logging. Closes #65. 319 tests. |
|---|---|---|---|---|
| backend-v0.2.13 | 0.2.13 | da6b6c8 | 2026-06-14 (fleet) | fix(offline,worker): local_files_only=True on all remaining from_pretrained render-pipeline + lora_train calls (Closes #66); patch RunPod SDK _transmit to send application/json instead of x-www-form-urlencoded so job-done callback no longer 400s (Closes #65). 319 tests. Verified: v0213-verify-00 standard render, LoRA training (2 slots) + 6 keyframes + i2v (all 6 shots) + assemble + upload clean. #66 CONFIRMED fixed (zero offline probe warnings). #65 patch did NOT take effect (SDK still 400s on job-done; output available via streaming channel, not callback -- see issue for follow-up). |
|---|---|---|---|---|
| backend-v0.2.12 | 0.2.12 | e8bf45b | 2026-06-14 (fleet) | fix(models): Lightning distill probes 6+7 -- add weight_name to I2V_DISTILL ModelSpec (confirmed from R2); pass it to load_lora_weights (probe 6: offline scan) and hf_hub_download (probe 7: wrong placeholder filename). Closes #64. 319 tests. Verified: v0212-00 6-shot standard render, zero offline probe failures end-to-end. |
|---|---|---|---|---|
| backend-v0.2.11 | 0.2.11 | 49c0dbc | 2026-06-14 (fleet) | fix(lora_train): UNet + DDPMScheduler from_pretrained model_info probe under HF_HUB_OFFLINE=1 -- add local_files_only=True; 5th offline probe, root cause of the failed verify render. 319 tests. |
|---|---|---|---|---|
| backend-v0.2.10 | 0.2.10 | d2f6b4e | 2026-06-14 (fleet) | fix(mirror): i2v prefetch self-join -- ensure_i2v_models skips join() when called from the prefetch thread itself (RuntimeError "cannot join current thread"); prefetch now overlaps LoRA training as intended. 319 tests. |
|---|---|---|---|---|
| backend-v0.2.9 | 0.2.9 | b2ef825 | 2026-06-14 (fleet) | fix(mirror): write .no_exist stubs at R2 revision not build-time HF revision -- stubs now written inside ensure_models() after R2 mirror so refs/main holds the correct (R2-seeded) revision. Fixes HF_HUB_OFFLINE=1 probe failures on cold start. 318 tests. |
|---|---|---|---|---|
| backend-v0.2.8 | 0.2.8 | 7aadeb2 | 2026-06-14 (fleet) | feat: eager Wan I2V prefetch + --multi-thread-streams (#61); HF offline support -- fix 4 HF Hub probes (#62). Was superseded same day by 0.2.9 (build-time stub revision mismatch). 318 tests. |
|---|---|---|---|---|
| backend-v0.2.7 | 0.2.7 | aee1ca9 | 2026-06-13 (fleet) | FBC context fallback: standard/final tier i2v retries uncached on ValueError("No context is set") from diffusers FirstBlockCache hook. Caught in load test. 310 tests. (#57) |
|---|---|---|---|---|
| backend-v0.2.6 | 0.2.6 | ebbf858 | 2026-06-13 (fleet) | Hash-gate keyframe cache invalidation; bump lora_scale_per_slot default 0.3->0.7 (fixes dual-shot dark-blob output). 310 tests. (#53) |
|---|---|---|---|---|
| backend-v0.2.5 | 0.2.5 | 2e60829 | 2026-06-13 (fleet) | Re-land orphaned #37 finishing-stage deps (gfpgan/basicsr/facexlib + RIFE vendor); fix RIFE load_model path; CI import smoke gate (#51). |
|---|---|---|---|---|
| backend-v0.2.4 | 0.2.4 | 997568a | 2026-06-12 (fleet) | First release tagged AND pushed to origin post-mindcrime. Render-hardening batch (#40-#45) + deploy fixes (#34/#35/#38). |
| backend-v0.2.3 | 0.2.3 | ~8919c79 (#33)* | 2026-06-12 09:21Z (mindcrime) | **git tag LOST** (cut local, never pushed; box released). Pipeline iteration. Was the image running on RunPod. |
| backend-v0.2.2 | 0.2.2 | ~8919c79 (#33)* | 2026-06-12 08:08Z (mindcrime) | git tag lost (as above). Build iteration. |
| backend-v0.2.1 | 0.2.1 | ~8919c79 (#33)* | 2026-06-12 05:26Z (mindcrime) | git tag lost (as above). Build iteration. |
| backend-v0.2.0 | 0.2.0 | (on origin) | -- | last tag that reached origin before the gap. |

\* Commit inferred from image build-time vs the `main` commit timeline -- the images carry
no `org.opencontainers.image.revision` label. All three predate the #34/#35/#38 deploy
fixes, so they are the #33-era backend; 0.2.4 supersedes them.

## Fix-forward
- Release step pushes tags to origin (the bug behind the 0.2.1-0.2.3 gap).
- Add `org.opencontainers.image.revision=$GIT_SHA` to the Dockerfile (build ARG) so future
  images are self-documenting even if a tag is lost. (fast-follow)
