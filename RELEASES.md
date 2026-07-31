# Releases -- vivijure-backend

Render backend for RunPod serverless. A release is an annotated git tag
`backend-v<semver>` **pushed to origin**; that tag push triggers GitHub Actions
(`.github/workflows/release.yml`), which builds and pushes a Docker image to
`ghcr.io/skyphusion-labs/vivijure-backend:<semver>` (the image tag drops the `backend-v`
prefix). Through backend-v0.2.24 this build ran on the fleet Jenkins; it was ported to
GitHub Actions in backend-v0.2.25 (#107) when Jenkins was decommissioned.

> **Lesson (2026-06-12):** the release step MUST push tags to origin. See the 0.2.1-0.2.3
> gap below -- those tags were cut on mindcrimes local clone, never pushed, and lost when
> the box was released.

## Cutting a release tag: check GHCR first (tags may trail GHCR, never collide)

The GHCR image semver can run AHEAD of the git release tags, and a GHCR tag in our setup is
immutable. Before cutting any `backend-v<X.Y.Z>` release tag, check GHCR for the next FREE semver
and cut the tag there. A git tag MAY trail the GHCR semver; it must NEVER reuse one.

**Why the drift exists.** `release.yml` also runs on `workflow_dispatch` with an explicit `version`
input (see the header of that workflow). The runtime / snapshot bakes (the S19/S20 runner work)
dispatch it against a tagless commit to publish an image at a chosen semver, so GHCR gains
`ghcr.io/skyphusion-labs/vivijure-backend:<X.Y.Z>` tags that have no matching git tag. The GHCR
package semver therefore advances past the `backend-v*` git tags.

**Why a collision is not OK.** A tag push derives `<X.Y.Z>` from the tag name and pushes
`vivijure-backend:<X.Y.Z>` (+ `:latest`). Pushing an `<X.Y.Z>` that already exists on GHCR collides
with / overwrites the published image at that tag. So cutting `backend-v<X.Y.Z>` for a semver already
on GHCR clobbers an existing image. Never do it.

**The check (do this BEFORE `git tag`):**

```bash
gh api "/orgs/skyphusion-labs/packages/container/vivijure-backend/versions" \
  --jq ".[].metadata.container.tags[]" | grep -E "^[0-9]+\.[0-9]+\.[0-9]+$" | sort -V | tail
```

(or the GHCR Packages UI.) Take the highest published semver, pick the next FREE one above it, and
cut `backend-v<that>`. Do NOT assume the next number after the latest git tag is free.

**Snapshot (2026-07-31):** git tag **`backend-v1.0.12`** / GHCR **`:1.0.12`** InstantID face path back on the GPU (#346). Runtime repinned to `runtime-1-bf16-t5`; prod repinned by the verify+promote gate (run 30644063094).

**Snapshot (2026-07-23):** git tag **`backend-v1.0.10`** / GHCR **`:1.0.10`** cast-registry `pretrained_loras` (#332). Src-only assemble; pin RunPod after verify+promote.

**Snapshot (2026-07-23):** git tag **`backend-v1.0.7`** / GHCR **`:1.0.7`** src-only assemble (#324 tenant R2 key binding). Next free semver: re-check GHCR before the next cut.

**Snapshot (2026-07-22):** git tag **`backend-v1.0.6`** / GHCR **`:1.0.6`** (#317 repo_id allowlist). Pin RunPod after verify+promote.

**Snapshot (2026-07-22):** git tag **`backend-v1.0.4`** / GHCR **`:1.0.4`** on overlay runtime
`runtime-1-bf16-t4` (Pillow 12.3.0; cf#178). Do **not** pin `:1.0.3` (tagless dispatch still on t3)
or `:1.0.1` (t2 weight-digest miss). Next free semver: re-check GHCR before the next cut.

**Snapshot (2026-07-15):** git + GHCR at **`1.0.2`** (`backend-v1.0.2`) on overlay runtime
`runtime-1-bf16-t3` (~104 GB shared with `1.0.0`; ~96 MB new). Do **not** pin `1.0.1` (t2
weight-digest miss). Superseded by the 2026-07-22 line above.

**Snapshot (2026-07-05):** git tags trail at `backend-v0.4.4`; GHCR is published through `0.4.7`. The
next release tag is `backend-v0.4.8`, NOT `backend-v0.4.5` (0.4.5 / 0.4.6 / 0.4.7 are taken).
(Historical; superseded by the 2026-07-15 line above.)

## Release contract (#537: the seed -> runtime -> backend chain)

Since #537 the backend image is the top of a three-image chain, so a release does NOT rebuild the
world. What you do depends on WHAT changed:

| What changed | Steps |
|---|---|
| **Toolchain / pip deps only** (safetensors, tokenizers, transformers, …; CUDA/torch/seed unchanged) | Dispatch `runtime-build.yml` with `overlay_from=<prior runtime@digest>` and bump `-t<N>`. Uses `deploy/runtime-overlay.Dockerfile` so weight layers inherit. Then merge the RUNTIME_REF repin PR + tag `backend-v<semver>`. Do **not** full-rebuild for deps-only (t2 lesson: ~101 GB new weight digests broke RunPod). |
| **CUDA / torch / apt / seed change** | Full `runtime-build.yml` (empty `overlay_from`), bump `-t<N>` or model version as appropriate, repin, tag. |
| **App code only** (`src/vivijure_backend`) -- the common case | Push the tag `backend-v<semver>`. `release.yml` builds `deploy/Dockerfile` (`FROM the pinned runtime base` + `COPY src`) and pushes. Only the app layers upload; the runtime + weight layers dedup on GHCR. Fast. |
| **Model weights** (new/changed curated set) | 1. Dispatch `seed-build.yml` (stages R2, rebuilds `vivijure-backend-seed`, bump `model_version`). 2. Repin `SEED_REF_<PREC>` (tag + `@sha256`) in `deploy/runtime.Dockerfile`. 3. Dispatch `runtime-build.yml`. 4. Repin `RUNTIME_REF_<PREC>` in `deploy/Dockerfile`. 5. Push the tag. |
| **Toolchain / deps / CUDA / torch / hf-configs** | 1. Dispatch `runtime-build.yml` (bump `toolchain_version` -> the `-t<N>` tag; NO R2, weights come from the existing seed). 2. Repin `RUNTIME_REF_<PREC>` in `deploy/Dockerfile`. 3. Push the tag. |

Why: the weights + full runtime live in `vivijure-backend-runtime`, pinned by digest; the seed
(`vivijure-backend-seed`) is the only image that stages from R2, and only on a weight change. This
makes a src-only release an assemble-and-push, stages R2 once per weight version, and keeps a
toolchain/CUDA bump a deliberate, revalidated base build (repin, not a tag silently rebuilding
everything). Full architecture: `docs/weights-base-and-snapshots.md`. NOTE (#537 dedup fix): the runtime is a `runtime-*` TAG in the `vivijure-backend` package (not a separate package), so a release `FROM` it dedups same-repo; pinned runtime tags must be retained (untagged digests are GC-bait). The base builds are
`workflow_dispatch` only on `vivijure-bake` (never fork-reachable). Promotion to prod is still the
separate pod-staging verify gate (`docs/release-gate.md`) -- unchanged.

**Re-bake cadence (CVE freshness):** the RUNTIME base is rebuilt monthly (a `runtime-build.yml` cron
floor) plus on demand, because every release inherits its layers so runtime age = shipped CVE posture.
The scheduled re-bake reads the currently shipped pin, rebuilds at the same tag with fresh layers, and
auto-opens a `RUNTIME_REF` repin PR (a human merges). The runner snapshot is event-coupled to each
re-bake (+ a monthly backstop). The SEED is exempt (content-addressed weight data, not software). Full
policy: `docs/weights-base-and-snapshots.md`.

| git tag | GHCR image | source commit | built | notes |
|---|---|---|---|---|
| backend-v1.0.12 | 1.0.12 | 470edd8 | 2026-07-31 (GHA) | **InstantID face path restored to the GPU (#346).** onnxruntime-gpu capped `>=1.21.0,<1.27.0`, CPU-wheel purge/repair, build-time provider gate (#347); two pins that could not resolve on py3.11 (#351); runtime repinned to `runtime-1-bf16-t5@sha256:0f3c9bd6` (#357). Verified by an InstantID render EXECUTED on B200 sm_100, not by the merge; prod repinned by verify+promote run 30644063094. |
| backend-v1.0.10 | 1.0.10 | (tag SHA) | 2026-07-23 (GHA) | **Cast-registry pretrained_loras (#332).** Accept `loras/cast-{id}/…` and `loras/lora-{slug}-…/…`; bundle_key tenancy unchanged. Src-only assemble; pin RunPod after verify+promote. |
| backend-v1.0.7 | 1.0.7 | (tag SHA) | 2026-07-23 (GHA) | **KF3 tenant R2 key binding (#324).** Harness scopes `bundle_key` / `keyframe_key` / `clip_key` / `pretrained_loras` to job project slug. Src-only assemble; no runtime rebuild. |
| backend-v1.0.6 | 1.0.6 | (tag SHA) | 2026-07-22 (GHA) | **HF repo_id allowlist at cold start (#317).** Pin RunPod after verify+promote. |
| backend-v1.0.4 | 1.0.4 | (tag SHA) | 2026-07-22 (GHA) | **Pillow 12.3.0** via runtime `runtime-1-bf16-t4` overlay (#295 / cf#178). Pin RunPod to `:1.0.4` after verify+promote. Do not pin `:1.0.3` (tagless; still t3). |
| backend-v1.0.2 | 1.0.2 | d62360e | 2026-07-15 (GHA) | **Fix for broken `1.0.1` RunPod pull.** Deps-overlay runtime `runtime-1-bf16-t3` FROM t1 (run 29466705701); weight layers share ~104 GB with `1.0.0`, ~96 MB new. Restores resolvable t2-era pins (safetensors 0.8.0 / tokenizers 0.22.2) without re-emitting weights. Digest `sha256:ea38bc9d...`. Pin RunPod to `:1.0.2` (not `:1.0.1`). |
| backend-v1.0.1 | 1.0.1 | 22cadfe | 2026-07-15 (GHA) | **DO NOT PIN on RunPod.** Full t2 rebuild re-emitted ~101 GB weight digests; cold pull dies with unexpected EOF. Superseded by `1.0.2` (t3 overlay). Digest `sha256:da037184...`. |
|---|---|---|---|---|
| backend-v0.4.1 | 0.4.1 | f3a0d41 | 2026-07-02 (GHA) | fix(verify): the pod-staging gate renders end to end -- #183 register the GPU pipeline for the verify render (route _pod_draft_render through worker.handler, the production entrypoint that registers the per-job pipeline; fixes the "no GPU Pipeline registered" the H200 watched pod hit in ~3s), #180+#182 loud verify_fatal {stage,missing} instead of a silent pre-emitter death, #181 cold-pull TTL 3000s + pod-state timing log. Emitter + R2 channel proven healthy on the pod (gpu_probe real H200 values, summary.json + events.ndjson in ~3s). Fixes on top of :0.4.0. |
|---|---|---|---|---|
| backend-v0.4.0 | 0.4.0 | c10df5b | 2026-07-02 (GHA) | feat(release-gate): the pod-staging verify gate goes LIVE end to end -- a pushed tag builds the baked image, the gate spins a SECURE pod, runs a draft render emitting the structured @event channel (#175 pod-side emitter: gpu_probe/first_frame/sharpness/complete on a run-scoped R2 key + byte-identical stdout mirror; events_from_summary feeds runpod_verify.evaluate with no prose parsing), asserts, and promotes to the prod serverless endpoint only on green then tears the pod to zero (#173/#174 live SECURE pod client, #176 live up\|verify\|promote\|down, #177 key normalisation, #178 pod entrypoint launch + paced polls). FIRST image carrying the verify emitter. Also: perf(finish) NVENC + streamed interpolation (#172), fix(harness) F17 terminal-snapshot race (#171), fix(pipeline) honest degrades + draft default + job-key pinning (#170), feat(derisk) egress guard (#17/#161), docs/legal hardening (#164-#168). |
|---|---|---|---|---|
| backend-v0.3.3 | 0.3.3 | b93cdbd | 2026-06-30 (GHA) | feat(bake): facexlib baked offline (#158) -- the FIRST BAKED image, confirmed in PRODUCTION. Weights baked in-layer (bf16), no R2 cold-pull; facexlib detection+parsing weights baked so GFPGAN face-restore runs offline. Deployed to prod endpoint t9wcvlxh8rc5la + confirmed via the real serverless handler: **28/28 clean full-pipeline renders, ZERO errors, ZERO R2 cold-pulls** (mirror_done pulled=false on all) -- H200/sm_90 17/17 under concurrent load + B200/sm_100 11/11. cu128 wheels carry sm_90/sm_100/sm_120 (get_arch_list build-gate); true-cold sm_100 i2v JIT measured (~3.45 s/step steady, beats H200 ~5.0 s/step). Prod serverless tier = H200 \| B200 only. Full record: docs/serverless-0.3.3.md. |
|---|---|---|---|---|
| backend-v0.3.2 | 0.3.2 | 17a66fa | 2026-06-30 (GHA) | fix(models): the baked path never references an un-baked repo (#155) + baked_probe hardening (#156). The baked fast-path loaded the -fp8 i2v repo the bf16 bake never staged -> LocalEntryNotFoundError offline; _select_i2v_weights() gates the fp8 path on the fp8 repo being cached, else loads baked bf16 offline. baked_probe now asserts the EXACT runtime repo is cached pre-GPU ($0); CI guards the driver sha + marker. |
|---|---|---|---|---|
| backend-v0.3.1 | 0.3.1 | 410441f | 2026-06-29 (GHA) | feat(bake): first REAL bf16 baked image (#14). Built from the staged bf16 seed (~105 GB, 15 multi-GB layers); #138 gates pass on real weights, .vj-baked stamped precision=bf16. 3-arch coverage via prebuilt cu128 wheels; build-time proof get_arch_list() == {sm_90,sm_100,sm_120}. Supersedes the burned :0.3.0. |
|---|---|---|---|---|
| backend-v0.3.0 | 0.3.0 | 3dd84eb | 2026-06-29 (GHA) | feat(bake): bake pipeline (#127) -- BURNED, DO NOT PIN. Empty seed prefix -> hollow image (config stubs, zero weight shards) with a lying .vj-baked sentinel; caught by the #4 manual de-risk. Fixed in :0.3.1+ by the #138 empty-bake gate. |
|---|---|---|---|---|
| backend-v0.2.27 | 0.2.27 | c25edc8 | 2026-06-22 (GHA) | fix(finish): pad RIFE input to a multiple of 64, crop back (#245, PR #113). RIFE's flownet downsamples then concatenates skip features, so each spatial dimension must be a multiple of 64; a non-64-divisible i2v output (Wan 2.6 emits 1270x726) crashed the finish chain at step 0 ("tensor a (1270) must match tensor b (1280) ... dimension 3") before lip-sync/upscale, and the raw clip shipped with applied=[] (the umbrella behind #246, which pulled five motion backends for Monday). _RifeInterpolator.interpolate now pads the pair (replicate, bottom/right) up to the next multiple of 64, runs the flownet, and crops back to the original dims so the pad never reaches the encoded clip -- rescues every non-64 resolution at any aspect ratio in one place. own-gpu/kling already emit 64-divisible dims (pad 0). _pad_to_multiple unit-tested against all five backends' resolutions; the torch path validated by a live verify render. |
|---|---|---|---|---|
| backend-v0.2.26 | 0.2.26 | 5afd66a | 2026-06-21 (GHA) | fix(harness): verify keyframe presence in R2 before honoring a state-claimed REUSE (#108, PR #110). A stale or partial per-project state.tar.gz could name a keyframe whose R2 object was since cleared; _restore_prior_state trusted the state tar, so the planner marked the shot REUSE, skipped its keyframe render, and reported a phantom keyframe key to a nonexistent R2 object -- the shard then hung to the deadline, silently shipping a scatter render with one shot unrendered (hung shot_02 on the musetalk showcase). New R2.exists check verifies each state-claimed keyframe is actually present; an absent one is omitted from existing_keyframes so the planner GENERATEs it (self-healing), absent-on-failure being the safe default. Regression test added. FIX B (per-render/per-shard state isolation) is a separate follow-up. |
|---|---|---|---|---|
| backend-v0.2.25 | 0.2.25 | a73cc67 | 2026-06-20 (GHA) | ci: port the image build from Jenkins to GitHub Actions (#107) + keyframe distill CFG fix (#106). Jenkins was decommissioned in the fleet consolidation but the image-build pipeline was never ported (the repo had only tests.yml + stale.yml), so this release tag would have built nothing. Adds .github/workflows/release.yml, a faithful port of the Jenkinsfile build+push: triggers on a pushed backend-v<X.Y.Z> tag only, build context = repo ROOT with deploy/Dockerfile, runs deploy/smoke_imports.py AFTER build BEFORE push, pushes :<X.Y.Z> AND :latest. The deploy/RunPod pin stays a SEPARATE manual step (scripts/pin-runpod-template.py), same boundary Jenkins kept. Runner: GitHub-hosted ubuntu-latest (public repo, fork-safe); GHCR auth uses the built-in GITHUB_TOKEN (packages:write), no long-lived PAT. FIRST GitHub-Actions-built release. |
|---|---|---|---|---|
| backend-v0.2.24 | 0.2.24 | 93d1976 | (pending) | fix(models-mirror): never crash when VJ_VOLUME_ROOT is set but not mounted (#55, PR #96). 0.2.23's self-preload would os.open() the lock on a non-existent dir -> uncaught FileNotFoundError -> worker crash if VJ_VOLUME_SELF_PRELOAD was on while /runpod-volume wasn't mounted (the env-set-before-attach window). Now _resolve_volume returns a clean R2 fallback when VJ_VOLUME_ROOT isn't a mounted dir, and _acquire_volume_lock catches OSError broadly. SUPERSEDES 0.2.23 (do not pin 0.2.23). This is the image pinned to the prod endpoint for the network-volume rollout. 363 passed, 3 skipped. |
|---|---|---|---|---|
| backend-v0.2.23 | 0.2.23 | 263e18b | (pending) | feat(models-mirror): self-preloading network volumes (#55 Phase D, PR #94). Opt-in VJ_VOLUME_SELF_PRELOAD: on a volume miss (empty/partial/version-mismatch) the FIRST cold worker wins a single-writer lock (_acquire_volume_lock, atomic O_CREAT|O_EXCL, 60-min stale-takeover TTL) and mirrors R2 -> the volume (base+i2v) so every later worker in that DC reads it hot; lock losers fall back to the local R2 mirror (no concurrent-write corruption). Self-heals on VJ_MODEL_VERSION bumps. Scale-out becomes "attach an empty volume to a DC." The both-sentinels READ path (0.2.22) is unchanged + default; self-preload only triggers on a miss AND when the flag is set, so inert otherwise. 361 passed, 3 skipped. |
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
