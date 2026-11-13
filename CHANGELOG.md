# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

New entries are written as one file each under `changelog.d/` (see `changelog.d/README.md`) and
folded into a release section by `scripts/changelog-assemble.py` when the release is cut, so
`## Unreleased` below holds only what predates that convention.

## Unreleased

## [1.1.21] -- 2026-11-12

**What is new in the 1.1.21 image against 1.1.20: one feature, `multi_char.pose_template` (#649), on the same stills base.** 1.1.21 is built at main `27de2b6` (thin-image run 43334099190) FROM `stills-t1-m1-b2@sha256:9380cc42...`, the base 1.1.20 uses. Between `c7a6f02` and `27de2b6` the only `src/` changes are #649's (`config.py`, `keyframe.py`, `orchestrator.py`, `pipeline.py`); `deploy/Dockerfile.thin` and `deploy/smoke_imports.py` did not change. Against 1.1.20 the image shares 27 of 29 layers; the 2 new layers are the code, 202,145 bytes compressed (GHCR manifests, read 2026-11-13 UTC). So a worker that ran 1.1.20 pulls about 0.2 MB, and no weight layer. The version is a PATCH number on a feature because the thin-image tag was pushed as 1.1.21 before the release, and the workflow refuses to overwrite a GHCR tag.

**Feat: `keyframe.multi_char.pose_template` picks the figure the two-person pose skeleton draws (vivijure-cf#1281).**

`stand` (the default) is the upright, full-length, head-on figure drawn before; it is byte-identical and keeps every reuse hash. `reach` draws a waist-up figure with the arm nearer the partner forward and the head turned toward the partner (the far eye and far ear dropped), and mirrors the figure on the right so both face the frame centre. It is for two people at a shared surface, where the head-on figure cannot turn a head toward the work. `auto` picks `reach` from the shot text. A knob, not a default change: a sweep on the studio decides that.

## [1.1.20] -- 2026-11-12

**What is new in the 1.1.20 image against 1.1.19: the stills weights base, not the code.** 1.1.20 is built at main `9f2505d` (thin-image run 43312194021); 1.1.19 at main `dd3e7fc` (thin-image run 43070289187). Between them nothing under `src/` changed and `deploy/smoke_imports.py` did not change; `deploy/Dockerfile.thin` changed one comment line (`git diff --stat dd3e7fc 9f2505d -- src deploy/Dockerfile.thin deploy/smoke_imports.py`). The code layers rebuild to new digests (198,038 and 2,411 bytes, against 198,029 and 2,412) with the same sources.

The base is NEW. 1.1.19 is built FROM `stills-t1-m1@sha256:6d0e19bd...`, made on the retired bake box. 1.1.20 is built FROM `stills-t1-m1-rc1@sha256:9380cc42...`, made by Actions (`stills-base.yml`, run 43299859408, #644). The two bases share their first 22 layers (the toolchain) and differ in the 4 weight layers above it. So 1.1.20 has 6 layers that 1.1.19 does not: 18,429,278,177 bytes compressed (17.16 GiB), almost all of it the three weight bins (5,020,773,145, 7,971,655,393 and 5,436,649,064 bytes). Every worker's first cold start on 1.1.20 pulls those layers once; a host that ran 1.1.19 keeps only the toolchain. That pull is before the worker starts, so it is not billed GPU time, but it is delay. Measured from the GHCR manifests of both images, 2026-11-12.

The rest of this section is the tooling merged since 1.1.19. None of it is in the image. (`ci-thin-image-toolchain-ref.md` describes the `toolchain_ref` input that built 1.1.19; that fragment was not consumed when 1.1.19 was cut, so it is listed here.)

**Tooling: `scripts/burst_watch.py`, a read-only watch of an endpoint's queue and workers through a burst (#545).**

The #545 ruling keeps `QUEUE_DELAY` 4 and asks for the first burst after a code-only release to be measured, with RUNNING and THROTTLED worker counts per minute. The script polls only `GET /v2/<id>/health` (no submit, no setting change, no cost). It prints `@event burst.sample` lines and one `@event burst.summary`: per minute, the most workers in each state and the most jobs in queue. It also flags the minutes with jobs queued while no worker was initializing (`queuedNotScaling`, the scaler question) and the minutes with jobs queued while a worker stood ready (`queuedWithReady`). A failed read is an error sample, not a gap, and the key is never printed. Tests: `tests/test_burst_watch.py`.

**CI: the stills weights base is built by Actions, to a candidate tag only (#564, fleet-chezmoi#2620).**

Conrad's ruling (2026-11-12): every deployed artifact is built by Actions. The live base `stills-t1-m1` was built on the retired bake box. `.github/workflows/stills-base.yml` runs the documented procedure on a GitHub-hosted runner: reclaim disk and refuse below 100 GiB free; stage with the worker's own pull through a read-only R2 token in the `stills-bake` environment; gate; pack; build push-only; read the digest back; check every layer against the GHCR limit from the registry. It pushes only `stills-t<T>-m<M>-rc<N>` and refuses `stills-t1-m1`, any final tag and any existing tag (`deploy/stills-base-resolve.sh`). Layers are reproducible: `bake_layers.bin_pack` sorts by (size, path) instead of size alone, and the build sets `SOURCE_DATE_EPOCH` with `rewrite-timestamp`. `deploy/image_bytes.py` gains `MODE=max-layer`. Tests: `tests/test_stills_base_resolve.py`, `tests/test_bake_layers.py` (walk order), `tests/test_image_bytes.py` (max layer).

**CI: the stills weights base appends reproducible bin tarballs with crane, no BuildKit build (#564, fleet-chezmoi#2620).**

The first `stills-base.yml` run (43292232874) staged, gated and packed, then ran out of disk in the BuildKit build (`No space left on device`; nothing was pushed). This is option B from the design: `deploy/bake_layers.py layer-tar` writes each bin as one reproducible, uncompressed tar (sorted entries, every mtime the commit time, uid and gid 0, fixed modes, no hardlink entries) and prints its diff_id; the stage and each bin are deleted as soon as their bytes are in a tar; and `crane mutate --append` (go-containerregistry v0.22.1, pinned by version and archive sha256) appends the tars onto the exact toolchain digest and pushes to the candidate tag, compressing as it uploads. Estimated peak disk about 37 GB, so the floor drops from 100 GiB to 50 GiB. The toolchain's entrypoint, env and labels are kept, and four labels name the commit, the stills manifest, the model version and the toolchain. `bake_layers.py assert-appended` checks the pushed config is the toolchain's with exactly the run's tars appended. The BuildKit path and `deploy/stills-weights.Dockerfile` are removed, not kept behind an input. Tests: `tests/test_bake_layers.py` (layer-tar, assert-appended), `tests/test_stills_weights_base.py` (the workflow); `tests/test_stills_base_resolve.py` unchanged.

**CI: the `thin-image.yml` CPU import smoke runs on the runner scratch disk, and can smoke a published digest without a re-push (#636).**

The 1.1.19 run (43070289187, toolchain_ref = the stills base) pushed the image, then died in the smoke step with no conclusion. The image is the full stills base plus the thin layers, 29 GiB compressed. The build never unpacks it, because `Dockerfile.thin` is FROM plus COPY and BuildKit pushes by layer reference. But `docker pull` and `docker run` must unpack it, and the runner root disk cannot hold it. A new step moves Docker storage to `/mnt` before the pull. On the current runner (Ubuntu 26.04) `/mnt` is the same disk as `/`, so the move gains nothing; the free-space check that follows it is what protects the runner (measured 2026-11-12, run 43178431220, fleet-chezmoi#2620). It reads the image's compressed size from the registry manifest (`deploy/image_bytes.py`) and refuses loudly when `/mnt` has less than 2.5 times that free, instead of filling the disk and losing the runner. A new `smoke_ref` input takes a full digest of an existing worker image: then nothing is built or pushed, and only the smoke runs. That is how a published version is smoked again, because the version tag check refuses a re-push by design. `smoke_ref` with `toolchain_ref` is refused. Tests: `tests/test_thin_image_resolve.py`, `tests/test_image_bytes.py`.

**CI: the `thin-image.yml` CPU import smoke leaves the stills base's weight layers out, so a worker on that base can be smoked on a hosted runner (#645).**

Run 43312194021 built and pushed 1.1.20 on `stills-t1-m1-rc1`, then refused the smoke: the worker is 31.8 GB compressed, `/mnt` had 43.8 GB free, and the pull needs about 2.5 times its size. The build fits because `Dockerfile.thin` is FROM plus COPY and BuildKit pushes by layer reference. Only the smoke pulled the whole image. `deploy/smoke_image.py` (new) streams the worker into `docker load` without the base's layers above its toolchain (18.4 GB for 1.1.20, leaving 13.4 GB): the toolchain layers and the worker's own thin layers, fetched by digest, each checked for size and sha256, under the worker's config with those layers' diff_ids and history removed and an `org.skyphusion.vivijure.smoke-of` label added. Nothing is rebuilt or pushed. Every prefix is checked: the worker starts with the base, and the base starts with the toolchain its `org.skyphusion.vivijure.toolchain` label names. The smoke step reads both pipe statuses and checks the loaded rootfs against the plan before it runs `smoke_imports.py`. That is sound because the weight layers add files under `/opt/models` only and the smoke reads no weight on CPU; the GPU proof on a scratch endpoint stays the check of the full image. The room check now asks for 3 times the compressed size, not 2.5: run 43315326937 measured the load of 1.1.20's 13.4 GB smoke image at 33.9 GB on disk, 2.53 times, so 2.5 was below the real use. A new `smoke_base` input names the base of a `smoke_ref` worker; outside smoke mode the base is `toolchain_ref`. A smoke-only run builds and pushes nothing, so the main-only check now skips it, and it can run from a branch. Tests: `tests/test_smoke_image.py`, `tests/test_thin_image_resolve.py`.

**CI: `thin-image.yml` can build the worker FROM an existing base by digest, and never overwrites a GHCR tag.**

A new `toolchain_ref` input takes a full digest (`ghcr.io/skyphusion-labs/<image>[:tag]@sha256:<64 hex>`). When it is set, the workflow builds no toolchain and pushes no toolchain tag. It builds `deploy/Dockerfile.thin` with that digest as `TOOLCHAIN_REF`, so the worker's lower layers are the base's. Without it, the `toolchain_tag` path is as before. `deploy/thin-image-resolve.sh` checks the inputs before any build. It fails closed on a bad version or ref shape, on a ref GHCR cannot read, on a version tag GHCR already has, and when GHCR cannot tell. After the push, the workflow reads the digest back by tag, prints it, writes it to the run summary and smokes the image by digest. Tests: `tests/test_thin_image_resolve.py`.

## [1.1.19] -- 2026-11-11

This is the first assembled section since 1.0.16. The images 1.1.0 to 1.1.18 were published to GHCR with no git tag and no CHANGELOG section (RELEASES.md, "The 1.1.x line"), so every change merged since 1.0.16 is listed here, including the endpoint records of those images. New in the 1.1.19 image against 1.1.18: the `src/` changes merged after GHCR made 1.1.18 (2026-11-06 22:39:39Z; its exact commit is not recorded, so this is by merge time):
- the keyframe reuse hash covers the LoRA bound to each slot (#612);
- a warm worker swaps its keyframe base instead of refusing the job (#614);
- finish stops writing the unread finished-clip pointer (#621);
- a bundle accepts only regular files and directories (#624).

Repo tooling, not in the image: deploy, promote and the image bake use RunPod REST v2 (fleet-chezmoi#2085), an endpoint create sends its routing type, and a promote names its endpoint with no default (#629).

**Fix: promote proof uses tenant-scoped verify bundle key.**

`SMOKE_BUNDLE_KEY` was `bundles/verify-smoke/...`. Serverless `_job_bundle_key`
rejects that for project `verify`. Use `bundles/verify/Verify_Smoke.tar.gz`.

**Fix: Dependabot ignores `torchao>=0.18.0` while the runtime is torch 2.7.1 (#443).**

Stops Dependabot re-proposing the bump that broke Wan i2v in 1.0.15 (see the 1.0.16 entry
below). `deploy/smoke_imports.py` now also lists `torchao`, and `tests/test_torchao_pin.py`
covers the pin, the Dependabot ignore and the smoke-import list. Merged after the
`backend-v1.0.16` tag.

**Fix: deploy, promote and image-bake move to RunPod REST v2 (fleet-chezmoi#2085).**

RunPod REST v1 retired on 2026-11-15. v2 lives on its own host, `https://api.runpod.io/v2`, and every list there is cursor-paginated.

- **`deploy.sh` creates or updates one endpoint, with no template.** On v2 an endpoint carries its own `image`, `env` and `disk`; a v2 template is applied once at create and keeps no link to the endpoint. The script finds the endpoint by name across every page, then PATCHes or POSTs `/v2/serverless`.
  - GPUs are chosen by `gpu.pools`. `GPU_TYPE_IDS` still works, mapped to pools from RunPod's GPU catalog; the new `GPU_POOLS` names pools directly.
  - `EXECUTION_TIMEOUT_MS` blank now means one hour. v2 has no "no limit", and its own default is far shorter than a render.
  - `flashboot` is set to `FLASHBOOT`. A leftover `<name>-template` from an older deploy is no longer used.
- **`deploy/runpod_verify.py`:** promote re-pins the ENDPOINT image and reads it back. Flush and restore PATCH `workers.max` and `workers.min` on `/v2/serverless/{id}`.
- **`scripts/runpod_image_bake.py`:** the CPU pod body uses v2 names (`cpu`, `image`, `disk`, `entrypoint`, `cmd`). A pod's state is read from `status`. Pods-by-name lists every page and matches the name locally, because v2's filters are tag-based.
- **`deploy.env.example`:** `GPU_TYPE_IDS` is quoted. Unquoted, the file stopped `deploy.sh` with "command not found" when it was sourced.
- `tests/test_no_rest_v1_2085.py` fails if a shipped file names the v1 host again.

**Fix: `deploy.sh` sends the v2 routing type on a first create (fleet-chezmoi#2085).**

A v2 endpoint create requires `type` (`QUEUE` here), along with `name` and `scaling`, so the first run of `deploy.sh` on an account with no endpoint was refused. The create body now carries it. The in-place update takes no type and is unchanged.

**Fix: `finish_clip` writes to the `output_key` it is given (vivijure-core#379).**

One film can no longer replace another film's finished clip, once core sends a film-scoped key. With no `output_key` the shared per-project key is used, as before. A key that is not an `.mp4` under this project's `clips/` is logged and the shared key is used, so the job does not fail.

`docs/runpod-endpoint-config.md` records the live pools and their serverless prices.

**Fix: the network-volume readiness gate asks what THIS job needs, and names what it found (#453).**

`_resolve_volume` required both the base and the i2v sentinel unconditionally. Conrad retired the
local i2v path on 2026-09-26, so a volume preloaded with the honest post-ruling corpus (keyframes,
no Wan; the i2v repos were 60% of the 107.8 GB weight set) has no i2v sentinel. That volume was
therefore refused on every call, forever: no error, every worker falling back to the R2 mirror, and
the only symptom a cold-start cost that reads as a volume nobody configured.

The i2v sentinel is now required only when the job's action can actually reach the Wan i2v stage
(`models_mirror.i2v_stage_possible`, fed by `handler` from the job's own action; no new endpoint
knob). The base sentinel is still always required, and a volume missing a sentinel the job NEEDS
still degrades WHOLESALE to the R2 mirror rather than being repointed -- the guard against a
half-preloaded volume producing mysterious i2v failures is unchanged, only its premise moved.

`_resolve_volume` now returns a named `VolumeState` (`ABSENT`, `READY`, `READY_NO_I2V`,
`INCOMPLETE`) instead of a bare bool, so a monitor can tell "the i2v corpus was deliberately
retired" from "the preload did not finish" without parsing the log line; the two are identical on
disk and used to render as one reassuring reading. The structured skip event carries the same
split (`"reason": "volume"` vs `"reason": "volume_no_i2v"`).

Because readiness is now a per-job question while the `HF_HOME` / `VJ_MODELS_ROOT` repoint is
process-global, a decline also takes back a repoint an earlier job on the same warm worker made;
otherwise a `render` following a `preview` would inherit the keyframe-only volume, skip the R2 pull
and try to mirror Wan onto the shared volume at i2v time.

NOT settled here, and it needs spend, not code: whether a RunPod global network volume preserves
the symlinks the HF cache corpus is built from (`deploy/bake_layers.py`). The vendor docs enumerate
the POSIX semantics a network volume lacks and say nothing about symlinks. That needs one real
volume and one probe; tracked on `skyphusion-labs/fleet-chezmoi#2087`.

**Fix: `R2.exists()` returns False only for a real not-found (#460).**

It caught every `ClientError` and returned False, so an authorization or expired-credential error
read as "object absent". Now only `404`, `NoSuchKey` and `NotFound` return False; any other error
propagates out of `exists()`. Classification is duck-typed on `ClientError.response`, so no botocore
import is needed.

**Fix: `finish_clip` forwards its Deadline to `run_finish_job` (#462).**

`handler()` built the Deadline and checked it before dispatch but did not pass it on, so the
fetch, upload and in-engine checks in `run_finish_job` were skipped and a stalled encode was bounded
only by the platform ceiling. The Deadline is now forwarded.

**Fix: a stored LoRA the planner counts as trained is now staged for keyframing (#464).**

The planner skips training a slot whose adapter exists at `lora_key(project, slot)`, but only
`pretrained_loras` were staged into the pipeline, so on a later render the stored adapter was
counted as reused and never loaded. `run_job` now stages those stored adapters (SDXL family only)
when a keyframe will be drawn; an unfetchable adapter fails fast. A `pretrained_loras` entry for
the same slot supersedes the stored one.

**Fix: `face_restore: true` selects the default backend in `finish_clip` (#466).**

The backend name was derived with `str(cfg.get("face_restore") or "gfpgan")`, so `true` became
`"True"` and `FaceRestore("true")` raised `ValueError`. Only a string now names a backend; `true`
selects the default (`gfpgan`). The off values (`None`, `False`, `"none"`, `""`) still disable it.

**Fix: a failed standalone job is recorded on the progress channel (#468).**

`run_finish_job` and `run_i2v_clip_job` now record the failure (`progress.error(<action>, e)`, then
re-raise), matching `run_job`. Previously a failed standalone job left the snapshot at `running`.
The wall-clock degrade path (`FinishDeadlineExceeded`) is unchanged and still ends `complete`.

**Fix: `face_restored` is reported only when a frame was actually restored (#470).**

`finish_clip` set it True unconditionally, even when the restorer failed on every frame (each
failure passes the raw frame through by design), so the result, `applied` list and regression
checks claimed a restore that did not happen. `face_restored` is now true only if at least one
frame was restored, and an all-failed clip prints an `@event face_restore_degraded` line. Per-frame
best-effort behaviour is unchanged.

**Fix: a stored-state existence error aborts the job instead of retraining (#475).**

Since #460 `R2.exists()` raises on anything but a real not-found, but `_restore_prior_state` still
absorbed the error and treated it as "nothing stored", so every LoRA was retrained and every
keyframe redrawn on a paid GPU before the same error failed the final upload. Any store error
during the existence probes now fails the job with a `HarnessError` before GPU work starts; a
genuine not-found still restores to empty and renders fresh. Failing to fetch a keyframe or its
hash sidecar after existence is confirmed keeps its best-effort behaviour (regenerate).
`docs/operations.md` and `docs/architecture.md` updated to match.

**Fix: `finish_clip` and `i2v_clip` use the process-global ModelServer; a diverging `i2v_clip` model config is now refused (#476).**

`run_finish_job` and `run_i2v_clip_job` built a fresh `ModelServer()` per job, so a warm worker
reloaded its models on every standalone job (paid GPU time), and `i2v_clip` ignored its
`config.model` / `distill_model`. Both now take the warm process-global server via
`worker.standalone_server()`. `i2v_clip`'s model config now takes effect on a cold worker.

**Behaviour change, an `i2v_clip` that used to succeed now fails.** On a warm worker whose loaded
i2v models differ from the job's `config.model` / `distill_model`, the job is refused with
`ModelDivergenceError`, the same refusal a render already got. Previously the foreign
`config.model` was silently ignored and the clip rendered on the loaded set. A refusal the caller
sees is deliberate: it beats a substitution the caller cannot see. Likewise a job-supplied i2v repo
id now passes `validate_repo_id` before use, so an invalid id is rejected where it used to be
parsed and dropped. `finish_clip` names no models and just reuses the singleton.

**Fix: `kf_hash` covers scheduler, identity scales, distill and distill_model (#479).**

`kf_hash` omitted `scheduler`, `ip_adapter_scale`, `instantid_ip_adapter_scale`, `distill` and
`distill_model`. Each changes the drawn keyframe (or the few-step path that draws it), so changing
only one of them reused the old PNG; `distill=False, steps=8` and `distill=True, distill_steps=8`
also hashed the same. All five are now in the hash.

**One-time cost.** The hash payload changed, so every keyframe cached before this change no longer
matches and is redrawn once on its next render (GPU seconds, once per cached shot). The bound
character LoRA is still not in the hash (#478).

**Fix: a still is redrawn when the character LoRA bound to its slot changes (#478)**

The keyframe reuse hash (`kf_hash`) covered the render params, the prompts and the cast, but not which LoRA each slot binds. So a shot drawn with one adapter was reused after `pretrained_loras` repointed that slot, or after a LoRA was added or removed. Now `kf_hash` takes `bound_loras`, a `{slot: adapter key}` map, narrowed to the shot's own slots. A new helper, `bound_lora_keys`, builds it once: `pretrained_loras` first, then the project key for a slot trained this run or reused from R2. The planner hashes with it and stores it on the plan (`RenderPlan.bound_loras`), and the pipeline's sidecar writer reads it from there, so the hash a still is stored with and the hash it is checked against cannot differ. A still with no LoRA bound for its slots keeps its exact old hash (pinned against main), so it stays reusable. A still drawn with a LoRA before this change is redrawn once. Limit, accepted: a retrain at the same key is not seen, because the key is unchanged. Use a new project or new shot ids to redraw after a retrain.

**Fix: `regen_shot` trains no LoRAs, per the contract (#482).**

`docs/contract.md` says `regen_shot` trains no LoRAs, but the planner trained every slot that was
neither pretrained nor already trained, on a paid GPU, before redrawing keyframes. It now zeroes
`to_train` for `regen_shot` (as it already does for `finalize`) and records the skip.

**Behaviour change.** A `regen_shot` on a project whose slot has no adapter available now draws
that shot without the character LoRA (IP-Adapter identity fallback, recorded on the
`keyframe_done` event) instead of training one first. A LoRA trained by an earlier render is used
as before.

**Fix: `ensure_models` carries the network-volume repoint to the real environment (#484).**

`ensure_models` repointed `VJ_MODELS_ROOT` and `HF_HOME` on a copy of the environment, so the
deferred model loads (which read `HF_HOME` at call time) never used the volume. When a volume
resolves, both keys are now written back to the environment it was asked to use (the caller's dict
when `env` is passed, `os.environ` otherwise). No change when no volume resolves; latent today
because no endpoint sets `VJ_VOLUME_ROOT`.

**Fix: `deploy.sh` first-time create reads the created template and endpoint ids (#486).**

The template and endpoint POSTs ran `rp` in a pipeline subshell, so its `RP_BODY` assignment was
lost and `json_field id` read the previous call's body; a first-time create died with "could not
determine the template id" right after creating the template. Both POSTs now run `rp` in the
current shell. PATCH branches are unchanged.

**Fix: a still with no hash is redrawn, never reused (#488).**

A job used to reuse any still at the flat key that had no hash sidecar.
film-a9939630 asked for SDXL, drew nothing, and returned a cloud door's stills as COMPLETED.
Reuse now needs a matching hash.
Old GPU stills with no sidecar are redrawn once.
A reused still is reported with `reused: true`, so the studio can stop stamping it.

**Fix: an unreadable keyframe hash sidecar no longer reads as "no sidecar" (#491).**

`_restore_prior_state` read each stored keyframe's `.hash` sidecar inside a bare `except`, so a 403,
an expired credential, a throttle or a transport error stored `None`, which the planner reads as
"reuse the cached keyframe conservatively": a keyframe drawn with different parameters was served
as current. Only a real not-found (the classification `R2.exists()` uses, now `is_not_found`)
still means "no sidecar". Any other error now raises `HarnessError`, so the job fails before GPU
work, as the LoRA and keyframe existence checks already do (#460, #475). Behaviour change: a render
whose sidecar read fails with anything but not-found previously succeeded (with a possibly stale
keyframe) and now fails.

**Feat: hosted keyframes are full-step 1280x720 InstantID stills.**

The still the movie keeps is no longer an 8-step square. `KeyframeParams` and
`KeyframeConfig` (standard and final) are 30 steps, guidance 5.0, 1280x720, InstantID.
Draft stays the 4-step opt-in. One person takes InstantID even when scene lock is on.
Two people stay on masked IP-Adapter. A `lora_only` still is refused before it is saved.
The bake manifest drops Wan2.2-I2V and Wan2.2-Lightning. The image-to-video code path
stays; this change does not rebuild the image. A baked image that does not hold the
selected image-to-video repo pulls it from the model mirror before the offline load.

### Fixed

- `bake`: the pod reap no longer reports success for any listing that cannot
  establish what is running. "Nothing is running", "pods came back and none
  matched", and "the listing failed" had the same shape and the same exit code;
  only the first may report success. A failed listing is the likelier of the two
  unanswerable cases, since a 429 or 5xx on a list endpoint is ordinary.
- `bake`: each rejection reason is reported separately, because one counter made
  four different facts print one sentence and in one reachable case that sentence
  was false. A pod carrying the name we asked for but no usable id is not a
  name-filter problem; it is the pod we wanted, arriving unaddressable.
- `bake`: a row with a real, different name is reported as the measurement it is.
  It can only come back if RunPod's `name` filter is not exact, which is the
  question a live read could not settle against an empty account.
- `bake`: a pods listing that is a bare `null`, number, boolean or non-JSON body
  no longer raises `TypeError` out of the reap. Inside `run()`'s `finally` that
  exception replaced the one that sent us there, so a create timeout was reported
  as a complaint about iteration.
- `bake`: the reap never renders a pod row. A Pod response carries 34 properties
  including `env`, and the branch that printed a row fires only for a row whose
  name is ours, so it was the one branch guaranteed to print our own bake pod and
  its `GHCR_TOKEN`. It now prints an allowlist of identifying fields, so a new
  response property cannot leak by default.
- `bake`: a dict listing body with no `pods` list is no longer read as an empty
  account. That branch exists to survive a response-shape change, and reading a
  missing key as `[]` was the one way it would survive one by reporting that
  nothing is running.

**Feat: anime stills may use Animagine XL 4.0 (#524).**

A job may set the keyframe base to `cagliostrolab/animagine-xl-4.0`.
The allowlist accepts that namespace.
LoRA train uses the job base.
A missing checkpoint fails with a clear error.
Animagine stays out of the default bake.

**Fix: anime `final` no longer auto-enables GFPGAN (#526).**

`FinishConfig.for_tier(final)` still turns GFPGAN on for photoreal.
Anime reads `style` from the finish or keyframe override.
No restorer pick keeps restore off.
An explicit restorer still wins.

**Fix: scene-lock Canny loads from a local snapshot and moves to CUDA (#528).**

`bake_hf_configs` writes a Canny config snapshot and can point `refs/main` at it.
`from_pretrained(..., local_files_only=True)` then raises `OSError` even when the
weight snapshot is on disk, or when only the stub is present.
The loader now picks a snapshot that holds tensors.
It loads from that path, not the hub id.
It then moves the ControlNet to CUDA.
If no tensors exist, it pulls `xinsir/controlnet-canny-sdxl-1.0` from the model
mirror, then from Hub.
`scene_lock` stays on. Off remains a debug hatch.

**Feat: thin image reads stills from a global volume (#530).**

The keyframe worker skips the bake and the cold pull.
A mounted `/runpod-volume` is the default weight root.
`VJ_VOLUME_KIND=global` refuses first-worker lock.
`--preload --stills --flatten` stages the base set as real files.
`deploy/Dockerfile.thin` FROMs the toolchain stage only.
This PR does not create, fill, or attach a volume.

**Feat: thin keyframe image builds on GitHub Actions (#530).**

The bake job stays disabled.
`thin-image.yml` builds `--target toolchain` and `Dockerfile.thin`.
It pushes to GHCR.
It does not pull the seed.
It does not need a GPU runner.

**Fix: concurrent renders of one project keep their own keyframes (#534).**

A render now writes `renders/<project>/keyframes/<job_id>/<shot>.png`.
Reuse now reads `<shot>.latest.json`, one object with the key and the param hash.
A reused still is copied to the new job's key.
A project from before this change is still read from the flat key.
Nothing new is written at the flat key.
This needs vivijure-core 1.27.0 for `own-gpu` clip submits and lost-poll recovery.

**Fix: stills load from container disk, and say what they are loading.**

A stills job on a global volume now copies the weights it reads to container disk once per worker.
The volume read was about 26 MB/s, and one job read the UNet twice.
Each file reports its MB/s.
Each model load now emits `load_start`, `load_heartbeat` and `load_done` with elapsed seconds.

**Feat: a stills job pulls only what it loads.**

`preview`, `regen_shot` and `train_lora` now pull 25.75 GB from R2 instead of 52.29 GB, plus `antelopev2`.
The list is `harness/stills_manifest.json`, by exact blob at the pinned revisions.
The volume hydrate reads the same list.
A missing file fails the job at the mirror, naming the file.
Other actions keep the full mirror. `VJ_STILLS_MIRROR=0` restores it for all.

**Fix: a cold `finish_clip` job pulls only the models its step loads, and a finished pull no longer costs the job.**

A RIFE step now pulls the 11.6 MiB `rife` leg instead of 53 GB. Each pulled leg has its own sentinel, so a warm worker does not copy it again, and a later job that needs the rest still pulls the rest.
The mirror has its own budget, `VJ_FINISH_MIRROR_MAX_SECONDS` (default 120). The 420 s budget starts when the mirror returns.
A hung `rclone` is killed at the mirror budget and the job soft-degrades at stage `model_mirror`.

**Fix: a warm worker swaps to a job's keyframe base instead of refusing it, and a base not on disk no longer poisons the worker (#559).**

The process-global `ModelServer` froze the first job's keyframe base, so `_check_model_divergence`
refused every later job with another `style` or `base_model` for the life of the worker. A cold
anime job (animagine is not on the GPU image, #557) also failed on missing weights while leaving
its spec loaded, so that worker then refused every photoreal job after it.

`build_pipeline` now calls `ModelServer.swap_keyframe_specs` first. A new base that is on local disk
swaps in: only the `keyframe` and `instantid` pipes are dropped, they reload on the next draw, and
`@event keyframe_model_swap` names the change. A new base that is not on disk raises the existing
missing-weights error before anything changes, so the loaded pipes stay warm. The repo id is
validated before either. An i2v divergence is still refused with `ModelDivergenceError`.

**Cost:** one SDXL base reload from local disk per change of base on a warm worker, where the job
used to be refused.

### fix(mirror): the main model pull has a deadline on every action

A cold worker pulled 26 GB for up to 2618 s on the GPU clock, with only the endpoint's 3600 s job timeout to end it. The pull in `ensure_models` outside `finish_clip` now has a budget, `VJ_MIRROR_MAX_SECONDS`, default 3000 s. Past it rclone is killed, the job fails by name, and the worker goes free.

Not covered: `ensure_i2v_models`, `ensure_hub_repos` and the volume self-preload have no deadline.

### feat(bake): a stills weights base, so a cold keyframe worker pulls nothing

A cold keyframe worker pulled the stills set, 26.22 GB, from R2 inside the handler: 104 to 2618 s on the GPU clock per worker, about 20 minutes per job when 15 pulled at once, and again after every release. RunPod pulls image layers before a worker starts, and does not bill that. So the stills set and the finish legs now go into a weights base image below the code layer (`deploy/stills-weights.Dockerfile`, runbook `docs/stills-weights-base.md`). A code-only release changes only the code layer.

`python -m vivijure_backend.harness.models_mirror --stage-stills` fills the tree with the worker's own pull and checks it; `--check-stills` checks a built image. The base carries no `.vj-baked`, so any action other than a stills job or a finish step still pulls what it needs. The warm stills path now checks the files behind its sentinel before it skips, and pulls again if one is missing.

Nothing changes on the endpoint until a base is built and a code image on it is pinned.

### fix(mirror): a model pull is killed on a stall, and reports progress while it runs

The pull budget (#564) bounds a whole pull, so it cannot tell a slow pull from a hung one: a hung pull held its GPU until the 3000 s budget, about $2.90. Every model pull now also ends when rclone's transferred bytes have not moved for `VJ_MIRROR_STALL_SECONDS`, default 300 s (about $0.29 of GPU). A slow pull that still moves is left alone. A stall fails a job by name (`model pull stalled: ...`), and a finish job soft-degrades as it does on its budget.

This covers every pull: the job pull, `ensure_i2v_models`, `ensure_hub_repos`, and the volume self-preload, which falls back to the job's own pull on a stall. While a pull runs, each rclone stats line becomes a `mirror_progress` event with the leg and the bytes moved, so a cold pull no longer looks like a silent job.

**Fix: a stored keyframe is reused only when nothing that draws it has changed.**

The keyframe reuse hash covered the render parameters only. So after a new image changed how prompts are built, a re-render with the same parameters returned the old stills as `reused: true`, and the run read COMPLETED (film-0afa679b on 1.1.6). An edited shot in the same project reused its old still the same way. Each shot's hash now also covers the prompts the engine builds for it (the plate, the identity pass and the plain pass), so an edited shot, a changed cast description, a new `setting` or new prompt-building code redraws, and a `KEYFRAME_ENGINE_VERSION` that is bumped for render changes that leave the prompts the same. Stills stored before this change carry the old hash, so they are redrawn once.

### fix(verify): a verify render removes its temp directory

`verify._pod_draft_render` downloaded the finished clip into a `vj-verify-*` directory from `tempfile.mkdtemp` and never removed it, so every verify run left its clip on the pod's disk and every test run left one in `$TMPDIR` (over 100 found in one account). The directory is now removed in a `finally`, after a normal return and after a failed download alike.

### feat(keyframe): a two-person scene-lock plate draws without the pose skeleton by default

Under the fixed standing skeleton, a two-person scene-lock plate posed the pair for the camera instead of drawing the shot's action, and Pass B kept that pose through canny. On vivijure-cf#1281 (18 jobs, one board, 6 seeds), 2 of 6 stills held the action with the right people at today's default, against 5 of 6 with no skeleton. The plate count guard caught every third person.

So `multi_char.pose_conditioning` now defaults to unset, which means each path's own default:
- A scene-lock plate for two-plus people draws without the pose ControlNet while the plate count guard runs.
- With `plate_count_retries` 0, it keeps the pose, so pose off with the guard off is never a default.
- The plain regional path (scene lock off) keeps its pose.
- An explicit `true` or `false` sets both paths, so a job can turn the plate's pose back on.

Reuse hash: only a two-plus-person, scene-lock, guarded still with the pose unset gets a new hash and redraws. Every other stored still keeps its hash, held by tests against values computed on the previous code.

### fix(keyframe): `multi_char.ip_adapter_scale_per_slot` reaches the two-person draw

The field was documented (default 0.7), parsed and hashed, but never reached the renderer: both two-person paths (scene-lock Pass B and the plain regional render) drew at `keyframe.ip_adapter_scale`, 0.65. Setting it forced a redraw that drew the same image. It now reaches both paths. Unset, the new default, falls back to `keyframe.ip_adapter_scale`, so a job that does not set it draws and hashes exactly as before and no stored still redraws. A job that sets it draws at that scale, and its stills redraw.

### fix(harness): a worker whose disk cannot hold the job fails that one job by number and is replaced

A worker's model cache persists across jobs on purpose. On the old 80 GB disk, one i2v-capable job's pull (Wan 2.2 I2V plus Lightning, 156.8 GiB measured on 1.1.17) filled it, and every later job on that worker failed within seconds on ENOSPC (vivijure-backend#600). Now a model pull is sized first (`rclone size` over exactly what the copy would copy) and fails its job with the numbers when it cannot fit with the job floor left free. A job also fails by number before it starts when free disk is below the floor (4 GiB, `VJ_MIN_FREE_GB`). Either failure, or an ENOSPC raised anywhere in a job, is returned with RunPod's `refresh_worker`, so the SDK stops that worker and RunPod replaces it instead of it eating the queue. Every job's `started`, `complete` and `error` progress events now carry `disk_free_gb`. The endpoint disk is 300 GB (docs/runpod-endpoint-config.md has the arithmetic).

**Fix: a finish job no longer writes `<shot>_finished.latest.json` (#613).**

Nothing read that object: core does not reuse a finished clip through a pointer (vivijure-core#382). The clip and its `.hash` sidecar are written as before.

**Docs: the hosted backend endpoint drops the B200 and B300 pools (#625, #606).**

`docs/runpod-endpoint-config.md` now records the backend pools as `BLACKWELL_96`, `ADA_80_PRO` and `HOPPER_141`, with the before list and the rollback. The current-state table is re-read on 2026-11-11, and the image row moves from 1.1.17 to 1.1.18@sha256:3ec38492. The Recommendation for the render backend no longer says B200 / H200: RunPod serves SDXL keyframes only, and keyframes are proven on 96 GB and 80 GB cards. This records a live endpoint change that the lead applies under CR #625. No code changes.

**Fix: a promote names its endpoint; there is no default (#629).**

`deploy/runpod_verify.py` defaulted the promote target to an endpoint that has been retired, and `runpod-verify.yml` passed none. So a `live=true promote=true` dispatch would spend on the verify pod and then fail at the first PATCH. There is no default now. `promote_image()` takes `endpoint_id` as a required keyword and refuses an empty or malformed id before any call. `--promote` without `--promote-endpoint` exits 2 before anything runs. The workflow takes a `promote_endpoint` input and refuses `promote=true` without one, before the pod. `docs/release-gate.md` describes the v2 endpoint image repin and points at `docs/runpod-endpoint-config.md` for the current endpoint.

**Fix: a bundle extract accepts only regular files and directories.**

Absolute names and names with a `..` component are refused. A link is refused, including one whose target resolves outside the destination. A fifo, a character device, and a block device are refused.

**Fix: a two-person keyframe keeps both people's hair and clothes, every person is said to be an adult once per shot, and a shot with no place gets the film's setting.**

The keyframe prompt is now encoded in two 75-token chunks, not cut at 77 tokens, so the identity pass of a two-person shot keeps both descriptions (vivijure-cf#1143). A chunk splits at a comma near its end, and a prompt past 150 tokens is cut and reported in a `keyframe.prompt_cut` event, never silently. Each shot with people now says once that they are all adults ("Mara and Dev, both adults"; "two adults" on the plate). That clause is never fitted away, and on a two-person shot it lets each description put hair and clothes ahead of its own age clause. The storyboard's optional `setting` follows any shot whose prompt and style prefix do not already say it, and only when it still fits after the descriptions (vivijure-cf#1145).

**Feat: a diagnostic job flag, `debug_intermediates`, saves the scene-lock plate beside the keyframe.**

A two-person shot sometimes draws a third person, and the knobs measured so far do not move how often (vivijure-cf#1143). To find which pass draws the extra person, a job with the top-level field `debug_intermediates: true` also writes the Pass A pose control, the Pass A plate and the Pass B canny map beside each drawn still, as `renders/<project>/keyframes/<job>/<shot>.<stage>.png`, and lists them under `debug_intermediates` in the result. It is not a generation knob: it is not under `render_overrides`, not in the reuse hash, and the studio never sends it, so only a direct `/run` caller sets it. Off by default, and with it on every input to every pass is unchanged.

**Feat: the scene-lock identity crop padding is a knob, `keyframe.face_crop_pad`.**

Pass B gives each person's IP-Adapter a crop of their cast portrait padded by this fraction of the face box on each side. The default stays 0.35, the crop shipped before, so nothing changes until it is set. Measured at seed 110501 on 1.1.6 (vivijure-cf#1143): the two-shot lost Mara's hair on the scene-lock path, whose 0.35 crop holds little hair, and kept it on the path that takes the whole portrait. The knob lets one image be swept live. It is clamped to 0..2 and is part of the keyframe reuse hash.

**Feat: with per-region text, the band between two people can read a people-free prompt, `keyframe.plate_people_free_background`.**

Off by default. With `plate_regional_text` on, the cells outside both people's regions read only the style and the setting, never the shot text that names them, while each person's region keeps the shot's action and the adult clause. Measured on 1.1.9 (vivijure-cf#1143): under per-region text the band read "Mara and Dev stand ..." and a third person appeared there at 2 of 2 seeds. The pooled text embedding stays the shared prompt's, so the plate is still told to draw the two people.

**Feat: the plate count guard is on by default (`keyframe.plate_count_retries` 3) for two-person-plus shots.**

Measured live on 1.1.13 at six seeds (vivijure-cf#1143): clean two-person stills rose from 2 of 6 to 5 of 6. Every still marked clean had exactly two people, and the one failure was reported as `clean: false`. The cost is about 1.7 extra plates per two-person keyframe, about 23 s on average, at most 3 plates. The value now enters the keyframe reuse hash for two-person-plus shots only, so each existing two-person still redraws once, and one-person and empty shots keep their hash. Set it to 0 to turn the guard off.

**Docs: the cold-start rule after a repin is corrected.** A warm-up batch does not keep workers warm after a new image: every new worker pulls about 24 GB. Keep about 4 to 6 cold workers at once by submitting in small batches, and budget one cold pull per worker (`docs/runpod-endpoint-config.md`, `docs/bake-box.md`).

**Feat: a plate count guard, `keyframe.plate_count_retries`, redraws a two-person plate that holds a third person.**

The scene plate (Pass A) draws a third person at about half of seeds on a two-person shot, and no conditioning lever measured moved that rate; Pass B never adds a person (vivijure-cf#1143). With this set to 1..3, the plate's faces are counted with the insightface analyser the scene-lock path already loads, and while the plate has more faces than the shot has people, only the plate is redrawn with a derived seed. Pass B runs once, on the first passing draw, or on the draw with the fewest faces, which the result's `plate_count` and a `keyframe.plate_count` event then report as `clean: false`. Default 0: no count and the same draw as before, with the reuse hash unchanged. Measured offline on 19 saved plates, the count caught 11 of 12 plates with a third person and read all 7 clean plates as clean; it misses a person facing away from the camera. Also documents the four earlier #1143 knobs in `docs/configuration.md`.

**Feat: a two-person plate can take each person's cast portrait, `keyframe.plate_ip_adapter_scale`.**

Off by default (0), so the plate is unchanged until it is set. When set (0 to 1), Pass A of a two-person shot conditions each region on that person's whole cast portrait through the masked IP-Adapter, using the same region boxes as the pose skeleton and Pass B. A person with no portrait gets scale 0. Measured on 1.1.6 and 1.1.7 (vivijure-cf#1143): the plate draws the two-person hair from shared text, and nothing in Pass B undoes it, while only the whole-portrait path restored the hair. The scale is in the keyframe reuse hash, and the `keyframe.scene_plate` event reports the scale used.

**Feat: a two-person plate can give each person's description to their own region, `keyframe.plate_regional_text`.**

Off by default, so the plate is unchanged until it is set. When on, on a two-person shot each cross-attention layer of Pass A reads the shared prompt (the style, the scene and its action, the setting and the adult clause) everywhere, and each person's own prompt (the shared text plus only their description) inside their region box, the same boxes the pose skeleton and Pass B use. The unconditioned half of guidance and an IP-Adapter's image embeddings are untouched, and the original processors are always put back. Measured on 1.1.8 (vivijure-cf#1143): one shared prompt bound Dev's "grey zip hoodie" to Mara. `scripts/check_regional_text.py` checks, on diffusers' own Attention, that each region reads only its own prompt.

**Feat: `keyframe.multi_char.pose_spread` moves the two pose figures toward the frame centre.**

The 1.1.11 plate diagnostic (vivijure-cf#1143) showed that the third person in a two-shot is drawn by the scene plate, in the empty band the pose skeleton leaves between two figures placed at the centres of the two halves of the frame. Above 0, this value places the figures' centres that fraction of the width apart around the frame centre, clamped so each figure stays inside its own region (the masks, LoRAs and per-region text are unchanged). The default stays 0: the skeleton is byte-identical to before, and the reuse hash is unchanged. A set value is part of the hash.

**Fix: the keyframe identity prompt carries each slot's character description (vivijure-cf#1147).**

The prompt was style, scene and LoRA triggers, so a shot that named only the character lost the hair and wardrobe the cast member is defined with. The description now follows the triggers on the identity pass and is left out of the plate pass.

**Fix: a shot that names only its character keeps that character's hair and clothes (vivijure-cf#1147).**

The scene plate now gets the character description. Rendering a plate alone at the same seed showed the identity pass keeps the plate's hair and outline through canny, so a description added only there came too late. The token budget now keeps a stated adult age first, then hair and garments, then other physical traits, and mood last. A portrait direction in a description, such as a studio background, no longer reaches a shot. The plate gets only the age, hair and garments, and says the scene again after them, so the shot keeps more of its action and its location. A garment named after a place, such as a field jacket, no longer draws that place.

**Fix: character descriptions fit the 77 token limit (vivijure-cf#1147).**

The description came before the location-lock clause and pushed it out of the prompt on long shots, and on a two-person shot the first character took the whole budget. The lock clause now comes first, and what is left is split evenly across the shot's slots, cut at clause boundaries.

**Feat: under per-region plate text, the plate count guard also redraws a plate that is short of a person.**

With `keyframe.plate_regional_text` on, a two-person scene-lock plate drew ONE person on 4 of 6 seeds of a hand-a-mug shot, and the plate count guard passed each of them, because it redraws only for an extra person (vivijure-cf#1281). The same board without per-region text drew one face on 1 of 6. So when per-region text is on and the guard runs (`plate_count_retries` > 0), a plate with fewer faces than people is redrawn as well, up to the same bound. The chosen draw is the first with the exact count, else the one with the most faces not over it, and only with none of those the fewest: a missing person is still preferred to an extra one. The `plate_count` record and event carry `floor: true`, and `clean` is false for a plate that is still short. Every other path is unchanged: a one-face plate can be a person facing away, and the guard accepts it there. Only per-region text stills with the guard on get a new reuse hash; the default still and a per-region text still with the guard off keep theirs (pinned against main 57a0f27).

**CI: dispatch inputs go through env and are validated; thin-image publishes only from main.**

`runtime-build.yml` and `thin-image.yml` read their `workflow_dispatch` inputs through `env` instead of
placing them in the script text, and check each against its shape: integers for the runtime
versions, a digest-pinned `ghcr.io/skyphusion-labs/vivijure-backend:runtime-*` ref for
`overlay_from`, `X.Y.Z` and `t<N>` for the thin image. `thin-image.yml` refuses a commit that is not
on `main`, as `runtime-build.yml` already does, and its push trigger on the deleted
`ci/530-thin-gha` branch is removed, so no branch build publishes a release-shaped tag.

**CI: every third-party Action is pinned by commit SHA.**

Each `uses:` reference in `.github/` now names a full commit SHA with the exact release tag as a
comment (for example `actions/checkout@<sha> # v7.0.1`), resolved from the tag the line used before.
A moved tag can no longer change what a workflow runs. Dependabot (`github-actions`) keeps proposing
updates against the pinned form.

**Docs: the backend endpoint runs 1.1.10, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.10 thin image built from main `1d78182` (#582, `plate_people_free_background`, default OFF);
- the fresh-draw check;
- the rollback to 1.1.9.

**Docs: the backend endpoint runs 1.1.12; the repin cold-start rule; the 1.1.x tag decision.**

`docs/runpod-endpoint-config.md`:
- records the 1.1.11 (#584) and 1.1.12 (#585) pins, with the 1.1.11 rollback;
- adds a standing repin procedure: wait 65 s after the release, then submit 2 or 3 jobs first because every worker starts cold and pulls about 24 GB.

`docs/bake-box.md` points at that procedure. `RELEASES.md` records that the 1.1.x line is GHCR tags only, with no git tag.

**Docs: the backend endpoint runs 1.1.13, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.13 thin image built from main `a3d48a6` (#587, `plate_count_retries`, default 0, which is OFF);
- that its face count uses the antelopev2 analyser already mirrored for stills jobs;
- the rollback to 1.1.12.

**Docs: the backend endpoint runs 1.1.14, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.14 thin image built from main `a2f3b18` (#589, `plate_count_retries` default 3 for two-person-plus shots);
- that the first re-render of each stored two-person still redraws it once;
- the rollback to 1.1.13.

**Docs: the backend endpoint runs 1.1.15 on the stills weights base; first-bake numbers.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.15 pin, built from main `82b2324` on `stills-t1-m1@6d0e19bd`, with its in-image checks;
- the rollback to 1.1.14.

`docs/stills-weights-base.md`:
- records the first bake (stage size and time, the gates, the bins, the three layer sizes, the push time);
- corrects step 1: stage with code that has `--stage-stills`, and keep the shell that reads the credential in the foreground.

**Docs: the backend endpoint runs 1.1.16, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.16 code-only image built from main `6edad52` (#597) on the existing `stills-t1-m1` base, which reused all three weight layers;
- that two-person scene-lock stills redraw once;
- the rollback to 1.1.15.

**Docs: the backend endpoint runs 1.1.17, with its rollback; the disk history.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.17 code-only image built from main `eed4b57` (#601 per-slot ip scale, #602 disk guards) on the existing `stills-t1-m1` base;
- the rollback to 1.1.16;
- the container disk raise of 2026-11-06 from the release history: 80 to 200 GB at 15:01:44Z (v41), then 200 to 300 GB at 15:20:17Z (v42), kept at 300 through the repin.

**Docs: the backend endpoint runs 1.1.5, with its rollback.**

`docs/runpod-endpoint-config.md` records the 1.1.5 thin image built on the bake box from main `1ac386e`,
its proof job, the live worker settings (max 12, idle 60 s), and the rollback image.

**Docs: the backend endpoint runs 1.1.6, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.6 thin image, built on the bake box from main `67e883e` (#572);
- its proof, a partial fix for vivijure-cf#1143 (film-1d0dad23);
- the rollback to 1.1.5.

**Docs: the backend endpoint runs 1.1.7, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.7 thin image built from main `7f66faa` (#575, #576);
- that #575 makes every stored still redraw once;
- the fresh-draw repin check;
- the rollback to 1.1.6.

**Docs: the backend endpoint runs 1.1.8, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.8 thin image built from main `957e4ce` (#578, `plate_ip_adapter_scale`, default 0, which is OFF);
- the 60 s wait after a repin;
- the fresh-draw check;
- the rollback to 1.1.7.

**Docs: the backend endpoint runs 1.1.9, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.9 thin image built from main `c63b76c` (#580, `plate_regional_text`, default OFF);
- the fresh-draw check, whose two-person still shows a duplicate person;
- the rollback to 1.1.8.

**Docs: separate the idle timeout from standby in the billing note.**

A worker bills through the idle timeout after a job, by RunPod's rule. A scaled-down worker does not. `docs/runpod-endpoint-config.md` cites that rule, and the hourly buckets for 18:00 to 20:00Z on 2026-10-30 as evidence on standby only. The image line reads 1.1.3.

**Docs: record the endpoint state and the real worker quota.**

`docs/runpod-endpoint-config.md` now states the two live endpoints, read from the API, with measured times.
The quota is 50, not 30. The serial behaviour of two cold submits is recorded.

**Fix: a slow cold start now names its own stage in the job's progress object.**

`mirror_done` carries the seconds per leg, total seconds, bytes, and the rclone passes that failed and were retried. A failed pull writes `mirror_failed` with the legs it finished. Before this, the numbers were only in the worker's log, which goes when the worker does.

## [1.0.16] -- 2026-08-20

**Fix: pin torchao 0.17.0; 0.18.0 breaks Wan i2v on torch 2.7.1 (#441).**

Dependabot #419 bumped torchao to 0.18.0. That release imports
`torch.nn.functional.ScalingType` (torch 2.10+) at package load, so
`from diffusers import WanImageToVideoPipeline` died on the 1.0.15 H200
smoke (`s1-41aa1926802e`). Revert to 0.17.0. Overlay runtime-2-bf16-t2
from t1 (deps only; seed 2-bf16 unchanged).

**Fix: verify report includes R2 channel error stage and message (#439).**

A `status=error` smoke no longer dies as a generic hold. `channel_error` lands in
the report. `print-verify-error.yml` dumps a prior run's stage+message from R2
with no GPU.

**Fix: verify pod pulls GHCR with the same RunPod registry auth as serverless (#436, #438).**

The GraphQL SDK cannot attach `containerRegistryAuthId`. The harness was rejecting it
and assuming GHCR was public. REST create now passes the serverless template's auth
id (`cmqbz5bba0018e11d6bpcnu4n`). The PAT stays in RunPod; we never mint a second one.
Pod create is REST v2 (`POST https://api.runpod.io/v2/pods`: `image`, `registry`,
`gpu`, `args`, `disk`). Promote/flush stay on v1 (v2 endpoint mutations 500).

**Fix: the verify pod's REST create body uses `dockerStartCmd` (#437).**

`dockerArgs` is not in the REST v1 `PodCreateInput`; the live gate got a 400 from
`https://rest.runpod.io/v1/pods`. The field is `dockerStartCmd` (argv). This was superseded within
the release by the REST v2 create described in the GHCR entry above (#438).

**Fix: `print-verify-error` uses boto3 only (#440).**

The dump job imported `vivijure_backend` and crashed on `yaml`. It is now standalone boto3 and
still prints only the stage and message.

**Build: runtime repinned to `runtime-2-bf16-t2` (#444).**

`RUNTIME_REF_BF16` in `deploy/Dockerfile` and `.runpod/Dockerfile` now names
`ghcr.io/skyphusion-labs/vivijure-backend:runtime-2-bf16-t2@sha256:babdb3821d7cbac5ac47dc281e4e3bba6cf18a259e693754ddb3c9ae4af80b8a`,
a deps-only overlay from `runtime-2-bf16-t1` carrying torchao 0.17.0; seed `2-bf16` unchanged.
This release is src-only on that runtime. Everything in this release other than the torchao pin
(#441) touches the verify harness under `deploy/` and `.github/workflows/`, not the worker code.

## [1.0.15] -- 2026-08-19

**Weights: seed `2-bf16` (canny ControlNet in the importable seed) (#434).**

`ghcr.io/skyphusion-labs/vivijure-backend-seed:2-bf16@sha256:45d036fd50c83dfa0347cf837c43bcccb07e4d2c52c3fb5c02d3deee962a7e66`.
Runtime imports this; `backend-v*` still only COPY src.

**Fix: seed-build and runtime-build register on Plane C before they bake (#433).**

Group 9 is workflow-restricted. `release.yml` already ran `sync-gpu-allowlist` on
ubuntu-latest first. The seed and runtime workflows did not, so a dispatch sat
queued on idle bake runners. Same job, `--apply` (these are main-ref, not tags).

**Fix: tests.yml concurrency no longer cancels against the dummy coverage check (#432).**

`tests.yml` and `coverage.yml` both used `group: coverage-${{ github.ref }}`. GitHub
concurrency is repository-wide, so the org-ruleset dummy `coverage` job cancelled the
real pytest + required `ci` job (PR 432). tests.yml now uses `tests-${{ github.ref }}`.

**Fix: keyframes stay in the scene (plate then canny face, not studio portraits) (#432).**

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

**Build: runtime repinned to `runtime-2-bf16-t1`, which imports seed `2-bf16` (#435).**

`RUNTIME_REF_BF16` in `deploy/Dockerfile` and `.runpod/Dockerfile` now names
`ghcr.io/skyphusion-labs/vivijure-backend:runtime-2-bf16-t1@sha256:c8528fbc7b0280ca5af3633dd4dbcfacb4dfb62d568744f7c44e6ec9b0452fb7`
(previously `runtime-1-bf16-t5@sha256:db752274...`). This release is src-only on that runtime,
which is what makes the keyframe scene-lock default above safe to ship. The 1.0.16 entry records
that this runtime also carried torchao 0.18.0, which fails `from diffusers import
WanImageToVideoPipeline` on torch 2.7.1 (found on the 1.0.15 H200 smoke, fixed in 1.0.16).

**CI: the adversarial-audit workflow is removed (#431).**

Deletes `.github/workflows/adversarial-audit.yml` (the PR states the decision: no automated paid
security scanners). No `src/` or `deploy/` change.

## [1.0.14] -- 2026-08-17

**Fix: job-authored bundle paths and job-supplied R2 endpoints stay in contract (#426).**

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

**Fix: `finish_clip` is bounded by one wall-clock guard (#423).**

`finish_clip` had no wall-clock guard: of the subprocess sites on its path only the two inside
the cached NVENC probe were timed. One monotonic budget now opens on the first line of the
RunPod handler (so the R2 client build and the cold-start model mirror are inside it), is checked
between stages and inside the per-frame loops, and on expiry the handler RETURNS
`{"ok": false, "detail": "finish_clip deadline: <elapsed>s of <budget>s budget at stage <stage>"}`
instead of raising, which is the shape the `vivijure-cf` soft-degrade reader treats as a
completed-with-degrade output rather than a failed job. `VJ_FINISH_MAX_SECONDS` sets the budget
(seconds, default 420; junk, empty, zero and negative values fall back to the default, and there
is no value that turns the guard off). `FinishDeadlineExceeded` derives from `BaseException` so
the broad `except Exception` handlers on the path cannot swallow an expiry.
`assemble.probe_has_audio`, a fifth untimed subprocess on this path, takes an optional `timeout`
that defaults to `None`, so its render/assemble callers are unchanged.

Coverage is stated in `docs/finish-deadline.md`: 13 of 14 compute steps are inside the budget and
6 of those 13 are interruptible mid-step. Not covered: R2 GET/PUT (botocore socket timeouts
only), the two model loads, a stall inside a single decoded frame or one `interp.interpolate`
call, and the best-effort `.hash` sidecar. The PR says this does NOT license declaring a
`max_invocation_seconds` for the module.

**Build: runtime base re-baked and repinned (#390).**

`RUNTIME_REF_BF16` in `deploy/Dockerfile` and `.runpod/Dockerfile` moved from
`runtime-1-bf16-t5@sha256:0f3c9bd6...` to `runtime-1-bf16-t5@sha256:db75227454bdd6196c978475309f353cb082f93929fdb195419efb5580f9cf65`:
the same `t5` tag re-baked on the scheduled cadence, which the PR says refreshes the shipped CVE
posture.

**Deps and CI changes in this release (no worker behaviour change beyond the two entries above).**

- **`deploy/requirements.txt` and `deploy/requirements-verify.txt`:** `huggingface_hub`
  1.25.1 -> 1.27.0 (#410, #419), `torchao` 0.17.0 -> 0.18.0 (#419), `boto3` floor 1.43.58 -> 1.43.69 (#411,
  #420). These install in the runtime base, and the bumps merged after the base repin above
  (#390, 2026-08-03), so whether they reached the 1.0.14 image is not recoverable from the
  record. The `torchao` 0.18.0 pin is the one that later broke Wan i2v on torch 2.7.1 (see 1.0.15
  and 1.0.16).
- **`github/codeql-action` bumps (#412, #421)** in a workflow that #424 removed later in the
  same release.
- **CI:** `seed-build.yml` gains a `runner_host` dispatch input to pin the Plane C bake host
  (#402, #404) and extracts rclone into `RUNNER_TEMP` with an exactly-one-match assertion after a
  `cp` failure on a host carrying two extracted versions (#406); bakes routed to the
  `bake-capable` label (#415), with the card-dependent release smoke and CUDA-EP steps moved back
  to `gpu-device` because a runner without a GPU makes the release smoke print SKIP and exit 0
  (#416); the adversarial audit no longer skips Dependabot PRs (#414); the CodeQL workflow is
  removed (#424) and a free `coverage` check job is restored in `coverage.yml` (#425).
- **Docs and repo hygiene:** `RELEASES.md` pre-cut GHCR check scoped to the active release line
  and the 1.0.13 snapshot recorded (#399); `CLAUDE.md` refreshed (#413), a pre-prod GPU test
  spend rule added (#418) and a tag-gated-deploy note added (#417, whose diff here is `CLAUDE.md`
  only); Cursor Cloud `AGENTS.md` removed (#408); `dependabot.yml` comment dashes replaced with
  `--` (#409).

## [1.0.13] -- 2026-08-01

**Feat: a per-job tenant R2 credential, so one pooled endpoint can serve many tenants (#393).**

A job payload may carry an `r2` block (`endpoint`, `access_key_id`, `secret_access_key`,
`bucket`, optional `session_token`). When present and valid it is used for that job's tenant I/O
only (bundle, LoRAs, keyframes, clips, manifest, progress channel, reuse probes, `finish_clip`
and `i2v_clip`), and the handler strips it from the payload before anything downstream sees it.
Absent, the endpoint's `R2_*` environment is used exactly as before. Present but malformed,
including an explicit `"r2": null`, fails the job rather than falling back to the environment.
Contract: `docs/contract.md`, "Tenant R2 credential (all job types)".

The models mirror is a separate purpose on a separate credential: it now reads
`MODELS_R2_ENDPOINT`, `MODELS_R2_ACCESS_KEY_ID`, `MODELS_R2_SECRET_ACCESS_KEY` and
`MODELS_R2_BUCKET`, each falling back field by field to its legacy `R2_*` name. That fallback
deliberately keeps today's behaviour, including the mirror following a tenant's `R2_BUCKET`. An
endpoint that carries `MODELS_R2_ACCESS_KEY_ID` refuses any job without an `r2` block, even if a
complete `R2_*` credential sits in its environment. The PR states the deploy order: the control
plane sends the block on every submit, then this image ships, then templates are repinned with
`MODELS_R2_*`. Not done here and filed separately: removing the legacy `R2_*` fallback (#395,
which the PR marks as gating the shared tier accepting a real tenant) and short-lived per-job
credentials (#396). A per-job credential is bounded by tenant and job, not absent, so it is
weaker than a credentialless design.

**Feat: `health` attests the build, and keyframe records say which identity path ran (#375).**

`health` returns `baked`, the parsed bake stamp (`baked_utc`, `precision`, `model_version`,
`overlay`, `base_runtime`), beside the existing booleans; `null` means no stamp and `{}` means a
stamp that yielded nothing readable. `vj_baked` and `ok` are unchanged and the probe still
answers before the harness import. `keyframe_done` now carries `identity_requested` and
`identity_path`, recorded separately because InstantID falls back to the shared IP-Adapter pipe
when insightface finds no face and a slot with no usable reference renders from its LoRA alone,
and `onnx_providers`, read from `session.get_providers()` after `prepare()` (never the requested
list) and summarised as `all_cuda`, or `null` when the analyzer never loaded on that worker. It
does not make a CPU-only face path fail a render (the CPU fallback in `face_analyzer()` is by
design); it makes the question answerable from render history after the CPU-bound face path of
1.0.9 to 1.0.11 (#346, fixed in 1.0.12). The tests run against a fake ONNX session; the PR names
the first live `health` call and InstantID render on the next image as the real proof, and that
check is not recorded here.

**Fix: the i2v precision is measured, not asserted (#367).**

`plan()` reported the card's best possible precision (`quant_for`) as if it were a state; it now
reports what the loaders load (new `loaded_quant_for()`), for example bf16 for a 96 GB card that
CPU-offloads the inactive Wan expert instead of quantizing. A `model_precision` event, which the
verify gate had always read and nothing in `src/` sent, is now emitted, measured as the dominant
parameter dtype across `transformer` and `transformer_2` with a per-module dtype histogram
(`i2v_dtype` is `null` when unmeasurable, never a guessed string). The load compares requested
against resident dtype and raises `I2VPrecisionMismatch` on a mismatch: under the pinned
diffusers a `float8_e4m3fn` request silently yields fp32, unreachable today because no fp8 repo
is baked. The verify gate's BAK-4 became three fail-closed checks (reported, valid, matches the
request). `ensure_i2v_models(force=)` lets the FINAL tier's bf16 pull happen on an fp8-baked image
(the mechanism half of #339), and the cold-start log line that claimed fp8 weights were baked in
on every production worker now says only what the sentinel proves. The PR did not observe the
resident dtype under torchao and the real Wan load on a GPU.

**Deps and CI changes in this release (no worker behaviour change beyond the three entries above).**

- **`deploy/requirements.txt`:** `numpy` 2.3.5 -> 2.4.6, `huggingface_hub` 1.24.0 -> 1.25.1,
  `peft` 0.19.1 -> 0.20.0 (#377); `github/codeql-action` 4 -> 4.37.3 (#383). This release did not
  repin the runtime base (`RUNTIME_REF_BF16` stayed at `runtime-1-bf16-t5@sha256:0f3c9bd6...`, as
  in 1.0.12), and `deploy/requirements.txt` installs there, so the record does not show these pins
  reaching the 1.0.13 image.
- **Resolve gate for `deploy/requirements.txt` (#365):** a PR gate on `ubuntu-latest` and a
  push/dispatch gate inside the pinned runtime base, plus Dependabot `ignore` entries for
  `numpy>=2.5.0` and `tokenizers>0.22.2` (the two bumps behind the eight-day unbuildable `main`
  of 1.0.12, #351). #380 and #388 only change what the gate says: the `onnxruntime` line in a
  resolve is the CPU wheel, the gate has only been seen red on the numpy pin, and a green does not
  mean a dependency fix is live.
- **Image smoke needs a GPU (#368):** the smoke steps in `release.yml` and `runtime-build.yml`
  run with `--gpus all`, and a new `deploy/smoke_cuda_ep.py` checks a CUDA session in the runtime
  base and fails rather than skips when no GPU is visible.
- **Repin step (#370):** the runtime-build repin edits both `deploy/Dockerfile` and
  `.runpod/Dockerfile` (`hub-files-sync` requires them byte-identical) and opens its PR through
  the REST API, failing the run if it cannot (`gh` is absent on the Plane C runners).
- **Stacked PRs (#387):** the `tests.yml`, `adversarial-audit.yml`, `hub-files-sync.yml` and
  `requirements-resolve.yml` `pull_request` triggers fire on any base, and `ci` refuses a PR whose
  base is not `main`.
- **Comment-only or repo hygiene:** `runtime-build.yml` and `release.yml` comments retargeted at
  Plane C (#382); `.coverage` and `coverage.xml` gitignored (#394).
- **Docs:** `RELEASES.md` ledger rows for 1.0.12 (#361), 1.0.11/1.0.9/1.0.8/1.0.5 (#366) and the
  four `(tag SHA)` placeholders (#373); `docs/derisk-3arch.md` R2-credential scaffold corrected
  (#369).

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
