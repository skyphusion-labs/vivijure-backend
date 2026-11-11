# RunPod serverless endpoint configuration

How to size and pin the RunPod serverless endpoints that back the Vivijure render
and finish chain. This is the operator-facing reference: the GPU-tier rule, the
economics behind it, the per-endpoint actuals, and the two hard deploy
constraints (version-tag pinning, the account worker quota).

See also `docs/configuration.md` (per-endpoint env + the i2v tier table) and
`docs/operations.md` (build/deploy/pin mechanics).

**Do not pin `:1.0.1`** (t2 weight-digest rewrite; RunPod `unexpected EOF`). Incident:
[`runpod-1.0.1-weight-digest-eof.md`](runpod-1.0.1-weight-digest-eof.md).

## Current state (both read with `get-endpoint` on 2026-11-11; backend pools as changed by #625)

| | `vivijure-backend-hosted` | `vivijure-upscale-hosted` |
|---|---|---|
| id | `xe5d1ivs40ohjo` | `0m7puc409hqw8a` |
| Image | `vivijure-backend:1.1.18@sha256:3ec3849279aa18e6c58ba0716bd768366162a70d8a6d61e9bdde2254ee4e049a` | `vivijure-upscale:1.1.3@sha256:8b8b514688830b53c446b211052ddce962c1587b6df13bbefc45bda9833a3a94` |
| Pools | `BLACKWELL_96`, `ADA_80_PRO`, `HOPPER_141` | `BLACKWELL_96` |
| `workers` min / max | 0 / 12 | 0 / 10 |
| Idle timeout | 60 s | 10 s |
| Scaler | `QUEUE_DELAY` 4 | `QUEUE_DELAY` 4 |
| FlashBoot | on | on |
| Container disk | 300 GB (read 2026-11-06, see below) | 40 GB |
| Mounts | none | none |
| Execution timeout | 3600 s | 900 s |
| Min CUDA | 12.8 | 12.0 |

**B200 and B300 dropped from the backend pools (#625, decision on #606, 2026-11-11).** Before: `BLACKWELL_96`,
`ADA_80_PRO`, `HOPPER_141`, `BLACKWELL_180`, `BLACKWELL_288`. RunPod serves SDXL keyframes only. Keyframe jobs
complete on the 96 GB RTX PRO 6000 and on an 80 GB H100 (measured below), so the 180 GB and 288 GB cards add
VRAM the workload does not use, at 3.1x and 3.4x the RTX PRO 6000 per-second price. Under concurrency, jobs
were reaching them: the 4-in-flight run of 2026-11-06 billed $0.0023 per runtime-second, above the H200 rate.
Rollback: `update-endpoint` with the five-pool list above, in that order.

Backend env: `HF_HOME=/opt/models/hf-cache`, `VJ_MODELS_ROOT=/opt/models`, `VJ_MODEL_VERSION=1`,
`HF_HUB_OFFLINE=1`. `VJ_VOLUME_ROOT` and `VJ_VOLUME_KIND` are not set on the endpoint. The thin image
still sets them, and nothing is mounted there, so the worker logs that the volume is not mounted and
takes the R2 mirror.

**Why the disk is 300 GB (vivijure-backend#600).** The backend container disk read 80 GB until 2026-11-06, when
one worker filled it and failed every later job within seconds on ENOSPC. It was raised that day in two steps, as
the endpoint's release history records (`list-endpoint-releases`, field `template.containerDiskInGb`):

- v41, 2026-11-06T15:01:44Z: from 80 to 200, after the full-disk worker;
- v42, 2026-11-06T15:20:17Z: from 200 to 300, to hold the whole set below with room to spare rather than only the Wan set.

`get-endpoint` on `xe5d1ivs40ohjo` read `disk: 300` on 2026-11-06, and the 1.1.17 repin (v43) kept it at 300. A repin changes only the image and the workers, so it leaves the disk alone.

**Measured on 1.1.17, 2026-11-06,** one cold `i2v_clip` job (the own-gpu motion live check, job
`5bedf426-a997-4a1e-8926-8796c0c718c9-e1`, worker `5po0bwtutn8k0e`). Every figure is that job's own event, in GiB
(300 GB is 279.4 GiB):

| Point in the job | Free | Used | Event |
|---|---|---|---|
| `started`, after the base mirror | 250.9 | 28.5 | `started.disk_free_gb` |
| The i2v pull: Wan 2.2 I2V plus Lightning | | +156.8 | `i2v_mirror_complete.total_bytes` 168,327,909,540 |
| `complete`, after the clip | 94.1 | 185.3 | `complete.disk_free_gb` |

The free space fell by exactly the bytes pulled (250.9 less 156.8 is 94.1), so each weight is stored once on disk.
Of the 156.8 GiB, the Lightning leg is 39.2 GiB (its final rclone stats line); the rest is the Wan 2.2 I2V repo,
which R2 lists at 126.2 GB (117.5 GiB). This record earlier carried 114 GiB for the whole i2v set and 171 GiB for a
full worker: both were too low, because Lightning is 39.2 GiB, not about 3.

**Why 80 GB and 200 GB both failed.** A worker that has run an i2v job holds 185.3 GiB, and the job floor keeps 4
GiB free, so it needs about 189 GiB. 80 GB is 74.5 GiB, so the i2v pull alone overflowed it. 200 GB is 186.3 GiB,
so a worker there ran out on its last leg, Lightning, which is what happened at 15:20Z. 300 GB (279.4 GiB) leaves
94 GiB free after the whole set. The model cache persists across jobs on purpose, so a worker that has run every
kind of job holds all of it.

Since #600 a too-small disk fails loudly instead of silently: a pull that cannot fit fails its job by number
before it writes, a disk below 4 GiB free fails a job before it starts (`VJ_MIN_FREE_GB` sets the floor), and
either, or any ENOSPC, returns `refresh_worker` so RunPod replaces the worker. Every job's `started`, `complete`
and `error` progress events carry `disk_free_gb`.

**Before: why 80 GB.** A stills job pulls 26.22 GB from R2 (`mirror_complete total_bytes`). Six jobs on two
80 GB workers ran with no ENOSPC. That held for stills-only workers and not for a worker that also took an i2v job.

**Why 180 s idle (historical).** The live endpoint reads 60 s idle and max 12 workers on 2026-11-04;
this section explains the earlier 180 s value. Measured on the same image: a cold keyframe cost $0.13 to $0.23 of worker time and
a warm one $0.02, at $3.49 per hour. Idle costs $0.058 per minute, so holding a worker pays when the next
job arrives within about 2 to 3.5 minutes. A one-off job pays up to about $0.17 of extra idle.

**Upscale rollback, moved 2026-10-30:** previous image
`vivijure-upscale:1.0.4@sha256:533e1f0e366d12b6230ab08c7b9a61a51e00f143ef3610b3c8d90b880a67a413`, all other
settings the same. 1.1.3 honours `target_height`, which 1.0.4 ignores and runs 2x. A host from vivijure-cf
v1.42.0 sends `target_height` by default, so rolling back under one delivers 2x against a budget priced at
the target. Tested on a separate endpoint in R2 mode: 1344x768 to 1890x1080 in 11.5 s, and 720x1280 to
1080x1920 in 6.0 s. The presigned path has not run on this image yet.

**Repinning the backend endpoint (standing procedure).**
1. Drain the pool to 0, repin by digest, then restore max 12 and idle 60 s.
2. Wait 65 s after the release before submitting anything. Renders through the studio got `keyframe /run -> 409` in that window after the 1.1.7 repin.
3. On a thin image built on the bare toolchain, after a repin, every worker is cold and pulls about 24 GB from R2 (on a code image built on the stills weights base, a stills job pulls nothing: [stills-weights-base.md](stills-weights-base.md), #564), and a warm-up batch does not change that: each NEW worker pulls again. So keep the number of concurrent cold workers small, about 4 to 6, by submitting in small batches, and budget one cold pull per worker. On 2026-11-06, 15 cold workers at once slowed pulls to 2.6 MiB/s and stretched cold jobs to about 20 minutes each; 6 at once pulled normally (cold delay about 2 to 3 minutes). Submitting 2 first and the rest once they ran did NOT help after a new image: the second batch started on new workers, cold.

**Backend 1.1.17, pinned 2026-11-06 (release v43, 16:05:16Z).** A code-only release on the existing stills base: no restage.

- **Code:** main `eed4b57`. It adds #601 (`ip_adapter_scale_per_slot` reaches the two-person draw) and #602 (disk guards):
  - a pull-size guard (`check_pull_fits`);
  - a 4 GiB job floor (`check_job_floor`, `DEFAULT_MIN_FREE_GB`);
  - `refresh_worker` on a disk failure;
  - `disk_free_gb` on events.
  The image is built from `deploy/Dockerfile.thin` with `TOOLCHAIN_REF=vivijure-backend:stills-t1-m1@sha256:6d0e19bd18bdba3318e510fcc0055d9d75c1a1353df918795dc92be0d70740b6`.
  Since `1ac386e`, `deploy/` changed only in #591 (`stills-weights.Dockerfile`, and a comment in `Dockerfile.thin`).
- **Image:** `vivijure-backend:1.1.17@sha256:2269c849c72f4f293ee6954765c8c6a98cc44a5f9aa27dd37be9b7c551332393`.
  The push took 3 s. It pushed 2 code layers and reused 26. The base is an exact prefix, and all three weight layers were reused.
  The push digest and the GHCR read-back match.
- **Checks in the image:**
  - `--check-stills` prints `complete` and exits 0;
  - `harness/disk.py` is present, with `DEFAULT_MIN_FREE_GB` 4.0, `check_job_floor` and `check_pull_fits`;
  - `refresh_worker` is in `harness/handler.py`, and `disk_free_gb` is in `harness/progress.py`;
  - `ip_adapter_scale_per_slot` is a `MultiCharConfig` field;
  - the code files hash the same as at `eed4b57`, and the image holds 30 .py files (#602 adds `harness/disk.py`);
  - `.vj-stills-complete` reads `1`, there is no `.vj-baked`, and the CPU import smoke passed.
- **Repin:** held until a live film finished at 16:02:40Z. Before the drain, 0 jobs were in progress and 0 queued.
  The pool was drained to 0, then restored to max 12 with a 60 s idle timeout, and the disk was kept at 300 GB.
  65 s after the release the rollout read 100% on 1.1.17, with 0 unhealthy. No render job was submitted.

**Backend rollback from 1.1.17:** image
`vivijure-backend:1.1.16@sha256:5d7ef08c9391ccece589c28b492b8092b5b701905fcdb71011770b5d627c541c` (same stills base), idle timeout 60 s,
max 12, disk 300 GB. Follow the repinning procedure above. 1.1.16 has no disk guards, so a full worker fails silently again.

**Backend 1.1.16, pinned 2026-11-06 (release v40, 13:45:37Z).** A code-only release on the existing stills base: no restage.

- **Code:** main `6edad52`, adding #597. A two-person scene-lock plate draws without the pose skeleton by default while the plate count guard runs.
  Built from `deploy/Dockerfile.thin` with
  `TOOLCHAIN_REF=vivijure-backend:stills-t1-m1@sha256:6d0e19bd18bdba3318e510fcc0055d9d75c1a1353df918795dc92be0d70740b6`.
  `deploy/` is unchanged since `82b2324`.
- **Image:** `vivijure-backend:1.1.16@sha256:5d7ef08c9391ccece589c28b492b8092b5b701905fcdb71011770b5d627c541c`.
  The push took 2 s (13:44:55Z to 13:44:57Z). It pushed 2 code layers (190 KB) and reused 26.
  The base's 26 layers are an exact prefix of the image. All three weight layers (`76c9fde8193f`, `f953853e0b64`, `89cf0b497764`) were reused.
  The push digest and the GHCR read-back match.
- **Checks in the image:**
  - `--check-stills` prints `complete` and exits 0;
  - `.vj-stills-complete` reads `1`, and there is no `.vj-baked`;
  - `plate_pose_default_off` is present, and returns True for a two-person shot and False for one person under the default config (`plate_count_retries` 3);
  - `orchestrator.py`, `keyframe.py`, `pipeline.py`, `contract.py` and `config.py` hash the same as at `6edad52`, and the image holds 29 .py files. The CPU import smoke passed.
- **Two-person scene-lock stills redraw once.** The reuse hash gains `plate_pose: off` for exactly those shots (`orchestrator.py`). One-person shots keep their hash.
- **Repin check:** 65 s after the release the rollout read 100% on 1.1.16, with 0 unhealthy. No render job was submitted.

**Backend rollback from 1.1.16:** image
`vivijure-backend:1.1.15@sha256:30c08660d643f2b7629b90ac92e686603c0d227aca257e11debed71820e20f18` (same stills base, so a host that pulled
1.1.16 keeps the weight layers), idle timeout 60 s, max 12, same disk. Follow the repinning procedure above.

**Backend 1.1.15, pinned 2026-11-06 (release v39, 13:01:31Z). The first image on the stills weights base (#591, #564).**

- **Code:** main `82b2324` (#591, #592 stall kill with `mirror_progress`, #594 verify temp dir). Built from `deploy/Dockerfile.thin`
  with `TOOLCHAIN_REF=vivijure-backend:stills-t1-m1@sha256:6d0e19bd18bdba3318e510fcc0055d9d75c1a1353df918795dc92be0d70740b6`.
  `Dockerfile.thin` changed only in comments since `1ac386e`, and `requirements.txt` is unchanged.
- **Image:** `vivijure-backend:1.1.15@sha256:30c08660d643f2b7629b90ac92e686603c0d227aca257e11debed71820e20f18`, 29.08 GiB
  compressed. The base's 26 layers are a prefix of it, and its two code layers are 190 KB. The push took 8 s
  (13:00:26Z to 13:00:34Z). The push digest and the GHCR read-back match.
- **Checks in the image:**
  - `--check-stills` prints `stills tree at /opt/models is complete` and exits 0;
  - `.vj-stills-complete` reads `1`, and there is no `.vj-baked`;
  - `orchestrator.py`, `keyframe.py`, `pipeline.py`, `contract.py`, `config.py`, `verify.py` and `harness/models_mirror.py` hash the same as at `82b2324`;
  - the image holds 29 .py files, and the CPU import smoke passed.
- **Repin check:** 65 s after the release the rollout read 100% on 1.1.15, with 0 unhealthy. No render job was submitted.
  The first worker started at 13:01:39Z and was still `INITIALIZING` at 13:02:46Z, because it pulls the 29 GB image. That pull is not billed.
  The time to ready and the cold job's `executionTime` are for the live check ([stills-weights-base.md](stills-weights-base.md)).

**Backend rollback from 1.1.15:** image
`vivijure-backend:1.1.14@sha256:fe34935b797d4582d3e9f838dca3337fcd8410bb9130a33a210476c36e3f7e85` (no stills base: workers pull the
stills set from R2 again), idle timeout 60 s, max 12, same disk. Follow the repinning procedure above.

**Backend 1.1.14, pinned 2026-11-06 (release v38, 11:09:01Z).** A thin image (`deploy/Dockerfile.thin`) at main `a2f3b18`, built on the
Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no change to `deploy/` or `requirements.txt` since `1ac386e`). It adds #589: `plate_count_retries` defaults to 3 for two-person-plus shots. The guard is on by default, and the cold-start doc is corrected.

In the image, `KeyframeConfig().plate_count_retries` and `KeyframeParams().plate_count_retries` both read 3.

**Two-person stills redraw once.** The reuse hash includes `plate_count_retries` when it is set on a shot the guard can run on (2 or more people, `orchestrator.py`). So the first re-render of each stored two-person still draws it again. One-person and empty shots keep their hash.

Image `vivijure-backend:1.1.14@sha256:fe34935b797d4582d3e9f838dca3337fcd8410bb9130a33a210476c36e3f7e85`. The push digest and the GHCR read-back match.
In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py`, `contract.py` and `config.py` hash the same as at `a2f3b18`, and the image holds 29 .py files. The CPU import smoke passed.
Repin check: the rollout read 100% on 1.1.14, with 0 unhealthy, 65 s after the release. No render job was submitted.

**Backend rollback from 1.1.14:** image
`vivijure-backend:1.1.13@sha256:7bd7293457c4e69c24c387072a9e3456a0a0ff528843c6eda562b6bfacda8911`, idle timeout 60 s,
max 12, same disk. Follow the repinning procedure above. On 1.1.13 the guard is off by default.

**Backend 1.1.13, pinned 2026-11-06 (release v37, 10:38:35Z).** A thin image (`deploy/Dockerfile.thin`) at main `a3d48a6`, built on the
Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no change to `deploy/` or `requirements.txt` since `1ac386e`). It adds #587, `plate_count_retries`: a guard that redraws a two-person plate when it has a third face. The default is 0, which is OFF.

The guard counts faces with `server.face_analyzer()`, the insightface antelopev2 analyser that the scene-lock identity crop already uses. It needs no new model: the model mirror pulls the `antelopev2` leg for every stills or full job (`harness/models_mirror.py`), and in the image `from insightface.app import FaceAnalysis` imports (onnxruntime 1.26.0).

Image `vivijure-backend:1.1.13@sha256:7bd7293457c4e69c24c387072a9e3456a0a0ff528843c6eda562b6bfacda8911`. The push digest and the GHCR read-back match.
In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py`, `contract.py` and `config.py` hash the same as at `a3d48a6`, and the image holds 29 .py files. The CPU import smoke passed.
Repin check: the rollout read 100% on 1.1.13, with 0 unhealthy, 65 s after the release. No render job was submitted.

**Backend rollback from 1.1.13:** image
`vivijure-backend:1.1.12@sha256:5164674342cdb841013aacc3498762ccf0322d67e306955874309071d179dde5`, idle timeout 60 s,
max 12, same disk. Follow the repinning procedure above.

**Backend 1.1.12, pinned 2026-11-06 (release v36, 09:49:22Z).** A thin image (`deploy/Dockerfile.thin`) at main `ec6dd0f`, built on the
Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds #585, `multi_char.pose_spread`. Image
`vivijure-backend:1.1.12@sha256:5164674342cdb841013aacc3498762ccf0322d67e306955874309071d179dde5`. The push digest and the GHCR read-back match.
In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py` and `contract.py` hash the same as at `ec6dd0f`, and the image holds 29 .py files, as the tree does. The CPU import smoke passed.
Repin check: the rollout read 100% on 1.1.12, with 0 unhealthy, 65 s after the release. No render job was submitted.

**Backend rollback from 1.1.12:** image
`vivijure-backend:1.1.11@sha256:bbc525a1b22f707f64601fae89a23c8641c39ef8fd9ca2202cd170ebdfe0da6c`, idle timeout 60 s,
max 12, same disk. Follow the repinning procedure above.

**Backend 1.1.11, pinned 2026-11-06 (release v35, 09:15:47Z).** A thin image at main `69ff143`, on the same toolchain. It adds #584,
`debug_intermediates`: a diagnostic job flag that saves the scene-lock plate beside the still. Image
`vivijure-backend:1.1.11@sha256:bbc525a1b22f707f64601fae89a23c8641c39ef8fd9ca2202cd170ebdfe0da6c`. The push digest and the GHCR read-back match.
`orchestrator.py`, `keyframe.py`, `pipeline.py` and `contract.py` hash the same as at `69ff143`, and the image holds 29 .py files. The import smoke passed.
Repin check: the rollout read 100% on 1.1.11, with 0 unhealthy, 65 s after the release. No render job was submitted.
Rollback from 1.1.11: `vivijure-backend:1.1.10@sha256:6bdbe03ad7653597768d55f7acbf1e01e698a21181a3d46dfb4cdd673e4da052`.

**Backend 1.1.10, pinned 2026-11-06.** A thin image (`deploy/Dockerfile.thin`) at main `1d78182`, built on
the Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds #582, `plate_people_free_background`: a people-free band between two people under regional text. The default is OFF, so nothing changes until a render sets it.

In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py` and `contract.py` hash the same as at `1d78182`, and the image holds 29 .py files, as the tree does. The CPU import smoke passed. The digest pinned is the one read back from GHCR.

Repin check: the pool was drained to 0, then restored to max 12 with a 60 s idle timeout. Nothing was submitted for 65 s after the release. A fresh-draw keyframes-only render through the studio, `film-2072dd45-98c7-4eb1-bfa2-a391ba7b481f` (seed 111006), drew all four stills on 1.1.10 worker `iizcpgx1m3hkn3`. RunPod jobs `78278008-...-u1` and `d9466de1-...-u1` report no `reused`. The applied config shows `plate_ip_adapter_scale: 0`, `plate_regional_text: false` and `plate_people_free_background: false`. Its two-person still shows TWO people (Mara and Dev), but Mara wears a black hoodie over a grey top rather than her reference outfit (vivijure-cf#1143).

**Backend rollback from 1.1.10:** image
`vivijure-backend:1.1.9@sha256:ed7bbcc42904908f908f88a4b34babbabac11fc46b76b7854000876bbb25dd95`, idle
timeout 60 s, max 12, same disk. Drain the pool to 0 and restore it after the repin, then wait 65 s before
submitting.

**Backend 1.1.9, pinned 2026-11-06.** A thin image (`deploy/Dockerfile.thin`) at main `c63b76c`, built on
the Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds #580, `plate_regional_text`: per-region text for a two-person plate. The default is OFF, so nothing changes until a render sets it.

In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py` and `contract.py` hash the same as at `c63b76c`, and the image holds 29 .py files, as the tree does (#580 adds one). The CPU import smoke passed.

Repin check: the pool was drained to 0, then restored to max 12 with a 60 s idle timeout. Nothing was submitted for 65 s after the release. A fresh-draw keyframes-only render through the studio, `film-a90c324a-f2b9-4364-97a3-061d19150995` (seed 110905), drew all four stills on 1.1.9 workers. RunPod jobs `bdf2c9ba-...-u2` and `bf4d25c9-...-u1` report no `reused`, and the applied config shows `plate_ip_adapter_scale: 0` and `plate_regional_text: false`. Its two-person still shows THREE people (Dev twice), the same duplicate-person failure seen on 1.1.8 with Mara twice. Both new knobs were off, so this is the existing path (vivijure-cf#1143).

**Backend rollback from 1.1.9:** image
`vivijure-backend:1.1.8@sha256:a7f9885f245a6db2fa1e4c749cf27da9c8b3d1e5e22494d141f68517b42231f6`, idle
timeout 60 s, max 12, same disk. Drain the pool to 0 and restore it after the repin, then wait 65 s before
submitting.

**Backend 1.1.8, pinned 2026-11-06.** A thin image (`deploy/Dockerfile.thin`) at main `957e4ce`, built on
the Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds #578, `KeyframeConfig.plate_ip_adapter_scale`: a two-person plate can take each person's cast portrait. The default is 0, which is OFF, so nothing changes until a render sets it.

In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py` and `contract.py` hash the same as at `957e4ce`, and the image holds 28 .py files, as the tree does. The CPU import smoke passed.

Repin check: the pool was drained to 0, then restored to max 12 with a 60 s idle timeout. Nothing was submitted for 60 s after the release, which avoids the window in which renders through the studio got `keyframe /run -> 409` after the 1.1.7 repin. A fresh-draw keyframes-only render through the studio, `film-ece55b2a-80a4-4cbc-ad7d-63266d13c24c` (seed 110804), drew all four stills on 1.1.8 workers. RunPod jobs `5392d53a-...-u2` and `4ca8b3fc-...-u1` report no `reused`, and the applied config shows `plate_ip_adapter_scale: 0`.

**Backend rollback from 1.1.8:** image
`vivijure-backend:1.1.7@sha256:d9da0b5a2d4276bb4de149169b513444ace7bf21f10009656cc45105e5596e6c`, idle
timeout 60 s, max 12, same disk. Drain the pool to 0 and restore it after the repin, then wait 60 s before
submitting.

**Backend 1.1.7, pinned 2026-11-06.** A thin image (`deploy/Dockerfile.thin`) at main `7f66faa`, built on
the Vultr bake box from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds:
- #575: the keyframe reuse hash covers the built prompts and `KEYFRAME_ENGINE_VERSION` (`kf-2026-11-06-chunked`). This closes #573.
- #576: `KeyframeConfig.face_crop_pad`, default 0.35, which is today's crop.

In the image, `orchestrator.py`, `keyframe.py`, `pipeline.py` and `prompt_chunks.py` hash the same as at `7f66faa`, and the image holds 28 .py files, as the tree does. The CPU import smoke passed.

**#575 makes every stored still redraw once.** Every hash stored before 1.1.7 lacks the engine version, so the first re-render of any existing project draws all its stills again at full GPU cost.

Repin check: the pool was drained to 0, then restored to max 12 with a 60 s idle timeout. A `preview` job on the verify bundle (`c33e4fb1-81e7-487b-beeb-bbcd411bffb6-u1`) ran on a fresh 1.1.7 worker and drew its still; it reports no `reused`. The same job reused its still on 1.1.6 (`ecd05663-...-u1`).

**Backend rollback from 1.1.7:** image
`vivijure-backend:1.1.6@sha256:cc4111358aea3d318ddd98330e5606b18d3184f42fd05fb68696837bb3702edd`, idle
timeout 60 s, max 12, same disk. Drain the pool to 0 and restore it after the repin. A rollback does NOT
restore reuse of stills drawn on 1.1.7, because their stored hashes carry the engine version.

**Backend 1.1.6, pinned 2026-11-06.** A thin image (`deploy/Dockerfile.thin`) at main `67e883e`, built on
the Vultr bake box ([bake-box.md](bake-box.md)) from the same `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`
(no `deploy/` change since `1ac386e`). It adds #572:
- the keyframe prompt is encoded in up to two 75-token chunks;
- one adult clause per shot;
- the film `setting` from the board.

In the image, `keyframe.py`, `prompt_chunks.py` and `contract.py` hash the same as at `67e883e`, and the image holds 28 .py files, as the tree does. The CPU import smoke passed.

**It is a PARTIAL fix for vivijure-cf#1143.** Proof: `film-1d0dad23-c650-4319-877d-04e9ea7076cc` (the film-076828c5 board under a new project, seed 110501, every still drawn fresh):
- in the two-shot, Mara's top is now black (it was maroon on 1.1.5);
- her hair is still pinned up rather than curls, and her jacket shows a grey hood;
- the single shots did not regress.

#1143 stays open. A re-render under the OLD project returned the 1.1.5 stills marked `reused` (film-0afa679b), because the reuse hash ignores prompt building (#573). To see a prompt change, use a new project or a new seed.

Repin check: the pool was drained to 0 and then restored to max 12 with a 60 s idle timeout. A fresh-draw run, `film-08fd6115-265d-4387-8046-add8d88d9650` (seed 110502, so the reuse hash misses), drew all four stills on 1.1.6 workers. RunPod jobs `3a37166d-...-u1` and `2fcb954b-...-u1` completed, and neither reports `reused`.

**Backend rollback from 1.1.6:** image
`vivijure-backend:1.1.5@sha256:5b3e7140ac6b7ddcd8ebf0d65c034cf6501aa5049b1a7c47db56b8b62b887ec6`, idle
timeout 60 s, max 12, same disk. Drain the pool to 0 and restore it after the repin, or old workers keep
serving.

**Backend 1.1.5, pinned 2026-11-04.** A thin image (`deploy/Dockerfile.thin`) at main `1ac386e`, built on
the Vultr bake box ([bake-box.md](bake-box.md)) from `toolchain-t1@sha256:456b417bffc7d80cef7e2f93d400beda9ebbea38dcb5e4efc7676460eee4d45d`,
the same toolchain as 1.1.4 (no `deploy/` dependency change since). It adds #560, #561 and #562 (the
character description in the keyframe and plate prompts, within the 77 token limit) and #565 (a deadline
on every model pull). Proof: a `preview` job on `bundles/verify/Verify_Smoke.tar.gz` (job
`82644d5a-dc84-446d-b4b5-ba898b447787-u1`) ran on a fresh 1.1.5 worker after the pool was drained to 0,
an H100 80GB, cold: 74 s delay, 527 s execution, COMPLETED with one keyframe.

**Backend rollback from 1.1.5:** image
`vivijure-backend:1.1.4-kf-e54b0d0@sha256:a9b86c7bfd8f886601fe07281fdd38801a7781be9bebd21aa367776aef3f8a09`
(1.1.4 with the #562 `keyframe.py` laid over it), idle timeout 60 s, max 12, same disk. Drain the pool to
0 and restore it after the repin, or old workers keep serving.

**Previous values, for rollback:** image
`vivijure-backend:1.1.4@sha256:44a571135d2f9f4ecd88e603a62e4bab2a98cb81e0b947614261fcfb5ba2cbc7`, same disk and idle timeout
as 1.1.3. Before that:
`vivijure-backend:1.1.3@sha256:dc04da20fb0ec28136b942c2dffe9608b75f631d8de597e3855e3165ad670e64`, same
disk and idle timeout. On 1.1.3 a cold `finish_clip` pulls the whole base set again (#553). Before that:
`vivijure-backend:1.1.2@sha256:69f06ab95ab75530af7e0fa135c768548d339435a3714036b6f32e95a2b0d586`, same
disk and idle timeout. Before that: 1.1.1, same disk and idle timeout, digest
`sha256:0ebdc0e19b48676e310a10c863c7dfc4e6a8c642b3fbd887e73e248a6780157f`. Before that: image
`vivijure-backend:1.1.0@sha256:b42084bb1e295976d4ad555566aee1240c5ce86f1692a33560a127ecb155e6ac`,
disk 300 GB, idle timeout 10 s. That image starts a 114 GiB Wan pull on every keyframe job and needs
the larger disk, so restore all three together.

### Pool prices

Serverless prices per hour, from `list-gpu-types` on 2026-10-30:

| Pool | GPU | Price |
|---|---|---|
| `BLACKWELL_96` | RTX PRO 6000 | $3.49 |
| `ADA_80_PRO` | H100 | $5.39 |
| `HOPPER_141` | H200 | $6.99 |
| `BLACKWELL_180` | B200 | $10.79 |
| `BLACKWELL_288` | B300 | $11.99 |

Nothing has been measured on `ADA_80_PRO` or `HOPPER_141`. The times below are not evidence for those pools, and a job that lands on one costs more per hour than `BLACKWELL_96`.

### Measured keyframe times, image 1.1.1

80 GB, no volume, RTX PRO 6000, one shot, `final` tier, scene lock, trained LoRA supplied.

| Run | Queue | Execution | First keyframe from submit |
|---|---|---|---|
| Cold | 161 s | 233 s | 6 min 31 s |
| Cold | 6 s | 130 s | 2 min 12 s |
| Warm | 0.2 s | 21 s | 22 s |
| Warm | 0 s | 25 s | 26 s |

Mirror on the cold run: 136 s at 192.5 MB/s. Each model load after that took 1.5 to 7 s.

### Measured finish and keyframe times, image 1.1.4

Three jobs on one worker of a separate test endpoint: 80 GB, no volume, the same pools, RTX PRO 6000,
`workers.max` 1. Clip: 854x480, 24 fps, 97 frames. Cost is execution time at $3.49 per hour. It does not
include cold start or idle.

| Job | Queue | Execution | Legs pulled | Bytes pulled | Cost |
|---|---|---|---|---|---|
| Cold `finish_clip`, RIFE 2x | 178 s | 14.6 s | `rife` 1.3 s | 12,186,817 | $0.014 |
| `regen_shot`, one shot, `draft`, no LoRA | 0 s | 117 s | `hf-cache` 78.9 s, `antelopev2` 4.3 s | 26,236,131,927 | $0.113 |
| Warm `finish_clip`, RIFE 2x | 0 s | 6.7 s | none | 0 | $0.006 |

The cold job wrote 193 frames at 48 fps to a film folder (`ffprobe`). On 1.1.3 the same kind of job
ran for 477 s and $0.46, and the step did not run (#553).

One run on one worker. Not run on this image: a face restore job, which pulls `GFPGANv1.4` and `facexlib`;
a full mirror after a finish job; and a mirror that hangs. Tests cover those, except that no live run
measures the 120 s mirror budget against a slow worker.

### Two cold submits ran serially

Tested twice at `QUEUE_DELAY` 4 with `workers.max` 2. The second job waited for the first worker, then
ran on it. On the cold pair, a second worker only began initializing about 2 min 12 s after submit, as
the first job finished. So a burst of cold jobs queues before it fans out. Tracked in #545.

## Recommendation

**Run every endpoint on a high-end Blackwell-line GPU. Never the budget
L4 / L40S tiers.**

Realized across the fleet as two tiers, both Blackwell-line:

- **Render backend** (`vivijure-backend`): **RTX PRO 6000 first, then H100 and H200; no B200 / B300**
  (#625). This line used to read "SDXL keyframes, i2v, LoRA training -> datacenter top tier, B200 / H200".
  i2v and LoRA training no longer run on RunPod (SDXL keyframes only, Conrad 2026-10-30; Wan training
  retired, cp#589). Keyframes are proven on 96 GB and 80 GB cards, so the B-class cards bought VRAM nothing
  used, at about three times the price.
- **Finish-chain endpoints** (upscale; audio-upscale and musetalk are both RETIRED, see below) ->
  **RTX PRO 6000 Blackwell (96 GB).**

This is the standing standard. Do not "save money" by dropping a finish endpoint
to L4 / L40S; as the economics below show, the cheaper card is the more
expensive choice for a GPU-bound job.

## Why (justification)

### 1. The render backend basically REQUIRES the high-end card

The SDXL keyframe gen + i2v + LoRA training are heavy on both VRAM and compute,
and the backend images are built `cu128` / torch 2.8 for the Blackwell line
(`sm_120`). It is not a preference; the workload needs the silicon. Running it on
anything lower either will not fit in VRAM or will not run the compiled kernels.

### 2. For every other endpoint, the high-end Blackwell tier SAVES money

Serverless billing is **per second of active execution**, not per hour. So the
unit cost of a job is:

```
$/job = ($/sec of the GPU tier) x (seconds the handler runs)
```

A faster card finishes a GPU-bound job in fewer seconds, and the shorter
wall-time more than offsets the higher per-second rate. The premium card comes
out **cheaper per job**, not just faster.

On top of that:

- **Supply, not just speed.** The RTX PRO 6000 Blackwell tier is high-supply, so
  jobs schedule immediately. The budget L4 / L40S tiers are low-stock and
  **throttle**: jobs queue waiting for a free worker, and a queued job holds its
  worker slot longer. That is effectively *more* cost and worse latency. (This is
  exactly why `vivijure-video-upscale` was moved onto Blackwell.)
- **Scale-to-zero means no idle penalty.** With `workersMin = 0` plus flashboot
  standby (see billing note below), an idle endpoint costs **$0** regardless of
  tier. There is no carrying cost to provisioning the premium card; you only pay
  premium seconds while a job is actually running, and there are fewer of them.

Net: fastest finish + no throttling + per-second billing = **lowest $/job AND
best latency.** The "expensive" card is the cheaper one when the work is
GPU-bound.

### 3. The caveat that makes or breaks the math: the handler must be GPU-bound

Everything above holds **only when the endpoint's handler is GPU-bound.** A
CPU-bound or IO-bound handler on a Blackwell card pays premium per-second rates
for idle silicon (the pre-fix `vivijure-video-upscale`, which was doing CPU
encode/lipsync work on a GPU card and bleeding cost). The fix in that case is to
**make the handler GPU-bound** (move the encode/restoration onto the GPU), NOT to
downgrade the card. Downgrading just trades one waste for slower waste.

Rule of thumb before trusting the economics: confirm the handler saturates the
GPU for the bulk of its wall-time. If it does not, fix the handler first.

## Per-endpoint reference (live config)

The two endpoints in "Current state" are the only ones on the account (2026-10-30). The endpoints this
page used to list (`t9wcvlxh8rc5la`, `uf4iwoen5r48zx`, `zqb7tougbqfkqa`, `4q8idwbk6tyqbq`) no longer
exist, and their IDs are history. Read `GET /v2/serverless` rather than this page before acting.

## Deploy constraint 1: pin by `:version`, never `:sha`

Pin each endpoint's template image to its **`:version` tag** (e.g.
`ghcr.io/skyphusion-labs/vivijure-upscale:0.1.0`). RunPod's immutable
`:sha-<digest>` pin does **not** work for these endpoints; the version tag is
what RunPod resolves on the next cold start. The image build CI must therefore
emit a `:version` tag (see the build-image workflow's "Compute tags" step); a
build that only pushes `:sha` / `:latest` cannot be pinned. The now-retired
`vivijure-musetalk` and `vivijure-audio-upscale` satellites followed the same
rule, and their build workflows are the reference if you add a satellite.

Pinning is a deliberate, separate step from building: a build does not touch the
live endpoint (`docs/operations.md`). You pin the template, and the endpoint pulls
the new image on its next cold start.

After an image change, jobs can land on stale workers until the rollout finishes. Read the worker's `version` and image before crediting a proof.

Keep a test endpoint until its logs are read. A worker's logs go when the endpoint is deleted: `stream-worker-logs` then returns 404.

## Deploy constraint 2: the account worker quota

**The quota is 50 `workers.max` across all endpoints** (measured 2026-10-30: `POST /v2/serverless` refused
with "Max workers across all endpoints must not exceed your workers quota (50)" when the sum was 50). The
sum now is **40** (30 + 10). The old figures on this page (30 cap, 25 plan, 19 sum) no longer hold.

Consequence: **adding an endpoint, or raising `workers.max`, can be refused at create or update.** Check
the sum first. Lower another endpoint deliberately if needed, and read the result back.

## Billing note (so nobody re-flags standby)

- **Per-second billing while running.** A worker bills per second while it is
  running, which includes the idle timeout after a job (see RunPod's rule below).
- **`workersMin > 0` = "Always Active" = billed 24/7**, idle or not. Keep
  `workersMin = 0` on every endpoint. This is the only knob that creates idle
  spend; check it (not standby) when worried about a bleed.
- **`workersStandby` + flashboot = warm idle workers that bill $0 while idle.**
  Standby is free cold-start insurance: a standby worker bills only when it
  actually picks up a job. Do not mistake a non-zero `workersStandby` for idle
  spend; it is not.
- **RunPod's rule ([worker lifecycle](https://docs.runpod.io/serverless/workers/overview#worker-lifecycle)):**
  a worker bills while it is running, which is cold start, processing, and waiting through the idle
  timeout after its last job. A scaled-down worker, shown as Idle, does not bill. So the $0.058 a minute
  under "Why 180 s idle" stands for the idle timeout, and it is not standby.
- **Evidence on standby only, 2026-10-30 on `xe5d1ivs40ohjo`:** hourly billing for 18:00 to 20:00Z read
  $0.14 and $0.46 while 11 to 19 workers read `IDLE` in `list-endpoint-workers`. These buckets do not
  measure the idle timeout. That part rests on RunPod's rule.
