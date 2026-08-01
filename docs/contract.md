# The contract

This is the data contract between the Vivijure control plane and the render backend: the
project bundle the backend reads, the render job it is given, and the result it returns.
Everything here is typed in [`contract.py`](../src/vivijure_backend/contract.py) and parsed
forgivingly: unknown keys are ignored and only fields the renderer actually consumes are
surfaced, so the control plane can add authored fields without breaking an older backend.

The contract is the *only* thing carried over from the control plane; it is a data shape, not
borrowed implementation. The backend is a clean-sheet implementation written against it.

## The project bundle

The control plane writes a gzipped tar to R2 and passes its key as `bundle_key`. The backend
extracts it (rejecting symlinks, hardlinks, and path-traversal entries) into the project
working tree.

```
bundle.tar.gz
├── storyboard.yaml              # the storyboard (required)
└── characters/
    ├── registry.json            # slot -> {name, prompt}
    └── refs/                     # default refs dir (override with storyboard.refs_dir)
        ├── A/                    # per-slot training / IP-Adapter references
        │   ├── ref_01.png
        │   └── ref_02.jpg
        └── B/
            └── ref_01.png
```

- `storyboard.yaml` is required; the bundle is rejected without it. YAML is a superset of
  JSON, so a `storyboard.json` is accepted verbatim under that filename.
- `characters/registry.json` is optional (a character-free storyboard is valid). Reference
  images are read from `characters/refs/<SLOT>/` unless the storyboard sets `refs_dir`.
  Accepted ref extensions: `.png`, `.jpg`, `.jpeg`, `.webp`.
- Character slots are the fixed set `A`, `B`, `C`, `D`. The renderer treats a slot as an
  opaque id for a region / LoRA / ref directory.

## Storyboard

`Storyboard` (`contract.py`). `title` and `scenes` are load-bearing; the style block applies
uniformly to every scene's prompt. A storyboard with no scenes is rejected.

| Field | Type | Default | Notes |
|---|---|---|---|
| `title` | str | `"untitled"` | Project title. |
| `scenes` | list[Scene] | (required) | One entry per shot; must be non-empty. |
| `full_prompt` | str | `""` | Authored, not consumed by the backend. |
| `duration_seconds` | float \| null | null | Optional total duration hint. |
| `clip_seconds` | float \| null | null | Optional per-clip duration hint. |
| `style_prefix` | str | `""` | Prepended to every scene's prompt. |
| `style_category` | str | `"None"` | Normalized to the literal `"None"` when absent. |
| `style_preset` | str | `"None"` | Normalized to the literal `"None"` when absent. |
| `use_characters` | list[str] | `[]` | Slots in play (subset of A-D); drives LoRA planning. |
| `cast_rules` | str | `""` | Authored cast guidance. |
| `refs_dir` | str \| null | null | Override for the refs root (default `characters/refs`). |

### Scene

`Scene` (`contract.py`). Only `prompt` is required; the rest are authored hints.

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | str | (required) | The shot's text prompt. |
| `id` | str | `shot_NN` | Shot id; auto-assigned `shot_01`, `shot_02`, ... when absent. |
| `character_slots` | list[str] | `[]` | Slots in this shot (subset of A-D). 2+ triggers the regional multi-character path. |
| `start` | float \| null | null | Beat start (seconds). |
| `end` | float \| null | null | Beat end (seconds). |
| `target_seconds` | float \| null | null | Desired clip length; derives the i2v frame count. |
| `act` | str \| null | null | Authored act label. |
| `start_image` | str \| null | null | An authored keyframe to inject (skips keyframe generation for this shot). |

A scene with two or more `character_slots` is "multi-character" and renders on the regional
no-bleed keyframe path; see [configuration.md](configuration.md#multicharconfig).

## Cast

`Cast` / `Character` (`contract.py`), parsed from `characters/registry.json`. The registry
maps each slot to a name and an identity prompt; the backend resolves each slot's reference
images from the refs directory and attaches them.

`registry.json`:

```json
{
  "characters": {
    "A": { "name": "Mara",  "prompt": "a woman in her 30s, short dark hair, weathered jacket" },
    "B": { "name": "Eli",   "prompt": "a man in his 40s, salt-and-pepper beard, wool coat" }
  }
}
```

| Field | Type | Notes |
|---|---|---|
| `slot` | str | A-D. Entries with an unknown slot are ignored. |
| `name` | str | Trigger token the keyframe prompt uses to summon this identity (defaults to the slot id). |
| `prompt` | str | Identity descriptor used in the training caption and prompt. |
| `ref_paths` | list[Path] | Resolved by the backend from `characters/refs/<SLOT>/` (not in the JSON). |

## Render job (request)

`RenderRequest` (`contract.py`), built from the job input dict via `from_dict`. The control
plane sends `render_overrides`; the backend parses it into the typed `config`
([configuration.md](configuration.md)) and keeps the raw dict only for the few non-generation
routing flags it still reads off it (e.g. `finish_offloaded`).

| Field | Type | Default | Notes |
|---|---|---|---|
| `action` | str | `"render"` | `render` \| `preview` \| `regen_shot` \| `finalize` \| `train_lora`. Selects the pipeline path. |
| `project` | str | `"untitled"` | Project name; keys every artifact (see [the key map](operations.md#r2-object-key-map)). |
| `bundle_key` | str | `""` | R2 key of the bundle tar. |
| `quality_tier` | str | `"draft"` | `draft` \| `standard` \| `final`. Sets every config baseline. The studio always sends it explicitly; the draft default is the wallet-safe floor for a direct caller that omits it (an omitted tier must never silently buy the most expensive render). |
| `config` | RenderConfig | tier baseline | Typed generation config; built from `quality_tier` + `render_overrides`. |
| `overrides` | dict | `{}` | The raw `render_overrides`; routing flags only. |
| `pretrained_loras` | dict[slot, str] | `{}` | Slot -> already-trained LoRA (R2 key or local path); skips that slot's training. |
| `process_shot_ids` | list[str] \| null | null | Restrict `finalize` / `regen_shot` to a subset of shots. |
| `audio_key` | str \| null | null | R2 key of an audio bed to mux under the final video. A REQUESTED bed that cannot be fetched FAILS the render. Set `render_overrides.audio_optional: true` to opt into soft-degrade instead: the film ships silent, `audio_missing: true` appears in the result, and an `audio_missing` event is emitted. |
| `keyframes_only` | bool | `false` | DEPRECATED, kept for compat: on a `render`, draw keyframes (training the LoRAs they need) then short-circuit before any i2v/finish GPU-seconds. The studio stopped sending it in v0.160.0 and sends `action:"preview"` instead; do not build new callers on it. |
| `model_family` | str | `sdxl` | `sdxl` \| `wan`. On the **render** backend, only SDXL inline training runs; `wan` fails loud (submit to `vivijure-wan-train` via `RUNPOD_WAN_TRAIN_ENDPOINT_ID`). On the **Wan train** satellite, defaults to Wan. Unknown strings fall back to SDXL. |

### Actions

| Action | Trains LoRAs | Keyframes | Animates (i2v) | Use |
|---|---|---|---|---|
| `render` | as needed | as needed | yes | Full render. |
| `preview` | as needed | as needed | no | Cheap keyframe preview before committing GPU-seconds. |
| `finalize` | no | reuse | yes | Animate over existing keyframes; zero training. |
| `regen_shot` | no | as needed | no | Redraw specific keyframes (with `process_shot_ids`), no motion. |
| `train_lora` | yes | no | no | Train character adapters only. |
| `i2v_clip` | no | no | yes | Standalone image-to-video on one keyframe. Separate job shape (no bundle); see [the `i2v_clip` job](#the-i2v_clip-job). |
| `finish_clip` | no | no | no | Standalone finishing pass on one existing clip. Separate job shape (no bundle); see [the `finish_clip` job](#the-finish_clip-job). |

Job-supplied R2 keys are validated against the render key map before any store I/O:
`bundle_key` must sit under `bundles/`, `pretrained_loras` values under `loras/` (project slug
**or** a cast-registry key such as `loras/cast-{id}/…` / `loras/lora-{slug}-…/…` resolved by the
control plane from cast rows), `audio_key` under `audio/` (a staged bed) or `renders/` (a
pipeline-produced bed), and the standalone jobs' `clip_key` / `keyframe_key` under `renders/`. A key
outside its prefix, an absolute key, or one carrying `..` fails the job before any transfer.

The first five actions ride the `RenderRequest` shape above and the render pipeline. `i2v_clip` and
`finish_clip` are sibling job types: the harness routes each directly to a standalone pass that needs
no bundle or planner, so each carries its own input/output shape, documented below.

## Tenant R2 credential (all job types)

The backend uses R2 for **two unrelated purposes**, and they have opposite sharing requirements:

1. **The models mirror** (`harness/models_mirror.py`) pulls shared weights (~120GB for Wan i2v) from
   OUR bucket. Every tenant pulls identical bytes. Correctly shared.
2. **Tenant job I/O** (bundle in, film out, LoRAs, keyframes, clips, manifest, progress and error
   snapshots) belongs to THE TENANT's bucket.

They are split by purpose. The models mirror reads `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` /
`R2_ENDPOINT` / `R2_BUCKET` from the endpoint environment, **always**, whatever a job carries. Tenant
job I/O takes an **optional per-job credential block** in the job input, which is what lets ONE
pooled RunPod endpoint serve many tenants without any of them sharing a credential or a bucket.

| Field | Type | Required | Notes |
|---|---|---|---|
| `r2.endpoint` | str | yes | The tenant's R2 S3 endpoint. |
| `r2.access_key_id` | str | yes | |
| `r2.secret_access_key` | str | yes | |
| `r2.bucket` | str | yes | The tenant's bucket. |
| `r2.session_token` | str | no | Set for R2 temporary access credentials, which issue one; omit for a static R2 API token. |

```json
{
  "input": {
    "action": "render",
    "project": "the-long-walk-home",
    "bundle_key": "bundles/the-long-walk-home/abc123.tar.gz",
    "r2": {
      "endpoint": "https://<account>.r2.cloudflarestorage.com",
      "access_key_id": "...",
      "secret_access_key": "...",
      "bucket": "tenant-bucket-name"
    }
  }
}
```

Rules the backend guarantees (`R2Config.from_payload_or_env`, `harness/r2.py`):

1. **Block absent -> the endpoint environment**, byte-identical to the behaviour before the block
   existed. A dedicated endpoint keeps working unchanged; backward compatibility here is
   load-bearing, not a courtesy.
2. **Block present and valid -> used for every tenant job I/O**, for that job only.
3. **Block present but malformed -> the job FAILS.** It does NOT fall back to the environment. This
   is the safety property of the whole split: a silent fallback would run a tenant's job against our
   bucket under our credential, the precise failure the per-job credential exists to prevent. For the
   same reason an explicit `"r2": null` is REFUSED rather than read as absent -- **omit the key**, do
   not send a null.
4. The models mirror ignores the block. This is structural, not policy: `models_mirror` builds its
   own rclone environment from `os.environ` and is never handed the store or the payload.
5. Refusal messages name **fields only, never values**, because the handler mirrors a config failure
   into the R2 progress channel and stdout.
6. The block is consumed at the handler boundary and **stripped from the payload** before anything
   downstream sees it, so no emitter, manifest, or error path can echo it.

### Why not presigned URLs

Presigned URLs are the right shape for a fixed, single-artifact flow, and that is where we already
use them (the video-finish tier and the `film-titles` module are credentialless by construction). They
do not fit this backend:

- `ProgressEmitter` writes `renders/<slug>/progress/<job_id>.ndjson` and `.json`
  (`harness/keys.py`), and `job_id` is **RunPod's**, assigned when RunPod accepts the job. The
  submitter learns it only after the payload is sealed, so those keys cannot be presigned.
- The incremental-reuse path runs **negative probes** (`store.exists()` per cast slot and per shot,
  plus hash-sidecar reads) over a slot/shot set the worker derives from the storyboard it extracts
  from the bundle, not from the payload.

Presigning would force the producer to enumerate every key the backend might touch, including keys
that may not exist, which inverts the ownership `harness/keys.py` exists to hold and makes every
future key a breaking two-repo change.

Stated plainly: a per-job credential is **weaker** than credentialless-by-construction. It is bounded
by tenant and by job lifetime rather than absent. It is still strictly better than the alternative it
replaces, a single long-lived credential in the RunPod template shared by every job on the endpoint.

## Render result (response)

`RenderResult` (`contract.py`), returned as a dict. The control plane polls for `output_key`
(the final MP4) plus the per-shot `keyframes` and `clips`.

| Field | Type | Notes |
|---|---|---|
| `project` | str | Echoed project name. |
| `output_key` | str \| null | R2 key of the final muxed MP4 (`null` for preview / train-only). |
| `seconds` | float \| null | Total duration of the final video. |
| `has_audio` | bool | Whether an audio bed was muxed. |
| `audio_missing` | bool | `true` ONLY on the explicit `audio_optional` soft-degrade: the requested bed could not be fetched and the job opted in, so the film shipped silent. Without the opt-in an unfetchable requested bed fails the render. |
| `keyframes` | list[{shot_id, key}] | Per-shot keyframe PNG keys (generated, reused, and injected). Naming seam: the studio's `motion.backend` hook calls this field `keyframe_key`; this backend returns `key`, and the studio's own-gpu module translates `key` -> `keyframe_key` when mapping the result into the hook output. |
| `clips` | list[{shot_id, key, target_seconds?}] | Per-shot clip keys (present when shots were animated). |
| `lora` | dict[slot, {lora_id}] | Trained and passed-through adapters by slot. |
| `state_key` | str \| null | Always `null` since #112 (kept for wire-compat): incremental state is per-artifact R2 objects, not a tarball. |

## The `i2v_clip` job

A standalone image-to-video pass on a single keyframe: Wan2.2-I2V animates one still into one clip,
nothing else. It bypasses the bundle and the planner, so it does not use the `RenderRequest` /
`RenderResult` shapes above. The harness routes `action == "i2v_clip"` straight to
`run_i2v_clip_job` (`harness/handler.py`), which fetches the keyframe from R2, animates it, and
uploads the clip. Unlike `finish_clip`, it DOES load the Wan models, so the worker keeps the
cold-start i2v prefetch for it. The studio's `motion` module (vivijure #81) dispatches this action
per shot: keyframe -> **i2v_clip** -> finish_clip -> assemble.

**Input** (the job `input` dict):

| Field | Type | Default | Notes |
|---|---|---|---|
| `action` | str | -- | Must be `"i2v_clip"`. |
| `project` | str | `"untitled"` | Keys the output clip. |
| `shot_id` | str | `"shot"` | Identifies the shot; part of the output key. |
| `prompt` | str | (required) | The motion description fed to Wan. |
| `keyframe_key` | str | `renders/<project>/keyframes/<shot_id>.png` | R2 key of the keyframe still to animate. |
| `config` | object | `{}` | The i2v knobs (below). |

`config` fields (a subset of [`I2VConfig`](configuration.md#i2vconfig), over the `quality` tier
baseline):

| Field | Type | Default | Notes |
|---|---|---|---|
| `quality` | str | `"final"` | `draft` \| `standard` \| `final`. Drives steps / guidance / distill / feature-cache via `I2VConfig.for_tier`. |
| `num_frames` | int | `81` | Snapped to the temporal-VAE-valid `4k+1` (e.g. 50 -> 53), clamped 1..256. |
| `fps` | int | `16` | Export fps, clamped 1..120. |
| `seed` | int | `0` | i2v RNG seed (independent of the keyframe seed). |
| `flow_shift` | float | `5.0` | FlowMatch scheduler shift; lower = faster motion. |
| `height` | int \| null | `null` | `null`/`0` = follow the keyframe's native dims. |
| `width` | int \| null | `null` | `null`/`0` = follow the keyframe's native dims. |
| `negative_prompt` | str \| null | `null` | Additive over the engine's anti-static guard, never a replacement. |

Caching can never combine with the 4-step distill path: when the tier (or an override) selects
`distill`, `feature_cache` is forced to `NONE` (nothing to cache at 4 steps), matching the full
render path.

**Output:**

| Field | Type | Notes |
|---|---|---|
| `clip_key` | str | R2 key of the clip (`renders/<project>/clips/<shot_id>_i2v.mp4`). |
| `shot_id` | str | Echoed shot id. |
| `num_frames` | int | Realized (snapped) frame count. |
| `fps` | int | Realized fps. |
| `seconds` | float | Clip length (`num_frames / fps`). |
| `distilled` | bool | Whether the 4-step Lightning distill path produced it. |

**Example** input and result:

```json
{
  "action": "i2v_clip",
  "project": "the-long-walk-home",
  "shot_id": "shot_02",
  "prompt": "slow dolly in as she turns toward the window",
  "config": { "quality": "final", "num_frames": 81, "flow_shift": 5.0 }
}
```

```json
{
  "clip_key": "renders/the-long-walk-home/clips/shot_02_i2v.mp4",
  "shot_id": "shot_02",
  "num_frames": 81,
  "fps": 16,
  "seconds": 5.062,
  "distilled": false
}
```

## The `finish_clip` job

A standalone finishing pass on a single, already-rendered clip: RIFE frame interpolation and/or
blind face restoration, nothing else. It bypasses the bundle, the planner, and Wan entirely, so it
does not use the `RenderRequest` / `RenderResult` shapes above. The harness routes `action ==
"finish_clip"` straight to `run_finish_job` (`harness/handler.py`), which fetches one clip from R2,
finishes it, and uploads the result. The studio's `finish-rife` module dispatches this action to the
same RunPod endpoint.

**Audio is preserved (#240).** The interpolation re-encode is video-only (it is fed a rawvideo stream),
so if the SOURCE clip carries an audio track (e.g. a dialogue shot lip-synced before finish, so the
MuseTalk audio reaches this stage), the finished clip muxes that track back with a stream copy; RIFE
keeps wall-clock duration fixed, so the audio lines up 1:1. A mux that fails FAILS the shot with the
real error (#245), never silently shipping a video-only clip when audio was present; a source with no
audio track is returned unchanged (no audio step, no failure).

**Input** (the job `input` dict):

| Field | Type | Default | Notes |
|---|---|---|---|
| `action` | str | -- | Must be `"finish_clip"`. |
| `project` | str | `"untitled"` | Keys the output clip. |
| `shot_id` | str | `"shot"` | Identifies the clip; part of the output key. |
| `clip_key` | str | (required) | R2 key of the clip to finish. |
| `config` | object | `{}` | The finishing knobs (below). |
| `output_hash` | str | (optional) | The studio's #583 param-hash of this step's inputs. When present, the handler writes it VERBATIM to `<clip_key_out>.hash` AFTER the finished clip (artifact first, sidecar last) as the reuse-provenance stamp. Opaque here (never parsed/recomputed); absent -> no sidecar. A sidecar write is best-effort (a failure never fails the render). |

`config` fields (a flat subset of [`FinishConfig`](configuration.md#finishconfig)):

| Field | Type | Default | Notes |
|---|---|---|---|
| `interpolate` | bool | `true` | Run RIFE interpolation. |
| `interpolation_factor` | int | `2` | Recursive 2x factor (1/2/4/8). |
| `target_fps` | int | `0` | 0 = `src*factor`; else a hard fps cap. |
| `face_restore` | str \| false | `false` | Restorer backend (`"gfpgan"` / `"codeformer"`); falsy or `"none"` disables it. |
| `face_fidelity` | float | `0.7` | Restoration / fidelity balance. |
| `only_faces` | bool | `true` | Restore detected faces only. |

**Output:**

| Field | Type | Notes |
|---|---|---|
| `shot_id` | str | Echoed shot id. |
| `clip_key` | str | R2 key of the finished clip (`renders/<project>/clips/<shot_id>_finished.mp4`). |
| `out_fps` | int | Realized fps of the finished clip. |
| `frames` | int | Output frame count. |
| `applied` | list[str] | What ran, e.g. `["interpolate:2x", "face_restore:gfpgan"]`. |

When both passes are off (`interpolate` false and `face_restore` disabled) the stage is a no-op:
`finish_clip` does no GPU work and the clip is returned unchanged (`FinishConfig.enabled` is the gate).

**Example** input and result:

```json
{
  "action": "finish_clip",
  "project": "the-long-walk-home",
  "shot_id": "shot_02",
  "clip_key": "renders/the-long-walk-home/clips/shot_02.mp4",
  "config": { "interpolate": true, "interpolation_factor": 2, "face_restore": "gfpgan" }
}
```

```json
{
  "shot_id": "shot_02",
  "clip_key": "renders/the-long-walk-home/clips/shot_02_finished.mp4",
  "out_fps": 32,
  "frames": 161,
  "applied": ["interpolate:2x", "face_restore:gfpgan"]
}
```

## Worked example

A two-character, two-shot render at standard quality.

**`storyboard.yaml`:**

```yaml
title: The Long Walk Home
style_prefix: "cinematic, 35mm film still, golden hour"
use_characters: [A, B]
scenes:
  - id: shot_01
    prompt: "Mara walks alone down an empty rural road, looking back over her shoulder"
    character_slots: [A]
    target_seconds: 5
  - id: shot_02
    prompt: "Mara and Eli meet at a crossroads and embrace"
    character_slots: [A, B]
    target_seconds: 6
```

**Job input** (what the control plane submits):

```json
{
  "action": "render",
  "project": "the-long-walk-home",
  "bundle_key": "bundles/the-long-walk-home/abc123.tar.gz",
  "quality_tier": "standard",
  "audio_key": "audio/the-long-walk-home/bed.m4a",
  "render_overrides": {
    "keyframe": { "ip_adapter_scale": 0.7 },
    "i2v": { "flow_shift": 5.0 }
  }
}
```

**Result** (what the backend returns):

```json
{
  "project": "the-long-walk-home",
  "output_key": "renders/the-long-walk-home/full.mp4",
  "seconds": 11.0,
  "has_audio": true,
  "keyframes": [
    { "shot_id": "shot_01", "key": "renders/the-long-walk-home/keyframes/shot_01.png" },
    { "shot_id": "shot_02", "key": "renders/the-long-walk-home/keyframes/shot_02.png" }
  ],
  "clips": [
    { "shot_id": "shot_01", "key": "renders/the-long-walk-home/clips/shot_01.mp4", "target_seconds": 5 },
    { "shot_id": "shot_02", "key": "renders/the-long-walk-home/clips/shot_02.mp4", "target_seconds": 6 }
  ],
  "lora": {
    "A": { "lora_id": "loras/the-long-walk-home/A/pytorch_lora_weights.safetensors" },
    "B": { "lora_id": "loras/the-long-walk-home/B/pytorch_lora_weights.safetensors" }
  },
  "state_key": null
}
```

`shot_01` is single-character (the plain SDXL + IP-Adapter path); `shot_02` has two slots and
renders on the regional no-bleed path. On a second render of the same project, the planner
derives reuse from the per-artifact R2 objects (adapter keys, keyframe PNGs + `.hash`
sidecars) and reuses both LoRAs and any keyframe whose params have not changed.

## See also

- The generation knobs `render_overrides` carries: [configuration.md](configuration.md).
- The exact R2 keys these artifacts land under: [operations.md](operations.md#r2-object-key-map).
- How the planner turns a request into work: [architecture.md](architecture.md#what-the-planner-decides-and-why).
