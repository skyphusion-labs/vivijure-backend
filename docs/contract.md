# The contract

This is the data contract between the Vivijure control plane and the render backend: the
project bundle the backend reads, the render job it is given, and the result it returns.
Everything here is typed in [`contract.py`](../src/vivijure_backend/contract.py) and parsed
forgivingly: unknown keys are ignored and only fields the renderer actually consumes are
surfaced, so the control plane can add authored fields without breaking an older backend.

The contract is the *only* thing carried over from the control plane; it is a data shape, not
borrowed implementation (see [CONTRIBUTING](../CONTRIBUTING.md) on the clean-room posture).

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
| `quality_tier` | str | `"final"` | `draft` \| `standard` \| `final`. Sets every config baseline. |
| `config` | RenderConfig | tier baseline | Typed generation config; built from `quality_tier` + `render_overrides`. |
| `overrides` | dict | `{}` | The raw `render_overrides`; routing flags only. |
| `pretrained_loras` | dict[slot, str] | `{}` | Slot -> already-trained LoRA (R2 key or local path); skips that slot's training. |
| `process_shot_ids` | list[str] \| null | null | Restrict `finalize` / `regen_shot` to a subset of shots. |
| `audio_key` | str \| null | null | R2 key of an audio bed to mux under the final video. |
| `user_email` | str \| null | null | Access-authenticated submitter; stamped on every artifact for the control plane's ownership-gated `/api/artifact` route. |

### Actions

| Action | Trains LoRAs | Keyframes | Animates (i2v) | Use |
|---|---|---|---|---|
| `render` | as needed | as needed | yes | Full render. |
| `preview` | as needed | as needed | no | Cheap keyframe preview before committing GPU-seconds. |
| `finalize` | no | reuse | yes | Animate over existing keyframes; zero training. |
| `regen_shot` | no | as needed | no | Redraw specific keyframes (with `process_shot_ids`), no motion. |
| `train_lora` | yes | no | no | Train character adapters only. |

## Render result (response)

`RenderResult` (`contract.py`), returned as a dict. The control plane polls for `output_key`
(the final MP4) plus the per-shot `keyframes` and `clips`; `state_key` is the project tree it
restores on the next render of this project.

| Field | Type | Notes |
|---|---|---|
| `project` | str | Echoed project name. |
| `output_key` | str \| null | R2 key of the final muxed MP4 (`null` for preview / train-only). |
| `seconds` | float \| null | Total duration of the final video. |
| `has_audio` | bool | Whether an audio bed was muxed. |
| `keyframes` | list[{shot_id, key}] | Per-shot keyframe PNG keys (generated, reused, and injected). |
| `clips` | list[{shot_id, key, target_seconds?}] | Per-shot clip keys (present when shots were animated). |
| `lora` | dict[slot, {lora_id}] | Trained and passed-through adapters by slot. |
| `state_key` | str \| null | R2 key of the project state tarball for incremental re-render. |

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
  "user_email": "director@example.com",
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
  "state_key": "projects/the-long-walk-home/state.tar.gz"
}
```

`shot_01` is single-character (the plain SDXL + IP-Adapter path); `shot_02` has two slots and
renders on the regional no-bleed path. On a second render of the same project, the planner
restores the state tarball and reuses both LoRAs and any keyframe whose params have not changed.

## See also

- The generation knobs `render_overrides` carries: [configuration.md](configuration.md).
- The exact R2 keys these artifacts land under: [operations.md](operations.md#r2-object-key-map).
- How the planner turns a request into work: [architecture.md](architecture.md#what-the-planner-decides-and-why).
