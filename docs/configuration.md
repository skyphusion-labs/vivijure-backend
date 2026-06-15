# Configuration

`RenderConfig` ([`config.py`](../src/vivijure_backend/config.py)) is the single source of
truth for what drives a render. It replaces an untyped `overrides` grab-bag with typed config
objects where **every field maps to a real model parameter**, carries a sane default and a
clamped range, and reads the same per quality tier on both the control plane and the backend.

## How a config is built

A render job carries a `quality_tier` and a `render_overrides` dict. The config is built in two
layers:

1. **Tier baseline.** `quality_tier` (`draft` / `standard` / `final`) sets every field via
   each stage's `for_tier()`. This alone is a complete, valid config.
2. **Overrides.** `render_overrides` then layers explicit knobs over the baseline. The payload
   is namespaced by stage and parsing is forgiving: unknown keys and unknown sections are
   ignored, numeric knobs are clamped to their documented range, and enums fall back to the
   baseline on anything unrecognized. A newer control plane never breaks an older backend.

```json
{
  "render_overrides": {
    "keyframe": { "steps": 36, "ip_adapter_scale": 0.7 },
    "i2v":      { "flow_shift": 6.0 },
    "finish":   { "interpolate": true, "interpolation_factor": 4 },
    "lora":     { "rank": 24, "max_steps": 1200 }
  }
}
```

Out-of-range or non-numeric values do not raise; they clamp or fall back. An invalid
combination is corrected, not rejected (e.g. a feature cache requested alongside the 4-step
distill path is forced off, because there is nothing to cache at 4 steps).

## Quality tiers at a glance

| | draft | standard | final |
|---|---|---|---|
| **Keyframe** | Hyper-SD 4-step, cfg 0 | Hyper-SD 8-step, cfg 0 | full 30-step, cfg 6.5 |
| **i2v** | Lightning 4-step, cfg ~1.0, no cache | full 20-step, cfg 3.5, EasyCache | full 40-step, cfg 3.5, MixCache |
| **Finish** | none (preview) | interpolate 2x | interpolate 2x + GFPGAN face restore |
| **i2v GPU tier** | RTX PRO 6000 | H200 | B200 |

Draft is a fast preview, standard is the balanced middle, final is the hero deliverable. LoRA
training is not tier-dependent (the adapter is the adapter), so the `lora` section is the same
across all tiers.

## KeyframeConfig

The SDXL keyframe stage. Fields map to diffusers SDXL parameters.

| Field | Type | Default (final) | Range | Maps to |
|---|---|---|---|---|
| `base_model` | str | RealVisXL V5.0 (`models.DEFAULT_SPECS`) | -- | SDXL checkpoint repo. |
| `steps` | int | 30 | 1-128 | `num_inference_steps` (full path). |
| `guidance_scale` | float | 6.5 | 0-30 | CFG. Few-step distill wants 0.0. |
| `scheduler` | enum | `dpmpp_2m_karras` | see [Scheduler](#scheduler) | The sampler. |
| `width` | int | 1024 | 512-2048 | Generated width (SDXL native 1024). |
| `height` | int | 1024 | 512-2048 | Generated height. |
| `distill` | bool | false | -- | Hyper-SD few-step path on/off. |
| `distill_model` | str | Hyper-SD (`DEFAULT_SPECS`) | -- | The few-step LoRA repo. |
| `distill_steps` | int | 8 | 1-8 | Hyper-SD fixed-step count. |
| `seed` | int | 424242 | >=0 | Base RNG seed. |
| `identity_method` | enum | `ip_adapter` | see [IdentityMethod](#identitymethod) | How a face is pinned (all shots). |
| `ip_adapter_scale` | float | 0.65 | 0-1 | Single-subject IP-Adapter identity pull. |
| `instantid_controlnet_scale` | float | 0.8 | 0-1.5 | InstantID face-ControlNet (single-char upgrade). |
| `instantid_ip_adapter_scale` | float | 0.8 | 0-1.5 | InstantID IP-Adapter (single-char upgrade). |
| `multi_char` | MultiCharConfig | see below | -- | Regional multi-character anti-bleed block. |

`from_dict` also accepts a `resolution` string `"WIDTHxHEIGHT"` (the control plane's shape) in
place of explicit `width` / `height`.

Tier baselines: **draft** = `distill=true, distill_steps=4, steps=4, guidance_scale=0,
scheduler=ddim_trailing`; **standard** = the same with `distill_steps=8, steps=8`; **final** =
`distill=false, steps=30, guidance_scale=6.5, scheduler=dpmpp_2m_karras`.

### MultiCharConfig

Anti-bleed config for the **regional multi-character path only** (2+ characters in one frame).
`keyframe.py` reads this block solely when a shot renders on the regional engine; a
single-character shot never touches it. InstantID is deliberately absent here (it is
single-face by nature), so the regional path stays masked-IP-Adapter only.

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `regional` | bool | true | -- | Use the regional no-bleed engine for multi-char shots. |
| `pose_conditioning` | bool | true | -- | OpenPose ControlNet to separate bodies. |
| `lora_scale_per_slot` | float | 0.7 | 0-2 | Per-character LoRA strength in a shared frame. |
| `ip_adapter_scale_per_slot` | float | 0.7 | 0-1 | Per-region masked IP-Adapter identity pull. |
| `max_slots` | int | 2 | 1-4 | Characters the no-bleed path supports at once. |
| `controlnet_pose_scale` | float | 0.55 | 0-1.5 | OpenPose ControlNet conditioning scale. |
| `region_gutter` | int | 64 | 0-256 | Dead-band px between masks (anti seam-blend). |

## I2VConfig

The Wan 2.2 image-to-video stage. Fields map to documented Wan 2.2 diffusers parameters.

| Field | Type | Default (final) | Range | Maps to |
|---|---|---|---|---|
| `model` | str | Wan2.2-I2V-A14B (`DEFAULT_SPECS`) | -- | i2v base repo. |
| `num_frames` | int | 81 | 1-256 | Frame count (Wan default = 5s at 16fps). |
| `fps` | int | 16 | 1-120 | Export fps. |
| `steps` | int | 40 | 1-64 | `num_inference_steps` (full path). |
| `guidance_scale` | float | 3.5 | 0-30 | CFG (full); distill ~1.0. |
| `flow_shift` | float | 5.0 | 0-20 | FlowMatch scheduler shift. |
| `seconds_per_shot` | float | 5.0 | 0.5-60 | Derives `num_frames` when a shot has no duration. |
| `distill` | bool | false | -- | Wan2.2-Lightning 4-step path on/off. |
| `distill_model` | str | Wan2.2-Lightning (`DEFAULT_SPECS`) | -- | The distill LoRA repo. |
| `distill_steps` | int | 4 | 1-8 | Lightning step count. |
| `loader` | enum | `diffusers` | see [I2VLoader](#i2vloader) | Which LoRA loader applies the distill. |
| `feature_cache` | enum | `mixcache` | see [FeatureCache](#featurecache) | Full-step inference cache. |
| `seed` | int | 0 | >=0 | i2v RNG seed (independent of the keyframe seed). |
| `negative_prompt` | str | `""` | -- | Optional; empty = the model's shipped default. |

A caching choice is force-cleared to `none` whenever `distill` is on, so an override cannot
create the invalid "cache a 4-step render" combination. `frames_for(seconds)` derives the
clamped frame count for a shot's duration.

Tier baselines: **draft** = `distill=true, distill_steps=4, steps=4, guidance_scale=1.0,
feature_cache=none`; **standard** = `distill=false, steps=20, guidance_scale=3.5,
feature_cache=easycache`; **final** = `distill=false, steps=40, guidance_scale=3.5,
feature_cache=mixcache`.

## FinishConfig

The post-i2v finishing passes (`finish.py`): RIFE frame interpolation and blind face
restoration, run per clip on the GPU before the off-GPU assemble.

| Field | Type | Default (final) | Range | Notes |
|---|---|---|---|---|
| `interpolate` | bool | true | -- | Turn on RIFE interpolation. |
| `interpolation_factor` | int | 2 | 1/2/4/8 | Recursive 2x multiple (snapped to a power of two). |
| `target_fps` | int | 0 | 0-120 | 0 = `src*factor`; else a hard cap on realized fps. |
| `face_restore` | enum | `gfpgan` | see [FaceRestore](#facerestore) | Blind face restorer (off by default). |
| `face_fidelity` | float | 0.7 | 0-1 | Restoration / fidelity balance (0 = max restoration). |
| `only_faces` | bool | true | -- | Restore detected faces only. |

`interpolation_factor` is snapped to a power of two at parse time (3 -> 2, 6 -> 4), so the
stored config is exactly what runs. The stage runs only when at least one pass is on (`enabled`);
otherwise the raw i2v clips ship unchanged.

Tier baselines: **draft** = nothing (a preview is not worth finishing); **standard** =
`interpolate=true, factor=2`; **final** = `interpolate=true, factor=2, face_restore=gfpgan`.

## LoraTrainConfig

Per-slot training knobs ([`lora_train.py`](../src/vivijure_backend/lora_train.py)). Not
quality-tier dependent. Defaults are tuned for a few-reference character fit (5-20 images).

| Field | Type | Default | Range | Notes |
|---|---|---|---|---|
| `rank` | int | 16 | 1-128 | LoRA rank; small enough to capture a face without memorizing backgrounds. |
| `resolution` | int | 1024 | 512-1536 | Training resolution. |
| `learning_rate` | float | 1e-4 | 1e-6 - 1e-2 | Optimizer LR. |
| `max_steps` | int | 1000 | 1-5000 | Training steps; converges before it overfits. |
| `batch_size` | int | 1 | 1-8 | Per-step batch. |
| `gradient_accumulation_steps` | int | 1 | 1-32 | Effective batch multiplier. |
| `seed` | int | 0 | >=0 | Training RNG seed. |
| `random_flip` | bool | true | -- | Horizontal-flip augmentation. |
| `gradient_checkpointing` | bool | true | -- | Trade compute for VRAM (fits a 1024 UNet on a mid card). |
| `caption_template` | str | `"{name}, {prompt}"` | -- | `{name}` / `{prompt}` filled from the slot's registry entry. |
| `save_every` | int | 0 | 0-5000 | 0 = only the final adapter; >0 writes intermediate checkpoints. |

Training adapts the UNet attention projections only (`to_k`, `to_q`, `to_v`, `to_out.0`); the
text encoders stay frozen. It runs in bf16 so the optimizer sees real gradients.

## Enums

### Scheduler
SDXL sampler. The few-step distill paths pin specific schedulers: Hyper-SD fixed-step LoRAs
want `ddim_trailing`, the unified 1-step LoRA wants `tcd`. The full-step path is free to use a
higher-order solver.

`euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_2m_karras`, `unipc`, `ddim`, `ddim_trailing`, `tcd`.

### IdentityMethod
How a character's face is pinned onto the keyframe. `ip_adapter` (the default, single and
multi-character) pulls identity from the reference embedding, masked per region on the regional
path. `instantid` is a single-character upgrade for face-critical shots (adds a face-ControlNet;
single-face by nature, so not on the regional path). `both` stacks them (advanced, future).

`ip_adapter`, `instantid`, `both`.

### I2VLoader
Which loader applies the Wan2.2-Lightning distill LoRA. `diffusers` uses `load_lora_weights`;
`lightx2v` is the documented fallback for a known diffusers compat issue (#12535).

`diffusers`, `lightx2v`.

### FeatureCache
Full-step Wan inference cache (a TeaCache successor); reuses block features across adjacent
steps for ~1.5-2x at high step counts. Never stacked on the 4-step distill path.

`none`, `mixcache`, `easycache`.

### FaceRestore
The blind face restorer in the finish stage. `gfpgan` is the redistribution-clean default;
`codeformer` is higher quality but ships under the S-Lab non-commercial license, so it is an
opt-in the deployer chooses, never bundled by default (same posture as the antelopev2 pack).

`none`, `gfpgan`, `codeformer`.

## Environment variables

Generation config arrives per job; these env vars tune the runtime. Most are baked into the
image (see [operations.md](operations.md#environment)); the R2 credential is the only secret.

| Var | Default | Purpose |
|---|---|---|
| `VJ_MODELS_ROOT` | `/opt/models` | Model cache root + mirror sentinels. |
| `VJ_MODEL_VERSION` | `1` | Mirror sentinel version; bump to force a re-mirror on warm workers. |
| `VJ_I2V_DISTILL` | `1` | Toggle the Wan2.2-Lightning distill LoRA. |
| `VJ_I2V_FP8` | `1` | Toggle Wan fp8 quantization. |
| `HF_HOME` | `/opt/models/hf-cache` | HuggingFace cache the mirror fills (read offline). |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE` | `1` | Read weights from the local mirror, never the Hub. |
| `R2_ENDPOINT` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET` | -- | R2 credentials (the one runtime secret; `R2_BUCKET` defaults to `vivijure`). |

## See also

- The request that carries `render_overrides`: [contract.md](contract.md#render-job-request).
- Where these stages run and at what precision: [architecture.md](architecture.md#capability-aware-precision).
