"""Typed generation config: the single source of truth for what drives a render.

The control plane and this backend used to meet over an untyped `overrides` dict (the backend
would `overrides.get("some_knob")` and the planner had no authoritative list of what it could
send). This module replaces that grab-bag with typed config objects where **every field maps to
a real, documented model parameter**, carries a sane default and a clamped range, and reads the
same per quality tier on both sides.

Two configs are load-bearing for the GPU work and are defined first:
  - `KeyframeConfig` -- the SDXL keyframe stage (diffusers SDXL + the multi-character identity
    stack: regional engine, pose ControlNet, per-slot LoRA / IP-Adapter, InstantID).
  - `I2VConfig` -- the Wan 2.2 image-to-video stage (frames / fps / steps / guidance / shift,
    the Wan2.2-Lightning distill path, and the final-tier feature cache).

Provenance (clean-room): field names and ranges mirror the control plane's own emitted shape
(`skyphusion-llm-public` `src/runpod-submit.ts`, Conrad's code); defaults are set from the
NEW backend's model stack, taken from each model's own public docs (June 2026):
  - SDXL keyframe steps/cfg/scheduler: diffusers SDXL pipeline; few-step distill: ByteDance
    Hyper-SD SDXL (2/4/8-step LoRAs run cfg=0 on DDIM `timestep_spacing="trailing"`; the
    unified 1-step LoRA runs on TCDScheduler).
  - Wan 2.2 I2V: Wan-AI/Wan2.2-I2V-A14B-Diffusers documents 40 steps, guidance 3.5, 81 frames
    at 16fps; the lightx2v/Wan2.2-Lightning distill runs 4 steps with CFG off (~1.0).
In-repo cross-refs: `models.DEFAULT_SPECS` (the real HF ids), `orchestrator.MULTI_CHAR_DEFAULTS`
(the anti-bleed scales), `routing.QualityTier` (draft / standard / final).

Pure dataclasses + enums, no model imports at module top: this loads and unit-tests on a CPU box.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from .models import DEFAULT_SPECS, ModelRole
from .routing import QualityTier


# --------------------------------------------------------------------------- helpers

def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    """Coerce `value` to a float inside [lo, hi]; fall back to `default` on junk. The contract
    stays forgiving: a planner that sends an out-of-range or non-numeric knob gets clamped, not
    an exception."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    return int(_clamp(value, lo, hi, default))


def _enum_or(enum_cls, value: Any, default):
    """Parse `value` into `enum_cls`, returning `default` for anything unrecognized. Accepts an
    already-typed member (so `from_dict(to_dict())` round-trips) as well as a wire string."""
    if value is None:
        return default  # absent key: keep the baseline (and never collide with a "none" member)
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower())
    except (ValueError, AttributeError):
        return default


def _model_id(role: ModelRole) -> str:
    return DEFAULT_SPECS[role].repo_id


# --------------------------------------------------------------------------- enums

class Scheduler(str, Enum):
    """Diffusers SDXL sampler. The few-step distill paths pin specific schedulers: Hyper-SD
    fixed-step LoRAs want `DDIM_TRAILING` (DDIM with timestep_spacing="trailing"), the unified
    1-step LoRA wants `TCD`. The full-step path is free to use a higher-order solver."""
    EULER = "euler"                    # EulerDiscreteScheduler
    EULER_A = "euler_ancestral"        # EulerAncestralDiscreteScheduler
    DPMPP_2M = "dpmpp_2m"              # DPMSolverMultistepScheduler
    DPMPP_2M_KARRAS = "dpmpp_2m_karras"  # DPMSolverMultistepScheduler(use_karras_sigmas=True)
    UNIPC = "unipc"                    # UniPCMultistepScheduler
    DDIM = "ddim"                      # DDIMScheduler
    DDIM_TRAILING = "ddim_trailing"    # DDIMScheduler(timestep_spacing="trailing") -- Hyper-SD
    TCD = "tcd"                        # TCDScheduler -- Hyper-SD unified LoRA


class IdentityMethod(str, Enum):
    """How a character's face is pinned onto the keyframe. `IP_ADAPTER` (the default for single
    AND multi-character shots) pulls identity from the reference embedding; in the regional
    multi-character path it is masked per region (one identity per mask), which is what holds two
    faces apart. `INSTANTID` is a single-character upgrade for face-critical shots (it adds a
    face-ControlNet for a stronger structural lock, but is single-face by nature, so it is NOT
    used on the regional multi-face path). `BOTH` stacks them: an advanced, future option, never
    a default. Mirrors the control plane's `face_lock` mode union."""
    IP_ADAPTER = "ip_adapter"
    INSTANTID = "instantid"
    BOTH = "both"


class I2VLoader(str, Enum):
    """Which loader applies the Wan2.2-Lightning distill LoRA. `DIFFUSERS` uses
    `pipe.load_lora_weights`; that path hits a known compat issue on some Wan LoRAs
    (diffusers #12535), so `LIGHTX2V` (the LightX2V / DiffSynth loader) is the documented
    fallback. The config only states the choice; the loader code lives in i2v.py."""
    DIFFUSERS = "diffusers"
    LIGHTX2V = "lightx2v"


class FeatureCache(str, Enum):
    """Final-tier inference cache for the full-step Wan path (a TeaCache successor). Reuses
    block features across adjacent steps for ~1.5-2x at high step counts. NEVER stack this on
    the 4-step distill path: at 4 steps there is nothing to cache."""
    NONE = "none"
    MIXCACHE = "mixcache"
    EASYCACHE = "easycache"


# ------------------------------------------------------------------ multi-character block

@dataclass
class MultiCharConfig:
    """Anti-bleed config for the **regional multi-character path only** (2+ characters in one
    frame). keyframe.py reads this block solely when `engine_for(scene) == "regional"`; a
    single-character shot never touches it, which is why the per-slot scales live here and not on
    `KeyframeConfig`. The identity method itself (IP-Adapter vs InstantID) is a `KeyframeConfig`
    field because it applies to every shot; here we only carry the per-region masked-IP-Adapter /
    pose knobs that separate two bodies. InstantID is deliberately absent: it is single-face by
    nature, so the regional path stays masked-IP-Adapter only.

    Defaults mirror `orchestrator.MULTI_CHAR_DEFAULTS` (engine=regional, pose_conditioning=True,
    lora_scale_per_slot=0.7, ip_adapter_scale_per_slot=0.7, max_slots=2)."""
    pose_conditioning: bool = True            # OpenPose ControlNet to separate bodies
    lora_scale_per_slot: float = 0.7          # 0..2; per-character LoRA strength in a shared frame
    ip_adapter_scale_per_slot: float = 0.7    # 0..1; per-region masked IP-Adapter identity pull
    max_slots: int = 2                        # 1..4; characters the no-bleed path supports at once
    controlnet_pose_scale: float = 0.55       # 0..1.5; OpenPose ControlNet conditioning scale
    region_gutter: int = 64                   # 0..256 px dead band between regional masks (anti seam-blend)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "MultiCharConfig":
        d = d if isinstance(d, dict) else {}
        base = cls()
        return cls(
            pose_conditioning=bool(d.get("pose_conditioning", base.pose_conditioning)),
            lora_scale_per_slot=_clamp(d.get("lora_scale_per_slot"), 0.0, 2.0, base.lora_scale_per_slot),
            ip_adapter_scale_per_slot=_clamp(d.get("ip_adapter_scale_per_slot"), 0.0, 1.0, base.ip_adapter_scale_per_slot),
            max_slots=_clamp_int(d.get("max_slots"), 1, 4, base.max_slots),
            controlnet_pose_scale=_clamp(d.get("controlnet_pose_scale"), 0.0, 1.5, base.controlnet_pose_scale),
            region_gutter=_clamp_int(d.get("region_gutter"), 0, 256, base.region_gutter),
        )


# --------------------------------------------------------------------------- keyframe

@dataclass
class KeyframeConfig:
    """SDXL keyframe-stage generation config.

    Fields map to documented diffusers SDXL parameters: `steps` -> num_inference_steps,
    `guidance_scale` -> CFG, `scheduler` -> the sampler, `width`/`height` -> the generated size,
    `seed` -> the RNG seed. `distill` toggles the Hyper-SD few-step path (`distill_steps` at
    cfg=0 on a DDIM-trailing / TCD scheduler). `base_model` defaults to the keyframe SDXL in
    `models.DEFAULT_SPECS`.

    Identity (applies to every shot): `identity_method` defaults to IP-Adapter; `ip_adapter_scale`
    is the single-subject identity pull; the `instantid_*` scales apply only when InstantID is
    selected (a single-character face-critical upgrade). The regional multi-character anti-bleed
    knobs live in `multi_char`, read only on the regional path.

    Built tier-aware via `for_tier`: draft = 4-step distilled, final = full-step high-cfg.
    """
    base_model: str = field(default_factory=lambda: _model_id(ModelRole.KEYFRAME_BASE))
    steps: int = 30                  # 1..128; diffusers num_inference_steps (full path)
    guidance_scale: float = 6.5      # 0..30; CFG. Hyper-SD few-step wants 0.0
    scheduler: Scheduler = Scheduler.DPMPP_2M_KARRAS
    width: int = 1024                # 512..2048; SDXL native is 1024
    height: int = 1024               # 512..2048
    distill: bool = False            # few-step Hyper-SD path on/off
    distill_model: str = field(default_factory=lambda: _model_id(ModelRole.KEYFRAME_FEWSTEP))
    distill_steps: int = 8           # 1..8; Hyper-SD fixed-step LoRA step count
    seed: int = 424242               # >=0; base RNG seed (control-plane default)
    # Identity stack (all shots). InstantID stays single-char; regional multi-char is masked IP-Adapter.
    identity_method: IdentityMethod = IdentityMethod.IP_ADAPTER
    ip_adapter_scale: float = 0.65   # 0..1; single-subject IP-Adapter identity pull (face_lock default)
    instantid_controlnet_scale: float = 0.8  # 0..1.5; InstantID face-ControlNet (single-char upgrade)
    instantid_ip_adapter_scale: float = 0.8  # 0..1.5; InstantID IP-Adapter (single-char upgrade)
    multi_char: MultiCharConfig = field(default_factory=MultiCharConfig)

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "KeyframeConfig":
        """The keyframe config for a quality tier. Draft and standard ride the Hyper-SD few-step
        LoRA (cfg=0, DDIM-trailing) to keep keyframes cheap; final drops distillation for a full
        high-CFG SDXL pass. Keyframes get animated downstream, so even final stays modest."""
        if tier is QualityTier.DRAFT:
            return cls(distill=True, distill_steps=4, steps=4, guidance_scale=0.0,
                       scheduler=Scheduler.DDIM_TRAILING)
        if tier is QualityTier.STANDARD:
            return cls(distill=True, distill_steps=8, steps=8, guidance_scale=0.0,
                       scheduler=Scheduler.DDIM_TRAILING)
        return cls(distill=False, steps=30, guidance_scale=6.5,
                   scheduler=Scheduler.DPMPP_2M_KARRAS)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, *, tier: QualityTier | None = None) -> "KeyframeConfig":
        """Build from a (forgiving) override dict, layered over the tier baseline. Unknown keys
        are ignored; numeric knobs are clamped to their documented range. Accepts either explicit
        `width`/`height` or a `resolution` "WIDTHxHEIGHT" string (the control plane's shape)."""
        base = cls.for_tier(tier) if tier is not None else cls()
        d = d if isinstance(d, dict) else {}
        w, h = base.width, base.height
        if isinstance(d.get("resolution"), str) and "x" in d["resolution"]:
            try:
                ws, hs = d["resolution"].lower().split("x", 1)
                w, h = int(ws), int(hs)
            except ValueError:
                pass
        return cls(
            base_model=str(d.get("base_model", base.base_model)),
            steps=_clamp_int(d.get("steps"), 1, 128, base.steps),
            guidance_scale=_clamp(d.get("guidance_scale"), 0.0, 30.0, base.guidance_scale),
            scheduler=_enum_or(Scheduler, d.get("scheduler"), base.scheduler),
            width=_clamp_int(d.get("width", w), 512, 2048, base.width),
            height=_clamp_int(d.get("height", h), 512, 2048, base.height),
            distill=bool(d.get("distill", base.distill)),
            distill_model=str(d.get("distill_model", base.distill_model)),
            distill_steps=_clamp_int(d.get("distill_steps"), 1, 8, base.distill_steps),
            seed=_clamp_int(d.get("seed"), 0, 2**31 - 1, base.seed),
            identity_method=_enum_or(IdentityMethod, d.get("identity_method"), base.identity_method),
            ip_adapter_scale=_clamp(d.get("ip_adapter_scale"), 0.0, 1.0, base.ip_adapter_scale),
            instantid_controlnet_scale=_clamp(d.get("instantid_controlnet_scale"), 0.0, 1.5, base.instantid_controlnet_scale),
            instantid_ip_adapter_scale=_clamp(d.get("instantid_ip_adapter_scale"), 0.0, 1.5, base.instantid_ip_adapter_scale),
            multi_char=MultiCharConfig.from_dict(d.get("multi_char")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------- i2v

@dataclass
class I2VConfig:
    """Wan 2.2 image-to-video generation config.

    Fields map to documented Wan 2.2 diffusers parameters: `steps` -> num_inference_steps,
    `guidance_scale` -> CFG, `num_frames` / `fps` -> the clip, `flow_shift` -> the
    FlowMatch scheduler shift. `distill` toggles the Wan2.2-Lightning 4-step path (CFG off, so
    `guidance_scale` ~1.0); `loader` selects the LoRA loader (diffusers vs the LightX2V fallback
    for issue #12535). `feature_cache` is the final-tier full-step accelerator and must stay
    NONE whenever `distill` is on (nothing to cache at 4 steps). `model` defaults to the i2v
    spec in `models.DEFAULT_SPECS`.

    Built tier-aware via `for_tier`: draft = Lightning 4-step, final = full 40-step + cache.
    """
    model: str = field(default_factory=lambda: _model_id(ModelRole.I2V))
    num_frames: int = 81             # 1..256; Wan2.2 documented default (5s at 16fps)
    fps: int = 16                    # 1..120; Wan2.2 export fps
    steps: int = 40                  # 1..64; Wan2.2 documented num_inference_steps (full)
    guidance_scale: float = 3.5      # 0..30; Wan2.2 documented CFG (full); distill ~1.0
    flow_shift: float = 5.0          # 0..20; FlowMatch scheduler shift
    seconds_per_shot: float = 5.0    # 0.5..60; derives num_frames when a shot has no duration
    distill: bool = False            # Wan2.2-Lightning 4-step path on/off
    distill_model: str = field(default_factory=lambda: _model_id(ModelRole.I2V_DISTILL))
    distill_steps: int = 4           # 1..8; Lightning runs 4
    loader: I2VLoader = I2VLoader.DIFFUSERS
    feature_cache: FeatureCache = FeatureCache.NONE
    seed: int = 0                    # >=0; i2v RNG seed, independent of the keyframe seed
    negative_prompt: str = ""        # optional; empty = the model's shipped default negative

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "I2VConfig":
        """The i2v config for a quality tier. Draft rides Wan2.2-Lightning (4 steps, CFG off, no
        cache -- nothing to cache at 4 steps); standard runs a reduced full-step pass with the
        EasyCache accelerator; final runs the full 40-step pass with MixCache (the long pole, the
        hero quality)."""
        if tier is QualityTier.DRAFT:
            return cls(distill=True, distill_steps=4, steps=4, guidance_scale=1.0,
                       loader=I2VLoader.DIFFUSERS, feature_cache=FeatureCache.NONE)
        if tier is QualityTier.STANDARD:
            return cls(distill=False, steps=20, guidance_scale=3.5,
                       feature_cache=FeatureCache.EASYCACHE)
        return cls(distill=False, steps=40, guidance_scale=3.5,
                   feature_cache=FeatureCache.MIXCACHE)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, *, tier: QualityTier | None = None) -> "I2VConfig":
        """Build from a (forgiving) override dict over the tier baseline. Unknown keys ignored,
        numeric knobs clamped. A caching choice is force-cleared to NONE whenever distillation is
        on, so an override cannot create the invalid 'cache a 4-step render' combination."""
        base = cls.for_tier(tier) if tier is not None else cls()
        d = d if isinstance(d, dict) else {}
        distill = bool(d.get("distill", base.distill))
        cache = _enum_or(FeatureCache, d.get("feature_cache"), base.feature_cache)
        if distill:
            cache = FeatureCache.NONE
        return cls(
            model=str(d.get("model", base.model)),
            num_frames=_clamp_int(d.get("num_frames"), 1, 256, base.num_frames),
            fps=_clamp_int(d.get("fps"), 1, 120, base.fps),
            steps=_clamp_int(d.get("steps"), 1, 64, base.steps),
            guidance_scale=_clamp(d.get("guidance_scale"), 0.0, 30.0, base.guidance_scale),
            flow_shift=_clamp(d.get("flow_shift"), 0.0, 20.0, base.flow_shift),
            seconds_per_shot=_clamp(d.get("seconds_per_shot"), 0.5, 60.0, base.seconds_per_shot),
            distill=distill,
            distill_model=str(d.get("distill_model", base.distill_model)),
            distill_steps=_clamp_int(d.get("distill_steps"), 1, 8, base.distill_steps),
            loader=_enum_or(I2VLoader, d.get("loader"), base.loader),
            feature_cache=cache,
            seed=_clamp_int(d.get("seed"), 0, 2**31 - 1, base.seed),
            negative_prompt=str(d.get("negative_prompt", base.negative_prompt)),
        )

    def frames_for(self, seconds: float | None) -> int:
        """Frame count for a shot of `seconds` at this fps, clamped to the documented 1..256
        ceiling; falls back to `num_frames` when no duration is given."""
        if not seconds or seconds <= 0:
            return self.num_frames
        return max(1, min(256, round(seconds * self.fps)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------- finish

class FaceRestore(str, Enum):
    """Which blind face restorer relocks the face over the i2v frames. NONE is off; the others name
    a model role the ModelServer loads. GFPGAN is the redistribution-clean default; CodeFormer is
    higher quality but ships under the S-Lab non-commercial license (so it is an opt-in the deployer
    chooses, NOT bundled by default -- same posture as the antelopev2 pack). The face_fidelity knob
    maps to the restorer's restoration/fidelity balance."""
    NONE = "none"
    GFPGAN = "gfpgan"
    CODEFORMER = "codeformer"


@dataclass
class FinishConfig:
    """Post-i2v finishing-stage config: the two clip-in/clip-out passes that run on the GPU worker
    after each shot animates and before the off-GPU assemble merges them (see `finish.py`).

    `interpolate` turns on RIFE frame interpolation; `interpolation_factor` is the recursive 2x
    multiple (2/4/8, snapped to a power of two) and `target_fps` (when > 0) caps the realized rate.
    `face_restore` selects a blind face restorer (off by default) that relocks the character's face
    over the motion frames; `face_fidelity` is its restoration/fidelity balance and `only_faces`
    keeps it to the detected faces.

    Built tier-aware via `for_tier`: draft finishes nothing (a fast preview), standard interpolates
    to smooth motion, final interpolates AND face-restores for the hero deliverable. Mirrors the
    namespaced `render_overrides` `{"finish": {...}}` section; parsing stays forgiving + clamped."""
    interpolate: bool = False
    interpolation_factor: int = 2     # 1/2/4/8; recursive RIFE doubling (1 = off)
    target_fps: int = 0               # 0 = src*factor; else a hard cap on the realized fps
    face_restore: FaceRestore = FaceRestore.NONE
    face_fidelity: float = 0.7        # 0..1; restorer balance (0 = max restoration, 1 = max fidelity)
    only_faces: bool = True           # restore detected faces only, leave the rest of the frame alone

    @property
    def enabled(self) -> bool:
        """Whether the finish stage runs at all for this render. When neither pass is on, the
        pipeline never calls `finish_clip` and the raw i2v clips ship unchanged."""
        return bool(self.interpolate or self.face_restore is not FaceRestore.NONE)

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "FinishConfig":
        if tier is QualityTier.DRAFT:
            return cls()  # a draft is a preview; do not spend GPU finishing it
        if tier is QualityTier.STANDARD:
            return cls(interpolate=True, interpolation_factor=2)  # smooth motion, no restore
        return cls(interpolate=True, interpolation_factor=2, face_restore=FaceRestore.GFPGAN)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, *, tier: QualityTier | None = None) -> "FinishConfig":
        # RIFE interpolates by recursive doubling, so the only valid factors are powers of two
        # (1/2/4/8). A plain [1,8] clamp would let 3/5/6/7 through and `finish` then silently rounds
        # them DOWN at render time, so the stored config would not match what actually runs. Snap
        # here, at config validation, so the typed config is the truth: factor 3 -> 2, 6 -> 4, etc.
        from .finish import snap_factor  # deferred: keeps finish CPU-light and avoids any cycle

        base = cls.for_tier(tier) if tier is not None else cls()
        d = d if isinstance(d, dict) else {}
        return cls(
            interpolate=bool(d.get("interpolate", base.interpolate)),
            interpolation_factor=snap_factor(
                _clamp_int(d.get("interpolation_factor"), 1, 8, base.interpolation_factor)),
            target_fps=_clamp_int(d.get("target_fps"), 0, 120, base.target_fps),
            face_restore=_enum_or(FaceRestore, d.get("face_restore"), base.face_restore),
            face_fidelity=_clamp(d.get("face_fidelity"), 0.0, 1.0, base.face_fidelity),
            only_faces=bool(d.get("only_faces", base.only_faces)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------- LoRA

def _lora_from_dict(d: dict[str, Any] | None):
    """Parse a LoRA training-config override block into the existing
    `lora_train.LoraTrainConfig` (the single source of truth for training knobs; not re-derived
    here). Forgiving + clamped, same as the other sections. The import is deferred to avoid a
    contract -> config -> lora_train -> contract import cycle."""
    from .lora_train import LoraTrainConfig  # deferred: lora_train imports contract

    base = LoraTrainConfig()
    d = d if isinstance(d, dict) else {}
    return LoraTrainConfig(
        rank=_clamp_int(d.get("rank"), 1, 128, base.rank),
        resolution=_clamp_int(d.get("resolution"), 512, 1536, base.resolution),
        learning_rate=_clamp(d.get("learning_rate"), 1e-6, 1e-2, base.learning_rate),
        max_steps=_clamp_int(d.get("max_steps"), 1, 5000, base.max_steps),
        batch_size=_clamp_int(d.get("batch_size"), 1, 8, base.batch_size),
        gradient_accumulation_steps=_clamp_int(d.get("gradient_accumulation_steps"), 1, 32, base.gradient_accumulation_steps),
        seed=_clamp_int(d.get("seed"), 0, 2**31 - 1, base.seed),
        random_flip=bool(d.get("random_flip", base.random_flip)),
        gradient_checkpointing=bool(d.get("gradient_checkpointing", base.gradient_checkpointing)),
        caption_template=str(d.get("caption_template", base.caption_template)),
        save_every=_clamp_int(d.get("save_every"), 0, 5000, base.save_every),
    )


def _default_lora():
    from .lora_train import LoraTrainConfig
    return LoraTrainConfig()


# ----------------------------------------------------------------------------- top level

@dataclass
class RenderConfig:
    """The whole typed generation contract for one render: the quality tier plus the three
    stage configs. This is the single source of truth both the control plane and the backend
    build to, replacing the untyped `render_overrides` grab-bag for generation parameters.

    `quality` drives the tier baselines (`KeyframeConfig.for_tier` / `I2VConfig.for_tier`); a
    `render_overrides` payload then layers explicit knobs over those baselines. The expected
    payload shape is namespaced -- `{"keyframe": {...}, "i2v": {...}, "lora": {...}}` -- and
    parsing stays forgiving: unknown keys and unknown sections are ignored, so a newer control
    plane never breaks an older backend.

    LoRA training is not quality-tier dependent (the adapter is the adapter), so the `lora`
    section is the same across tiers; it leans on `lora_train.LoraTrainConfig` rather than
    re-deriving the training knobs.
    """
    quality: QualityTier = QualityTier.FINAL
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    i2v: I2VConfig = field(default_factory=I2VConfig)
    finish: FinishConfig = field(default_factory=FinishConfig)
    lora: Any = field(default_factory=_default_lora)  # lora_train.LoraTrainConfig (deferred type)

    @classmethod
    def for_tier(cls, tier: QualityTier) -> "RenderConfig":
        return cls(
            quality=tier,
            keyframe=KeyframeConfig.for_tier(tier),
            i2v=I2VConfig.for_tier(tier),
            finish=FinishConfig.for_tier(tier),
            lora=_default_lora(),
        )

    @classmethod
    def from_request(cls, quality_tier: object, overrides: dict[str, Any] | None) -> "RenderConfig":
        """Build the full config from a request's `quality_tier` and `render_overrides`. The tier
        sets every baseline; the (forgiving) overrides layer explicit knobs over it."""
        tier = QualityTier.parse(quality_tier)
        d = overrides if isinstance(overrides, dict) else {}
        return cls(
            quality=tier,
            keyframe=KeyframeConfig.from_dict(d.get("keyframe"), tier=tier),
            i2v=I2VConfig.from_dict(d.get("i2v"), tier=tier),
            finish=FinishConfig.from_dict(d.get("finish"), tier=tier),
            lora=_lora_from_dict(d.get("lora")),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderConfig":
        """Reconstruct from a dict produced by `to_dict`, completing the round-trip."""
        tier = QualityTier.parse(d.get("quality", "final"))
        return cls(
            quality=tier,
            keyframe=KeyframeConfig.from_dict(d.get("keyframe"), tier=tier),
            i2v=I2VConfig.from_dict(d.get("i2v"), tier=tier),
            finish=FinishConfig.from_dict(d.get("finish"), tier=tier),
            lora=_lora_from_dict(d.get("lora")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality": self.quality.value,
            "keyframe": self.keyframe.to_dict(),
            "i2v": self.i2v.to_dict(),
            "finish": self.finish.to_dict(),
            "lora": asdict(self.lora),
        }
