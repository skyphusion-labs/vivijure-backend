"""Render planning: decide everything on the CPU, touch the GPU only for the irreducible.

The expensive resource is GPU-seconds, and the cheapest way to save them is to never spend
them on a decision a CPU can make. So this module computes a complete RenderPlan up front,
on the CPU, from the request plus the storyboard: which LoRAs actually need training (versus
reused), which keyframes need generating (versus reused or injected), which scenes take the
multi-character path, where each stage should run, and roughly what it will cost. Only after
the plan is settled does any GPU work fire, and it fires only for the work the plan could not
eliminate.

This is deliberate. A render that validates late, retrains a LoRA it already has, regenerates
a keyframe that already exists, or assembles on the GPU what a CPU container could mux, is
burning the most expensive resource on the cheapest kind of mistake. The planner exists to
make those mistakes unreachable, and it records every elimination so the savings are visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .contract import Cast, RenderRequest, Scene, Storyboard
from .routing import QualityTier, Stage, Tier, gpu_for
import hashlib
import json

# Anti-bleed defaults for the multi-character keyframe path (the standard, not cranked,
# per-slot scales + pose conditioning that hold two identities apart).
MULTI_CHAR_DEFAULTS: dict[str, object] = {
    "engine": "regional",
    "pose_conditioning": True,
    "lora_scale_per_slot": 0.7,
    "ip_adapter_scale_per_slot": 0.7,
    "max_slots": 2,
    "region_gutter": 64,
}


class Action(str, Enum):
    """What a render job asks for; selects which stages the planner schedules."""
    RENDER = "render"          # full pipeline: train -> keyframes -> i2v -> assemble
    PREVIEW = "preview"        # keyframes-only preview: train -> keyframes, NO i2v, no MP4
    FINALIZE = "finalize"      # i2v over existing keyframes (no keyframe gen)
    REGEN_SHOT = "regen_shot"  # regenerate named keyframes only, no i2v
    TRAIN_LORA = "train_lora"  # train LoRAs only

    @classmethod
    def parse(cls, value: str) -> "Action":
        try:
            return cls(value)
        except ValueError:
            return cls.RENDER


class KeyframeMode(str, Enum):
    """How a shot's keyframe is obtained: drawn fresh, reused from disk, or an authored inject."""
    GENERATE = "generate"  # SDXL has to draw it (GPU)
    REUSE = "reuse"        # already on disk from a prior pass (no GPU)
    INJECT = "inject"      # authored start_image supplied (no GPU)


# Rough per-unit GPU-second estimates. Deliberately conservative and easy to retune; the point
# is an order-of-magnitude "what will this cost" before anything burns, not a stopwatch.
_COST = {
    "lora_train_per_slot": 180.0,
    "keyframe_generate": 6.0,
    "i2v_per_target_second": {QualityTier.DRAFT: 4.0, QualityTier.STANDARD: 9.0, QualityTier.FINAL: 18.0},
}


@dataclass
class ScenePlan:
    """The planner's decision for one shot: how to get its keyframe, whether it animates, and the
    GPU tier its i2v should run on."""
    shot_id: str
    keyframe_mode: KeyframeMode
    is_multi_character: bool
    needs_i2v: bool
    target_seconds: float | None = None
    multi_char: dict[str, object] = field(default_factory=dict)
    i2v_tier: Tier | None = None  # which GPU tier should run this shot's i2v


@dataclass
class LoraPlan:
    """Which character slots must train versus which are reused (already trained or pretrained)."""
    train: list[str] = field(default_factory=list)   # slots that actually have to train (GPU)
    reuse: list[str] = field(default_factory=list)   # slots skipped: already trained / pretrained


@dataclass
class RenderPlan:
    """The complete CPU-decided plan for a render: the LoRA plan, the per-scene plans, whether
    assembly is off-GPU, a GPU-seconds estimate, and the GPU work the plan eliminated."""
    action: Action
    project: str
    quality: QualityTier
    lora: LoraPlan
    scenes: list[ScenePlan]
    assemble_off_gpu: bool          # merge on a CPU container, never burn GPU on ffmpeg
    estimated_gpu_seconds: float
    skips: list[str] = field(default_factory=list)   # every GPU unit the plan eliminated, in words

    @property
    def keyframes_to_generate(self) -> int:
        return sum(1 for s in self.scenes if s.keyframe_mode is KeyframeMode.GENERATE)

    @property
    def shots_to_animate(self) -> int:
        return sum(1 for s in self.scenes if s.needs_i2v)

    def summary(self) -> str:
        lines = [
            f"plan[{self.action.value}] {self.project} ({self.quality.value}): "
            f"{len(self.lora.train)} LoRA train, {self.keyframes_to_generate} keyframe gen, "
            f"{self.shots_to_animate} i2v, assemble={'off-GPU' if self.assemble_off_gpu else 'on-GPU'}",
            f"  est GPU: ~{self.estimated_gpu_seconds / 60:.1f} min",
        ]
        for s in self.skips:
            lines.append(f"  saved: {s}")
        return "\n".join(lines)


def kf_hash(kc) -> str:
    """Short SHA-256 hash of the keyframe render params (steps, guidance, seed, model, and the
    multi-char anti-bleed knobs). Stored alongside each keyframe in state so an incremental
    re-run with changed params forces regeneration instead of silently reusing a stale image.

    Covers every input that changes the SDXL output but NOT the prompt or character refs -- those
    are project-level inputs that change the bundle, not the config. If only prompt/refs changed,
    the user supplies a new bundle and the state tar is naturally stale (different project slot).

    Accepts any object with the KeyframeConfig attribute shape to avoid a circular import; the
    type is enforced by the callers."""
    mc = kc.multi_char
    payload = {
        "steps": kc.distill_steps if kc.distill else kc.steps,
        "guidance": kc.guidance_scale,
        "seed": kc.seed,
        "base_model": kc.base_model,
        "identity_method": kc.identity_method.value,
        "width": kc.width,
        "height": kc.height,
        "lora_scale": mc.lora_scale_per_slot,
        "ip_scale": mc.ip_adapter_scale_per_slot,
        "pose_cond": mc.pose_conditioning,
        "cn_scale": mc.controlnet_pose_scale,
        "gutter": mc.region_gutter,
        "max_slots": mc.max_slots,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def validate(request: RenderRequest, storyboard: Storyboard, *, cast: Cast | None = None) -> list[str]:
    """Cheap preflight. Catch on the CPU what would otherwise fail minutes into a GPU job."""
    errs: list[str] = []
    if not storyboard.scenes:
        errs.append("storyboard has no scenes")
    for sc in storyboard.scenes:
        if not sc.prompt or not sc.prompt.strip():
            errs.append(f"scene {sc.id!r} has an empty prompt")
    slots_used = {sl for sc in storyboard.scenes for sl in sc.character_slots}
    unknown = slots_used - set(storyboard.use_characters)
    if unknown:
        errs.append(f"scenes reference slots not in use_characters: {sorted(unknown)}")
    if cast is not None:
        for slot in storyboard.use_characters:
            if slot not in cast.characters:
                errs.append(f"use_characters slot {slot!r} has no entry in the cast registry")
    if request.process_shot_ids:
        known = {sc.id for sc in storyboard.scenes}
        missing = [s for s in request.process_shot_ids if s not in known]
        if missing:
            errs.append(f"process_shot_ids not in storyboard: {missing}")
    return errs


def plan(
    request: RenderRequest,
    storyboard: Storyboard,
    *,
    trained_slots: set[str] = frozenset(),       # slots whose LoRA already exists (prior state)
    existing_keyframes: dict[str, str | None] = {},  # shot_id -> stored hash (None = old state, no hash)
) -> RenderPlan:
    """Decide the whole render on the CPU. Nothing here touches a GPU."""
    action = Action.parse(request.action)
    quality = QualityTier.parse(request.quality_tier)
    skips: list[str] = []

    # --- which scenes are in scope (finalize / regen can target a subset) ---
    scope = set(request.process_shot_ids) if request.process_shot_ids else None
    scenes = [s for s in storyboard.scenes if scope is None or s.id in scope]
    if scope is not None:
        dropped = len(storyboard.scenes) - len(scenes)
        if dropped:
            skips.append(f"{dropped} scene(s) out of process_shot_ids scope: not rendered")

    # --- LoRA plan: train only what is neither pretrained nor already trained ---
    needed = list(storyboard.use_characters)
    already = set(trained_slots) | set(request.pretrained_loras.keys())
    to_train = [s for s in needed if s not in already]
    reused = [s for s in needed if s in already]
    if action in (Action.FINALIZE,):  # i2v-only: keyframes (and their LoRAs) already happened
        if to_train:
            skips.append(f"finalize: {len(to_train)} LoRA(s) not trained (i2v-only pass)")
        to_train = []
    for s in reused:
        why = "pretrained passthrough" if s in request.pretrained_loras else "already trained"
        skips.append(f"LoRA slot {s}: reused ({why})")
    lora = LoraPlan(train=to_train, reuse=reused)

    # --- per-scene keyframe + i2v plan ---
    # PREVIEW is RENDER minus motion: it draws keyframes (and trains the LoRAs they need, since
    # to_train above is only zeroed for FINALIZE) but never runs i2v, so `_finish` assembles no
    # MP4 (no clips) and the user gets a keyframe preview before committing GPU-seconds to Wan.
    # keyframes_only (issue #119): a `render` (or finalize) carrying the flag draws its keyframes
    # but STOPS before i2v/finish, the same motion-skip the PREVIEW action takes. Without this the
    # flag is silently dropped and the caller pays a full ~27min render for a draft keyframe pass.
    want_i2v = action in (Action.RENDER, Action.FINALIZE) and not request.keyframes_only
    want_keyframe = action in (Action.RENDER, Action.PREVIEW, Action.REGEN_SHOT)
    # A train-only job (neither keyframe nor i2v) has no per-scene render: building scene
    # plans would estimate keyframe GPU-seconds for work that never fires, which is exactly
    # the wasteful over-estimate this planner exists to prevent.
    scene_plans: list[ScenePlan] = []
    for sc in (scenes if (want_keyframe or want_i2v) else []):
        cur_hash = kf_hash(request.config.keyframe)
        mode = _keyframe_mode(sc, action, existing_keyframes, want_keyframe, cur_hash)
        if mode is not KeyframeMode.GENERATE and want_keyframe:
            skips.append(f"{sc.id} keyframe: {mode.value} (no SDXL pass)")
        multi = sc.is_multi_character
        scene_plans.append(ScenePlan(
            shot_id=sc.id,
            keyframe_mode=mode,
            is_multi_character=multi,
            needs_i2v=want_i2v,
            target_seconds=sc.target_seconds,
            multi_char=dict(MULTI_CHAR_DEFAULTS) if multi else {},
            i2v_tier=gpu_for(Stage.I2V, quality) if want_i2v else None,
        ))

    assemble_off_gpu = True  # ffmpeg is cheap CPU work; never burn GPU on the merge
    plan_obj = RenderPlan(
        action=action,
        project=request.project,
        quality=quality,
        lora=lora,
        scenes=scene_plans,
        assemble_off_gpu=assemble_off_gpu,
        estimated_gpu_seconds=0.0,
        skips=skips,
    )
    plan_obj.estimated_gpu_seconds = _estimate_cost(plan_obj)
    return plan_obj


def _keyframe_mode(
    scene: Scene, action: Action,
    existing: dict[str, str | None],
    want_keyframe: bool,
    current_hash: str = "",
) -> KeyframeMode:
    if action is Action.FINALIZE:
        return KeyframeMode.REUSE  # i2v-only reuses the preview's keyframe
    if scene.start_image:
        return KeyframeMode.INJECT  # authored / cloud-made keyframe handed in
    if want_keyframe and scene.id in existing:
        stored = existing[scene.id]
        if stored is None:
            return KeyframeMode.REUSE  # old state without a hash; reuse conservatively
        if stored == current_hash:
            return KeyframeMode.REUSE  # params unchanged; safe to reuse
        # hash mismatch: render params changed since the cached keyframe was drawn
    return KeyframeMode.GENERATE


def _estimate_cost(p: RenderPlan) -> float:
    # These per-second constants reflect uncached, full-step timing. EASYCACHE (standard tier)
    # and MIXCACHE (final tier) reduce actual i2v GPU-seconds by 30-50%; the estimate is
    # therefore conservative (high) for those tiers and is intended only as an order-of-magnitude
    # pre-flight check, not a billing figure.
    secs = len(p.lora.train) * _COST["lora_train_per_slot"]
    secs += p.keyframes_to_generate * _COST["keyframe_generate"]
    per_sec = _COST["i2v_per_target_second"][p.quality]
    for s in p.scenes:
        if s.needs_i2v:
            secs += (s.target_seconds or 5.0) * per_sec
    return round(secs, 1)
