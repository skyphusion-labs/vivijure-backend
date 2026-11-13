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
from .harness.keys import lora_key
from .routing import QualityTier, Stage, Tier, gpu_for
import hashlib
import json

# Bump on ANY change to how a keyframe is drawn that leaves the prompts the same (the encoder, the passes, the
# pipelines, the identity path). A change to prompt building needs no bump: the built prompts are in the hash
# (vivijure-backend#573). Without this, a new image re-rendering the same board returned the old stills as
# `reused` and the run read COMPLETED (film-0afa679b on 1.1.6).
KEYFRAME_ENGINE_VERSION = "kf-2026-11-06-chunked"

# Anti-bleed defaults for the multi-character keyframe path (the standard, not cranked,
# per-slot scales + pose conditioning that hold two identities apart).
MULTI_CHAR_DEFAULTS: dict[str, object] = {
    "engine": "regional",
    "pose_conditioning": None,  # per-path default: regional on, a guarded scene-lock plate off (#596)
    "lora_scale_per_slot": 0.35,
    "ip_adapter_scale_per_slot": None,  # unset: the keyframe ip_adapter_scale (vivijure-backend#599)
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
    # Wan 2.2 A14B LoRA training is a different order of magnitude from the SDXL adapter: the
    # Phase-0 spike (vivijure-cf #29) bound identity in ~40min on an 80GB A100 (a full to-2000-step
    # run is longer). This estimate is GPU-HOURS-scale, deliberately conservative (the planner
    # over-estimates by design), and used only when a train job is model_family="wan".
    "wan_lora_train_per_slot": 3600.0,
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
    lora_family: str = "sdxl"                        # "sdxl" (in-process) or "wan" (ai-toolkit A14B)
    # vivijure-backend#478: slot -> the adapter KEY a drawn keyframe binds for that slot (bound_lora_keys).
    # The planner hashes with it and the pipeline's sidecar writer reads it from here, so the two cannot differ.
    bound_loras: dict[str, str] = field(default_factory=dict)

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


# The value kf_hash's `ip_scale` key has always carried (the per-slot field's old default, which never reached a draw).
# Fixed, not derived: changing it would change every stored still's hash and redraw the fleet (vivijure-backend#599).
LEGACY_IP_SCALE_HASH = 0.7


def bound_lora_keys(project: str, lora: LoraPlan, trained_slots, pretrained_loras: dict[str, str],
                    lora_family: str = "sdxl") -> dict[str, str]:
    """vivijure-backend#478: which character LoRA a keyframe draw binds, per slot, as the adapter KEY.

    The one place the mapping is built. `plan` hashes with it and stores it on the RenderPlan, and the
    pipeline's sidecar writer reads it from the plan, so the hash a still is stored with and the hash it is
    checked against come from the same map. `trained_slots` is what `_restore_prior_state` found in R2
    (the harness adds the pretrained slots to it before calling `plan`).

    - A slot trained this run, or reused from R2, binds the project key `lora_key(project, slot)`.
    - A `pretrained_loras` entry supersedes it, as `_stage_pretrained_loras` stages it.
    - A Wan-family job draws no SDXL keyframe with a project adapter, so it binds only pretrained ones.

    LIMIT, accepted on #478: a retrain at the SAME key keeps the same map, so it is not caught. The fix for
    that is a versioned LoRA key on retrain, which is not this change."""
    out: dict[str, str] = {}
    if str(lora_family).strip().lower() != "wan":
        trained = set(trained_slots)
        for slot in list(lora.train) + [s for s in lora.reuse if s in trained]:
            out[slot] = lora_key(project, slot)
    for slot, ref in pretrained_loras.items():
        if isinstance(ref, str) and ref:
            out[slot] = ref
    return out


def kf_hash(kc, scene: Scene | None = None, cast: Cast | None = None, storyboard: Storyboard | None = None,
            bound_loras: dict[str, str] | None = None) -> str:
    """Short SHA-256 hash of what draws ONE shot's keyframe. Stored beside each keyframe so a re-run reuses a
    still only when nothing that draws it has changed.

    It covers the render params (steps, guidance, scheduler, seed, models, the distill flag, the identity
    scales, the multi-char anti-bleed knobs, the scene-lock two-pass knobs), the keyframe engine version, and,
    given the shot, the prompts the engine would build for it: the plate, the identity pass and the plain
    pass. So an edited shot, a changed cast description, a new setting, or new prompt-building code redraws
    (vivijure-backend#573). Before, the prompt was left out on the theory that a changed board means a new
    project; a board keeps its project when it is edited, so an edited shot reused its old still.

    The writer (pipeline) and the reader (plan) must pass the same shot, cast and board. If they ever
    differ, the hashes differ and the still is redrawn: the safe direction.

    `bound_loras` (vivijure-backend#478, from `bound_lora_keys`) is narrowed to the shot's own slots, so a
    repointed, added or removed LoRA redraws only the shots its character is in. With no LoRA bound for those
    slots it adds no key, so the hash of a still drawn without a LoRA is the hash it had before #478.

    Flipping `scene_lock` MUST change this hash so incremental preview does not reuse grey
    studio portraits. Accepts any object with the KeyframeConfig attribute shape to avoid a
    circular import; the type is enforced by the callers."""
    mc = kc.multi_char
    payload = {
        "steps": kc.distill_steps if kc.distill else kc.steps,
        "guidance": kc.guidance_scale,
        "scheduler": kc.scheduler.value,
        "seed": kc.seed,
        "base_model": kc.base_model,
        "distill": kc.distill,
        "distill_model": kc.distill_model,
        "identity_method": kc.identity_method.value,
        "ip_adapter_scale": kc.ip_adapter_scale,
        "instantid_ip_adapter_scale": kc.instantid_ip_adapter_scale,
        "width": kc.width,
        "height": kc.height,
        "lora_scale": kc.lora_scale,
        "lora_scale_per_slot": mc.lora_scale_per_slot,
        # vivijure-backend#599: this key always read 0.7, the old field default, and the field never reached a draw.
        # It stays 0.7 so every stored hash holds; a set per-slot value is a new key below, so it redraws.
        "ip_scale": LEGACY_IP_SCALE_HASH,
        # Unset reads as on, its meaning before #596, so this key keeps every existing hash; where the new
        # default changes a drawing, `plate_pose` below says so.
        "pose_cond": mc.pose_conditioning is not False,
        "cn_scale": mc.controlnet_pose_scale,
        "gutter": mc.region_gutter,
        "max_slots": mc.max_slots,
        "scene_lock": kc.scene_lock,
        "canny_scale": kc.canny_scale,
        "face_crop_pad": kc.face_crop_pad,
        "plate_ip_adapter_scale": kc.plate_ip_adapter_scale,
        "plate_regional_text": kc.plate_regional_text,
        "plate_people_free_background": kc.plate_people_free_background,
        "two_pass": "scene_lock_v1",
        "engine": KEYFRAME_ENGINE_VERSION,
    }
    # vivijure-backend#599: a set per-slot identity scale now changes the drawing, so it says so; unset adds nothing.
    if mc.ip_adapter_scale_per_slot is not None:
        payload["ip_scale_per_slot"] = mc.ip_adapter_scale_per_slot
    # Only when set: at 0 the skeleton is byte-identical to the one drawn before this field existed, so a
    # still drawn then is still the same image and keeps its hash (no fleet-wide redraw for a no-op).
    if mc.pose_spread:
        payload["pose_spread"] = mc.pose_spread
    # Same rule (vivijure-cf#1281): "stand" is the figure drawn before this field existed, byte for byte, so it adds
    # nothing and every stored still keeps its hash. "reach" and "auto" draw a different control, so they say so.
    if mc.pose_template != "stand":
        payload["pose_template"] = mc.pose_template
    # Same rule (vivijure-cf#1442): "off" is the Pass B prompt as before, byte for byte, so it adds nothing. A set mode
    # says so here; on a shot the built Pass B prompt below carries the clause itself.
    if kc.pass_b_gaze != "off":
        payload["pass_b_gaze"] = kc.pass_b_gaze
    # Same rule (vivijure-cf#1442): "off" is the plate as before, byte for byte. The per-region plate prompts are not
    # in the hash below (only the flag that turns them on is), so a set mode says so here.
    if kc.plate_gaze != "off":
        payload["plate_gaze"] = kc.plate_gaze
    # Same rule: off draws exactly what it drew before, so the hash stays. And only on a shot the guard can run on
    # (2..max_slots people, keyframe.engine_for): a one-person or empty shot draws the same with it on, so turning
    # the default on must not redraw those stills. With no scene (a config-only hash) it is included when set.
    guarded = scene is None or 2 <= len(scene.character_slots) <= mc.max_slots
    if kc.plate_count_retries and guarded:
        payload["plate_count_retries"] = kc.plate_count_retries
        # vivijure-cf#1281: under per-region text the guard also redraws a plate short of a person, so only those
        # stills change and only they get this key; every other still keeps its hash (keyframe.plate_face_floor_on).
        if kc.plate_regional_text:
            payload["plate_face_floor"] = 1
    # backend#596: a guarded scene-lock plate for two-plus people now draws without the pose skeleton when the
    # job leaves pose_conditioning unset. Only those stills change, so only they get this key and redraw; the
    # rule is keyframe.plate_pose_default_off, the same function the renderer reads.
    from .keyframe import plate_pose_default_off
    if kc.scene_lock and plate_pose_default_off(
            None if scene is None else len(scene.character_slots), max_slots=mc.max_slots,
            pose_conditioning=mc.pose_conditioning, plate_count_retries=kc.plate_count_retries):
        payload["plate_pose"] = "off"
    if bound_loras:
        slots = bound_loras.keys() if scene is None else scene.character_slots
        bound = {s: bound_loras[s] for s in sorted(set(slots)) if s in bound_loras}
        if bound:
            payload["bound_loras"] = bound
    if scene is not None and storyboard is not None:
        from .keyframe import DEFAULT_NEGATIVE, build_prompt, pass_b_gaze_clause, scene_lock_negative

        c = cast if cast is not None else Cast(characters={})
        payload["prompts"] = [
            build_prompt(scene, c, storyboard, include_triggers=False),
            build_prompt(scene, c, storyboard, include_triggers=True, lock_clause=True,
                         gaze=pass_b_gaze_clause(scene, kc.pass_b_gaze, mc.max_slots, c) if kc.scene_lock else ""),
            build_prompt(scene, c, storyboard),
            scene_lock_negative(scene, DEFAULT_NEGATIVE),
        ]
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


def resolve_lora_family(action: Action, model_family: str) -> str:
    """Pick SDXL vs Wan training for this job.

    Wan character LoRA training lives on the ``vivijure-wan-train`` satellite endpoint
    (``RUNPOD_WAN_TRAIN_ENDPOINT_ID``). This render backend trains SDXL adapters inline only.
    Explicit ``model_family:"wan"`` is rejected at the pipeline layer on this image.
    """
    explicit = str(model_family or "").strip().lower()
    if explicit == "wan":
        return "wan"
    if explicit == "sdxl":
        return "sdxl"
    if explicit:
        return "sdxl"
    return "sdxl"


def plan(
    request: RenderRequest,
    storyboard: Storyboard,
    *,
    trained_slots: set[str] = frozenset(),       # slots whose LoRA already exists (prior state)
    existing_keyframes: dict[str, str | None] = {},  # shot_id -> stored hash (None = unproven, redrawn; old state, no hash)
    cast: Cast | None = None,  # the bundle's cast: its descriptions are in each shot's keyframe hash (#573)
) -> RenderPlan:
    """Decide the whole render on the CPU. Nothing here touches a GPU."""
    action = Action.parse(request.action)
    quality = QualityTier.parse(request.quality_tier)
    lora_family = resolve_lora_family(action, request.model_family)
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
    if action is Action.REGEN_SHOT:  # keyframe redraw only: the contract says regen_shot trains no LoRAs
        if to_train:
            skips.append(f"regen_shot: {len(to_train)} LoRA(s) not trained (keyframe redraw only)")
        to_train = []
    for s in reused:
        why = "pretrained passthrough" if s in request.pretrained_loras else "already trained"
        skips.append(f"LoRA slot {s}: reused ({why})")
    lora = LoraPlan(train=to_train, reuse=reused)
    bound = bound_lora_keys(request.project, lora, trained_slots, request.pretrained_loras, lora_family)

    # --- per-scene keyframe + i2v plan ---
    # PREVIEW is RENDER minus motion: it draws keyframes (and trains the LoRAs they need, since
    # to_train above is only zeroed for FINALIZE and REGEN_SHOT) but never runs i2v, so `_finish` assembles no
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
        cur_hash = kf_hash(request.config.keyframe, scene=sc, cast=cast, storyboard=storyboard, bound_loras=bound)
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
        lora_family=lora_family,
        bound_loras=bound,
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
        # Reuse needs PROOF the parameters match (#488). A still with no hash may be another render's:
        # film-a9939630 asked for SDXL and returned a cloud door's stills as COMPLETED, drawing nothing.
        if existing[scene.id] == current_hash:
            return KeyframeMode.REUSE  # params unchanged; safe to reuse
        # no hash, or a hash for other params: draw it
    return KeyframeMode.GENERATE


def _estimate_cost(p: RenderPlan) -> float:
    # These per-second constants reflect uncached, full-step timing. EASYCACHE (standard tier)
    # and MIXCACHE (final tier) reduce actual i2v GPU-seconds by 30-50%; the estimate is
    # therefore conservative (high) for those tiers and is intended only as an order-of-magnitude
    # pre-flight check, not a billing figure.
    per_slot = _COST["wan_lora_train_per_slot"] if p.lora_family == "wan" else _COST["lora_train_per_slot"]
    secs = len(p.lora.train) * per_slot
    secs += p.keyframes_to_generate * _COST["keyframe_generate"]
    per_sec = _COST["i2v_per_target_second"][p.quality]
    for s in p.scenes:
        if s.needs_i2v:
            secs += (s.target_seconds or 5.0) * per_sec
    return round(secs, 1)
