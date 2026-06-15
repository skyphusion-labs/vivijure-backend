"""Per-scene SDXL keyframe: the still each shot's i2v animates from.

One scene, one image. For a single character the job is ordinary: SDXL with the slot's trained
LoRA and an IP-Adapter pulling identity from the bundle refs. The hard case is two characters
in one frame, where the naive result is *bleed*: one identity's features smear onto the other
(both faces drift toward an average, hair color crosses over). The fix this module implements
is regional: each character is bound to its own half of the canvas with a per-region IP-Adapter
mask and its own LoRA, at deliberately moderate scales (a strong per-slot scale is what causes
the cross-contamination), with optional OpenPose conditioning to plant two distinct bodies so
the regions have something to attach to.

Clean-room: the SDXL + IP-Adapter masking + ControlNet wiring is built from diffusers' own
documented interfaces (the IP-Adapter regional/masking guide, the SDXL ControlNet pipeline), not
any prior pipeline. The prompt building, region geometry, and engine-path decision are pure and
unit-tested on a CPU box; the generation body defers torch/diffusers/PIL and is validated on a
pod. Engine knobs live in `KeyframeParams` here; the control plane's typed `KeyframeConfig`
(separate work) maps into them, it does not replace them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .contract import Cast, Scene, Storyboard

# Anti-bleed defaults: a character LoRA pushed hard in a shared frame is exactly what bleeds, so
# the per-slot scales are deliberately moderate and the identity comes mostly from the masked
# IP-Adapter. Mirrors orchestrator.MULTI_CHAR_DEFAULTS (the planner's view of the same numbers).
DEFAULT_LORA_SCALE = 0.3
DEFAULT_IP_ADAPTER_SCALE = 0.7
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, extra limbs, fused faces, two heads, deformed, blurry, watermark, text"
)
# The persistent base adapter name the ModelServer loads the Hyper-SD few-step distill LoRA under
# (models.ModelServer.keyframe_pipeline). _bind_loras keeps it active at 1.0 on the few-step path and
# 0.0 on the full-step (final) path, so one warm pipe renders every tier.
DISTILL_ADAPTER = "distill"


@dataclass
class KeyframeParams:
    """Engine knobs for one keyframe render. Defaults suit the few-step distilled base (the
    keyframe gets animated, so 6-8 steps is plenty); the control plane's KeyframeConfig fills
    these per job."""
    steps: int = 8
    guidance_scale: float = 5.0
    width: int = 1024                # SDXL native side; non-square (16:9 / vertical) is honored
    height: int = 1024               # the two dims are threaded independently through the engine
    seed: int = 0
    few_step: bool = True            # use the Hyper-SD/DMD2 distilled path the ModelServer loaded
    scheduler: str = "ddim_trailing" # config.Scheduler value; few-step wants ddim_trailing, final a solver
    lora_scale: float = DEFAULT_LORA_SCALE
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE
    # Identity method for the SINGLE-character path: "ip_adapter" (h94 IP-Adapter, the default) or
    # "instantid" (insightface face-embed projected through InstantID's image-projection + IP attn,
    # higher face fidelity). Multi-character shots ignore this and always take the masked-IP-Adapter
    # regional path.
    identity_method: str = "ip_adapter"
    instantid_ip_adapter_scale: float = 0.8  # InstantID face-embedding (image-proj) scale
    pose_conditioning: bool = True   # ControlNet-OpenPose to separate bodies in multi-char shots
    controlnet_pose_scale: float = 0.55  # OpenPose ControlNet conditioning scale (regional path)
    region_gutter: int = 64          # px dead band between regional masks so they do not seam-blend
    negative_prompt: str = DEFAULT_NEGATIVE
    max_slots: int = 2               # regional path is tuned for 2; more is future work


@dataclass
class KeyframeResult:
    """The outcome of rendering one shot's keyframe: where the PNG landed, which slots and engine
    path it used, and the prompt and seed that produced it."""
    shot_id: str
    path: Path
    slots: list[str]
    multi_char: bool
    prompt: str
    seed: int
    engine: str                      # "single" | "regional"


# ------------------------------------------------------------------------- prompt building

def slot_trigger(cast: Cast, slot: str) -> str:
    """The token that summons a slot's identity in the prompt: its trained LoRA trigger, which
    is the character's name (see lora_train.TrainedLora.trigger), falling back to the slot id."""
    char = cast.characters.get(slot)
    return (char.name if char and char.name else slot)


def build_prompt(scene: Scene, cast: Cast, storyboard: Storyboard) -> str:
    """Compose the SDXL prompt: the storyboard style prefix, the scene's own prompt, and the
    trigger token for each character in the shot. Pieces are comma-joined and de-duplicated of
    empty fragments so a missing style or prompt never leaves a dangling comma."""
    triggers = ", ".join(slot_trigger(cast, s) for s in scene.character_slots)
    parts = [storyboard.style_prefix, scene.prompt, triggers]
    if storyboard.style_preset and storyboard.style_preset != "None":
        parts.append(storyboard.style_preset)
    # Fragments may carry their own edge commas (the control plane's style_prefix ends in ","),
    # so strip leading/trailing commas per fragment before joining; internal commas (the trigger
    # list) are untouched, so two characters never collapse into one.
    cleaned = [c for c in (p.strip().strip(",").strip() for p in parts if p) if c]
    return ", ".join(cleaned)


# ------------------------------------------------------------------------- region geometry

def region_boxes(width: int, height: int, n: int, *, orientation: str = "vertical",
                 gutter: int = 0) -> list[tuple[int, int, int, int]]:
    """Split the canvas into `n` regions, one per character, as (left, top, right, bottom) boxes.
    Vertical orientation (side-by-side strips) is the default for the common two-shot; horizontal
    stacks them. `gutter` carves a dead band of that many pixels between adjacent regions (the
    interior edges are inset by gutter//2; the outer canvas edges stay flush) so the IP-Adapter
    masks do not abut and blend at the seam. Pure geometry, so the mask generation that consumes it
    is testable without an image library."""
    if n <= 1:
        return [(0, 0, width, height)]
    g = max(0, gutter) // 2
    boxes = []
    if orientation == "vertical":
        step = width // n
        for i in range(n):
            left = i * step + (g if i > 0 else 0)
            right = (width if i == n - 1 else (i + 1) * step) - (g if i < n - 1 else 0)
            boxes.append((left, 0, right, height))
    else:
        step = height // n
        for i in range(n):
            top = i * step + (g if i > 0 else 0)
            bottom = (height if i == n - 1 else (i + 1) * step) - (g if i < n - 1 else 0)
            boxes.append((0, top, width, bottom))
    return boxes


def engine_for(scene: Scene, params: KeyframeParams) -> str:
    """Which path this scene takes: 'regional' (the masked multi-identity, anti-bleed path) for a
    two-plus-character shot within the slot cap, else 'single'."""
    n = len(scene.character_slots)
    return "regional" if 2 <= n <= params.max_slots else "single"


# --------------------------------------------------------------------------- render (GPU)

def render_keyframe(
    scene: Scene,
    cast: Cast,
    storyboard: Storyboard,
    server,
    out_path: Path,
    *,
    params: KeyframeParams | None = None,
    lora_paths: dict[str, Path] | None = None,
    pose_image: "Path | None" = None,
) -> KeyframeResult:
    """Render one scene's keyframe to `out_path`.

    `server` is a `models.ModelServer` (provides the fp8 SDXL pipeline with the distill LoRA);
    `lora_paths` maps slot -> trained adapter file. Single-character shots take the ordinary
    identity path; two-character shots take the regional anti-bleed path. Heavy imports are
    deferred; the body is validated on a pod.
    """
    cfg = params or KeyframeParams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(scene, cast, storyboard)
    slots = list(scene.character_slots)
    engine = engine_for(scene, cfg)

    import torch  # deferred: keep this module CPU-importable

    loras = {s: Path(p) for s, p in (lora_paths or {}).items()}
    single_slot = scene.character_slots[0] if scene.character_slots else None

    # Single-character + identity_method "instantid" + a real reference face -> the InstantID pipe
    # (its own SDXL + face-keypoints ControlNet + face-embedding injection). Everything else (the
    # default IP-Adapter single path, and all multi-character shots) takes the shared keyframe pipe.
    if engine == "single" and cfg.identity_method == "instantid" and single_slot is not None \
            and _ref_images(cast, single_slot, count=1):
        image = _render_instantid(server, prompt, scene, cast, cfg, loras, single_slot)
    else:
        pipe = server.keyframe_pipeline()
        _apply_scheduler(pipe, cfg)
        generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        if engine == "single":
            image = _render_single(pipe, prompt, scene, cast, cfg, loras, generator)
        else:
            image = _render_regional(pipe, prompt, scene, cast, cfg, loras, generator, pose_image)

    image.save(out_path)
    return KeyframeResult(shot_id=scene.id or "shot", path=out_path, slots=slots,
                          multi_char=(engine == "regional"), prompt=prompt, seed=cfg.seed, engine=engine)


def _render_single(pipe, prompt, scene, cast, cfg, loras, generator):
    """One character (or none): the slot's LoRA bound, identity from a single IP-Adapter image.
    The plain SDXL identity path."""
    slot = scene.character_slots[0] if scene.character_slots else None
    _bind_loras(pipe, {slot: loras[slot]} if (slot and slot in loras) else {}, cfg.lora_scale,
                few_step=cfg.few_step)

    ip_images = _ref_images(cast, slot, count=1) if slot else None
    if ip_images:
        _ensure_ip_adapter(pipe, 1)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
    else:
        _ensure_ip_adapter(pipe, 0)  # clear any IP-Adapter a prior scene left on the shared pipe
    # The shared keyframe pipe is a ControlNet pipeline; a single subject wants no pose control, so
    # hand it a blank control image at conditioning_scale 0.0 -> the ControlNet is inert (zero
    # residual) and this path renders exactly like plain SDXL.
    return pipe(
        prompt=prompt, negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        height=cfg.height, width=cfg.width, generator=generator,
        image=_blank_control(cfg.width, cfg.height),
        controlnet_conditioning_scale=0.0,
        **({"ip_adapter_image": ip_images[0]} if ip_images else {}),
    ).images[0]


def _render_regional(pipe, prompt, scene, cast, cfg, loras, generator, pose_image):
    """Two characters, no bleed: each slot's identity is confined to its own region with a
    per-region IP-Adapter mask, each LoRA bound at the moderate per-slot scale, and (when on)
    OpenPose conditioning planting two bodies. The masks are what keep one face off the other."""
    from diffusers.image_processor import IPAdapterMaskProcessor

    slots = scene.character_slots[:cfg.max_slots]
    w, h = cfg.width, cfg.height
    boxes = region_boxes(w, h, len(slots), orientation="vertical", gutter=cfg.region_gutter)

    _bind_loras(pipe, {s: loras[s] for s in slots if s in loras}, cfg.lora_scale,
                few_step=cfg.few_step)

    # One IP-Adapter image per slot (its refs), each confined to its region's mask.
    _ensure_ip_adapter(pipe, n=len(slots))
    pipe.set_ip_adapter_scale([cfg.ip_adapter_scale] * len(slots))
    ip_images = [_ref_images(cast, s, count=1)[0] for s in slots]
    masks = IPAdapterMaskProcessor().preprocess(
        [_box_mask(w, h, b) for b in boxes], height=h, width=w)

    # Plant two distinct bodies: an OpenPose skeleton with one standing figure per region box, so
    # the masked IP-Adapter identities land on SEPARATE people instead of merging into one (the
    # "older Aria" collapse). An explicit pose_image overrides the generated skeleton. With
    # pose_conditioning off, a blank control image at scale 0.0 makes the ControlNet inert and the
    # path degrades to masked-IP-Adapter only.
    if cfg.pose_conditioning:
        if pose_image is not None:
            from PIL import Image
            control = Image.open(pose_image).convert("RGB").resize((w, h))
        else:
            control = _pose_skeleton(w, h, boxes)
        cn_scale = cfg.controlnet_pose_scale
    else:
        control = _blank_control(w, h)
        cn_scale = 0.0
    return pipe(
        prompt=prompt, negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        height=h, width=w, generator=generator,
        ip_adapter_image=ip_images,
        cross_attention_kwargs={"ip_adapter_masks": masks},
        image=control,
        controlnet_conditioning_scale=cn_scale,
    ).images[0]


def _render_instantid(server, prompt, scene, cast, cfg, loras, slot):
    """Single-character InstantID keyframe: the slot's reference face is embedded by insightface,
    projected to identity tokens, and injected into the UNet cross-attention (InstantID's face IP-
    Adapter) on a plain SDXL pipe. The slot's character LoRA still binds for body/style.

    Identity tokens reach the UNet through the IP attn processors' side channel (`proc.id_embeds`), so
    the prompt embeds stay clean. Uses the identity IP-Adapter only (no IdentityNet ControlNet -- see
    models.instantid_pipeline). GPU body, validated on a pod."""
    import torch
    from . import instantid as _iid

    pipe = server.instantid_pipeline()
    _apply_scheduler(pipe, cfg)
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
    _bind_loras(pipe, {slot: loras[slot]} if slot in loras else {}, cfg.lora_scale,
                few_step=cfg.few_step)

    ref = _ref_images(cast, slot, count=1)[0]
    analyzed = _iid.analyze_face(server.face_analyzer(), ref)
    if analyzed is None:  # no face detected: fall back to the shared keyframe pipe's IP-Adapter path
        kpipe = server.keyframe_pipeline()
        _apply_scheduler(kpipe, cfg)
        return _render_single(kpipe, prompt, scene, cast, cfg, loras, generator)
    embedding = analyzed[0]
    id_tokens = _iid.faceid_tokens(pipe._vj_image_proj, embedding)  # (1, num_tokens, 2048)

    # Feed identity through the IP attn SIDE channel. CFG batch is [uncond; cond]: zero identity for
    # the uncond pass so guidance does not invent a face, the real tokens for cond.
    id_embeds = torch.cat([torch.zeros_like(id_tokens), id_tokens], dim=0)
    for proc in pipe._vj_id_attn.values():
        proc.scale = cfg.instantid_ip_adapter_scale
        proc.id_embeds = id_embeds
    try:
        return pipe(
            prompt=prompt, negative_prompt=cfg.negative_prompt,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
            height=cfg.height, width=cfg.width, generator=generator,
        ).images[0]
    finally:
        for proc in pipe._vj_id_attn.values():
            proc.id_embeds = None  # clear so a later render on the warm pipe never reuses them


# --------------------------------------------------------------------------- internals (GPU)

def _apply_scheduler(pipe, cfg: KeyframeParams) -> None:
    """Pin the SDXL sampler for this render on the warm shared pipe. The Hyper-SD few-step LoRA
    needs DDIM with timestep_spacing="trailing" (cfg=0); the full-step final tier wants a higher-
    order solver. Every scheduler is rebuilt from the pristine full-step config stashed by the
    ModelServer (`_vj_default_scheduler`), so repeated draft->final->draft swaps on the warm pipe
    never compound. Unrecognized values fall back to that pristine scheduler. GPU path: diffusers is
    imported lazily so this module stays CPU-importable; the mapping is validated on a pod."""
    from diffusers import (
        DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
    )
    base = getattr(pipe, "_vj_default_scheduler", None) or pipe.scheduler
    base_cfg = base.config
    sched = getattr(cfg, "scheduler", "ddim_trailing")
    if sched == "ddim_trailing":
        pipe.scheduler = DDIMScheduler.from_config(base_cfg, timestep_spacing="trailing")
    elif sched == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(base_cfg)
    elif sched == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_cfg, use_karras_sigmas=True)
    elif sched == "dpmpp_2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_cfg)
    elif sched == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(base_cfg)
    elif sched == "euler_ancestral":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(base_cfg)
    elif sched == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(base_cfg)
    else:
        pipe.scheduler = base  # tcd is handled by the unified-LoRA path (not wired here); restore base


def _normalize_lora_state_dict(sd: dict) -> dict:
    """Convert a raw-PEFT LoRA state dict to the diffusers SDXL convention.

    Raw-PEFT format (save_file(get_peft_model_state_dict(unet)) without convert_state_dict_to_diffusers):
    keys have a `base_model.model.` prefix (or none), no `unet.` scope, and use `lora_A`/`lora_B`
    weight names. Diffusers format: `unet.<layer>.lora.down.weight` / `unet.<layer>.lora.up.weight`.

    Returns sd unchanged when it is already in diffusers format (no `.lora_A.` keys present).
    Pure key remapping -- no tensors allocated, CPU-safe.
    """
    if not any(".lora_A." in k or ".lora_B." in k for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        k = f"unet.{k}"
        k = k.replace(".lora_A.", ".lora.down.")
        k = k.replace(".lora_B.", ".lora.up.")
        out[k] = v
    return out


def _bind_loras(pipe, slot_paths: dict, scale: float, *, few_step: bool = True) -> list[str]:
    """Load each slot's character LoRA and activate it alongside whatever base adapter is already
    on the pipe (a few-step distill LoRA, if the ModelServer loaded one). Character LoRAs go on at
    the moderate per-slot `scale` (the anti-bleed weight); base adapters stay at 1.0, EXCEPT the
    distill adapter, whose weight is gated by `few_step`: 1.0 on the draft/standard few-step path,
    0.0 on the full-step (final) path, so the one warm pipe renders every tier without reloading.
    Discovering the base adapter instead of hardcoding a name means this is correct whether or not a
    distill LoRA is present, and a missing one never silently drops the character identity.

    The keyframe pipeline is a process-global shared across every scene on a warm worker, so the
    previous scene's character adapters are still attached on entry. Record the true base adapters
    once (whatever is present before any character is bound), then drop the prior scene's character
    adapters before rebinding; otherwise peft rejects the duplicate name ("Adapter name A already
    in use"). Clearing them also stops a no-character scene inheriting the last scene's identity."""
    if not hasattr(pipe, "_vj_base_adapters"):
        pipe._vj_base_adapters = _adapter_names(pipe)
    base = pipe._vj_base_adapters  # persistent adapters (e.g. a distill LoRA); never a character
    stale = [n for n in _adapter_names(pipe) if n not in base]
    if stale:
        pipe.delete_adapters(stale)
    loaded = []
    for slot, path in slot_paths.items():
        p = Path(path)
        if p.is_file():
            # On the GPU path the LoRA always exists on disk; load it so we can detect and convert
            # raw-PEFT format (cast-builder writes via save_file(get_peft_model_state_dict()) without
            # convert_state_dict_to_diffusers). CPU unit-test stubs use fictional paths that are
            # never on disk, so they take the else branch and the fake pipe handles them as before.
            from safetensors.torch import load_file as _sf_load
            state_dict = _normalize_lora_state_dict(_sf_load(str(p)))
            pipe.load_lora_weights(state_dict, adapter_name=slot)
        else:
            pipe.load_lora_weights(str(path), adapter_name=slot)
        # diffusers silently loads ZERO modules when the safetensors keys do not match the pipe's
        # convention. Left unchecked the slot never registers and set_adapters below explodes with
        # an opaque "not in the list of present adapters: set()". The normalization above should
        # handle all known raw-PEFT variants; this guard catches anything genuinely unrecognized.
        if slot not in _adapter_names(pipe):
            raise ValueError(
                f"LoRA for slot {slot!r} ({path}) registered no adapter: its safetensors keys did "
                f"not match the diffusers convention after normalization. Refusing to render the "
                f"character without its identity adapter.")
        loaded.append(slot)
    # The distill base adapter rides at 1.0 on the few-step path and 0.0 (inert) on the full-step
    # path; any other base adapter always stays at 1.0. Set the active set whenever ANYTHING is on
    # the pipe (base and/or characters), so the distill weight tracks the tier even on a scene with
    # no character. A pipe with no base and no character (the bare no-identity scene) sets nothing.
    base_weights = [(1.0 if few_step else 0.0) if n == DISTILL_ADAPTER else 1.0 for n in base]
    names = base + loaded
    if names:
        pipe.set_adapters(names, adapter_weights=base_weights + [scale] * len(loaded))
    return loaded


def _adapter_names(pipe) -> list[str]:
    """Names of the LoRA adapters currently loaded on the pipeline, de-duplicated across its
    components. Empty when none are loaded."""
    try:
        listed = pipe.get_list_adapters()  # {component: [adapter_name, ...]}
    except Exception:
        return []
    seen: list[str] = []
    for names in listed.values():
        for n in names:
            if n not in seen:
                seen.append(n)
    return seen


def _ensure_ip_adapter(pipe, n: int = 1):
    """Bring the shared keyframe pipe to EXACTLY n IP-Adapter encoders (n=0 clears it).

    The pipe is process-global across scenes, so a prior regional scene can leave 2 encoders
    attached while a later single-character scene needs 1. set_ip_adapter_scale and the call-time
    ip_adapter_image must both match the loaded count, or diffusers raises ("Cannot assign 1
    scale_configs to 2 IP-Adapter"). So reload whenever the count differs in either direction (not
    only when fewer are present, the old `>= n` bug), and unload entirely for a no-character scene."""
    if getattr(pipe, "_vj_ip_loaded", 0) == n:
        return
    if getattr(pipe, "_vj_ip_loaded", 0):
        pipe.unload_ip_adapter()
        pipe._vj_ip_loaded = 0
    if n:
        from .models import DEFAULT_SPECS, ModelRole
        repo = DEFAULT_SPECS[ModelRole.IP_ADAPTER].repo_id
        pipe.load_ip_adapter(repo, subfolder="sdxl_models",
                             weight_name=["ip-adapter_sdxl.safetensors"] * n if n > 1 else "ip-adapter_sdxl.safetensors")
        pipe._vj_ip_loaded = n


def _ref_images(cast, slot, count: int):
    """The first `count` reference images for a slot, as PIL images (the IP-Adapter identity)."""
    from PIL import Image
    char = cast.characters.get(slot)
    if not char or not char.ref_paths:
        return []
    return [Image.open(p).convert("RGB") for p in char.ref_paths[:count]]


def _box_mask(width, height, box):
    """A white-in-box, black-elsewhere mask image for one region (the IP-Adapter attends only
    where the mask is white)."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def _blank_control(width, height):
    """Black control image. A ControlNet at conditioning_scale 0 ignores it, so the single /
    no-pose paths keep the shared ControlNet pipe behaving like plain SDXL."""
    from PIL import Image
    return Image.new("RGB", (width, height), (0, 0, 0))


# COCO-18 OpenPose layout (the format the xinsir SDXL OpenPose ControlNet follows). Keypoint order
# 0..17: nose, neck, R-shoulder, R-elbow, R-wrist, L-shoulder, L-elbow, L-wrist, R-hip, R-knee,
# R-ankle, L-hip, L-knee, L-ankle, R-eye, L-eye, R-ear, L-ear. `_POSE_KP` is one upright,
# front-facing standing figure as (x, y) normalized within a region box.
_POSE_KP = [
    (0.50, 0.10), (0.50, 0.20), (0.40, 0.22), (0.34, 0.38), (0.32, 0.54),
    (0.60, 0.22), (0.66, 0.38), (0.68, 0.54), (0.44, 0.54), (0.43, 0.74),
    (0.43, 0.94), (0.56, 0.54), (0.57, 0.74), (0.57, 0.94), (0.47, 0.085),
    (0.53, 0.085), (0.44, 0.095), (0.56, 0.095),
]
_POSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
               (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_POSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
                (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
                (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
                (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def _pose_skeleton(width, height, boxes):
    """Render an OpenPose skeleton with ONE standing COCO-18 figure per region box on black, so the
    ControlNet plants exactly that many distinct bodies (the fix for two characters merging into
    one). Each figure scales into its box with a small vertical margin. Geometry is deterministic;
    visual fidelity to the ControlNet's training distribution is validated on the pod."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (l, t, r, b) in boxes:
        bw, bh = r - l, b - t
        my = int(bh * 0.04)  # top/bottom margin so head + ankles are not clipped at the box edge
        pts = [(l + x * bw, t + my + y * (bh - 2 * my)) for (x, y) in _POSE_KP]
        lw = max(2, bw // 70)
        for i, (a, c) in enumerate(_POSE_LIMBS):
            draw.line([pts[a], pts[c]], fill=_POSE_COLORS[i % len(_POSE_COLORS)], width=lw)
        rad = max(2, bw // 90)
        for i, (px, py) in enumerate(pts):
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=_POSE_COLORS[i % len(_POSE_COLORS)])
    return img
