"""Image-to-video: animate each keyframe into its shot's clip.

The keyframe is the still; this turns it into motion. Wan 2.2 image-to-video takes the keyframe
as the first frame and the scene prompt as the motion description and produces N frames at a
target fps. This is the long pole of the whole render, so the speed knobs are the point: the
draft and standard tiers run a few-step distilled path (the Wan2.2-Lightning LoRA, ~4 steps) for
the big throughput win, while the final tier runs full steps for the hero clip. The planner
already decided which shots animate and on which GPU tier; this just executes one shot.

Clean-room: built from diffusers' WanImageToVideoPipeline + export_to_video, the Wan2.2-Lightning
distill card, and the LightX2V fallback loader (diffusers LoRA-load issue #12535), not any prior
pipeline. The frame-count / duration math and the tier->steps decision are pure and CPU-tested;
the generation body defers torch/diffusers and is validated on a pod. Engine knobs live in
`I2VParams`; the control plane's typed `I2VConfig` (separate work) maps into them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import FeatureCache
from .contract import Scene
from .routing import QualityTier

# Wan's temporal VAE compresses time by 4, so a clip's frame count must be 4k+1 (e.g. 81 frames
# = 4*20+1, ~5s at 16 fps). FPS and the frame ceiling follow the A14B i2v defaults.
TEMPORAL_STRIDE = 4
DEFAULT_FPS = 16
MAX_FRAMES = 81  # ~5s at 16 fps, the model's comfortable clip length


@dataclass
class I2VParams:
    """Engine knobs for one shot's animation. Defaults are the few-step distilled path (the
    throughput win); the final tier flips `distill` off for full steps. The control plane's
    I2VConfig fills these per job."""
    num_frames: int = MAX_FRAMES
    fps: int = DEFAULT_FPS
    steps: int = 4                   # 4-step Wan2.2-Lightning distill
    guidance_scale: float = 1.0      # distilled sampling runs (near-)guidance-free
    distill: bool = True
    seed: int = 0
    height: int | None = None        # default: follow the keyframe's size
    width: int | None = None
    negative_prompt: str = "static, still, frozen, jpeg artifacts, blurry, watermark"
    feature_cache: FeatureCache = FeatureCache.NONE  # final=MIXCACHE, standard=EASYCACHE, draft=NONE
    flow_shift: float = 5.0           # FlowMatch scheduler shift; Wan2.2 default; lower=faster motion


@dataclass
class I2VResult:
    """The outcome of animating one keyframe: where the clip landed, its frame count / fps /
    length, and whether the few-step distill path produced it."""
    shot_id: str
    path: Path
    num_frames: int
    fps: int
    seconds: float
    distilled: bool


# --------------------------------------------------------------------------- pure helpers

def snap_frames(n: int, max_frames: int = 256) -> int:
    """Snap a frame count to the nearest valid 4k+1 the temporal VAE accepts (rounding up so a
    clip never comes out shorter than asked), clamped to [1, max_frames].

    Rounding up from 256 would yield 257, which is valid temporally but exceeds the documented
    Wan2.2 ceiling. snap-then-clamp to max_frames (defaulting to 256) rather than clamp-then-snap
    so the result is always 4k+1 even after the ceiling is applied: if n=256 would round up to 257,
    we step down to the previous valid value (253 = 4*63+1)."""
    n = max(1, int(n))
    snapped = n if (n - 1) % TEMPORAL_STRIDE == 0 else n + (TEMPORAL_STRIDE - (n - 1) % TEMPORAL_STRIDE)
    if snapped <= max_frames:
        return snapped
    # snapped exceeded ceiling: step back to the largest 4k+1 <= max_frames
    prev = max_frames - (max_frames - 1) % TEMPORAL_STRIDE
    return max(1, prev)


def frames_for(target_seconds: float | None, fps: int = DEFAULT_FPS, *, max_frames: int = MAX_FRAMES) -> int:
    """Frame count for a target duration at `fps`: snap to 4k+1 and cap at the model ceiling.
    Falls back to the ceiling when the scene gives no target."""
    if not target_seconds or target_seconds <= 0:
        return max_frames
    return min(max_frames, snap_frames(round(target_seconds * fps)))


def clip_seconds(num_frames: int, fps: int = DEFAULT_FPS) -> float:
    """The realized clip length. i2v fixes the first frame to the keyframe, so N frames play as
    N/fps seconds."""
    return round(num_frames / fps, 3)


def params_for(scene: Scene, quality: QualityTier, *, base: I2VParams | None = None) -> I2VParams:
    """Resolve the per-shot params: frame count from the scene's target duration, and the
    step/guidance/distill profile from the quality tier (draft/standard distilled for throughput,
    final full-step for the hero clip)."""
    p = base or I2VParams()
    p.num_frames = frames_for(scene.target_seconds, p.fps)
    if quality is QualityTier.FINAL:
        p.distill, p.steps, p.guidance_scale = False, 40, 5.0
    else:  # draft / standard: the few-step distilled path
        p.distill, p.steps, p.guidance_scale = True, 4, 1.0
    return p


# --------------------------------------------------------------------------- animate (GPU)

def animate(
    scene: Scene,
    keyframe: Path,
    prompt: str,
    server,
    out_path: Path,
    *,
    params: I2VParams | None = None,
    progress_cb=None,
) -> I2VResult:
    """Animate `keyframe` into a clip at `out_path` for one scene.

    `server` is a `models.ModelServer` (provides the Wan i2v pipeline with the Lightning distill
    LoRA). The keyframe is the first frame; `prompt` describes the motion. Heavy imports are
    deferred; the body is validated on a pod.

    `progress_cb(step, total)`, when given, is called once per denoise step (i2v is the long pole;
    at final tier each step is ~30s, so the live `step/total` is what distinguishes a slow shot
    from a hung one). It is wired through diffusers' `callback_on_step_end` hook, best-effort: a
    progress failure never breaks the render.
    """
    cfg = params or I2VParams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: keep this module CPU-importable
    from diffusers.utils import export_to_video, load_image

    image = load_image(str(keyframe))
    height = cfg.height or image.height
    width = cfg.width or image.width

    pipe = server.i2v_pipeline()
    _set_distill(pipe, cfg.distill)
    _apply_flow_shift(pipe, cfg.flow_shift)
    _set_feature_cache(pipe, cfg.feature_cache)  # per-shot: reset + (re)install, never leak across shots
    step_callback = _step_callback(progress_cb, cfg.steps)

    def _run_pipe():
        return pipe(
            image=image, prompt=prompt, negative_prompt=cfg.negative_prompt,
            height=height, width=width, num_frames=cfg.num_frames,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(cfg.seed),
            **({"callback_on_step_end": step_callback} if step_callback else {}),
        ).frames[0]

    try:
        frames = _run_pipe()
    except ValueError as _exc:
        if "No context is set" not in str(_exc):
            raise
        # FBC context not set up in this diffusers build; clear the hook and run uncached
        print(f"i2v FBC context error on {scene.id}; retrying without feature cache", flush=True)
        _set_feature_cache(pipe, FeatureCache.NONE)
        frames = _run_pipe()

    export_to_video(frames, str(out_path), fps=cfg.fps)
    return I2VResult(
        shot_id=scene.id or "shot", path=out_path, num_frames=cfg.num_frames,
        fps=cfg.fps, seconds=clip_seconds(cfg.num_frames, cfg.fps), distilled=cfg.distill,
    )


def _step_callback(progress_cb, total: int):
    """Wrap a `(step, total)` progress callback in diffusers' `callback_on_step_end` signature
    `(pipe, step_index, timestep, callback_kwargs) -> dict`. Returns None when there is no callback,
    so the pipe call omits the kwarg entirely (zero overhead). The callback is best-effort: a
    progress failure is swallowed and never breaks the denoise, and `callback_kwargs` is returned
    unchanged so diffusers' loop is unaffected."""
    if progress_cb is None:
        return None

    def on_step_end(pipe, step_index, timestep, callback_kwargs):
        try:
            progress_cb(step_index + 1, total)  # step_index is 0-based; report 1..total
        except Exception:
            pass
        return callback_kwargs

    return on_step_end


# FirstBlockCache thresholds per cache mode. FBCache (a TeaCache successor) skips the expensive
# later DiT blocks when the first block's step-to-step output delta is below the threshold, reusing
# the prior residual; a higher threshold skips more steps (faster, slightly lower fidelity). These
# are starting points to tune on the pod for the speed/quality knee. The enum names are ours; the
# mechanism is diffusers FirstBlockCache.
_FBCACHE_THRESHOLD = {
    FeatureCache.MIXCACHE: 0.20,   # final tier: aim ~1.5-2x
    FeatureCache.EASYCACHE: 0.10,  # standard tier: more conservative
}


def _feature_cache_targets(pipe):
    """The DiT(s) to cache. Wan 2.2 A14B is a Mixture-of-Experts: a high-noise expert
    (`transformer`) runs the early denoise steps and a low-noise expert (`transformer_2`) the rest,
    swapping at the boundary ratio mid-denoise. Caching only `transformer` leaves the back ~70% of
    steps (the low-noise expert) uncached -- the step-~12 hit-cliff we measured (only ~1.2x instead
    of ~1.8x). So return every expert the pipe exposes. getattr-safe: a single-DiT pipe (or a future
    attribute rename) just yields whatever is present, never crashes."""
    targets = []
    for attr in ("transformer", "transformer_2"):
        t = getattr(pipe, attr, None)
        if t is not None:
            targets.append(t)
    return targets


def _set_feature_cache(pipe, feature_cache) -> None:
    """Install (or clear) the denoise feature cache on the Wan DiT(s) for this shot.

    Wan 2.2 A14B is a two-expert MoE, so the cache is installed on BOTH experts (`transformer`
    high-noise + `transformer_2` low-noise); caching only the first leaves the later steps -- the
    bulk of the denoise -- uncached (see `_feature_cache_targets`).

    The i2v pipe is process-global and reused across every shot, and each expert's cache holds
    per-timestep state, so it MUST reset each call or it leaks across shots -- the per-scene-state
    bug class that bit keyframes in v0.1.4/v0.1.5. Via the MATCHED
    `CacheMixin.enable_cache(FirstBlockCacheConfig)` / `disable_cache()` pair (guarded by
    `is_cache_enabled`), so the reset is real and silent. (NOT the standalone
    `apply_first_block_cache`, whose hooks `disable_cache()` does not clear -> re-apply raises
    "hook already exists" and the shot runs uncached.) FasterCache / PyramidAttentionBroadcast are
    NOT wired for WanTransformer3DModel (diffusers #11134), so FBCache is the path. NONE bypasses.
    Best-effort PER EXPERT like `_set_distill`: an expert that cannot cache runs full rather than
    failing the render."""
    targets = _feature_cache_targets(pipe)
    if not targets:
        return

    # Reset: clear every expert's prior-shot cache (matched pair), silent when none is on.
    for t in targets:
        try:
            if getattr(t, "is_cache_enabled", False):
                t.disable_cache()
        except Exception:
            pass

    if feature_cache is FeatureCache.NONE:
        return

    try:
        from diffusers.hooks import FirstBlockCacheConfig
    except Exception as e:  # noqa: BLE001
        print(f"i2v feature cache {getattr(feature_cache, 'value', feature_cache)} unavailable "
              f"({e}); running full uncached.", flush=True)
        return

    threshold = _FBCACHE_THRESHOLD.get(feature_cache, 0.20)
    for t in targets:
        try:
            t.enable_cache(FirstBlockCacheConfig(threshold=threshold))
        except Exception as e:  # noqa: BLE001
            print(f"i2v feature cache {getattr(feature_cache, 'value', feature_cache)} not applied "
                  f"to {type(t).__name__} ({e}); that expert runs uncached.", flush=True)



def _apply_flow_shift(pipe, flow_shift: float) -> None:
    """Apply the FlowMatch scheduler shift for this shot on the warm shared pipe.

    Wan2.2 defaults to shift=5.0 at load time; lower values produce faster motion, higher
    values slow it. The shift is reset per shot so a warm pipe never carries a prior shot's
    value. Best-effort: a scheduler that does not expose `shift` (or where the rebuild fails)
    runs with whatever shift it was initialized with rather than failing the render.

    Pod-validate: confirm FlowMatchEulerDiscreteScheduler.from_config(sched.config, shift=x)
    accepts the Wan2.2 scheduler config without error."""
    try:
        sched = pipe.scheduler
        current = getattr(sched, "shift", None)
        if current is None:
            print(f"i2v: scheduler {type(sched).__name__} has no `shift` attr; "
                  f"flow_shift={flow_shift} skipped (scheduler default applies).", flush=True)
            return
        if abs(current - flow_shift) < 1e-6:
            return  # already at the right shift; skip the rebuild
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched.config, shift=flow_shift)
        print(f"i2v: flow_shift set to {flow_shift} (was {current:.4f})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"i2v: flow_shift {flow_shift} not applied ({e!r}); using scheduler default.", flush=True)


def _set_distill(pipe, distill: bool) -> None:
    """Gate the Wan2.2-Lightning distill LoRA for this shot.

    ModelServer.i2v_pipeline() records `_vj_i2v_distill_loaded` (True/False) and
    `_vj_i2v_distill_fused` (True when baked before fp8 quant). Raises when the job
    requests the distilled path but no LoRA loaded -- 4-step-no-distill ships garbage
    motion and must never silently reach the user.

    On the fused path, weights are permanently baked in and toggling back to full-step
    is not possible. distill=False on a fused pipe is noted and proceeds -- the model
    renders with the distilled weights at the caller's step count, which is suboptimal
    but not garbage. On the unfused adapter path (LightX2V future), set_adapters toggles."""
    fused = getattr(pipe, "_vj_i2v_distill_fused", False)
    loaded = getattr(pipe, "_vj_i2v_distill_loaded", None)

    if loaded is False:
        # ModelServer confirmed no distill LoRA -- the pipe runs full-step regardless.
        if distill:
            raise RuntimeError(
                "i2v: 4-step distilled render requested but no Wan2.2-Lightning LoRA loaded "
                "(both diffusers and LightX2V loaders failed). Refusing to ship 4-step-no-distill "
                "output. Set VJ_I2V_DISTILL=0 in the pod env to force full-step rendering.")
        return  # distill=False + no LoRA: already running full-step, nothing to do

    if fused:
        return  # baked-in distill; can't toggle; caller controls step count

    # Unfused adapter path (LightX2V loader, or old pipe without state tags)
    try:
        if distill:
            pipe.set_adapters(["distill"], adapter_weights=[1.0])
        else:
            pipe.set_adapters(["distill"], adapter_weights=[0.0])
    except Exception as e:
        if distill and loaded is True:
            # Unfused adapter that claimed it loaded but can't be activated
            raise RuntimeError(
                f"i2v: distill adapter toggle failed with distill=True ({e!r}); "
                "refusing 4-step-no-distill") from e
        # distill=False toggle failed: running full-step -- acceptable
