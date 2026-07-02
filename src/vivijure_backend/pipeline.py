"""GpuPipeline: the convergence. Wire the five engines into one render.

The harness decides the I/O and the order (bundle in -> plan -> `pipeline.execute` -> finish ->
out); the orchestrator decides, on the CPU, exactly what work survives (which LoRAs to train,
which keyframes to draw, which shots to animate). This module is the glue between that plan and
the GPU engines: it trains the kept LoRAs, draws the GENERATE keyframes with them, animates the
needs_i2v shots, and reuses everything the plan said to reuse, all on ONE shared `ModelServer`
so models load once per worker.

The typed `RenderConfig` drives the engines through two pure mappers (`keyframe_params_from`,
`i2v_params_from`): the control plane's config in, the engines' `KeyframeParams` / `I2VParams`
out. The three GPU stages sit behind small overridable methods so the orchestration is testable
on a CPU box with the engines stubbed (the same fake-stage pattern `tests/test_harness.py` uses
for the `Pipeline` protocol).

Clean-room: built only from our own modules (config / orchestrator / keyframe / i2v / lora_train
/ harness) and their documented signatures; no fork.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import i2v as _i2v
from . import keyframe as _keyframe
from . import lora_train as _lora_train
from .config import RenderConfig
from .device import current as _current_device
from .harness.handler import HarnessError, Outputs
from .harness.progress import NullEmitter
from .keyframe import KeyframeParams, build_prompt
from .i2v import I2VParams, frames_for, snap_frames
from .orchestrator import KeyframeMode, RenderPlan, kf_hash
from .contract import Bundle

# Shared no-op emitter for a pipeline run with no progress channel wired (the test default).
_NULL_PROGRESS = NullEmitter()


# --------------------------------------------------------------- config -> engine params

def keyframe_params_from(config: RenderConfig) -> KeyframeParams:
    """Map the typed `KeyframeConfig` onto the keyframe engine's `KeyframeParams`. `distill`
    selects the few-step path (its `distill_steps`) and `few_step`; the per-slot anti-bleed
    scales come from the multi_char block (the engine uses one IP-Adapter / LoRA scale for both
    the single and the masked-regional path, which is what KeyframeParams models)."""
    kc = config.keyframe
    mc = kc.multi_char
    return KeyframeParams(
        steps=kc.distill_steps if kc.distill else kc.steps,
        guidance_scale=kc.guidance_scale,
        width=kc.width,                            # both dims flow through; non-square is honored
        height=kc.height,                          # (16:9 / vertical), not collapsed to a square
        seed=kc.seed,
        few_step=kc.distill,
        scheduler=kc.scheduler.value,              # ddim_trailing on the few-step path, a solver on final
        lora_scale=mc.lora_scale_per_slot,
        ip_adapter_scale=kc.ip_adapter_scale,      # single-char field; multi-char path uses mc.ip_adapter_scale_per_slot via region override
        identity_method=kc.identity_method.value,  # single-char path: ip_adapter (default) or instantid
        instantid_ip_adapter_scale=kc.instantid_ip_adapter_scale,
        pose_conditioning=mc.pose_conditioning,
        controlnet_pose_scale=mc.controlnet_pose_scale,
        region_gutter=mc.region_gutter,
        max_slots=mc.max_slots,
    )


def i2v_params_from(config: RenderConfig, scene) -> I2VParams:
    """Map the typed `I2VConfig` (+ the scene's duration) onto the i2v engine's `I2VParams`.

    Frame count derives from the scene's target seconds via `I2VConfig.frames_for`, which
    respects the configured `seconds_per_shot` fallback and the 1..256 ceiling (the engine
    snaps to 4k+1 from there). `flow_shift` is threaded to the scheduler per shot. `seed` is
    the i2v-specific seed so motion can be re-rolled without re-rolling the keyframe. `distill`
    selects the 4-step Lightning path; `feature_cache` carries the tier's accelerator."""
    ic = config.i2v
    target = scene.target_seconds if (scene.target_seconds and scene.target_seconds > 0)         else ic.seconds_per_shot
    p = I2VParams(
        num_frames=snap_frames(ic.frames_for(target)),
        fps=ic.fps,
        steps=ic.distill_steps if ic.distill else ic.steps,
        guidance_scale=ic.guidance_scale,
        distill=ic.distill,
        seed=ic.seed,
        flow_shift=ic.flow_shift,
        feature_cache=ic.feature_cache,
    )
    if ic.negative_prompt:
        p.negative_prompt = ic.negative_prompt + ", " + I2VParams.negative_prompt
    return p


def finish_params_from(config: RenderConfig):
    """Map the typed `FinishConfig` onto the finish engine's `FinishParams`. The enum-shaped config
    (`interpolate` flag + `face_restore` as a `FaceRestore` member) becomes the engine's bool + the
    chosen restorer backend string: `FaceRestore.NONE` -> face_restore off, GFPGAN/CodeFormer ->
    on with that backend. The interpolation factor is already snapped to a power of two at config
    validation, so it is passed straight through. One resolved params object finishes every clip in
    the render the same way, so all clips stay codec/fps-uniform for the stream-copy concat."""
    from .finish import FinishParams  # deferred: keep finish CPU-light and avoid an import cycle
    from .config import FaceRestore

    fc = config.finish
    restore_on = fc.face_restore is not FaceRestore.NONE
    return FinishParams(
        interpolate=fc.interpolate,
        factor=fc.interpolation_factor,
        target_fps=fc.target_fps,
        face_restore=restore_on,
        face_restore_backend=(fc.face_restore.value if restore_on else FaceRestore.GFPGAN.value),
        face_fidelity=fc.face_fidelity,
        only_faces=fc.only_faces,
    )


# --------------------------------------------------------------------------- the pipeline

@dataclass
class GpuPipeline:
    """The deployed GPU `Pipeline`. Built from a job's `RenderConfig`; holds the shared
    `ModelServer` so a warm worker loads each model once. `pretrained_loras` (slot -> reference)
    lets a prior adapter feed keyframing when it is staged locally."""
    config: RenderConfig
    pretrained_loras: dict[str, str] = field(default_factory=dict)
    server: Any = None  # models.ModelServer; created lazily on first GPU use

    def _model_server(self):
        if self.server is None:
            from .models import ModelServer  # deferred: keep this module CPU-importable
            self.server = ModelServer()
        return self.server

    def set_progress(self, emitter) -> None:
        """Wire a progress emitter (the harness calls this per job). Default is a no-op emitter, so
        the pipeline runs unchanged without one."""
        self._progress = emitter

    @property
    def progress(self):
        return getattr(self, "_progress", None) or _NULL_PROGRESS

    def set_pretrained_loras(self, mapping: dict[str, str]) -> None:
        """Replace the reused-LoRA refs with the harness's local-path map (it stages them from R2
        before execute). The pipeline never touches R2 itself; it just loads the local files the
        `if p.is_file()` check in `execute` already understands."""
        self.pretrained_loras = dict(mapping)

    # --- GPU stages, behind overridable methods (stubbed in CPU tests) ---

    def _train_slot(self, char, out_dir: Path) -> Path:
        # Throttled per-step training progress (the long pole); lora_train calls the cb every N steps.
        cb = self.progress.train_step_cb(char.slot)
        result = _lora_train.train_slot(char, out_dir, config=self.config.lora, progress_cb=cb)
        # result.checkpoint_dirs contains any save_every intermediate adapters; they live inside
        # out_dir (which is inside the workdir), so the harness's workdir teardown cleans them up.
        return result.path

    def _render_keyframe(self, scene, cast, storyboard, out_path: Path, lora_paths: dict[str, Path]) -> Path:
        return _keyframe.render_keyframe(
            scene, cast, storyboard, self._model_server(), out_path,
            params=keyframe_params_from(self.config), lora_paths=lora_paths,
        ).path

    def _animate(self, scene, keyframe_path: Path, prompt: str, out_path: Path) -> Path:
        # Per-step i2v progress (every step; i2v is 4-40 steps, ~30s/step at final tier).
        cb = self.progress.i2v_step_cb(scene.id)
        return _i2v.animate(
            scene, keyframe_path, prompt, self._model_server(), out_path,
            params=i2v_params_from(self.config, scene), progress_cb=cb,
        ).path

    def _finish_clip(self, shot_id: str, in_path: Path, out_path: Path) -> Path:
        # Finishing stage (RIFE interpolation + face restore), clip in / clip out, on the warm
        # ModelServer so the finish models load once. Per-clip finish progress is best-effort.
        from . import finish as _finish  # deferred: finish defers torch + the finish models

        cb = self.progress.finish_cb(shot_id)
        return _finish.finish_clip(
            shot_id, in_path, out_path, self._model_server(),
            params=finish_params_from(self.config), progress_cb=cb,
        ).path

    # --- orchestration (CPU; the stages above are the only GPU touch points) ---

    def execute(self, plan: RenderPlan, bundle: Bundle, workdir: Path) -> Outputs:
        """Run the plan on the GPU over the shared `ModelServer`: train the kept LoRAs, then per
        shot draw / reuse / inject the keyframe, animate it, and finish the clip. Returns the
        `Outputs` (LoRA paths, keyframes, clips) the harness uploads; assembly happens off-GPU in
        the harness, not here."""
        out = Outputs()
        workdir = Path(workdir)
        cast, storyboard = bundle.cast, bundle.storyboard
        scenes_by_id = {s.id: s for s in storyboard.scenes}

        # Warn once if the running card doesn't match the tier the planner targeted for i2v.
        # A mismatch means the job landed on the wrong pool (cheaper card for a final-tier job,
        # or an expensive card burning money on a draft). Does not block the render.
        planned_i2v_tiers = {sp.i2v_tier for sp in plan.scenes if sp.i2v_tier is not None}
        if planned_i2v_tiers:
            actual_tier = _current_device().tier
            if actual_tier not in planned_i2v_tiers:
                self.progress.emit("tier_mismatch",
                                   actual=actual_tier.value,
                                   planned=[t.value for t in planned_i2v_tiers])

        # 1) Train the LoRAs the plan kept; collect adapter paths for keyframing.
        lora_paths: dict[str, Path] = {}
        for slot in plan.lora.train:
            char = cast.characters.get(slot)
            if char is None:
                raise HarnessError(
                    f"plan requires LoRA training for slot {slot!r} but the cast has no "
                    "such character; validate(cast=bundle.cast) should have caught this")
            path = self._train_slot(char, workdir / "loras" / slot)
            out.loras[slot] = path
            lora_paths[slot] = path
            self.progress.emit("train_done", slot=slot, path=str(path))
        # Reused / pretrained adapters feed keyframing too, when staged on disk locally (the
        # adapter is portable .safetensors; an R2-key reference that is not a local file is left
        # to the deploy to stage, and the shot falls back to IP-Adapter identity if absent).
        for slot, ref in self.pretrained_loras.items():
            p = Path(ref)
            if p.is_file():
                lora_paths.setdefault(slot, p)

        # 2) Per scene: draw the keyframe (or resolve a reused/injected one), then animate.
        for sp in plan.scenes:
            scene = scenes_by_id.get(sp.shot_id)
            if scene is None:
                continue
            if sp.keyframe_mode is KeyframeMode.GENERATE:
                kf_path = self._render_keyframe(
                    scene, cast, storyboard, workdir / "keyframes" / f"{sp.shot_id}.png", lora_paths)
                out.keyframes[sp.shot_id] = kf_path
                # Write a param hash alongside the PNG so the next warm-worker run can skip
                # regeneration when config is unchanged (_finish copies both into state.tar.gz).
                try:
                    kf_path.with_suffix(".hash").write_text(kf_hash(self.config.keyframe))
                except Exception:
                    pass  # hash write is best-effort; a missing file = old-state reuse behavior
                self.progress.emit("keyframe_done", shot=sp.shot_id)
            else:
                kf_path = self._resolve_keyframe(sp, scene, bundle, workdir, required=sp.needs_i2v)
            if sp.needs_i2v:
                clip = self._animate(
                    scene, kf_path, build_prompt(scene, cast, storyboard),
                    workdir / "clips" / f"{sp.shot_id}.mp4")
                out.clips.append((sp.shot_id, clip))
                self.progress.emit("i2v_done", shot=sp.shot_id)

        # 3) Finishing stage (RIFE interpolation + face restore), gated on the tier/override config.
        # Clip in / clip out: each animated clip is lifted to delivery quality and the result REPLACES
        # the raw clip in `out.clips`, so the off-GPU assemble merges the finished clips. Every clip
        # runs the same finish params and is re-encoded uniformly, so the stream-copy concat stays
        # valid. Skipped entirely when neither pass is on (draft), so the raw i2v clips ship as-is.
        if self.config.finish.enabled and out.clips:
            finished: list[tuple[str, Path]] = []
            for shot_id, clip in out.clips:
                fin = self._finish_clip(shot_id, clip, workdir / "finished" / f"{shot_id}.mp4")
                finished.append((shot_id, fin))
                self.progress.emit("finish_done", shot=shot_id)
            out.clips = finished
        return out

    def _resolve_keyframe(self, sp, scene, bundle: Bundle, workdir: Path,
                          *, required: bool = True) -> Path | None:
        """The keyframe to animate when the plan did not (re)generate it: the authored
        `start_image` for an INJECT shot, or a keyframe a prior pass already left on disk for a
        REUSE shot.

        HONEST FAILURE (#245/#249 applies to this backend too): when the shot NEEDS its keyframe
        (`required`, i.e. the plan animates it) and nothing can be staged, that is a HARD per-shot
        error naming the shot and the reason -- silently skipping it used to ship a film MISSING a
        shot under a success status, the exact dishonest-degrade class the studio refuses. A shot
        the plan does not animate (required=False) may resolve to None harmlessly."""
        if sp.keyframe_mode is KeyframeMode.INJECT and scene.start_image:
            cand = bundle.root / scene.start_image
            if cand.is_file():
                return cand
            if required:
                raise HarnessError(
                    f"shot {sp.shot_id}: authored start_image {scene.start_image!r} is not in "
                    "the bundle; refusing to render the film without this shot")
            return None
        for cand in (
            workdir / "keyframes" / f"{sp.shot_id}.png",
            bundle.root / "keyframes" / f"{sp.shot_id}.png",
        ):
            if cand.is_file():
                return cand
        if required:
            raise HarnessError(
                f"shot {sp.shot_id}: no keyframe staged for {sp.keyframe_mode.value} mode "
                "(not in the workdir or the bundle keyframes/); refusing to render the film "
                "without this shot")
        return None
