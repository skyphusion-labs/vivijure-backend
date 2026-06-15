"""The worker's job flow: bundle in -> plan -> stages -> finish -> results out.

This is the spine that turns a RunPod job into a render and back into R2 keys. It owns the I/O
contract (what comes in, what goes out, where) and the order of operations; it owns no model
code. The GPU stages sit behind the `Pipeline` protocol and are injected, so this module
imports and tests on a CPU box: `run_job` is exercised with a fake pipeline and a fake store,
and the real `handler` entry point wires the live R2 client, the cold-start model mirror, and
the deployed GPU pipeline.

The finish is deliberately off-GPU (see the planner's `assemble_off_gpu`): a normal render
merges the clips here with ffmpeg, while an offloaded finish (`finish_offloaded`) just uploads
the per-shot clips plus a manifest for a separate CPU container to merge.
"""
from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..assemble import ClipInput, assemble, build_manifest, order_for_storyboard, write_manifest
from ..contract import Bundle, RenderRequest, RenderResult, Keyframe, Clip
from ..orchestrator import RenderPlan, plan as make_plan, validate
from . import keys
from .progress import NullEmitter, ProgressEmitter


class HarnessError(RuntimeError):
    """A job that failed in the harness layer (bad bundle, validation, missing stage output)."""


@dataclass
class Outputs:
    """What a `Pipeline.execute` produced on disk. The harness turns these into R2 objects.

    `clips` are (shot_id, path) the harness orders by the storyboard before merging. A pipeline
    that already merged the film can set `final_video`; otherwise the harness assembles it."""
    loras: dict[str, Path] = field(default_factory=dict)        # slot -> adapter file
    keyframes: dict[str, Path] = field(default_factory=dict)    # shot_id -> png
    clips: list[tuple[str, Path]] = field(default_factory=list)  # (shot_id, mp4)
    final_video: Path | None = None
    audio: Path | None = None


@runtime_checkable
class Pipeline(Protocol):
    """The GPU stages, injected. Given the plan and the extracted bundle, run only the work the
    plan did not eliminate (train the listed LoRAs, generate the GENERATE keyframes, animate the
    needs_i2v shots) and return where the artifacts landed. Implemented by the model layer; the
    harness never imports torch."""

    def execute(self, plan: RenderPlan, bundle: Bundle, workdir: Path) -> Outputs: ...


def run_job(
    job: dict,
    *,
    pipeline: Pipeline,
    store,
    workdir: Path,
    job_id: str = "local",
    mirrored: bool = False,
    on_progress=None,
    trained_slots: set[str] = frozenset(),
    existing_keyframes: dict[str, str | None] = {},
) -> dict:
    """Run one render job end to end and return the control-plane response dict.

    `store` is an R2-like object with `get_file`, `put_file`, `put_dir_as_tar` (the real `R2`,
    or a fake in tests). Nothing here touches a GPU; the GPU work is `pipeline.execute`.

    Progress is emitted to the structured channel keyed by `(project, job_id)`: `mirrored` records
    whether the cold-start model mirror ran, `on_progress` is the optional RunPod hook. The whole
    channel is best-effort and never fails the render; a real render failure still propagates (an
    `error` event is recorded first).
    """
    req = RenderRequest.from_dict(job)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    progress = ProgressEmitter(store, req.project, job_id, on_progress=on_progress)
    progress.emit("started", action=req.action, quality=req.quality_tier, project=req.project)
    progress.emit("mirror_done", pulled=bool(mirrored))
    try:
        # --- bundle in ---
        tar = store.get_file(req.bundle_key, workdir / "bundle.tar.gz")
        bundle = Bundle.extract(Path(tar), workdir / "project")

        # --- validate + plan (CPU) ---
        errs = validate(req, bundle.storyboard, cast=bundle.cast)
        if errs:
            raise HarnessError("invalid render job: " + "; ".join(errs))
        plan = make_plan(
            req, bundle.storyboard,
            trained_slots=set(trained_slots) | set(req.pretrained_loras),
            existing_keyframes=existing_keyframes,
        )
        # Post-plan ref check: validate can't know which slots actually need training (that
        # depends on prior trained_slots from R2), so check refs here where the plan is settled.
        ref_errs = [
            f"character slot {s!r} has no reference images; LoRA training will fail"
            for s in plan.lora.train
            if not bundle.cast.characters.get(s) or not bundle.cast.characters[s].ref_paths
        ]
        if ref_errs:
            raise HarnessError("invalid render job: " + "; ".join(ref_errs))

        # --- stage reused LoRAs from R2 (the harness owns R2; the GPU layer never touches it) ---
        # The plan skipped training for these slots; their adapters live as R2 keys, so pull each
        # to local disk before keyframing and hand the local-path map to the pipeline. Fail-fast
        # (before any GPU work) if a requested adapter cannot be fetched, rather than silently
        # rendering the character without its identity LoRA.
        staged = _stage_pretrained_loras(req, store, workdir, progress)

        # --- GPU stages (only what the plan kept) ---
        _inject_progress(pipeline, progress)
        _inject_pretrained_loras(pipeline, staged)
        outputs = pipeline.execute(plan, bundle, workdir)

        # --- finish + results out ---
        result = _finish(req, plan, bundle, outputs, store, workdir, progress)
        progress.complete(output_key=result.output_key, seconds=result.seconds,
                          clips=len(result.clips), keyframes=len(result.keyframes))
        return result.to_dict()
    except Exception as e:
        progress.error("render", e)  # best-effort failure marker, then let the render fail
        raise


def _inject_progress(pipeline, progress) -> None:
    """Hand the emitter to a pipeline that wants per-stage progress (GpuPipeline), duck-typed so
    the `Pipeline` protocol and the test fakes stay unchanged. Best-effort."""
    setter = getattr(pipeline, "set_progress", None)
    if callable(setter):
        try:
            setter(progress)
        except Exception:
            pass


def _stage_pretrained_loras(req: RenderRequest, store, workdir: Path, progress) -> dict[str, str]:
    """Download each reused-LoRA R2 key to a local file so the GPU pipeline can load it without
    touching R2. Returns slot -> local path.

    A ref that is already a local file (a pre-staged deploy, or a test) is taken as-is. A ref the
    store cannot serve is a hard error (HarnessError): the plan already skipped training that slot,
    so rendering on without its adapter would silently produce the wrong identity, and that is
    worse than failing the job here, cheaply, before any GPU work. (R2 transient failures are the
    store's own retry concern.)"""
    staged: dict[str, str] = {}
    for slot, ref in req.pretrained_loras.items():
        if Path(ref).is_file():
            staged[slot] = str(ref)
            continue
        dest = workdir / "pretrained" / slot / (Path(ref).name or "pytorch_lora_weights.safetensors")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            store.get_file(ref, dest)
        except Exception as e:
            raise HarnessError(f"could not stage pretrained LoRA for slot {slot} from {ref!r}: {e}")
        staged[slot] = str(dest)
        progress.emit("lora_staged", slot=slot, key=ref)
    return staged


def _inject_pretrained_loras(pipeline, staged: dict[str, str]) -> None:
    """Hand the local-path adapter map to the pipeline so it loads the reused LoRAs.

    Unlike `_inject_progress`, this is NOT best-effort, and the asymmetry is deliberate: progress
    is optional (a dropped event does not change the render), but a dropped LoRA map silently
    renders the character without its identity, which is the exact outcome `_stage_pretrained_loras`
    fails fast to prevent. So when there is something to deliver, a pipeline that cannot receive it
    (no `set_pretrained_loras`, or a setter that throws) is a hard error, not a swallow. A job with
    no reused LoRAs injects nothing, so a pipeline without the setter is fine."""
    if not staged:
        return
    setter = getattr(pipeline, "set_pretrained_loras", None)
    if not callable(setter):
        raise HarnessError(
            f"pipeline cannot receive staged reused LoRAs {sorted(staged)} "
            "(no set_pretrained_loras); refusing to render them without their identity adapters")
    setter(staged)  # a setter failure propagates: dropping staged LoRAs is silent-wrong-identity


def _finish(req: RenderRequest, plan: RenderPlan, bundle: Bundle, outputs: Outputs,
            store, workdir: Path, progress=None) -> RenderResult:
    progress = progress or NullEmitter()
    project = req.project
    result = RenderResult(project=project)
    # Stamp the owner on every uploaded artifact. The control plane's /api/artifact route 403s
    # any object whose customMetadata.user_email != the caller, so without this the user cannot
    # fetch back their own keyframes/clips/mp4 (they render fine, then show blank in the UI). The
    # control plane sends user_email in the job input; the contract now parses it. None when absent
    # (e.g. a local/test run) leaves uploads untagged exactly as before.
    owner_meta = {"user_email": req.user_email} if req.user_email else None

    # LoRA adapters: upload trained ones, pass pretrained through.
    # Also write a zero-byte marker into the project tree so the next incremental render's
    # state restore can derive trained_slots without an R2 list call.
    for slot, path in outputs.loras.items():
        key = store.put_file(Path(path), keys.lora_key(project, slot), metadata=owner_meta)
        result.lora[slot] = {"lora_id": key}
        marker = bundle.root / "loras" / slot / ".trained"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    for slot, lora_id in req.pretrained_loras.items():
        result.lora.setdefault(slot, {"lora_id": lora_id})

    # Keyframes: upload whatever the stage drew; also persist into the project tree so the
    # next incremental render's state restore can derive existing_keyframes without an R2 list.
    reported_shots: set[str] = set()
    for shot_id, path in outputs.keyframes.items():
        key = store.put_file(Path(path), keys.keyframe_key(project, shot_id),
                             content_type="image/png", metadata=owner_meta)
        result.keyframes.append(Keyframe(shot_id=shot_id, key=key))
        reported_shots.add(shot_id)
        state_kf = bundle.root / "keyframes" / f"{shot_id}.png"
        state_kf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, state_kf)
        hash_src = Path(path).with_suffix(".hash")
        if hash_src.is_file():
            shutil.copy2(hash_src, state_kf.with_suffix(".hash"))

    # Reused keyframes: already in R2 from a prior run, not re-uploaded, but the caller needs
    # their keys to animate them. Report every REUSE/INJECT shot that wasn't freshly generated.
    from ..orchestrator import KeyframeMode
    for sc in plan.scenes:
        if sc.shot_id not in reported_shots and sc.keyframe_mode in (KeyframeMode.REUSE, KeyframeMode.INJECT):
            result.keyframes.append(Keyframe(shot_id=sc.shot_id, key=keys.keyframe_key(project, sc.shot_id)))

    # Clips ordered by the storyboard (never the stage's incidental order).
    ordered = order_for_storyboard(
        [ClipInput(shot_id=s, path=Path(p)) for s, p in outputs.clips], bundle.storyboard)

    # Audio bed: the pipeline's own track if it made one, else the job's audio_key fetched from the
    # store. Best-effort -- a missing/failed audio fetch records a marker and ships the video silent
    # rather than failing the whole render.
    audio_path = outputs.audio
    if audio_path is None and req.audio_key:
        try:
            audio_dest = workdir / ("audio" + (Path(req.audio_key).suffix or ".m4a"))
            store.get_file(req.audio_key, audio_dest)
            audio_path = audio_dest
        except Exception as e:  # noqa: BLE001 -- audio is optional; never fail the render on it
            progress.emit("audio_missing", key=req.audio_key, error=str(e)[:200])

    offloaded = bool(req.overrides.get("finish_offloaded"))
    if offloaded:
        # Off-GPU finish elsewhere: emit per-shot clips + a manifest, no merge here.
        for c in ordered:
            key = store.put_file(c.path, keys.clip_key(project, c.shot_id),
                                 content_type="video/mp4", metadata=owner_meta)
            result.clips.append(Clip(shot_id=c.shot_id, key=key))
        manifest = build_manifest(ordered, output_name="full.mp4",
                                  audio=str(audio_path) if audio_path else None)
        man_path = write_manifest(manifest, workdir / "manifest.json")
        man_key = keys.join("renders", project, "manifest.json")
        store.put_file(man_path, man_key, content_type="application/json", metadata=owner_meta)
        progress.emit("assemble_done", offloaded=True, clips=len(ordered))
        progress.emit("upload_done", key=man_key)
    elif ordered or outputs.final_video:
        # Normal finish: merge here (off-GPU) unless the pipeline already produced the film.
        final = Path(outputs.final_video) if outputs.final_video else \
            assemble(ordered, workdir / "full.mp4", audio=audio_path).output_path
        from ..assemble import probe_duration, probe_has_audio
        result.output_key = store.put_file(final, keys.output_key(project),
                                           content_type="video/mp4", metadata=owner_meta)
        result.seconds = probe_duration(final)
        result.has_audio = probe_has_audio(final)
        progress.emit("assemble_done", offloaded=False, seconds=result.seconds)
        progress.emit("upload_done", key=result.output_key)

    # Project state for the next incremental render.
    result.state_key = store.put_dir_as_tar(bundle.root, keys.state_key(project), metadata=owner_meta)
    progress.emit("upload_done", key=result.state_key)
    return result


def _restore_prior_state(store, project: str, workdir: Path) -> tuple[set[str], dict[str, str | None]]:
    """Fetch and extract the prior render's state_key into workdir/project, then derive the sets
    the planner needs to skip already-done GPU work.

    Returns (trained_slots, existing_keyframes). Both are empty on a fresh project (no prior state)
    or if the fetch / extraction fails for any reason (best-effort: a stale state is worse than a
    redundant re-render, but a failed fetch should not abort the job).

    The state tar is extracted into the SAME directory run_job will later use for the bundle
    (workdir/project). The fresh bundle tar (sent by the control plane) contains storyboard.yaml,
    characters/, and refs/ -- it does NOT contain keyframes/ or loras/, so extracting the bundle
    on top of the restored state leaves the prior keyframe PNGs and lora markers intact. The
    pipeline's _resolve_keyframe checks bundle.root/keyframes/{shot_id}.png, which lands exactly
    there."""
    trained_slots: set[str] = set()
    existing_keyframes: dict[str, str | None] = {}
    try:
        state_tar = workdir / "prior_state.tar.gz"
        store.get_file(keys.state_key(project), state_tar)
        state_root = workdir / "project"
        state_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(state_tar, "r:gz") as tf:
            from ..contract import _safe_extract
            _safe_extract(tf, state_root)
        state_tar.unlink(missing_ok=True)
        loras_dir = state_root / "loras"
        if loras_dir.is_dir():
            trained_slots = {d.name for d in loras_dir.iterdir()
                             if d.is_dir() and (d / ".trained").is_file()}
        kf_dir = state_root / "keyframes"
        if kf_dir.is_dir():
            # Build shot_id -> stored_hash dict. Hash files are written alongside PNGs in
            # _finish() so a warm worker can compare render params before reusing a keyframe.
            # Old state (no .hash files) gets None as the value -- _keyframe_mode treats that
            # as "reuse conservatively" so upgrading never forces a full regeneration.
            for png in kf_dir.iterdir():
                if png.suffix == ".png":
                    hash_file = png.with_suffix(".hash")
                    stored = hash_file.read_text().strip() if hash_file.is_file() else None
                    existing_keyframes[png.stem] = stored
    except Exception:  # noqa: BLE001 -- any fetch/extract failure -> fresh render (safe default)
        pass
    return trained_slots, existing_keyframes


def run_finish_job(
    job: dict,
    *,
    store,
    workdir: Path,
    job_id: str = "local",
    on_progress=None,
) -> dict:
    """Standalone finish pass: download a clip from R2, run RIFE interpolation and/or face restore,
    upload the result, return the output key. No bundle, no pipeline, no Wan needed.

    Input shape (the `input` dict from the RunPod job):
      { action, project, shot_id, clip_key, config: { interpolate, interpolation_factor,
        face_restore, face_fidelity, only_faces } }
    """
    from ..finish import FinishParams, finish_clip
    from ..models import ModelServer

    project = str(job.get("project") or "untitled")
    shot_id = str(job.get("shot_id") or "shot")
    clip_key_in = str(job.get("clip_key") or "")
    cfg = job.get("config") or {}

    progress = ProgressEmitter(store, project, job_id, on_progress=on_progress)
    progress.emit("started", action="finish_clip", project=project)

    if not clip_key_in:
        raise HarnessError("finish_clip: clip_key is required")

    params = FinishParams(
        interpolate=bool(cfg.get("interpolate", True)),
        factor=int(cfg.get("interpolation_factor", 2)),
        target_fps=int(cfg.get("target_fps", 0)),
        face_restore=bool(cfg.get("face_restore") not in (None, False, "none", "")),
        face_restore_backend=str(cfg.get("face_restore") or "gfpgan") if cfg.get("face_restore") not in (None, False, "none", "") else "gfpgan",
        face_fidelity=float(cfg.get("face_fidelity", 0.7)),
        only_faces=bool(cfg.get("only_faces", True)),
    )

    local_in = workdir / "input.mp4"
    local_out = workdir / "output.mp4"

    try:
        store.get_file(clip_key_in, local_in)
    except Exception as e:
        raise HarnessError(f"finish_clip: could not fetch clip {clip_key_in!r}: {e}")

    server = ModelServer()
    result = finish_clip(shot_id, local_in, local_out, server, params=params)

    safe = lambda s: (s or "x").replace("/", "_").replace(" ", "_")
    clip_key_out = f"renders/{safe(project)}/clips/{safe(shot_id)}_finished.mp4"
    store.put_file(local_out, clip_key_out)

    applied: list[str] = []
    if result.interpolated:
        applied.append(f"interpolate:{params.factor}x")
    if result.face_restored:
        applied.append(f"face_restore:{params.face_restore_backend}")

    progress.complete(output_key=clip_key_out)
    # Pointer-only return: keep the job-done payload small so RunPod's job-done endpoint
    # does not reject it. All state lives in R2; the caller only needs the output key.
    return {
        "clip_key": clip_key_out,
        "out_fps": result.out_fps,
        "frames": result.frames_out,
        "applied": applied,
    }


def run_i2v_clip_job(
    job: dict,
    *,
    store,
    workdir: Path,
    job_id: str = "local",
    on_progress=None,
) -> dict:
    """Standalone per-shot image-to-video pass: fetch a keyframe from R2, animate it into one clip
    with Wan2.2-I2V, upload the clip, return its key. No bundle, no render pipeline -- just the
    ModelServer's i2v pipeline (the backend half of studio #81). Mirrors `run_finish_job`.

    Input shape (the `input` dict from the RunPod job):
      { action, project, shot_id, prompt, keyframe_key?,
        config: { quality, num_frames?, fps?, seed?, flow_shift?, height?, width?, negative_prompt? } }

    The keyframe defaults to the project/shot convention `renders/<project>/keyframes/<shot>.png`
    when `keyframe_key` is omitted. The clip lands at `renders/<project>/clips/<shot>_i2v.mp4`.
    """
    from .. import i2v as i2v_mod
    from ..config import I2VConfig
    from ..contract import Scene
    from ..models import ModelServer
    from ..routing import QualityTier

    project = str(job.get("project") or "untitled")
    shot_id = str(job.get("shot_id") or "shot")
    prompt = str(job.get("prompt") or "")
    cfg = job.get("config") or {}

    progress = ProgressEmitter(store, project, job_id, on_progress=on_progress)
    progress.emit("started", action="i2v_clip", project=project, shot_id=shot_id)

    if not prompt:
        raise HarnessError("i2v_clip: prompt is required (the motion description)")

    keyframe_key = str(job.get("keyframe_key") or "") or keys.keyframe_key(project, shot_id)
    local_kf = workdir / "keyframe.png"
    try:
        store.get_file(keyframe_key, local_kf)
    except Exception as e:
        raise HarnessError(f"i2v_clip: could not fetch keyframe {keyframe_key!r}: {e}")

    # Build the engine params from the tier baseline + the job's overrides, reusing the typed
    # I2VConfig so clamping AND the distill<->feature-cache invariant (no caching a 4-step render)
    # are enforced exactly as in the full render path. height/width live only on the engine
    # I2VParams (I2VConfig follows the keyframe's native dims), so they are read from cfg directly;
    # a falsy value (null/0/"") means "follow the keyframe".
    tier = QualityTier.parse(cfg.get("quality"))
    ic = I2VConfig.from_dict(cfg, tier=tier)
    params = i2v_mod.I2VParams(
        num_frames=i2v_mod.snap_frames(ic.num_frames),  # temporal VAE wants 4k+1
        fps=ic.fps,
        steps=ic.distill_steps if ic.distill else ic.steps,
        guidance_scale=ic.guidance_scale,
        distill=ic.distill,
        seed=ic.seed,
        height=int(cfg["height"]) if cfg.get("height") else None,
        width=int(cfg["width"]) if cfg.get("width") else None,
        feature_cache=ic.feature_cache,
        flow_shift=ic.flow_shift,
    )
    # Custom negative is additive over the engine's anti-static guard (the #25 fix), never a
    # replacement -- a bare custom negative would drop the anti-freeze default and risk a still clip.
    if ic.negative_prompt:
        params.negative_prompt = ic.negative_prompt + ", " + i2v_mod.I2VParams.negative_prompt

    out_path = workdir / "out.mp4"
    result = i2v_mod.animate(
        Scene(id=shot_id, prompt=prompt), local_kf, prompt, ModelServer(), out_path,
        params=params, progress_cb=progress.i2v_step_cb(shot_id),
    )

    safe = lambda s: (s or "x").replace("/", "_").replace(" ", "_")
    clip_key_out = f"renders/{safe(project)}/clips/{safe(shot_id)}_i2v.mp4"
    store.put_file(result.path, clip_key_out, content_type="video/mp4")

    progress.complete(output_key=clip_key_out)
    # Pointer-only return (same rationale as run_finish_job): small job-done payload; R2 holds state.
    return {
        "clip_key": clip_key_out,
        "shot_id": shot_id,
        "num_frames": result.num_frames,
        "fps": result.fps,
        "seconds": result.seconds,
        "distilled": result.distilled,
    }


def handler(job: dict) -> dict:
    """RunPod serverless entry point. Mirrors models on a cold worker, builds the live R2
    client, runs the job through the deployed GPU pipeline, returns the response. RunPod passes
    `{"input": {...}}`; the render request is the inner dict.

    The R2 client and the cold-start model mirror both run BEFORE run_job's own emitter exists,
    yet a failure there (a broken mirror / missing weight, the exact class the channel must
    surface) is the most opaque kind. So build the store first and wrap each gate with an emitter
    that writes an `error` snapshot before re-raising. A bad R2 config is the one failure we cannot
    record to R2 (R2 is the failure), so it degrades to stdout + the RunPod hook."""
    import tempfile

    from .models_mirror import ensure_models
    from .r2 import R2, R2Config
    from .pipeline_registry import get_pipeline  # the deploy registers its GPU pipeline here

    payload = job.get("input", job)
    project = str(payload.get("project") or "untitled")
    job_id = str(job.get("id") or "unknown")
    on_progress = _runpod_progress_hook(job)

    try:
        store = R2(R2Config.from_env())
    except Exception as e:
        ProgressEmitter(None, project, job_id, on_progress=on_progress).error("config", e)
        raise
    try:
        mirrored = ensure_models()
    except Exception as e:
        ProgressEmitter(store, project, job_id, on_progress=on_progress).error("mirror", e)
        raise

    # Eager-start the Wan I2V pull in the background so it overlaps LoRA training: training is
    # GPU-bound with the network idle, while the pull (~120GB from R2) is network-bound. The two
    # run concurrently; ensure_i2v_models() joins the thread before loading the Wan pipeline.
    # finish_clip never loads the Wan pipeline, so skip the prefetch for that action.
    if str(payload.get("action", "render")) != "finish_clip":
        from .models_mirror import start_i2v_prefetch
        start_i2v_prefetch()

    workdir = Path(tempfile.mkdtemp(prefix="vj-job-"))
    try:
        action = str(payload.get("action", "render"))
        if action == "finish_clip":
            return run_finish_job(payload, store=store, workdir=workdir,
                                  job_id=job_id, on_progress=on_progress)
        if action == "i2v_clip":
            return run_i2v_clip_job(payload, store=store, workdir=workdir,
                                    job_id=job_id, on_progress=on_progress)

        trained_slots, existing_keyframes = _restore_prior_state(store, project, workdir)
        return run_job(payload, pipeline=get_pipeline(), store=store, workdir=workdir,
                       job_id=job_id, mirrored=bool(mirrored), on_progress=on_progress,
                       trained_slots=trained_slots, existing_keyframes=existing_keyframes)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _runpod_progress_hook(job: dict):
    """Option A: mirror each snapshot into RunPod's status `progress` field, best-effort. The
    `runpod` import is deferred so the harness stays CPU-importable; a missing SDK or a failed
    update is swallowed (the R2 channel is the source of truth)."""
    def hook(snapshot: dict) -> None:
        try:
            import runpod
            runpod.serverless.progress_update(job, snapshot)
        except Exception:
            pass
    return hook
