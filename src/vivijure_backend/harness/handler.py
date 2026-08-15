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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..assemble import ClipInput, assemble, build_manifest, order_for_storyboard, write_manifest
from ..contract import Bundle, RenderRequest, RenderResult, Keyframe, Clip
from ..finish import Deadline, FinishDeadlineExceeded
from ..orchestrator import Action, RenderPlan, plan as make_plan, resolve_lora_family, validate
from . import keys
from .progress import NullEmitter, ProgressEmitter
from . import job_done_diag


class HarnessError(RuntimeError):
    """A job that failed in the harness layer (bad bundle, validation, missing stage output)."""


@dataclass
class Outputs:
    """What a `Pipeline.execute` produced on disk. The harness turns these into R2 objects.

    `clips` are (shot_id, path) the harness orders by the storyboard before merging. A pipeline
    that already merged the film can set `final_video`; otherwise the harness assembles it."""
    loras: dict[str, Path] = field(default_factory=dict)        # slot -> SDXL adapter file
    wan_loras: dict[str, tuple[Path, Path]] = field(default_factory=dict)  # slot -> (high, low) Wan experts
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
) -> dict:
    """Run one render job end to end and return the control-plane response dict.

    `store` is an R2-like object with `get_file`, `put_file`, `exists`, `get_bytes` (the real
    `R2`, or a fake in tests). Nothing here touches a GPU; the GPU work is `pipeline.execute`.

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
        # Validate the required input up front (like finish_clip/i2v_clip do for their keys), so a
        # malformed job -- e.g. action=render with no bundle_key -- fails with a clear message
        # instead of a botocore ParamValidationError ("Invalid length for parameter Key") deep in R2.
        if not req.bundle_key:
            raise HarnessError(f"{req.action}: bundle_key is required (no project bundle to fetch)")
        _job_bundle_key(req.bundle_key, req.project, what=f"{req.action}: bundle_key")
        tar = store.get_file(req.bundle_key, workdir / "bundle.tar.gz")
        bundle = Bundle.extract(Path(tar), workdir / "project")

        # --- prior state from R2's per-artifact objects (#112) ---
        # Derived AFTER the bundle lands so the storyboard names every candidate key, and so
        # bundle-provided keyframes take precedence over restored ones (hybrid lane contract).
        # Resolve LoRA family before restore so auto train_lora on the train image reads Wan keys.
        lora_family = resolve_lora_family(Action.parse(req.action), req.model_family)
        trained_slots, existing_keyframes = _restore_prior_state(
            store, req.project, bundle, model_family=lora_family)

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


def _job_key(key: str, *, prefixes: tuple[str, ...], what: str) -> str:
    """keys.check_job_key surfaced as a HarnessError: a mis-scoped job key is the same malformed-
    input class as a missing bundle_key, and it must fail before any store I/O."""
    try:
        return keys.check_job_key(key, prefixes=prefixes, what=what)
    except ValueError as e:
        raise HarnessError(str(e)) from None


def _job_bundle_key(bundle_key: str, project: str, *, what: str) -> str:
    try:
        return keys.check_bundle_key_for_project(bundle_key, project, what=what)
    except ValueError as e:
        raise HarnessError(str(e)) from None


def _job_scoped_key(key: str, *, project: str, prefixes: tuple[str, ...], what: str) -> str:
    try:
        return keys.check_scoped_job_key(key, project=project, prefixes=prefixes, what=what)
    except ValueError as e:
        raise HarnessError(str(e)) from None


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
        _job_scoped_key(ref, project=req.project, prefixes=("loras/",),
                        what=f"pretrained LoRA for slot {slot}")
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
    # No owner stamp: the control plane completed the anti-SaaS identity strip (#292), so there is
    # no submitter identity to stamp and /api/artifact is served by key (no per-row ownership check).
    # Artifacts therefore carry no identity metadata. (See SECURITY.md.)

    # LoRA adapters: upload trained ones, pass pretrained through. The adapter object's
    # existence at lora_key IS the "trained" record the next render's restore reads (#112);
    # no local marker, no shared state object.
    for slot, path in outputs.loras.items():
        key = store.put_file(Path(path), keys.lora_key(project, slot))
        result.lora[slot] = {"lora_id": key}
    # Wan 2.2 A14B adapters are TWO experts; upload both and record both keys under the slot. The
    # existence of BOTH objects is the "trained" record the next job's restore reads (a half-upload
    # is never treated as trained -- _restore_prior_state checks both keys). #29.
    for slot, (high, low) in outputs.wan_loras.items():
        key_high = store.put_file(Path(high), keys.wan_lora_key(project, slot, "high"))
        key_low = store.put_file(Path(low), keys.wan_lora_key(project, slot, "low"))
        result.lora[slot] = {"lora_id_high": key_high, "lora_id_low": key_low, "family": "wan"}
    for slot, lora_id in req.pretrained_loras.items():
        result.lora.setdefault(slot, {"lora_id": lora_id})

    # Keyframes: upload whatever the stage drew, each with its param-hash sidecar (#112).
    # Every artifact this shard authored lands at its OWN per-identity key -- concurrent shards
    # of one scattered render write disjoint objects, so nothing here can clobber a sibling.
    reported_shots: set[str] = set()
    for shot_id, path in outputs.keyframes.items():
        key = store.put_file(Path(path), keys.keyframe_key(project, shot_id),
                             content_type="image/png")
        result.keyframes.append(Keyframe(shot_id=shot_id, key=key))
        reported_shots.add(shot_id)
        hash_src = Path(path).with_suffix(".hash")
        if hash_src.is_file():
            store.put_file(hash_src, keys.keyframe_hash_key(project, shot_id),
                           content_type="text/plain")

    # Reused keyframes: already in R2 from a prior run, not re-uploaded, but the caller needs
    # their keys to animate them. Report every REUSE/INJECT shot that wasn't freshly generated.
    from ..orchestrator import Action, KeyframeMode, resolve_lora_family
    for sc in plan.scenes:
        if sc.shot_id not in reported_shots and sc.keyframe_mode in (KeyframeMode.REUSE, KeyframeMode.INJECT):
            result.keyframes.append(Keyframe(shot_id=sc.shot_id, key=keys.keyframe_key(project, sc.shot_id)))

    # Clips ordered by the storyboard (never the stage's incidental order).
    ordered = order_for_storyboard(
        [ClipInput(shot_id=s, path=Path(p)) for s, p in outputs.clips], bundle.storyboard)

    # Audio bed: the pipeline's own track if it made one, else the job's audio_key fetched from
    # the store. A REQUESTED bed that cannot be fetched FAILS the render: shipping a silent film
    # under a success status is the dishonest-degrade class (#245/#249) this backend refuses.
    # render_overrides.audio_optional=true is the EXPLICIT opt-in to soft-degrade instead; that
    # path ships the video silent and surfaces audio_missing in BOTH the event stream and the
    # top-level result (a degrade is never silent).
    audio_path = outputs.audio
    if audio_path is None and req.audio_key:
        # staged beds live under audio/ (the studio's upload route); pipeline-produced beds
        # (music/dialogue/master chains) live under renders/ -- both are inside the key map.
        _job_scoped_key(req.audio_key, project=req.project,
                        prefixes=("audio/", "renders/"), what="audio_key")
        try:
            audio_dest = workdir / ("audio" + (Path(req.audio_key).suffix or ".m4a"))
            store.get_file(req.audio_key, audio_dest)
            audio_path = audio_dest
        except Exception as e:  # noqa: BLE001 -- classified below: hard error or explicit opt-in degrade
            if not bool(req.overrides.get("audio_optional")):
                raise HarnessError(
                    f"could not fetch the requested audio bed {req.audio_key!r}: {e} "
                    "(a requested bed failing is a render failure; set "
                    "render_overrides.audio_optional=true to ship silent instead)") from e
            progress.emit("audio_missing", key=req.audio_key, error=str(e)[:200])
            result.audio_missing = True

    offloaded = bool(req.overrides.get("finish_offloaded"))
    if offloaded:
        # Off-GPU finish elsewhere: emit per-shot clips + a manifest, no merge here.
        for c in ordered:
            key = store.put_file(c.path, keys.clip_key(project, c.shot_id),
                                 content_type="video/mp4")
            result.clips.append(Clip(shot_id=c.shot_id, key=key))
        manifest = build_manifest(ordered, output_name="full.mp4",
                                  audio=str(audio_path) if audio_path else None)
        man_path = write_manifest(manifest, workdir / "manifest.json")
        man_key = keys.join("renders", project, "manifest.json")
        store.put_file(man_path, man_key, content_type="application/json")
        progress.emit("assemble_done", offloaded=True, clips=len(ordered))
        progress.emit("upload_done", key=man_key)
    elif ordered or outputs.final_video:
        # Normal finish: merge here (off-GPU) unless the pipeline already produced the film.
        final = Path(outputs.final_video) if outputs.final_video else \
            assemble(ordered, workdir / "full.mp4", audio=audio_path).output_path
        from ..assemble import probe_duration, probe_has_audio
        result.output_key = store.put_file(final, keys.output_key(project),
                                           content_type="video/mp4")
        result.seconds = probe_duration(final)
        result.has_audio = probe_has_audio(final)
        progress.emit("assemble_done", offloaded=False, seconds=result.seconds)
        progress.emit("upload_done", key=result.output_key)

    # No monolithic project-state object (#112): incremental-render state IS the per-artifact
    # objects uploaded above (keyframe PNG + .hash sidecar per shot, adapter per slot).
    # `result.state_key` stays None; the old shared projects/<slug>/state.tar.gz -- which
    # concurrent shards raced last-writer-wins -- is no longer written or read.
    return result


def _restore_prior_state(store, project: str, bundle: Bundle, *, model_family: str = "sdxl") -> tuple[set[str], dict[str, str | None]]:
    """Derive the planner's skip sets straight from R2's per-artifact objects (#112) and stage
    the reusable keyframe PNGs into the bundle tree.

    Returns (trained_slots, existing_keyframes). Both are empty on a fresh project (nothing in
    R2 yet), and every per-item failure degrades to "regenerate" (best-effort: a redundant
    re-render is the safe default; a fetch hiccup must not abort the job).

    R2 is the ONLY source of truth. The old design extracted a shared projects/<slug>/state.tar.gz
    that every shard of a scattered render rewrote whole -- last-writer-wins, so concurrent
    keyframe-authoring shards clobbered each other's persisted state (#112). Now the storyboard
    names every candidate key directly (no R2 list call needed):

    - LoRA slot `s` is trained iff its adapter object exists at lora_key(project, s). The upload
      in _finish happens only after a successful train, so existence == trained.
    - Shot `sc.id` has a reusable keyframe iff its PNG exists at keyframe_key(project, sc.id);
      its param hash comes from the .hash sidecar (absent sidecar -> None, which _keyframe_mode
      treats as "reuse conservatively", the same contract as the old no-hash-file state).

    Trusting R2 existence is also what #108 wanted: there is no stale state object left to name
    a phantom keyframe.

    PRECEDENCE: a keyframe file the BUNDLE already provided is never overwritten. The hybrid
    keyframe lane splices exact frames into the bundle (overlayKeyframesIntoBundle in the
    control plane) and relies on bundle-wins ordering; skipping the fetch when the local file
    exists preserves that contract with the extraction order inverted."""
    trained_slots: set[str] = set()
    existing_keyframes: dict[str, str | None] = {}
    wan_family = str(model_family).strip().lower() == "wan"
    for slot in bundle.storyboard.use_characters:
        try:
            if wan_family:
                # A Wan slot is trained iff BOTH experts exist; a half-upload is not "trained".
                if (store.exists(keys.wan_lora_key(project, slot, "high"))
                        and store.exists(keys.wan_lora_key(project, slot, "low"))):
                    trained_slots.add(slot)
            elif store.exists(keys.lora_key(project, slot)):
                trained_slots.add(slot)
        except Exception:  # noqa: BLE001 -- unknown -> retrain (safe default)
            pass
    kf_dir = bundle.root / "keyframes"
    for sc in bundle.storyboard.scenes:
        try:
            if not store.exists(keys.keyframe_key(project, sc.id)):
                continue
            local_png = kf_dir / f"{sc.id}.png"
            if not local_png.exists():  # bundle-provided frames win; only fill the gaps
                local_png.parent.mkdir(parents=True, exist_ok=True)
                store.get_file(keys.keyframe_key(project, sc.id), local_png)
            stored: str | None = None
            try:
                raw = store.get_bytes(keys.keyframe_hash_key(project, sc.id))
                stored = raw.decode("utf-8", "replace").strip() or None
                if stored:
                    local_png.with_suffix(".hash").write_text(stored)
            except Exception:  # noqa: BLE001 -- no/unreadable sidecar -> reuse conservatively
                pass
            existing_keyframes[sc.id] = stored
        except Exception:  # noqa: BLE001 -- any per-shot failure -> the planner GENERATEs it
            continue
    return trained_slots, existing_keyframes


def run_finish_job(
    job: dict,
    *,
    store,
    workdir: Path,
    job_id: str = "local",
    on_progress=None,
    deadline: Deadline | None = None,
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
    _job_scoped_key(clip_key_in, project=project, prefixes=("renders/",),
                    what="finish_clip: clip_key")

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

    if deadline is not None:
        deadline.check("fetch_clip")
    try:
        store.get_file(clip_key_in, local_in)
    except Exception as e:
        raise HarnessError(f"finish_clip: could not fetch clip {clip_key_in!r}: {e}")
    if deadline is not None:
        deadline.check("fetch_clip")

    server = ModelServer()
    result = finish_clip(shot_id, local_in, local_out, server, params=params, deadline=deadline)

    # keys._slug via the shared helper: the SAME slug as the full-render path, so one project
    # never scatters its clips across two slug spellings ("My  Film" -> My_Film everywhere).
    clip_key_out = keys.finished_clip_key(project, shot_id)
    # Checked BEFORE the upload and never during it: once the artifact exists the GPU time is
    # already banked, and aborting mid-PUT would discard finished work for nothing. Past the budget
    # HERE we still degrade, because the ceiling has to mean what it says and core has moved on.
    # The PUT itself is bounded by botocore socket timeouts (harness/r2.py sets retries only), NOT
    # by this guard -- see docs/finish-deadline.md.
    if deadline is not None:
        deadline.check("upload")
    store.put_file(local_out, clip_key_out)
    # #583 provenance: stamp the core-computed param-hash to `<clip_key_out>.hash` AFTER the artifact
    # (artifact first, sidecar last -- the only safe order; studio CONTRACT.md 3.3.1). Opaque: write
    # `output_hash` verbatim, never recompute it. Best-effort -- a failed sidecar only disables reuse (the
    # core re-runs), it must NEVER fail a good render. Absent output_hash (legacy core) -> no sidecar.
    output_hash = job.get("output_hash")
    if output_hash:
        try:
            store.put_bytes(str(output_hash).encode("utf-8"), f"{clip_key_out}.hash", content_type="text/plain")
        except Exception:  # noqa: BLE001 -- best-effort provenance; a miss = safe re-run, never a failed render
            pass

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

    HARD INVARIANT (#129): this request/result shape + the two key templates are the `i2v_clip`
    wire contract shared byte-for-byte with the local-gpu doors (vivijure-local-12gb / -16gb),
    which vendor a parallel copy in `vivijure_local.core.contract`. The single reference is
    `tests/fixtures/i2v_clip_contract.json` (byte-identical across all three repos);
    `tests/test_i2v_clip_conformance.py` asserts this door matches it so the two cannot drift.
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

    keyframe_key = str(job.get("keyframe_key") or "")
    if keyframe_key:  # a job-supplied key must sit under this project's renders/ prefix
        _job_scoped_key(keyframe_key, project=project, prefixes=("renders/",),
                        what="i2v_clip: keyframe_key")
    else:
        keyframe_key = keys.keyframe_key(project, shot_id)
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

    # Same shared slug as run_finish_job (see the comment there).
    clip_key_out = keys.i2v_clip_key(project, shot_id)
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


def _wants_i2v_prefetch(action: str) -> bool:
    """Whether a job will load the Wan i2v pipeline, and so benefits from eager-starting the
    ~120GB i2v model prefetch that overlaps LoRA training. False for jobs that never reach an i2v
    stage, so the pull is not started for nothing:
      - finish_clip: RIFE interpolation / face-restore only, no Wan pipeline.
      - train_lora:  a train-only job -- it fits a LoRA and stops; no keyframe/i2v stage runs, and
                     the Wan-LoRA train image carries no i2v weights to mirror (cf#29 D2a).
    Every other action (render / finalize / preview / regen_shot) keeps the prefetch."""
    return str(action) not in ("finish_clip", "train_lora")


def _finish_deadline_degrade(expiry: FinishDeadlineExceeded, *, project: str, job_id: str,
                             store=None, on_progress=None) -> dict:
    """Turn a finish wall-clock expiry into a STRUCTURED soft degrade. Returns data, never raises.

    The shape is dictated by the CONSUMER, vivijure-cf modules/_shared/finish-soft-degrade.ts, read
    at cf 219cd6b6352a0535a64d76fc228c41fd6c43ee06:

      - ok: False is the entire discriminator. softDegradeInCompletedOutput requires an object
        output whose ok is exactly false. A genuine crash leaves NO structured output and keeps
        failing loud, so this widens nothing: only an expiry reaches here.
      - detail is the FIRST key its degradeReason reads, and it slices to 120 characters.
      - NO top-level error key. RunPod lifts a top-level error in a handler RETURN into a job-level
        FAILED envelope, which forces recovery through the FAILED branch for no gain; detail keeps
        the envelope COMPLETED and the output arrives intact.

    The cf module builds the passthrough TAG itself, from its own stored input key; this reason
    only becomes the human note in FinishOutput.degraded. KNOWN LIMIT: that tag is the generic
    passthrough:backend-soft-degrade for every cause, so the CAUSE is not machine-visible
    downstream (vivijure-core#226, cf#595). Not fixable from this repo.

    A raise instead of this return would leave no structured output, classify deterministic in
    vivijure-core and fail the WHOLE render after the GPU time is already banked -- strictly worse
    than the unbounded hang this guard replaces, which core phase ceiling at least recovers."""
    import json
    print("@event finish_deadline_exceeded " + json.dumps({
        "stage": expiry.stage,
        "elapsed_seconds": round(expiry.elapsed, 1),
        "budget_seconds": expiry.budget,
        "project": project,
        "job_id": job_id,
    }), flush=True)
    try:
        # Terminal record on the R2 progress channel so the run does not sit at started forever.
        # COMPLETE rather than error, matching the RunPod envelope, which also stays COMPLETED.
        # Best-effort: a progress failure must NEVER change what this returns.
        ProgressEmitter(store, project, job_id, on_progress=on_progress).complete(
            degraded=expiry.reason, stage=expiry.stage)
    except Exception:  # noqa: BLE001 -- diagnostics only; the degrade return is what matters
        pass
    return {"ok": False, "detail": expiry.reason}


def handler(job: dict) -> dict:
    """RunPod serverless entry point. Mirrors models on a cold worker, builds the live R2
    client, runs the job through the deployed GPU pipeline, returns the response. RunPod passes
    `{"input": {...}}`; the render request is the inner dict.

    TWO R2 credentials, split by PURPOSE, because they have opposite sharing requirements. The
    models mirror pulls identical shared weights from OUR bucket and reads its own credential from
    the environment, always. Tenant job I/O (bundle in, film out, LoRAs, keyframes, clips, progress)
    uses the per-job payload block when present, so ONE pooled endpoint can serve many tenants
    without any of them sharing a credential or a bucket, and falls back to the endpoint env when
    absent, which is what a dedicated endpoint keeps doing unchanged.

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

    # ONE wall-clock budget for the WHOLE finish_clip invocation (#422), opened HERE at the first
    # line of the invocation so the R2 client build and the cold-start model mirror are INSIDE it
    # and the number is a real invocation ceiling rather than a stage total. ONE shared deadline,
    # not a timeout per step: per-step timeouts multiply, and the invocation bound is then a sum
    # nobody computes.
    #
    # SCOPED TO finish_clip ALONE (#422 D). Every other action gets None, and every check here and
    # in finish.py is guarded on deadline-is-not-None, so render / i2v_clip / train_lora / health
    # cannot raise FinishDeadlineExceeded at all. Unchanged by CONSTRUCTION, not by a promise.
    action_in = str(payload.get("action", "render"))
    deadline = Deadline.start() if action_in == "finish_clip" else None
    store = None  # bound below; named here so the expiry handler can pass it to the emitter

    # Everything below runs inside the quiesce guard (#90): on EVERY exit -- return or raise --
    # in-flight progress-mirror posts are drained before the SDK posts the terminal result to
    # the same /job-done endpoint, so a straggler mirror can never race the finalization.
    try:
        try:
            # Tenant job I/O only. The payload block when the job carries one (a POOLED endpoint
            # serving many tenants), the endpoint env when it does not (a dedicated endpoint,
            # unchanged). A present-but-malformed block raises here rather than degrading to env:
            # see R2Config.from_payload_or_env for why that must never be a fallback.
            store = R2(R2Config.from_payload_or_env(payload))
        except Exception as e:
            ProgressEmitter(None, project, job_id, on_progress=on_progress).error("config", e)
            raise
        # Everything below the store gets the payload WITHOUT the credential block, so no
        # downstream emitter, manifest, or error path can echo it even by accident.
        payload = R2Config.strip_from_payload(payload)
        # Register the run context (the live store + identity) so the SDK's job-done _transmit
        # patch can mirror a callback rejection into this run's R2 channel; the patch itself has
        # no run context (#90). Best-effort; a config failure above never reaches here (R2 is the
        # failure, so its own rejection can only degrade to stdout).
        job_done_diag.register(store, project, job_id)
        if deadline is not None:
            deadline.check("r2_client")
        try:
            mirrored = ensure_models()
        except Exception as e:
            ProgressEmitter(store, project, job_id, on_progress=on_progress).error("mirror", e)
            raise
        # The mirror is one un-interruptible call, so it is bounded only at its EDGES: it cannot be
        # cut short, but it can no longer be followed by compute that ignores how long it took. On
        # the baked production image it short-circuits in milliseconds (models_mirror.is_baked).
        if deadline is not None:
            deadline.check("model_mirror")

        # Eager-start the Wan I2V pull in the background so it overlaps LoRA training: training is
        # GPU-bound with the network idle, while the pull (~120GB from R2) is network-bound. The two
        # run concurrently; ensure_i2v_models() joins the thread before loading the Wan pipeline.
        # SKIP the prefetch for jobs that never load the Wan i2v pipeline (see _wants_i2v_prefetch),
        # so the ~120GB pull is not started for nothing.
        if _wants_i2v_prefetch(payload.get("action", "render")):
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

            # Prior-state restore happens INSIDE run_job now (#112): it needs the extracted
            # bundle's storyboard to name the per-artifact R2 keys it checks.
            return run_job(payload, pipeline=get_pipeline(), store=store, workdir=workdir,
                           job_id=job_id, mirrored=bool(mirrored), on_progress=on_progress)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    except FinishDeadlineExceeded as expiry:
        # The ONE catch site for the guard. RETURN DATA, NEVER RAISE -- see the docstring on
        # _finish_deadline_degrade for why a raise here is worse than the hang it replaces.
        return _finish_deadline_degrade(expiry, project=project, job_id=job_id,
                                        store=store, on_progress=on_progress)
    finally:
        quiesce = getattr(on_progress, "quiesce", None)
        if callable(quiesce):
            quiesce()


def _runpod_progress_hook(job: dict):
    """Option A: mirror each snapshot into RunPod's status `progress` field, best-effort. The
    `runpod` import is deferred so the harness stays CPU-importable; a missing SDK or a failed
    update is swallowed (the R2 channel is the source of truth).

    A TERMINAL snapshot (status error/complete) is NEVER mirrored: the SDK's progress_update
    posts {"status": "IN_PROGRESS", "output": snapshot} from a fresh daemon thread (new event
    loop + new TLS session), which races the SDK's own terminal result POST on the same endpoint
    and can OVERWRITE a FAILED/COMPLETED verdict back to IN_PROGRESS. Observed live (F17): a
    155ms config-error job read IN_PROGRESS forever with the error snapshot in `output`, holding
    the billed worker 344s until manual cancel, while the studio's poll translated it to "job
    not found". The terminal record still reaches R2 + stdout; RunPod's terminal status comes
    from the handler's own return/raise, which must stand unclobbered.

    Mirror threads are TRACKED and joined via `hook.quiesce()` before the handler returns (#90):
    the SDK's progress_update fires an UNTRACKED daemon thread at the same /job-done endpoint the
    terminal result posts to, so a late mirror can arrive at (or race) a finalizing job and get
    rejected 400 "internal server error" -- logged by the SDK's shared _handle_result as "Failed
    to return job results.", misattributed to the result post (which succeeded: /status showed
    COMPLETED with the full payload every time). Draining our own tracked threads before return
    means no mirror is ever in flight when the result posts. Falls back to the untracked SDK
    call if the internals move (best-effort doctrine; the R2 channel stays the source of
    truth)."""
    import threading
    import time

    threads: list = []

    def hook(snapshot: dict) -> None:
        if snapshot.get("status") in ("error", "complete"):
            return
        try:
            import runpod
            try:
                from runpod.serverless.modules import rp_progress
                t = threading.Thread(target=rp_progress._thread_target,
                                     args=(job, dict(snapshot)), daemon=True)
                t.start()
                threads.append(t)
            except ImportError:  # SDK internals moved: untracked fallback, mirror still works
                runpod.serverless.progress_update(job, snapshot)
        except Exception:
            pass

    def quiesce(timeout: float = 10.0) -> None:
        """Join every in-flight mirror post (bounded): called before the handler returns so the
        SDK's terminal result post never shares the endpoint with a straggler mirror."""
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(max(0.0, deadline - time.monotonic()))

    hook.quiesce = quiesce
    return hook
