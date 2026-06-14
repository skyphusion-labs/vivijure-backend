"""Finishing pass: lift each animated clip to delivery quality, on the GPU worker, clip in / clip out.

Wan i2v emits its frames at 16 fps, and a character's face -- sharp in the SDXL keyframe -- can
soften or drift over those frames. This stage fixes both, cheaply, AFTER i2v and BEFORE the off-GPU
assemble merges the shots:

  - frame interpolation (RIFE) resamples the choppy 16 fps up to a smooth target. This is the single
    biggest perceived-quality jump per GPU-second we can buy, and low frame rate is the thing that
    most reads as "AI video"; the commercial tools paywall smooth frame rate, we give it away.
  - face restoration (a blind face restorer over the detected faces) re-locks the identity the
    keyframe established but the motion model blurred -- the identity-through-motion fix that serves
    the consistent-character goal directly.

Each pass is light next to i2v and is independently toggled by `config.FinishConfig`; the planner
estimates their cost. Crucially, every clip in one render runs the SAME finish params, so all clips
still share fps + codec and `assemble`'s stream-copy concat stays valid (no re-encode fallback).

Clean-room: built from RIFE's documented recursive 2x interpolation interface, a blind-face-restorer
inference API + facelib detection, and ffmpeg/imageio for decode/encode -- not from any prior
pipeline. The frame / fps math and the run/skip decisions are pure and CPU-tested; the GPU body
(`finish_clip`) defers torch + the model imports and is validated on a pod.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# RIFE interpolates recursively by doubling: one pass turns N frames into 2N-1, doubling the frame
# rate. A factor of 4 is two passes, 8 is three. So the only valid factors are powers of two, and we
# cap at 8x (16 -> 128 fps is already past any delivery need). 1x means "interpolation off".
VALID_FACTORS = (1, 2, 4, 8)
MAX_FACTOR = 8


# --------------------------------------------------------------------------- pure helpers

def snap_factor(factor: int) -> int:
    """Snap an interpolation factor to the nearest valid power of two in [1, 8], rounding DOWN so a
    request never silently buys more interpolation (more GPU) than asked. Junk falls back to 1 (off)."""
    try:
        f = int(factor)
    except (TypeError, ValueError):
        return 1
    if f <= 1:
        return 1
    f = min(f, MAX_FACTOR)
    # largest power of two <= f
    return 1 << (f.bit_length() - 1)


def interpolation_passes(factor: int) -> int:
    """Number of recursive 2x RIFE passes to reach `factor` (a power of two). 1x -> 0 passes,
    2x -> 1, 4x -> 2, 8x -> 3."""
    f = snap_factor(factor)
    return f.bit_length() - 1  # log2 of a power of two


def interpolated_frame_count(num_frames: int, factor: int) -> int:
    """Frames out after recursive 2x interpolation: each pass inserts one frame between every
    adjacent pair (N -> 2N-1), so after p passes a clip of N frames is (N-1)*2^p + 1. A 1-frame
    (or empty) clip is returned unchanged -- there is no pair to interpolate between."""
    f = snap_factor(factor)
    n = max(0, int(num_frames))
    if n <= 1 or f == 1:
        return n
    return (n - 1) * f + 1


def interpolated_fps(src_fps: int, factor: int) -> int:
    """Output fps after interpolation. Interpolation keeps the clip's DURATION fixed and multiplies
    the frame count, so the realized fps is the source fps times the (snapped) factor."""
    return max(1, int(src_fps)) * snap_factor(factor)


def output_fps(src_fps: int, params: "FinishParams") -> int:
    """The fps the finished clip is encoded at. Interpolation sets it to src*factor; an explicit
    `target_fps` (when > 0) overrides that as a hard cap on the realized rate, so a caller can ask
    for, say, exactly 30 fps regardless of the source. With interpolation off, the source fps is
    unchanged (a face-restore-only pass does not touch timing)."""
    if not params.interpolate:
        return max(1, int(src_fps))
    base = interpolated_fps(src_fps, params.factor)
    return min(base, params.target_fps) if params.target_fps and params.target_fps > 0 else base


# --------------------------------------------------------------------------- engine params

@dataclass
class FinishParams:
    """Engine knobs for one clip's finishing pass (the per-shot resolved form of the typed
    `config.FinishConfig`). Both passes default OFF here; `pipeline.finish_params_from` fills them
    from the tier config so a single warm worker finishes every clip the same way."""
    interpolate: bool = False
    factor: int = 2                 # 2 / 4 / 8; recursive RIFE doubling (snapped to a power of two)
    target_fps: int = 0             # 0 = src*factor; else a hard cap on the realized fps
    face_restore: bool = False
    face_restore_backend: str = "gfpgan"  # which restorer when face_restore is on (gfpgan / codeformer)
    face_fidelity: float = 0.7      # restorer balance: 0 = max restoration, 1 = max fidelity to input
    only_faces: bool = True         # restore detected faces only, leave the rest of the frame untouched

    @property
    def enabled(self) -> bool:
        """Whether this clip needs the GPU finish stage at all. When neither pass is on, the
        pipeline skips `finish_clip` entirely and the raw i2v clip is delivered as-is."""
        return bool(self.interpolate or self.face_restore)


# --------------------------------------------------------------------------- finish (GPU)

@dataclass
class FinishResult:
    """The outcome of finishing one clip: where the re-encoded clip landed, the source and output
    fps / frame counts, and which passes (interpolation, face restore) actually ran."""
    shot_id: str
    path: Path
    src_fps: int
    out_fps: int
    frames_in: int
    frames_out: int
    interpolated: bool
    face_restored: bool


def finish_clip(
    shot_id: str,
    in_path: Path,
    out_path: Path,
    server,
    *,
    params: FinishParams | None = None,
    progress_cb=None,
) -> FinishResult:
    """Finish one animated clip: decode -> (face restore) -> (interpolate) -> uniform re-encode.

    `server` is a `models.ModelServer` (provides the cached RIFE interpolator and face restorer).
    Heavy imports (torch / imageio / the restorer) are deferred so this module stays CPU-importable;
    the body is validated on a pod. `progress_cb(stage, done, total)` is optional and best-effort.

    Load-failure policy: a CONFIGURED pass whose model cannot load FAILS the render loud, it is not
    silently downgraded to a no-op. The whole point of this stage is the quality lift; a job that
    asked for smooth motion or a relocked face and got neither, with no error, is the worst outcome.
    (Per-FRAME hiccups inside a pass stay best-effort -- see `_restore_frame` -- so one bad frame
    does not sink a clip; but a missing MODEL is a deploy error worth surfacing.)

    Face restoration runs BEFORE interpolation deliberately: restore the real, model-generated frames
    (where the face detail lives), then let interpolation synthesize the in-between frames from
    already-cleaned anchors, so it never amplifies a restoration artifact across the inserted frames.

    Every clip is re-encoded to a uniform (codec, pix_fmt, fps) regardless of which passes ran, so
    the off-GPU `assemble` stream-copy concat stays valid across the whole render: a 1-frame or
    interpolation-skipped clip is encoded the SAME way as a fully interpolated one, so they never
    disagree on parameters and force the slow re-encode fallback.
    """
    cfg = params or FinishParams()
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio  # deferred: keep this module CPU-importable

    meta = iio.immeta(in_path, plugin="pyav")
    src_fps = int(round(meta.get("fps", 16))) or 16
    frames = list(iio.imiter(in_path, plugin="pyav"))  # list of HxWx3 uint8 arrays
    frames_in = len(frames)

    face_restored = False
    if cfg.face_restore and frames:
        # A configured restorer that cannot load is fatal (no `_safe` swallow): the loader RAISES and
        # we let it propagate. Restore the whole clip through ONE restorer instance for a stable
        # identity (see `_restore_clip`); per-frame independent restoration causes identity flicker.
        restorer = server.face_restorer(cfg.face_restore_backend)
        frames = _restore_clip(restorer, frames, cfg, progress_cb)
        face_restored = True

    interpolated = False
    if cfg.interpolate and len(frames) > 1:
        interp = server.frame_interpolator()  # configured-but-unloadable -> raises, not a silent skip
        passes = interpolation_passes(cfg.factor)
        for p in range(passes):
            frames = _interpolate_once(interp, frames)
            _tick(progress_cb, "interpolate", p + 1, passes)
        interpolated = passes > 0

    out_fps = output_fps(src_fps, cfg) if interpolated else src_fps
    _encode_uniform(frames, out_path, out_fps)
    return FinishResult(
        shot_id=shot_id, path=out_path, src_fps=src_fps, out_fps=out_fps,
        frames_in=frames_in, frames_out=len(frames),
        interpolated=interpolated, face_restored=face_restored,
    )


# --------------------------------------------------------------------------- GPU helpers (deferred)

def _interpolate_once(interp, frames):
    """One recursive 2x pass: insert an interpolated frame between every adjacent pair, so N frames
    become 2N-1 (the last real frame is appended unduplicated)."""
    out = []
    for a, b in zip(frames, frames[1:]):
        out.append(a)
        out.append(interp.interpolate(a, b))  # the RIFE midpoint frame
    out.append(frames[-1])
    return out


def _restore_clip(restorer, frames, cfg: FinishParams, progress_cb=None):
    """Face-restore a whole clip through ONE restorer instance, in clip order. Restoring every frame
    with the SAME loaded model (rather than re-loading or re-detecting per call) keeps the relocked
    identity consistent across the clip, which is what avoids the frame-to-frame identity flicker a
    naive per-frame restore produces. Per-frame errors stay best-effort (one bad frame passes
    through untouched) so a single detector miss does not drop the clip."""
    n = len(frames)
    out = []
    for i, f in enumerate(frames):
        out.append(_restore_frame(restorer, f, cfg))
        _tick(progress_cb, "face_restore", i + 1, n)
    return out


def _restore_frame(restorer, frame, cfg: FinishParams):
    """Run the blind face restorer over one frame's detected faces. Best-effort per frame: a frame
    the restorer chokes on passes through untouched rather than dropping the clip.

    The fidelity-to-backend-argument mapping (GFPGAN `weight` vs CodeFormer `w`) and the paste-back
    wiring live in the restorer wrapper (models.py), so this passes the uniform knobs only:
    `fidelity` and `only_faces`. `only_faces` is now LIVE -- the old `paste_back=not only_faces or
    True` was always True, making the flag dead; the wrapper honors it."""
    try:
        return restorer.restore(frame, fidelity=cfg.face_fidelity, only_faces=cfg.only_faces)
    except Exception:  # noqa: BLE001
        return frame


def _encode_uniform(frames, out_path: Path, fps: int) -> None:
    """Encode `frames` to `out_path` at a fixed (codec, pix_fmt, fps) so every finished clip in a
    render shares the same parameters and the downstream stream-copy concat in `assemble` stays
    valid (no per-clip re-encode fallback). H.264 / yuv420p is the broadly playable baseline the
    re-encode path in `assemble` also targets, so a finished clip and an assembler re-encode agree.
    Deferred imports keep this module CPU-importable."""
    import imageio.v3 as iio  # deferred

    iio.imwrite(
        str(out_path), list(frames), plugin="pyav", fps=max(1, int(fps)),
        codec="libx264", out_pixel_format="yuv420p",
    )


def _tick(progress_cb, stage: str, done: int, total: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(stage, done, total)
    except Exception:  # noqa: BLE001
        pass
