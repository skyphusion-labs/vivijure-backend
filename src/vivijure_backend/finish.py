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

# Tri-state warm-worker cache for the NVENC probe: None = unprobed, True/False = h264_nvenc usable.
_NVENC = None


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

    Memory: the source clip is decoded to a list (bounded -- one i2v shot), but the interpolated
    frames are STREAMED into the encoder as they are produced (see `_finished_stream`), so an 8x pass
    never holds its full 8N-frame expansion in RAM at once.

    Every clip is re-encoded to a uniform (codec, pix_fmt, fps) regardless of which passes ran, so
    the off-GPU `assemble` stream-copy concat stays valid across the whole render: a 1-frame or
    interpolation-skipped clip is encoded the SAME way as a fully interpolated one, so they never
    disagree on parameters and force the slow re-encode fallback.

    Audio: the re-encode is fed a rawvideo stream and is therefore video-only, so if the SOURCE
    clip carries an audio track (dialogue shots lipsync before finish since core v0.17.0, so
    MuseTalk audio reaches this stage) it is muxed back onto the finished clip with a stream copy.
    RIFE keeps the wall-clock duration fixed, so the audio lines up 1:1. If that mux fails the shot
    FAILS loud (#245): it never silently ships a video-only clip when audio was present.
    """
    cfg = params or FinishParams()
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v3 as iio  # deferred: keep this module CPU-importable

    meta = iio.immeta(in_path, plugin="pyav")
    src_fps = int(round(meta.get("fps", 16))) or 16
    frames = list(iio.imiter(in_path, plugin="pyav"))  # source frames (bounded: one i2v clip)
    frames_in = len(frames)

    face_restored = False
    if cfg.face_restore and frames:
        # A configured restorer that cannot load is fatal (no `_safe` swallow): the loader RAISES and
        # we let it propagate. Restore the whole clip through ONE restorer instance for a stable
        # identity (see `_restore_clip`); per-frame independent restoration causes identity flicker.
        restorer = server.face_restorer(cfg.face_restore_backend)
        frames = _restore_clip(restorer, frames, cfg, progress_cb)
        face_restored = True

    interp = None
    passes = 0
    interpolated = False
    if cfg.interpolate and len(frames) > 1:
        interp = server.frame_interpolator()  # configured-but-unloadable -> raises, not a silent skip
        passes = interpolation_passes(cfg.factor)
        interpolated = passes > 0

    out_fps = output_fps(src_fps, cfg) if interpolated else src_fps
    frames_out = interpolated_frame_count(frames_in, cfg.factor) if interpolated else frames_in

    # The re-encode rebuilds the clip from a rawvideo stream, so it is video-only. Dialogue shots
    # now lipsync BEFORE finish (core v0.17.0 / vivijure#595: lipsync -> rife -> upscale), so
    # MuseTalk muxes the dialogue audio onto `in_path`; without this step it is silently dropped and
    # the shot -- plus every clip after it in the stream-copy concat -- plays silent (#240). RIFE
    # keeps wall-clock duration fixed, so the source audio lines up 1:1 with the finished video.
    has_audio = _source_has_audio(in_path)
    encode_target = out_path.with_name(out_path.stem + ".noaudio" + out_path.suffix) if has_audio else out_path
    _encode_uniform(_finished_stream(interp, frames, passes, progress_cb), encode_target, out_fps)
    if has_audio:
        # Honest failure (#245): a mux that fails RAISES; never ship a video-only clip when the
        # source carried audio (a silent drop is exactly the defect this fixes).
        _mux_audio(encode_target, in_path, out_path)
        encode_target.unlink(missing_ok=True)
    return FinishResult(
        shot_id=shot_id, path=out_path, src_fps=src_fps, out_fps=out_fps,
        frames_in=frames_in, frames_out=frames_out,
        interpolated=interpolated, face_restored=face_restored,
    )


# --------------------------------------------------------------------------- GPU helpers (deferred)

def _between(interp, a, b, depth):
    """Recursively bisect the gap between two adjacent frames to `depth` levels, yielding the
    interpolated frames strictly between a and b in playback order. `depth` == interpolation passes:
    each level inserts the RIFE midpoint then recurses into both halves, so a gap expands to
    2**depth - 1 frames -- identical to the old whole-list recursive doubling, done pair by pair."""
    if depth <= 0:
        return
    mid = interp.interpolate(a, b)  # the RIFE midpoint frame
    yield from _between(interp, a, mid, depth - 1)
    yield mid
    yield from _between(interp, mid, b, depth - 1)


def _finished_stream(interp, frames, passes, progress_cb=None):
    """Yield the finished frame sequence lazily: each source frame followed by the interpolated
    frames that bridge it to the next (none when interpolation is off / `passes` == 0). Only a small
    per-pair recursion window is live at once, so the 8x-expanded sequence is never all in RAM.
    Interpolation progress is emitted per source pair as it is consumed by the encoder."""
    total = max(0, len(frames) - 1)
    prev = None
    for i, cur in enumerate(frames):
        if prev is None:
            prev = cur
            continue
        yield prev
        if interp is not None and passes > 0:
            yield from _between(interp, prev, cur, passes)
            _tick(progress_cb, "interpolate", i, total)
        prev = cur
    if prev is not None:
        yield prev


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


def _probe_nvenc() -> bool:
    """True only if h264_nvenc is BOTH listed by ffmpeg AND actually encodes a test clip on this
    worker. An old NVENC API (e.g. an old ffmpeg build) can list the encoder yet fail at runtime on
    some GPU/driver combos, so a real test encode is the only honest check. Any failure means "not
    usable" and we fall back to libx264. Ported from the vivijure-upscale satellite."""
    import subprocess  # deferred: keep this module CPU-importable
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30)
        if "h264_nvenc" not in (enc.stdout or ""):
            return False
        test = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi",
             "-i", "testsrc=size=320x240:rate=10:duration=1",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        return test.returncode == 0
    except Exception:  # noqa: BLE001 -- any probe failure means "not usable", fall back honestly
        return False


def _nvenc_available() -> bool:
    """Whether this worker should encode with h264_nvenc, probed once and cached for the warm worker."""
    global _NVENC
    if _NVENC is None:
        _NVENC = _probe_nvenc()
    return _NVENC


def _encoder_argv(out_path: Path, width: int, height: int, fps: int, encoder: str) -> list[str]:
    """ffmpeg argv to encode a rawvideo rgb24 stream on stdin to a uniform H.264 / yuv420p mp4.
    `encoder` picks the GPU (h264_nvenc) or CPU (libx264) path; both target H.264 / yuv420p at the
    same fps so finished clips stay concat-compatible. Pure (no I/O) so it is unit-testable."""
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{int(width)}x{int(height)}", "-framerate", str(max(1, int(fps))), "-i", "-"]
    if encoder == "h264_nvenc":
        argv += ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19"]
    else:
        argv += ["-c:v", "libx264", "-crf", "19", "-preset", "fast"]
    argv += ["-pix_fmt", "yuv420p", str(out_path)]
    return argv


def _frame_bytes(frame):
    """One HxWx3 uint8 rgb frame as contiguous raw bytes for the encoder pipe."""
    import numpy as np  # deferred: keep this module CPU-importable
    return np.ascontiguousarray(frame, dtype=np.uint8).tobytes()


def _encode_uniform(frames, out_path: Path, fps: int) -> None:
    """Encode an iterable of HxWx3 uint8 rgb frames to `out_path` at a fixed (codec, pix_fmt, fps)
    through an ffmpeg rawvideo pipe, so every finished clip in a render shares the same parameters
    and the downstream stream-copy concat in `assemble` stays valid (no per-clip re-encode fallback).

    Uses h264_nvenc when this worker can actually encode with it (honest probe + libx264 fallback,
    the upscale satellite's pattern), which takes the finish encode off the CPU of the GPU-billed
    worker. H.264 / yuv420p is the broadly playable baseline `assemble`'s re-encode path also targets.

    Encoder choice is per-worker; within one render every clip is finished on the same worker (or a
    GPU-homogeneous endpoint), so all clips still share codec/pix_fmt/fps. In the pathological case of
    a heterogeneous pool splitting encoders, `assemble`'s existing re-encode fallback keeps the result
    correct -- it only costs the fast path, never correctness.

    Frames are streamed to the pipe as produced, so an 8x-interpolated clip is never fully resident."""
    import subprocess  # deferred: keep this module CPU-importable

    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration as e:
        raise RuntimeError("finish encode: no frames to write") from e
    height, width = first.shape[0], first.shape[1]
    encoder = "h264_nvenc" if _nvenc_available() else "libx264"
    proc = subprocess.Popen(_encoder_argv(out_path, width, height, fps, encoder), stdin=subprocess.PIPE)
    try:
        proc.stdin.write(_frame_bytes(first))
        for fr in frames:
            proc.stdin.write(_frame_bytes(fr))
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"finish encode failed (encoder={encoder}, rc={rc})")


def _source_has_audio(path: Path) -> bool:
    """Whether the source clip carries an audio stream that must survive the finish re-encode.
    Reuses the `assemble` probe so audio detection is defined once for the whole backend; the
    deferred import keeps this module CPU-importable."""
    from .assemble import probe_has_audio  # deferred: keep this module CPU-importable
    return probe_has_audio(path)


def _mux_audio_argv(video_path: Path, audio_src: Path, out_path: Path) -> list[str]:
    """ffmpeg argv to remux the finished (video-only) clip with the audio track from `audio_src`,
    copying both streams with NO re-encode. `-map 0:v` takes the finished video, `-map 1:a?` takes
    the source audio (the optional `?` so a source that unexpectedly lost its audio still yields a
    valid video-only file instead of a hard failure). Pure (no I/O) so it is unit-testable, the
    upscale satellite's audio-copy pattern."""
    return ["ffmpeg", "-v", "error", "-y",
            "-i", str(video_path), "-i", str(audio_src),
            "-map", "0:v", "-map", "1:a?", "-c", "copy", str(out_path)]


def _mux_audio(video_path: Path, audio_src: Path, out_path: Path) -> None:
    """Remux the finished video-only clip with `audio_src`'s audio (`-c copy`) into `out_path`.
    Honest failure (#245): a mux that fails RAISES rather than silently shipping a video-only clip
    when the source carried audio -- the exact silent drop this fixes."""
    import subprocess  # deferred: keep this module CPU-importable
    proc = subprocess.run(_mux_audio_argv(video_path, audio_src, out_path),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"finish audio mux failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")


def _tick(progress_cb, stage: str, done: int, total: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(stage, done, total)
    except Exception:  # noqa: BLE001
        pass
