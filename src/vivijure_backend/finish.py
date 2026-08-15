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

import os
import time
from dataclasses import dataclass
from pathlib import Path

# RIFE interpolates recursively by doubling: one pass turns N frames into 2N-1, doubling the frame
# rate. A factor of 4 is two passes, 8 is three. So the only valid factors are powers of two, and we
# cap at 8x (16 -> 128 fps is already past any delivery need). 1x means "interpolation off".
VALID_FACTORS = (1, 2, 4, 8)
MAX_FACTOR = 8

# Tri-state warm-worker cache for the NVENC probe: None = unprobed, True/False = h264_nvenc usable.
_NVENC = None


# --------------------------------------------------------------------------- wall-clock guard

# ONE wall-clock budget for a WHOLE finish_clip invocation (#422). Before this, the only timeouts
# in this file were the two on the cached NVENC capability probe (_probe_nvenc), which is not the
# compute path: a stalled ffmpeg at the encode pipe or the mux, or a spinning RIFE recursion, held
# the GPU worker until something outside this repo gave up.
#
# WHY 420 SECONDS. Three bounds, and the default has to clear all three.
#
#   1. PLATFORM. The endpoint serving this image reports timeout: 0 (RunPod v2 per-request
#      execution timeout, milliseconds). What 0 MEANS is not settled: the queue-endpoint docs give
#      the execution timeout as default 600s, range 5s..7 days, while the Flash SDK gives
#      execution_timeout_ms default 0 = no limit -- a DIFFERENT product surface, and this endpoint
#      is type QUEUE. Control for the reading: of the 6 endpoints on the account, 5 report 0 and
#      vivijure-blender reports 600000, so 0 is a distinct stored state and the field can report a
#      real value. That control does not settle the SEMANTICS. Staying strictly under 600s makes
#      the guard fire under BOTH readings, so the open question cannot turn it into decoration.
#      See docs/finish-deadline.md.
#
#   2. CORE. vivijure-core retries a finish step FINISH_STEP_MAX_ATTEMPTS = 3 times, and a retry
#      moves its attempts counter, never the idx progress marker, so a guard G costs up to 3*G of
#      wall clock with no marker movement. Core raises the phase deadline to 3*G when that exceeds
#      its PHASE_HARD_DEADLINE_SECONDS floor of 5400s; 3*420 = 1260s leaves the floor untouched.
#
#   3. THE WORK. The largest clip that can reach this stage is 256 frames (config clamps num_frames
#      to 1..256; the default is 81, five seconds at 16 fps). 256 frames at 8x with face restore is
#      256 restores plus 255*7 = 1785 RIFE midpoints, which is minutes on the Hopper/Blackwell
#      pools this endpoint runs, not hours. 420s is multiples of the worst LEGAL configuration.
DEFAULT_MAX_SECONDS = 420
MAX_SECONDS_ENV = "VJ_FINISH_MAX_SECONDS"


def max_finish_seconds(env: dict | None = None) -> float:
    """The finish wall-clock budget in seconds: VJ_FINISH_MAX_SECONDS when it parses to a positive
    number, DEFAULT_MAX_SECONDS otherwise. Configurable rather than hardcoded because the ceiling a
    door enforces is deployment state, not a source constant (vivijure-core modules/types.ts says so
    for max_invocation_seconds). There is deliberately NO value that turns the guard OFF: junk, an
    empty string, zero and negatives all fall back to the default, because a guard an operator can
    disable by typo is the defect this closes, not a feature. Pure; env defaults to os.environ."""
    e = os.environ if env is None else env
    raw = e.get(MAX_SECONDS_ENV)
    if raw is None:
        return float(DEFAULT_MAX_SECONDS)
    try:
        v = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(DEFAULT_MAX_SECONDS)
    return v if v > 0 else float(DEFAULT_MAX_SECONDS)


class FinishDeadlineExceeded(BaseException):
    """The finish wall-clock budget ran out. Carries the stage, the elapsed seconds and the budget.

    Derives from BaseException, NOT Exception, ON PURPOSE. This path carries deliberate broad
    except-Exception handlers that are correct for their own job and would silently eat an expiry:
    _restore_frame (a frame the restorer chokes on passes through untouched), _tick (progress is
    best-effort), _probe_nvenc, and several in the harness. A guard a pre-existing handler swallows
    is a guard that is present, tested, and INERT on the exact path it exists for. Subclassing
    BaseException makes that structurally impossible instead of something every future handler on
    this path has to remember, which is the same reason KeyboardInterrupt and SystemExit sit there.

    It is caught in exactly ONE place, harness.handler.handler, which turns it into a structured
    soft degrade. It must never escape as a raise: a raise leaves no structured output, classifies
    deterministic in vivijure-core and fails the WHOLE render after the GPU time is already banked,
    which is strictly worse than the unbounded hang this replaces (that one the phase ceiling at
    least recovers)."""

    def __init__(self, stage: str, elapsed: float, budget: float) -> None:
        self.stage = str(stage)
        self.elapsed = float(elapsed)
        self.budget = float(budget)
        # The consumer caps the degrade reason at 120 characters (vivijure-cf
        # modules/_shared/finish-soft-degrade.ts degradeReason), so the guard name, the stage and
        # both numbers have to fit inside the first 120. This form is ~70; tested.
        self.reason = (f"finish_clip deadline: {self.elapsed:.1f}s of {self.budget:.0f}s"
                       f" budget at stage {self.stage}")
        super().__init__(self.reason)


@dataclass
class Deadline:
    """One monotonic budget shared by every stage of one finish_clip invocation.

    ONE deadline established at entry and checked between stages, rather than a per-step timeout:
    per-step timeouts multiply, so N steps of T each bound the invocation at N*T, and the number a
    door could honestly declare is the sum nobody computes. A shared deadline means the invocation
    ceiling IS the budget.

    time.monotonic, never time.time: a wall-clock step (ntp, a suspended worker) must not shorten or
    extend a budget."""

    budget_seconds: float
    started_at: float

    @classmethod
    def start(cls, budget_seconds: float | None = None, env: dict | None = None) -> "Deadline":
        """Open a budget now. budget_seconds overrides the env/default resolution (tests, callers
        that already resolved it); everything else reads max_finish_seconds."""
        b = max_finish_seconds(env) if budget_seconds is None else float(budget_seconds)
        return cls(budget_seconds=b, started_at=time.monotonic())

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining(self) -> float:
        """Seconds left, floored at 0 so it is always a legal subprocess timeout / Timer interval."""
        return max(0.0, self.budget_seconds - self.elapsed())

    def expired(self) -> bool:
        return self.elapsed() >= self.budget_seconds

    def check(self, stage: str) -> None:
        """Raise FinishDeadlineExceeded if the budget is gone. Called at every stage boundary and
        inside both per-frame loops, so the reported stage names where the time actually went."""
        if self.expired():
            raise FinishDeadlineExceeded(stage, self.elapsed(), self.budget_seconds)


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
    deadline: "Deadline | None" = None,
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

    if deadline is not None:
        deadline.check("start")
    meta = iio.immeta(in_path, plugin="pyav")
    src_fps = int(round(meta.get("fps", 16))) or 16
    # Consume the decode ITERATOR rather than list()-ing it, so the budget is checked per decoded
    # frame. list() is ONE un-interruptible call: a decoder that stalls between frames would sit
    # inside it with no check reachable. A stall INSIDE a single frame decode is still not bounded
    # here; see docs/finish-deadline.md, coverage row decode.
    frames = []
    for _fr in iio.imiter(in_path, plugin="pyav"):  # source frames (bounded: one i2v clip)
        frames.append(_fr)
        if deadline is not None:
            deadline.check("decode")
    frames_in = len(frames)

    face_restored = False
    if cfg.face_restore and frames:
        # A configured restorer that cannot load is fatal (no `_safe` swallow): the loader RAISES and
        # we let it propagate. Restore the whole clip through ONE restorer instance for a stable
        # identity (see `_restore_clip`); per-frame independent restoration causes identity flicker.
        if deadline is not None:
            deadline.check("restorer_load")
        restorer = server.face_restorer(cfg.face_restore_backend)
        frames = _restore_clip(restorer, frames, cfg, progress_cb, deadline=deadline)
        face_restored = True

    interp = None
    passes = 0
    interpolated = False
    if cfg.interpolate and len(frames) > 1:
        if deadline is not None:
            deadline.check("interpolator_load")
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
    has_audio = _source_has_audio(in_path, deadline=deadline)
    encode_target = out_path.with_name(out_path.stem + ".noaudio" + out_path.suffix) if has_audio else out_path
    _encode_uniform(_finished_stream(interp, frames, passes, progress_cb, deadline=deadline),
                    encode_target, out_fps, deadline=deadline)
    if has_audio:
        # Honest failure (#245): a mux that fails RAISES; never ship a video-only clip when the
        # source carried audio (a silent drop is exactly the defect this fixes).
        _mux_audio(encode_target, in_path, out_path, deadline=deadline)
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


def _finished_stream(interp, frames, passes, progress_cb=None, deadline=None):
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
            # Per SOURCE PAIR: one pair is at most 2**passes - 1 RIFE calls (7 at the 8x cap), so
            # this bounds the recursion without a check inside _between. A single
            # interp.interpolate call that never returns is NOT bounded here
            # (docs/finish-deadline.md, row interpolate). The check sits BEFORE the yield-from and
            # never inside _tick, which swallows every exception from the progress callback.
            if deadline is not None:
                deadline.check("interpolate")
            yield from _between(interp, prev, cur, passes)
            _tick(progress_cb, "interpolate", i, total)
        prev = cur
    if prev is not None:
        yield prev


def _restore_clip(restorer, frames, cfg: FinishParams, progress_cb=None, deadline=None):
    """Face-restore a whole clip through ONE restorer instance, in clip order. Restoring every frame
    with the SAME loaded model (rather than re-loading or re-detecting per call) keeps the relocked
    identity consistent across the clip, which is what avoids the frame-to-frame identity flicker a
    naive per-frame restore produces. Per-frame errors stay best-effort (one bad frame passes
    through untouched) so a single detector miss does not drop the clip."""
    n = len(frames)
    out = []
    for i, f in enumerate(frames):
        # BEFORE _restore_frame, never inside it: _restore_frame swallows Exception by design so a
        # bad frame passes through untouched, and a check placed inside it would be eaten by that
        # handler. (FinishDeadlineExceeded is a BaseException so it would survive anyway; keeping
        # the check outside means the guard does not DEPEND on that.)
        if deadline is not None:
            deadline.check("face_restore")
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


def _encode_uniform(frames, out_path: Path, fps: int, deadline=None) -> None:
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

    import threading  # deferred: only the watchdog below needs it

    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration as e:
        raise RuntimeError("finish encode: no frames to write") from e
    height, width = first.shape[0], first.shape[1]
    encoder = "h264_nvenc" if _nvenc_available() else "libx264"
    proc = subprocess.Popen(_encoder_argv(out_path, width, height, fps, encoder), stdin=subprocess.PIPE)
    # A per-frame check CANNOT interrupt a BLOCKED write. If ffmpeg stalls, its stdin pipe buffer
    # fills and proc.stdin.write never returns, so control never reaches the next check; proc.wait
    # has the same shape. ONE watchdog covers both: at expiry it kills ffmpeg, the blocked write
    # fails BrokenPipeError and wait returns. Without it the guard would be present, tested and
    # INERT on the single most likely hang in this file, which is the defect #422 is about.
    watchdog = None
    if deadline is not None:
        watchdog = threading.Timer(deadline.remaining(), proc.kill)
        watchdog.daemon = True
        watchdog.start()
    try:
        try:
            proc.stdin.write(_frame_bytes(first))
            for fr in frames:
                if deadline is not None:
                    deadline.check("encode")
                proc.stdin.write(_frame_bytes(fr))
        except BrokenPipeError:
            # A watchdog-killed ffmpeg looks exactly like a crashed one. If the budget is gone it
            # WAS the watchdog and the expiry is the honest report; otherwise re-raise the real
            # encoder failure unchanged.
            if deadline is not None:
                deadline.check("encode")
            raise
        finally:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
            rc = proc.wait()
    finally:
        if watchdog is not None:
            watchdog.cancel()
    # The expiry OUTRANKS the exit code: a killed encoder reports a nonzero rc, and reporting that
    # as an encode failure would hide the cause AND fail the render loud instead of degrading.
    if deadline is not None:
        deadline.check("encode")
    if rc != 0:
        raise RuntimeError(f"finish encode failed (encoder={encoder}, rc={rc})")


def _source_has_audio(path: Path, deadline=None) -> bool:
    """Whether the source clip carries an audio stream that must survive the finish re-encode.
    Reuses the `assemble` probe so audio detection is defined once for the whole backend; the
    deferred import keeps this module CPU-importable.

    The probe is an ffprobe subprocess and had NO timeout, so it was a FIFTH untimed subprocess on
    this path -- reached from here, which is why the four counted inside finish.py missed it. It
    gets the remaining budget. probe_has_audio keeps its no-timeout DEFAULT, so the render/assemble
    callers are byte-identical (#422 D)."""
    import subprocess  # deferred: keep this module CPU-importable
    from .assemble import probe_has_audio  # deferred: keep this module CPU-importable
    if deadline is None:
        return probe_has_audio(path)
    deadline.check("probe_audio")
    try:
        return probe_has_audio(path, timeout=deadline.remaining())
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child on timeout, so the worker is released either way. The
        # budget is what ran out; report THAT, not a probe failure. NEVER return False here: a
        # false negative silently drops the dialogue track (#240).
        deadline.check("probe_audio")
        raise


def _mux_audio_argv(video_path: Path, audio_src: Path, out_path: Path) -> list[str]:
    """ffmpeg argv to remux the finished (video-only) clip with the audio track from `audio_src`,
    copying both streams with NO re-encode. `-map 0:v` takes the finished video, `-map 1:a?` takes
    the source audio (the optional `?` so a source that unexpectedly lost its audio still yields a
    valid video-only file instead of a hard failure). Pure (no I/O) so it is unit-testable, the
    upscale satellite's audio-copy pattern."""
    return ["ffmpeg", "-v", "error", "-y",
            "-i", str(video_path), "-i", str(audio_src),
            "-map", "0:v", "-map", "1:a?", "-c", "copy", str(out_path)]


def _mux_audio(video_path: Path, audio_src: Path, out_path: Path, deadline=None) -> None:
    """Remux the finished video-only clip with `audio_src`'s audio (`-c copy`) into `out_path`.
    Honest failure (#245): a mux that fails RAISES rather than silently shipping a video-only clip
    when the source carried audio -- the exact silent drop this fixes."""
    import subprocess  # deferred: keep this module CPU-importable
    if deadline is None:
        proc = subprocess.run(_mux_audio_argv(video_path, audio_src, out_path),
                              capture_output=True, text=True)
    else:
        deadline.check("mux_audio")
        try:
            proc = subprocess.run(_mux_audio_argv(video_path, audio_src, out_path),
                                  capture_output=True, text=True, timeout=deadline.remaining())
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child, so a stalled mux never holds the worker past the
            # budget. Report the expiry, not a mux failure.
            deadline.check("mux_audio")
            raise
    if proc.returncode != 0:
        raise RuntimeError(f"finish audio mux failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")


def _tick(progress_cb, stage: str, done: int, total: int) -> None:
    if progress_cb is None:
        return
    try:
        progress_cb(stage, done, total)
    except Exception:  # noqa: BLE001
        pass
