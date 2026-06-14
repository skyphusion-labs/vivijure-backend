"""Off-GPU finish: merge the per-shot clips into one video and report what came out.

The GPU's job ends at the last i2v clip. Stitching those clips into the final movie,
muxing an audio track, and measuring the result is plain container work that a CPU does
for cents, so it never belongs on a rented Blackwell. This module is that finish line: it
orders the shots, concatenates them with ffmpeg (stream-copy when the clips already share a
codec, which the single i2v pipeline guarantees, re-encoding only as a fallback), optionally
lays an audio track over the cut, and probes the duration and audio presence the control
plane reports back.

It also speaks the *offloaded* path: a render can stop after emitting per-shot clips plus a
small JSON manifest, and a separate CPU container picks that up and runs `assemble` from it.
So the same logic both finishes a render in-process and finishes one handed off to a worker.

Clean-room: the ffmpeg/ffprobe invocations are built from those tools' own documented
interfaces (the concat demuxer, the ffprobe stream/format printers), not from any prior
pipeline. The command builders are pure functions so they unit-test without spawning ffmpeg;
only the thin `assemble`/`probe_*` wrappers actually shell out, validated where ffmpeg lives.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

from .contract import Clip, Storyboard

# Default x264 settings for the re-encode fallback: visually lossless-ish, fast, broadly
# playable. Stream-copy is always tried first, so this only fires when the clips disagree.
_REENCODE_VIDEO = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]


@dataclass
class ClipInput:
    """One shot's rendered clip on disk, plus the order key the storyboard assigned it."""
    shot_id: str
    path: Path
    target_seconds: float | None = None


@dataclass
class AssembleResult:
    """What the finish produced. Mirrors the fields the control plane reads off a render."""
    output_path: Path
    seconds: float | None
    has_audio: bool
    clip_count: int

    def to_clips(self, key_for) -> list[Clip]:
        """Adapt back to contract `Clip`s using a caller-supplied shot_id -> object-key map."""
        return [Clip(shot_id=c, key=key_for(c)) for c in self._shot_ids]

    _shot_ids: list[str] = field(default_factory=list, repr=False)


# ------------------------------------------------------------------------------- ordering

def order_for_storyboard(clips: list[ClipInput], storyboard: Storyboard) -> list[ClipInput]:
    """Put the clips in storyboard scene order. Clips whose shot_id is not in the storyboard
    are dropped (a stray clip must not wander into the cut); scenes with no clip are simply
    absent. Order follows the storyboard, never the clip list's incidental order."""
    scene_ids = {s.id for s in storyboard.scenes}
    by_id = {c.shot_id: c for c in clips}
    stray = [c.shot_id for c in clips if c.shot_id not in scene_ids]
    if stray:
        log.warning("order_for_storyboard: dropping %d clip(s) not in storyboard: %s", len(stray), stray)
    return [by_id[s.id] for s in storyboard.scenes if s.id in by_id]


# ------------------------------------------------------------------------------- manifest

def build_manifest(clips: list[ClipInput], *, output_name: str, audio: str | None = None) -> dict:
    """The handoff record a CPU finisher consumes: the ordered clips (shot_id + path), the
    target name, and an optional audio track. Pure data; writing/merge happen elsewhere."""
    return {
        "version": 1,
        "output_name": output_name,
        "audio": audio,
        "clips": [
            {"shot_id": c.shot_id, "path": str(c.path),
             **({"target_seconds": c.target_seconds} if c.target_seconds is not None else {})}
            for c in clips
        ],
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_manifest(path: Path) -> tuple[list[ClipInput], str, str | None]:
    """Read a manifest back into (clips, output_name, audio) for `assemble`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = [ClipInput(shot_id=c["shot_id"], path=Path(c["path"]),
                       target_seconds=c.get("target_seconds")) for c in data.get("clips", [])]
    return clips, data["output_name"], data.get("audio")


# ----------------------------------------------------------------- ffmpeg/ffprobe builders

def concat_list_text(clips: list[ClipInput]) -> str:
    """The ffmpeg concat-demuxer playlist: one `file '<abs path>'` per clip. Single quotes in
    a path are escaped the way the demuxer requires ('\\'' ), so odd filenames can't break out
    of the quoting or inject a directive."""
    lines = []
    for c in clips:
        p = str(Path(c.path).resolve()).replace("'", "'\\''")
        lines.append(f"file '{p}'")
    return "\n".join(lines) + "\n"


def ffmpeg_concat_cmd(list_file: Path, out_path: Path, *,
                      reencode: bool = False, audio: Path | None = None) -> list[str]:
    """argv for concatenating the playlist into `out_path`. Stream-copies the video by default
    (the i2v clips share a codec); re-encodes only when asked. An audio track is mixed in as a
    second input, AAC-encoded, padded with `apad` and trimmed with `-shortest` so the output is
    exactly the picture length (a short bed gets trailing silence; a long one is trimmed)."""
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    if audio is not None:
        cmd += ["-i", str(audio)]
    cmd += _REENCODE_VIDEO if reencode else ["-c:v", "copy"]
    if audio is not None:
        # Video length wins: `apad` pads a short bed with trailing silence and `-shortest` then
        # trims the (now >= video) audio to the picture, so the film is never cut short by a
        # shorter track nor extended past its last frame by a longer one.
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-af", "apad", "-shortest"]
    cmd.append(str(out_path))
    return cmd


def ffprobe_duration_cmd(path: Path) -> list[str]:
    return ["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nokey=1:noprint_wrappers=1", str(path)]


def ffprobe_has_audio_cmd(path: Path) -> list[str]:
    return ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
            "stream=index", "-of", "csv=p=0", str(path)]


# ---------------------------------------------------------------------------- run (ffmpeg)

def probe_duration(path: Path) -> float | None:
    out = subprocess.run(ffprobe_duration_cmd(path), capture_output=True, text=True)
    if out.returncode != 0:
        log.warning("ffprobe duration failed for %s (rc=%d): %s", path, out.returncode, out.stderr.strip()[-500:])
        return None
    try:
        return round(float(out.stdout.strip()), 3)
    except (ValueError, AttributeError):
        return None


def probe_has_audio(path: Path) -> bool:
    out = subprocess.run(ffprobe_has_audio_cmd(path), capture_output=True, text=True)
    if out.returncode != 0:
        log.warning("ffprobe audio check failed for %s (rc=%d): %s", path, out.returncode, out.stderr.strip()[-500:])
        return False
    return bool(out.stdout.strip())


def assemble(clips: list[ClipInput], out_path: Path, *, audio: Path | None = None) -> AssembleResult:
    """Concatenate `clips` (already in the order they should play) into `out_path`, optionally
    muxing `audio`, and return the measured result. Tries a stream copy first and falls back to
    a re-encode if the copy fails (mismatched clip params). Raises on empty input or a missing
    clip file, on the CPU, before spawning ffmpeg."""
    if not clips:
        raise ValueError("no clips to assemble")
    missing = [str(c.path) for c in clips if not Path(c.path).is_file()]
    if missing:
        raise FileNotFoundError(f"clip file(s) not found: {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as lf:
        lf.write(concat_list_text(clips))
        list_file = Path(lf.name)
    try:
        copy = subprocess.run(ffmpeg_concat_cmd(list_file, out_path, reencode=False, audio=audio),
                              capture_output=True, text=True)
        if copy.returncode != 0:
            # Stream copy refused (clips disagree on codec/params): re-encode to a common form.
            redo = subprocess.run(ffmpeg_concat_cmd(list_file, out_path, reencode=True, audio=audio),
                                  capture_output=True, text=True)
            if redo.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed:\n{redo.stderr[-2000:]}")
    finally:
        list_file.unlink(missing_ok=True)

    return AssembleResult(
        output_path=out_path,
        seconds=probe_duration(out_path),
        has_audio=probe_has_audio(out_path),
        clip_count=len(clips),
        _shot_ids=[c.shot_id for c in clips],
    )
