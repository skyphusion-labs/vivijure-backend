"""assemble.py's decision/command surface is pure and tested without ffmpeg; the actual
merge is exercised live when ffmpeg is on the box (skipped otherwise so the suite is portable)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from vivijure_backend.assemble import (
    ClipInput,
    assemble,
    build_manifest,
    concat_list_text,
    ffmpeg_concat_cmd,
    ffprobe_has_audio_cmd,
    load_manifest,
    order_for_storyboard,
    write_manifest,
)
from vivijure_backend.contract import Storyboard

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _clips(*ids):
    return [ClipInput(shot_id=i, path=Path(f"/clips/{i}.mp4"), target_seconds=5.0) for i in ids]


# --------------------------------------------------------------------------- ordering

def test_order_follows_storyboard_not_clip_list():
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "a"},
                                          {"id": "shot_02", "prompt": "b"},
                                          {"id": "shot_03", "prompt": "c"}]})
    out = order_for_storyboard(_clips("shot_03", "shot_01", "shot_02"), sb)
    assert [c.shot_id for c in out] == ["shot_01", "shot_02", "shot_03"]


def test_order_drops_stray_clips_and_skips_missing_scenes():
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "a"},
                                          {"id": "shot_02", "prompt": "b"}]})
    out = order_for_storyboard(_clips("shot_02", "ghost"), sb)  # ghost not in sb, shot_01 has no clip
    assert [c.shot_id for c in out] == ["shot_02"]


# --------------------------------------------------------------------------- manifest

def test_manifest_orders_and_carries_targets_and_audio():
    m = build_manifest(_clips("shot_01", "shot_02"), output_name="final.mp4", audio="music.m4a")
    assert m["output_name"] == "final.mp4"
    assert m["audio"] == "music.m4a"
    assert [c["shot_id"] for c in m["clips"]] == ["shot_01", "shot_02"]
    assert m["clips"][0]["target_seconds"] == 5.0


def test_manifest_omits_target_when_absent():
    m = build_manifest([ClipInput("s1", Path("/c/s1.mp4"))], output_name="o.mp4")
    assert "target_seconds" not in m["clips"][0]
    assert m["audio"] is None


def test_manifest_round_trips(tmp_path):
    m = build_manifest(_clips("shot_01", "shot_02"), output_name="final.mp4", audio="a.m4a")
    clips, name, audio = load_manifest(write_manifest(m, tmp_path / "manifest.json"))
    assert name == "final.mp4" and audio == "a.m4a"
    assert [c.shot_id for c in clips] == ["shot_01", "shot_02"]
    assert clips[0].target_seconds == 5.0


# ------------------------------------------------------------- ffmpeg/ffprobe builders

def test_concat_list_text_quotes_and_escapes():
    path = Path("/tmp/a'b.mp4")
    text = concat_list_text([ClipInput("s", path)])
    # the single quote is escaped so it can't break out of the demuxer's quoting.
    # Match the resolved path: on macOS /tmp is a symlink to /private/tmp.
    rendered = str(path.resolve()).replace("'", r"'\''")
    assert f"file '{rendered}'" in text
    assert text.endswith("\n")


def test_concat_cmd_stream_copies_by_default():
    cmd = ffmpeg_concat_cmd(Path("/l.txt"), Path("/o.mp4"))
    assert cmd[:6] == ["ffmpeg", "-y", "-f", "concat", "-safe", "0"]
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "libx264" not in cmd


def test_concat_cmd_reencode_uses_x264():
    cmd = ffmpeg_concat_cmd(Path("/l.txt"), Path("/o.mp4"), reencode=True)
    assert "libx264" in cmd
    assert "copy" not in cmd


def test_concat_cmd_with_audio_maps_and_shortens():
    cmd = ffmpeg_concat_cmd(Path("/l.txt"), Path("/o.mp4"), audio=Path("/a.m4a"))
    assert "/a.m4a" in cmd
    for flag in ("-map", "0:v:0", "1:a:0", "-c:a", "aac", "-shortest"):
        assert flag in cmd


def test_ffprobe_audio_cmd_selects_audio_stream():
    cmd = ffprobe_has_audio_cmd(Path("/o.mp4"))
    assert "-select_streams" in cmd and "a" in cmd


# ----------------------------------------------------------------------- guards (no ffmpeg)

def test_assemble_rejects_empty():
    with pytest.raises(ValueError, match="no clips"):
        assemble([], Path("/tmp/out.mp4"))


def test_assemble_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        assemble([ClipInput("s1", tmp_path / "nope.mp4")], tmp_path / "out.mp4")


# ---------------------------------------------------------------------- live ffmpeg merge

def _make_clip(path: Path, *, seconds: int, color: str, with_audio: bool = False):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=64x64:d={seconds}:r=24"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:d={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", "-t", str(seconds), str(path)]
    subprocess.run(cmd, capture_output=True, check=True)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_assemble_concatenates_and_measures(tmp_path):
    a, b = tmp_path / "shot_01.mp4", tmp_path / "shot_02.mp4"
    _make_clip(a, seconds=1, color="red")
    _make_clip(b, seconds=1, color="blue")
    res = assemble([ClipInput("shot_01", a), ClipInput("shot_02", b)], tmp_path / "final.mp4")
    assert res.output_path.is_file()
    assert res.clip_count == 2
    assert res.has_audio is False
    assert res.seconds == pytest.approx(2.0, abs=0.3)  # two 1s clips concatenated


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")
def test_assemble_muxes_audio_track(tmp_path):
    a = tmp_path / "shot_01.mp4"
    _make_clip(a, seconds=2, color="green")
    audio = tmp_path / "score.m4a"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:d=2",
                    "-t", "2", str(audio)], capture_output=True, check=True)
    res = assemble([ClipInput("shot_01", a)], tmp_path / "final.mp4", audio=audio)
    assert res.has_audio is True
    assert res.seconds == pytest.approx(2.0, abs=0.3)


def test_concat_cmd_pads_and_trims_audio_to_video_length():
    cmd = ffmpeg_concat_cmd(Path("list.txt"), Path("out.mp4"), audio=Path("bed.m4a"))
    # video length wins: apad pads a short bed, -shortest trims a long one
    assert "-af" in cmd and "apad" in cmd and "-shortest" in cmd
    assert cmd[cmd.index("-af") + 1] == "apad"
    # no audio -> no apad / -shortest
    plain = ffmpeg_concat_cmd(Path("list.txt"), Path("out.mp4"))
    assert "apad" not in plain and "-shortest" not in plain
