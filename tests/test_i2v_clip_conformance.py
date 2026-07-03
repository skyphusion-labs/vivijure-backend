"""Conformance guard (#129): the backend's `i2v_clip` contract MUST match the shared golden that the
local-gpu doors (vivijure-local-12gb / -16gb) also assert against. The two doors are hand-kept copies
with no shared source; the golden is the single reference so neither can silently drift (a drift = the
control plane builds a body one door rejects, or a keyframe 404 under a mis-slugged key). CPU-only:
the GPU engine + ModelServer are faked, exactly like test_i2v_clip.py.
"""
import json
from pathlib import Path

import pytest

from vivijure_backend import i2v as i2v_mod
from vivijure_backend.harness import handler as h, keys
from vivijure_backend.harness.handler import HarnessError

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "i2v_clip_contract.json").read_text())


class _KFStore:
    """Serves a stand-in keyframe for any get; records puts. Self-contained (no cross-test import)."""
    def __init__(self):
        self.puts: list[str] = []
        self.cts: dict[str, str | None] = {}

    def get_file(self, key, dest):
        Path(dest).write_bytes(b"PNG")
        return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        self.puts.append(key)
        self.cts[key] = content_type
        return key


@pytest.fixture
def _fake_engine(monkeypatch):
    def fake_animate(scene, keyframe, prompt, server, out_path, *, params=None, progress_cb=None):
        Path(out_path).write_bytes(b"MP4")
        return i2v_mod.I2VResult(
            shot_id=scene.id or "shot", path=Path(out_path),
            num_frames=params.num_frames, fps=params.fps,
            seconds=i2v_mod.clip_seconds(params.num_frames, params.fps), distilled=params.distill)

    monkeypatch.setattr(i2v_mod, "animate", fake_animate)
    monkeypatch.setattr("vivijure_backend.models.ModelServer", lambda *a, **k: object())


def test_slug_rule_matches_golden():
    # Assert the project slug via the PUBLIC key path (never reaching into _slug), for each golden case.
    for name, expected in GOLDEN["slug_examples"].items():
        assert keys.keyframe_key(name, "x") == f"renders/{expected}/keyframes/x.png", f"slug({name!r})"


def test_key_templates_match_golden():
    s = GOLDEN["sample"]
    assert keys.keyframe_key(s["project"], s["shot_id"]) == s["keyframe_key"]
    assert keys.i2v_clip_key(s["project"], s["shot_id"]) == s["clip_key"]


def test_prompt_is_required_per_golden(tmp_path, _fake_engine):
    assert GOLDEN["request"]["required"] == ["prompt"]
    store = _KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "config": {}}   # no prompt
    with pytest.raises(HarnessError):
        h.run_i2v_clip_job(job, store=store, workdir=tmp_path, job_id="jc0")


def test_result_pointer_fields_match_golden(tmp_path, _fake_engine):
    store = _KFStore()
    job = {"action": "i2v_clip", "project": "neon city", "shot_id": "shot 01",
           "prompt": "camera pushes in", "config": {"quality": "draft"}}
    out = h.run_i2v_clip_job(job, store=store, workdir=tmp_path, job_id="jc1")
    assert set(out.keys()) == set(GOLDEN["result_pointer_fields"])
    # and the clip landed under the golden clip-key template for this sample project/shot
    assert out["clip_key"] == keys.i2v_clip_key("neon city", "shot 01")
