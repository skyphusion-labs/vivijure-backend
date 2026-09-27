"""finish_clip and i2v_clip share the process-global ModelServer (worker._SERVER).

A warm worker keeps its models loaded across jobs (docs/architecture.md, "Warm worker, warm
models"). The standalone jobs used to build a fresh `ModelServer()` each, which reloaded the models
per job, ignored the job's `config.model` / `config.distill_model`, and skipped the warm-worker
divergence refusal that render jobs get. The GPU bodies and `ModelServer` are faked, exactly like
tests/test_i2v_clip.py.
"""
import types
from pathlib import Path

import pytest

from vivijure_backend import finish as finish_mod
from vivijure_backend import i2v as i2v_mod
from vivijure_backend import worker
from vivijure_backend.harness import handler as h
from vivijure_backend.models import DEFAULT_SPECS, ModelRole


class Store:
    def get_file(self, key, dest):
        Path(dest).write_bytes(b"X")
        return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        return key

    def put_bytes(self, data, key, *, content_type=None, metadata=None):
        return None


@pytest.fixture
def engines(monkeypatch):
    """Fake both GPU bodies; record the server each was handed and every ModelServer construction."""
    seen = {"finish": [], "i2v": [], "built": []}

    def fake_finish_clip(shot_id, in_path, out_path, server, params=None, deadline=None):
        seen["finish"].append(server)
        Path(out_path).write_bytes(b"MP4")
        return types.SimpleNamespace(interpolated=False, face_restored=False, out_fps=16, frames_out=8)

    def fake_animate(scene, keyframe, prompt, server, out_path, *, params=None, progress_cb=None):
        seen["i2v"].append(server)
        Path(out_path).write_bytes(b"MP4")
        return i2v_mod.I2VResult(shot_id="s", path=Path(out_path), num_frames=params.num_frames,
                                 fps=params.fps, seconds=1.0, distilled=params.distill)

    def fake_model_server(*a, **k):
        srv = types.SimpleNamespace(specs={**DEFAULT_SPECS, **(k.get("specs") or {})})
        seen["built"].append(srv)
        return srv

    monkeypatch.setattr(finish_mod, "finish_clip", fake_finish_clip)
    monkeypatch.setattr(i2v_mod, "animate", fake_animate)
    monkeypatch.setattr("vivijure_backend.models.ModelServer", fake_model_server)
    return seen


def _finish(tmp_path):
    return h.run_finish_job(
        {"action": "finish_clip", "project": "neon", "shot_id": "s1",
         "clip_key": "renders/neon/clips/s1_i2v.mp4", "config": {}},
        store=Store(), workdir=tmp_path)


def _i2v(tmp_path, config=None):
    return h.run_i2v_clip_job(
        {"action": "i2v_clip", "project": "neon", "shot_id": "s1", "prompt": "push in",
         "config": config or {}},
        store=Store(), workdir=tmp_path)


def test_finish_job_uses_the_warm_process_global_server(tmp_path, engines):
    warm = types.SimpleNamespace(specs=dict(DEFAULT_SPECS))
    worker._SERVER = warm
    _finish(tmp_path)
    assert engines["finish"] == [warm]
    assert engines["built"] == []


def test_i2v_job_uses_the_warm_process_global_server(tmp_path, engines):
    warm = types.SimpleNamespace(specs=dict(DEFAULT_SPECS))
    worker._SERVER = warm
    _i2v(tmp_path)
    assert engines["i2v"] == [warm]
    assert engines["built"] == []


def test_back_to_back_standalone_jobs_build_one_server(tmp_path, engines):
    _finish(tmp_path)
    _i2v(tmp_path)
    _finish(tmp_path)
    assert len(engines["built"]) == 1
    assert engines["finish"][0] is engines["i2v"][0] is engines["finish"][1] is worker._SERVER


def test_i2v_job_model_config_reaches_the_cold_server(tmp_path, engines):
    _i2v(tmp_path, {"model": "Wan-AI/custom-i2v", "distill_model": "Wan-AI/custom-distill"})
    (built,) = engines["built"]
    assert built.specs[ModelRole.I2V].repo_id == "Wan-AI/custom-i2v"
    assert built.specs[ModelRole.I2V_DISTILL].repo_id == "Wan-AI/custom-distill"
    # only the repo id is overridden; the rest of the spec (weight_name etc.) is preserved
    assert (built.specs[ModelRole.I2V_DISTILL].weight_name
            == DEFAULT_SPECS[ModelRole.I2V_DISTILL].weight_name)


def test_i2v_job_is_refused_when_the_warm_server_has_other_models(tmp_path, engines):
    worker._SERVER = types.SimpleNamespace(specs=dict(DEFAULT_SPECS))
    with pytest.raises(worker.ModelDivergenceError, match="Wan-AI/custom-i2v"):
        _i2v(tmp_path, {"model": "Wan-AI/custom-i2v"})
    assert engines["i2v"] == []          # refused BEFORE any GPU work


def test_i2v_job_rejects_a_disallowed_model_repo_before_building_a_server(tmp_path, engines):
    from vivijure_backend.models import InvalidModelRepoId
    with pytest.raises(InvalidModelRepoId):
        _i2v(tmp_path, {"model": "/etc/passwd"})
    assert engines["built"] == [] and worker._SERVER is None
