"""CPU tests for the standalone i2v_clip harness action (backend half of studio #81).

The GPU body (`i2v.animate`) and the `ModelServer` are faked, so `run_i2v_clip_job` is exercised
without a GPU, R2, or torch: we assert the R2 I/O contract (keyframe fetched, clip uploaded as
video/mp4 under the `_i2v` key) and that the typed tier config + per-shot overrides reach the
engine `I2VParams` intact (incl. the distill<->feature-cache invariant and the additive negative).
"""
from pathlib import Path

import pytest

from vivijure_backend import i2v as i2v_mod
from vivijure_backend.harness import handler as h
from vivijure_backend.harness.handler import HarnessError


class KFStore:
    """Serves a stand-in keyframe for any get; records puts (key, content_type)."""
    def __init__(self):
        self.gets: list[str] = []
        self.puts: list[str] = []          # artifact (put_file) keys only
        self.cts: dict[str, str | None] = {}
        self.order: list[tuple[str, str]] = []  # ("artifact"|"sidecar", key) -- write ordering
        self.bodies: dict[str, bytes] = {}      # put_bytes payloads (the .hash sidecar)

    def get_file(self, key, dest):
        self.gets.append(key)
        Path(dest).write_bytes(b"PNG")
        return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(), f"uploading a nonexistent file: {path}"
        self.puts.append(key)
        self.cts[key] = content_type
        self.order.append(("artifact", key))
        return key

    def put_bytes(self, data, key, *, content_type=None, metadata=None):
        self.bodies[key] = data
        self.order.append(("sidecar", key))
        return key


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace the GPU engine: animate() writes a stub mp4 and echoes its params back; ModelServer
    is a no-op. Returns a dict the test reads the captured call out of."""
    captured: dict = {}

    def fake_animate(scene, keyframe, prompt, server, out_path, *, params=None, progress_cb=None):
        Path(out_path).write_bytes(b"MP4")
        captured.update(scene=scene, keyframe=Path(keyframe), prompt=prompt,
                        params=params, progress_cb=progress_cb)
        return i2v_mod.I2VResult(
            shot_id=scene.id or "shot", path=Path(out_path),
            num_frames=params.num_frames, fps=params.fps,
            seconds=i2v_mod.clip_seconds(params.num_frames, params.fps), distilled=params.distill)

    monkeypatch.setattr(i2v_mod, "animate", fake_animate)
    monkeypatch.setattr("vivijure_backend.models.ModelServer", lambda *a, **k: object())
    return captured


def test_run_i2v_clip_job_io_contract(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "neon city", "shot_id": "shot 01",
           "prompt": "camera pushes in", "config": {"quality": "draft", "seed": 7}}
    out = h.run_i2v_clip_job(job, store=store, workdir=tmp_path, job_id="j1")

    # clip lands under the _i2v key with project/shot slugged and the right content type
    assert out["clip_key"] == "renders/neon_city/clips/shot_01_i2v.mp4"
    assert store.puts == ["renders/neon_city/clips/shot_01_i2v.mp4"]
    assert store.cts[out["clip_key"]] == "video/mp4"
    # pointer-only response carries the shot + realized clip facts
    assert out["shot_id"] == "shot 01"
    assert out["fps"] == 16 and out["num_frames"] == 81 and out["distilled"] is True
    assert out["seconds"] == pytest.approx(81 / 16, rel=1e-3)


def test_keyframe_key_defaults_to_project_shot_convention(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "neon", "shot_id": "shot_03",
           "prompt": "pan left", "config": {"quality": "draft"}}
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    assert store.gets == ["renders/neon/keyframes/shot_03.png"]  # default convention


def test_explicit_keyframe_key_is_used(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "neon", "shot_id": "s1", "prompt": "drift",
           "keyframe_key": "renders/neon/keyframes/hero.png", "config": {"quality": "draft"}}
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    assert store.gets == ["renders/neon/keyframes/hero.png"]


def test_i2v_clip_rejects_cross_project_keyframe_key(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "neon", "shot_id": "s1", "prompt": "drift",
           "keyframe_key": "renders/other/keyframes/hero.png", "config": {"quality": "draft"}}
    with pytest.raises(h.HarnessError, match="keyframe_key"):
        h.run_i2v_clip_job(job, store=store, workdir=tmp_path)


def test_draft_tier_drives_distill_and_overrides_reach_engine(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "move",
           "config": {"quality": "draft", "seed": 9, "flow_shift": 3.0,
                      "num_frames": 49, "fps": 24, "height": 720, "width": 1280}}
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    p = fake_engine["params"]
    assert p.distill is True and p.steps == 4 and p.guidance_scale == 1.0
    assert p.feature_cache is i2v_mod.FeatureCache.NONE   # never cache a 4-step render
    assert p.seed == 9 and p.flow_shift == 3.0
    assert p.num_frames == 49 and p.fps == 24             # 49 = 4*12+1, already valid
    assert p.height == 720 and p.width == 1280


def test_final_tier_full_step_with_cache(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "move",
           "config": {"quality": "final"}}
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    p = fake_engine["params"]
    assert p.distill is False and p.steps == 40
    assert p.feature_cache is i2v_mod.FeatureCache.MIXCACHE
    assert p.height is None and p.width is None           # null size -> follow keyframe


def test_num_frames_snapped_to_4k_plus_1(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "move",
           "config": {"quality": "draft", "num_frames": 50}}  # 50 -> 53 (4*13+1)
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    assert fake_engine["params"].num_frames == 53


def test_custom_negative_is_additive_over_anti_static_default(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "move",
           "config": {"quality": "draft", "negative_prompt": "rain"}}
    h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    neg = fake_engine["params"].negative_prompt
    assert neg.startswith("rain, ")                       # custom first
    assert "static" in neg and "frozen" in neg            # anti-static guard retained


def test_missing_prompt_raises(tmp_path, fake_engine):
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "config": {"quality": "draft"}}
    with pytest.raises(HarnessError, match="prompt is required"):
        h.run_i2v_clip_job(job, store=store, workdir=tmp_path)


def test_keyframe_fetch_failure_raises_harness_error(tmp_path, fake_engine):
    class BrokenStore(KFStore):
        def get_file(self, key, dest):
            raise RuntimeError("no such key")
    with pytest.raises(HarnessError, match="could not fetch keyframe"):
        h.run_i2v_clip_job(
            {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "x",
             "config": {"quality": "draft"}},
            store=BrokenStore(), workdir=tmp_path)


def test_job_supplied_keyframe_key_outside_renders_is_rejected(tmp_path, fake_engine):
    # A job-supplied keyframe_key is pinned to the render key map before any store I/O.
    store = KFStore()
    job = {"action": "i2v_clip", "project": "p", "shot_id": "s", "prompt": "push in",
           "keyframe_key": "loras/p/A/adapter.safetensors", "config": {"quality": "draft"}}
    with pytest.raises(HarnessError, match="keyframe_key"):
        h.run_i2v_clip_job(job, store=store, workdir=tmp_path)
    assert store.gets == []                            # rejected BEFORE any fetch


def test_finish_clip_key_outside_renders_is_rejected(tmp_path):
    # Same pin on the standalone finish job's input clip key.
    store = KFStore()
    job = {"action": "finish_clip", "project": "p", "shot_id": "s",
           "clip_key": "bundles/p/b.tar.gz", "config": {}}
    with pytest.raises(HarnessError, match="clip_key"):
        h.run_finish_job(job, store=store, workdir=tmp_path)
    assert store.gets == []                            # rejected BEFORE any fetch


# --- #583 provenance sidecar (run_finish_job) --------------------------------------------------

@pytest.fixture
def fake_finish(monkeypatch):
    """Replace the GPU finish body + ModelServer so run_finish_job runs on CPU. finish_clip writes a
    stub mp4 and returns a result with the fields the harness reads."""
    import types as _t
    from vivijure_backend import finish as _finish_mod

    def _fake_finish_clip(shot_id, in_path, out_path, server, params=None):
        Path(out_path).write_bytes(b"MP4")
        return _t.SimpleNamespace(interpolated=True, face_restored=False, out_fps=32, frames_out=160)

    monkeypatch.setattr(_finish_mod, "finish_clip", _fake_finish_clip)
    monkeypatch.setattr("vivijure_backend.models.ModelServer", lambda *a, **k: object())


def _finish_job(**over):
    return {"action": "finish_clip", "project": "neon", "shot_id": "shot_01",
            "clip_key": "renders/neon/clips/shot_01_i2v.mp4", "config": {"interpolation_factor": 2}, **over}


def test_run_finish_job_stamps_sidecar_after_artifact_with_output_hash(tmp_path, fake_finish):
    store = KFStore()
    out = h.run_finish_job(_finish_job(output_hash="deadbeef"), store=store, workdir=tmp_path)
    art = "renders/neon/clips/shot_01_finished.mp4"
    hkey = f"{art}.hash"
    assert out["clip_key"] == art
    # the sidecar value is the hash VERBATIM, at <artifact>.hash (the progress channel also uses
    # put_bytes, so assert on the .hash key specifically, not the whole bodies map)
    assert store.bodies.get(hkey) == b"deadbeef"
    # artifact FIRST, sidecar LAST (filter out the progress writes)
    assert [x for x in store.order if x[1] in (art, hkey)] == [("artifact", art), ("sidecar", hkey)]


def test_run_finish_job_writes_no_sidecar_without_output_hash(tmp_path, fake_finish):
    store = KFStore()
    h.run_finish_job(_finish_job(), store=store, workdir=tmp_path)
    assert not any(k.endswith(".hash") for k in store.bodies)  # legacy core -> no sidecar, safe re-run


def test_run_finish_job_sidecar_failure_never_fails_the_render(tmp_path, fake_finish):
    class BoomStore(KFStore):
        def put_bytes(self, data, key, *, content_type=None, metadata=None):
            if key.endswith(".hash"):
                raise RuntimeError("r2 down")  # only the sidecar fails, not the progress channel
            return super().put_bytes(data, key, content_type=content_type, metadata=metadata)
    store = BoomStore()
    out = h.run_finish_job(_finish_job(output_hash="deadbeef"), store=store, workdir=tmp_path)
    assert out["clip_key"].endswith("_finished.mp4")  # artifact up; sidecar miss is best-effort, no raise
