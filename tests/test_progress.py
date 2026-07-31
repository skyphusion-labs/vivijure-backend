"""CPU tests for the structured progress channel: the emitter's R2 writes + snapshot, the
best-effort/never-fatal guarantee, project+job-id keying, the throttled train callback, the
optional RunPod hook, and an end-to-end run_job that drives a stubbed GPU pipeline. No R2, no GPU."""
import io
import json
import tarfile

import yaml

from vivijure_backend.config import RenderConfig
from vivijure_backend.contract import Bundle, RenderRequest
from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import run_job
from vivijure_backend.harness.progress import NullEmitter, ProgressEmitter, read_snapshot
from vivijure_backend.pipeline import GpuPipeline
from vivijure_backend.routing import QualityTier


class RecordingStore:
    """Captures put_bytes; can be told to fail every write to exercise best-effort."""
    def __init__(self, fail=False):
        self.objects: dict[str, bytes] = {}
        self.fail = fail
        self.writes = 0

    def put_bytes(self, data, key, *, content_type=None, metadata=None):
        self.writes += 1
        if self.fail:
            raise RuntimeError("R2 is down")
        self.objects[key] = data
        return key

    def get_bytes(self, key):
        return self.objects[key]


def _emitter(store=None, **kw):
    ticks = iter(range(1, 10_000))
    return ProgressEmitter(store, "neon rain", "job-1", clock=lambda: next(ticks), **kw)


# --------------------------------------------------------------------------- key layout

def test_progress_keys_are_scoped_by_project_and_job():
    assert keys.progress_log_key("neon rain", "job-1") == "renders/neon_rain/progress/job-1.ndjson"
    assert keys.progress_snapshot_key("neon rain", "job-1") == "renders/neon_rain/progress/job-1.json"


def test_concurrent_jobs_do_not_clobber():
    store = RecordingStore()
    ProgressEmitter(store, "neon", "A").emit("started")
    ProgressEmitter(store, "neon", "B").emit("started")
    # distinct job ids -> distinct keys, both present
    assert "renders/neon/progress/A.json" in store.objects
    assert "renders/neon/progress/B.json" in store.objects


# --------------------------------------------------------------------- writes + snapshot

def test_emit_writes_ndjson_stream_and_snapshot():
    store = RecordingStore()
    e = _emitter(store)
    e.emit("started", action="render")
    e.emit("train_done", slot="A")
    log_key = keys.progress_log_key("neon rain", "job-1")
    snap_key = keys.progress_snapshot_key("neon rain", "job-1")
    lines = store.objects[log_key].decode().strip().split("\n")
    assert len(lines) == 2                                   # append-style: grows each emit
    assert json.loads(lines[0])["event"] == "started"
    assert json.loads(lines[1])["slot"] == "A"
    snap = json.loads(store.objects[snap_key])
    assert snap["status"] == "running"
    assert snap["counts"]["train_done"] == 1
    assert snap["last_event"]["event"] == "train_done"


def test_snapshot_tracks_complete_and_error():
    store = RecordingStore()
    e = _emitter(store)
    e.emit("keyframe_done", shot="shot_01")
    e.complete(output_key="renders/x/full.mp4")
    assert json.loads(store.objects[keys.progress_snapshot_key("neon rain", "job-1")])["status"] == "complete"

    store2 = RecordingStore()
    e2 = _emitter(store2)
    e2.error("render", "boom at keyframe")
    snap = json.loads(store2.objects[keys.progress_snapshot_key("neon rain", "job-1")])
    assert snap["status"] == "error"
    assert snap["error"]["stage"] == "render" and "boom" in snap["error"]["message"]


# ----------------------------------------------------------------- best-effort guarantee

def test_a_failing_store_is_swallowed_never_raises():
    e = _emitter(RecordingStore(fail=True))
    e.emit("started")          # must not raise
    e.complete(output_key=None)
    e.error("render", "x")     # still must not raise


def test_emitter_with_no_store_is_a_noop_but_still_logs():
    logged = []
    e = _emitter(store=None, log=logged.append)
    e.emit("train_done", slot="A")
    assert logged and logged[0].startswith("@event train_done ")   # human @event line still emitted


def test_human_log_is_event_name_plus_json():
    logged = []
    _emitter(RecordingStore(), log=logged.append).emit("i2v_done", shot="shot_02")
    assert logged[0] == '@event i2v_done {"ts":1,"shot":"shot_02"}'


# ----------------------------------------------------------------------- option A hook

def test_runpod_hook_gets_each_snapshot_and_failures_are_swallowed():
    seen = []
    _emitter(RecordingStore(), on_progress=seen.append).emit("started")
    assert seen and seen[0]["status"] == "running"

    def boom(_snap):
        raise RuntimeError("runpod down")
    _emitter(RecordingStore(), on_progress=boom).emit("started")   # must not raise


# --------------------------------------------------------------------- throttled train cb

def test_train_step_cb_emits_a_throttled_event():
    store = RecordingStore()
    e = _emitter(store)
    cb = e.train_step_cb("A")
    cb(50, 1000, 0.1234)
    rec = json.loads(store.objects[keys.progress_log_key("neon rain", "job-1")].decode().strip())
    assert rec["event"] == "train_step" and rec["slot"] == "A"
    assert rec["step"] == 50 and rec["total"] == 1000


def test_null_emitter_train_cb_is_none():
    assert NullEmitter().train_step_cb("A") is None   # pipeline passes None -> lora_train no-ops


def test_i2v_step_cb_emits_every_step():
    store = RecordingStore()
    e = _emitter(store)
    cb = e.i2v_step_cb("shot_01")
    cb(12, 40)
    rec = json.loads(store.objects[keys.progress_log_key("neon rain", "job-1")].decode().strip())
    assert rec["event"] == "i2v_step" and rec["shot"] == "shot_01"
    assert rec["step"] == 12 and rec["total"] == 40
    # i2v_step is counted in the snapshot (like train_step), so /progress shows it climbing
    snap = json.loads(store.objects[keys.progress_snapshot_key("neon rain", "job-1")])
    assert snap["counts"]["i2v_step"] == 1


def test_null_emitter_i2v_cb_is_none():
    assert NullEmitter().i2v_step_cb("shot_01") is None  # pipeline passes None -> i2v omits the hook


def test_read_snapshot_round_trips_and_tolerates_missing():
    store = RecordingStore()
    _emitter(store).emit("started")
    assert read_snapshot(store, "neon rain", "job-1")["status"] == "running"
    assert read_snapshot(store, "neon rain", "absent") is None


# ----------------------------------------------------------------- run_job integration

STORYBOARD = {
    "title": "neon", "use_characters": ["A", "B"],
    "scenes": [
        {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
        {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
    ],
}


class FullStore(RecordingStore):
    """RecordingStore plus the get_file/put_file/put_dir_as_tar surface run_job needs."""
    def __init__(self, bundle_tar):
        super().__init__()
        self.bundle_tar = bundle_tar

    def get_file(self, key, dest):
        import shutil
        shutil.copy(self.bundle_tar, dest); return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        return key

    def put_dir_as_tar(self, src_dir, key, *, metadata=None):
        return key


class StubPipeline(GpuPipeline):
    def __init__(self, config):
        super().__init__(config=config, server=object())

    def _train_slot(self, char, out_dir):
        from pathlib import Path
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        f = Path(out_dir) / "l.safetensors"; f.write_bytes(b"x"); return f

    def _render_keyframe(self, scene, cast, storyboard, out_path, lora_paths):
        from pathlib import Path
        from vivijure_backend.keyframe import KeyframeResult
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return KeyframeResult(shot_id=scene.id, path=Path(out_path),
                              slots=list(scene.character_slots), multi_char=False, prompt="p",
                              seed=0, engine="single", identity_requested="instantid",
                              identity_path="lora_only")

    def _animate(self, scene, keyframe_path, prompt, out_path):
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x"); return Path(out_path)


def _bundle_tar(path):
    members = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {
            "A": {"name": "Vesper", "prompt": "teal"}, "B": {"name": "Rhode", "prompt": "orange"}}}).encode(),
        "characters/refs/A/ref_01.png": b"PNG-ish",
        "characters/refs/B/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_run_job_emits_the_full_stage_timeline(tmp_path):
    store = FullStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(
        {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "quality_tier": "draft", "render_overrides": {"finish_offloaded": True}},
        pipeline=StubPipeline(RenderConfig.for_tier(QualityTier.DRAFT)),
        store=store, workdir=tmp_path / "work", job_id="job-xyz", mirrored=True)

    assert res["clips"]                                   # the render still produced its result
    snap = json.loads(store.objects["renders/neon/progress/job-xyz.json"])
    assert snap["status"] == "complete"
    c = snap["counts"]
    assert c["train_done"] == 2                           # A + B
    assert c["keyframe_done"] == 2                        # both GENERATE
    assert c["i2v_done"] == 2                             # both animated
    assert c["upload_done"] >= 1                          # manifest + state
    # the event stream recorded mirror_done -> ... -> complete in order
    events = [json.loads(l)["event"]
              for l in store.objects["renders/neon/progress/job-xyz.ndjson"].decode().strip().split("\n")]
    assert events[0] == "started" and events[1] == "mirror_done"
    assert events.index("train_done") < events.index("keyframe_done") < events.index("i2v_done")
    assert events[-1] == "complete"


def test_run_job_records_error_event_then_reraises(tmp_path):
    import pytest
    store = FullStore(_bundle_tar(tmp_path / "b.tar.gz"))

    class Boom(StubPipeline):
        def _train_slot(self, char, out_dir):
            raise RuntimeError("gpu exploded")

    with pytest.raises(RuntimeError, match="gpu exploded"):
        run_job({"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz", "quality_tier": "draft"},
                pipeline=Boom(RenderConfig.for_tier(QualityTier.DRAFT)),
                store=store, workdir=tmp_path / "work", job_id="job-err")
    snap = json.loads(store.objects["renders/neon/progress/job-err.json"])
    assert snap["status"] == "error" and "gpu exploded" in snap["error"]["message"]


# ----------------------------------------------- pre-render gate failures (handler())

def test_handler_writes_mirror_error_snapshot_before_reraising(monkeypatch):
    import pytest
    from vivijure_backend.harness import handler as H, models_mirror, r2, pipeline_registry

    store = RecordingStore()
    monkeypatch.setattr(r2.R2Config, "from_env", lambda *a, **k: object())
    monkeypatch.setattr(r2, "R2", lambda cfg=None: store)
    monkeypatch.setattr(models_mirror, "ensure_models",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("rclone is not installed")))
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda: None)

    # The cold-start mirror fails BEFORE run_job's emitter would exist; the channel must still
    # record it. The render failure still propagates.
    with pytest.raises(RuntimeError, match="rclone is not installed"):
        H.handler({"input": {"project": "neon rain"}, "id": "job-mir"})

    snap = json.loads(store.objects["renders/neon_rain/progress/job-mir.json"])
    assert snap["status"] == "error"
    assert snap["error"]["stage"] == "mirror"
    assert "rclone" in snap["error"]["message"]


def test_handler_config_failure_surfaces_to_stdout_and_reraises(monkeypatch, capsys):
    import pytest
    from vivijure_backend.harness import handler as H, r2

    # A bad R2 config cannot be recorded TO R2 (R2 is the failure), so it degrades to stdout +
    # the RunPod hook, and still re-raises rather than running a render against no store.
    monkeypatch.setattr(r2.R2Config, "from_env",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("R2 config incomplete; missing env: R2_BUCKET")))
    with pytest.raises(RuntimeError, match="R2 config incomplete"):
        H.handler({"input": {"project": "neon"}, "id": "job-cfg"})
    out = capsys.readouterr().out
    assert "@event error" in out and '"stage":"config"' in out
