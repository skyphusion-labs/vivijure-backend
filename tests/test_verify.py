"""CPU tests for the pod-side verify @event emitter (vivijure_backend.verify): the run-scoped R2
channel (summary.json + events.ndjson), the stdout mirror, the pure probe/metric helpers, the
run_verify orchestration + failure paths, the VJ_VERIFY gate on main(), and -- the load-bearing
cross-module test -- that a run_verify channel feeds straight into runpod_verify.evaluate and PASSES
with zero prose parsing. No GPU, no R2, no network."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vivijure_backend import verify
from vivijure_backend.harness import keys


def _load_runpod_verify():
    """Load the harness module the same way its own test does (it lives in deploy/, not the src
    package), so we can prove the emitter's output is exactly what evaluate() reads."""
    path = Path(__file__).resolve().parents[1] / "deploy" / "runpod_verify.py"
    spec = importlib.util.spec_from_file_location("runpod_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class RecordingStore:
    """Captures put_bytes; can be told to fail every write to exercise the best-effort guarantee."""
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


def _clock():
    ticks = iter(range(1, 100_000))
    return lambda: float(next(ticks))


def _emitter(store=None, logs=None):
    return verify.VerifyEmitter(store, "run-abc", log=(logs.append if logs is not None else (lambda _s: None)),
                                clock=_clock())


# --------------------------------------------------------------------------- key layout

def test_verify_keys_are_run_scoped():
    assert keys.verify_events_key("run-abc") == "verify/run-abc/events.ndjson"
    assert keys.verify_summary_key("run-abc") == "verify/run-abc/summary.json"


def test_verify_key_prefix_override_and_slug_safety():
    assert keys.verify_summary_key("r 1", prefix="stage") == "stage/r_1/summary.json"
    # a run id that tries to smuggle a slash cannot scatter the channel across prefixes
    assert keys.verify_events_key("a/b").startswith("verify/a_b/")
    assert keys.verify_summary_key("   ") == "verify/unkeyed/summary.json"


# --------------------------------------------------------------------------- env gate

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("no", False), ("off", False),
])
def test_is_enabled(val, expected):
    assert verify.is_enabled({"VJ_VERIFY": val}) is expected


def test_is_enabled_unset():
    assert verify.is_enabled({}) is False


def test_env_helpers():
    assert verify.run_id_from({"VJ_VERIFY_RUN_ID": " r9 "}) == "r9"
    assert verify.run_id_from({}) is None
    assert verify.key_prefix_from({"VJ_VERIFY_KEY_PREFIX": "gate"}) == "gate"
    assert verify.key_prefix_from({}) == "verify"
    assert verify.baseline_from({"VJ_SHARPNESS_BASELINE": "42.5"}) == 42.5
    assert verify.baseline_from({"VJ_SHARPNESS_BASELINE": "nan-bad"}) == verify.DEFAULT_SHARPNESS_BASELINE
    assert verify.baseline_from({}) == verify.DEFAULT_SHARPNESS_BASELINE


# --------------------------------------------------------------------------- pure probe + metrics

def test_gpu_probe_payload_defaults_and_passthrough():
    p = verify.gpu_probe_payload({"torch_cuda": True, "device_name": "H200", "extra": "dropped"})
    assert set(p) == set(verify.GPU_PROBE_KEYS)          # exactly the contract keys, no extras
    assert p["torch_cuda"] is True and p["device_name"] == "H200"
    assert p["kernel_ok"] is False and p["vram_free_gb"] == 0.0


def test_frame_sharpness_flat_is_zero_textured_is_positive():
    flat = np.full((16, 16), 128.0)
    assert verify.frame_sharpness(flat) == 0.0
    rng = np.arange(256, dtype=np.float64).reshape(16, 16)
    textured = (rng * 7) % 255
    assert verify.frame_sharpness(textured) > 0.0


def test_frame_sharpness_reduces_color_and_guards_tiny():
    color = np.zeros((8, 8, 3), dtype=np.float64)
    color[::2, :, 0] = 255.0                              # sharp stripes in one channel
    assert verify.frame_sharpness(color) > 0.0
    assert verify.frame_sharpness(np.zeros((2, 2))) == 0.0   # too small -> 0, never raises


# --------------------------------------------------------------------------- emitter channel

def test_emit_writes_summary_and_ndjson_and_mirrors_stdout():
    store, logs = RecordingStore(), []
    em = _emitter(store, logs)
    em.gpu_probe({"torch_cuda": True})
    em.first_frame(5.0)

    summary = json.loads(store.objects[keys.verify_summary_key("run-abc")])
    assert summary["schema"] == verify.SCHEMA
    assert summary["run_id"] == "run-abc"
    assert summary["status"] == "running"
    assert [e["event"] for e in summary["events"]] == ["gpu_probe", "first_frame"]
    assert [e["seq"] for e in summary["events"]] == [0, 1]        # monotonic seq
    assert summary["events"][0]["payload"]["torch_cuda"] is True

    ndjson = store.objects[keys.verify_events_key("run-abc")].decode()
    lines = [json.loads(ln) for ln in ndjson.splitlines()]
    assert [ln["event"] for ln in lines] == ["gpu_probe", "first_frame"]

    # stdout mirror is the byte-identical wire runpod_verify.parse_events reads
    assert logs[0].startswith("@event gpu_probe ")
    assert logs[1].startswith("@event first_frame ")


def test_complete_and_error_set_terminal_status():
    store = RecordingStore()
    em = _emitter(store)
    em.complete("renders/x/full.mp4", 4096)
    s = json.loads(store.objects[keys.verify_summary_key("run-abc")])
    assert s["status"] == "complete" and s["error"] is None
    assert s["events"][-1]["payload"] == {"output_key": "renders/x/full.mp4", "clip_bytes": 4096}

    em2 = _emitter(RecordingStore())
    rec = em2.error("render", "boom")
    assert em2._summary["status"] == "error"
    assert em2._summary["error"] == {"stage": "render", "message": "boom"}
    assert rec["event"] == "error"


def test_channel_writes_are_best_effort():
    # A failing store must never raise out of emit -- the render can't die on a logging hiccup.
    em = _emitter(RecordingStore(fail=True))
    em.gpu_probe({"torch_cuda": True})
    em.complete("k", 1)                                   # no exception
    assert em._summary["status"] == "complete"
    # And a None store is a valid no-op sink (stdout still mirrors).
    verify.VerifyEmitter(None, "r", clock=_clock()).first_frame(1.0)


# --------------------------------------------------------------------------- channel readers

def test_events_from_summary_and_ndjson_round_trip():
    store = RecordingStore()
    em = _emitter(store)
    em.gpu_probe({"torch_cuda": True})
    em.complete("k", 9)
    raw_summary = store.objects[keys.verify_summary_key("run-abc")]
    raw_ndjson = store.objects[keys.verify_events_key("run-abc")]

    ev_s = verify.events_from_summary(raw_summary)
    ev_n = verify.events_from_ndjson(raw_ndjson)
    assert ev_s == ev_n
    assert ev_s[0][0] == "gpu_probe" and ev_s[0][1]["torch_cuda"] is True
    assert ev_s[-1] == ("complete", {"output_key": "k", "clip_bytes": 9})


def test_readers_tolerate_garbage():
    assert verify.events_from_summary(b"not json") == []
    assert verify.events_from_summary(json.dumps([1, 2]).encode()) == []      # not a dict
    assert verify.events_from_ndjson("junk\n{bad}\n") == []


# --------------------------------------------------------------------------- run_verify orchestration

def _pass_facts():
    return {"torch_cuda": True, "kernel_ok": True, "vj_baked": True, "weights_on_disk": True,
            "vram_free_gb": 120.0, "vram_total_gb": 141.0, "device_name": "NVIDIA H200"}


def _good_render(seconds=5.0, value=100.0):
    def render(em):
        em.first_frame(seconds)
        return verify.RenderOutcome(output_key="renders/verify/full.mp4", clip_bytes=500_000,
                                    sharpness_value=value)
    return render


def test_run_verify_happy_path_event_order_and_status():
    store = RecordingStore()
    em = verify.run_verify(store, run_id="run-abc", render=_good_render(),
                           probe=_pass_facts, baseline=100.0, clock=_clock())
    names = [e["event"] for e in em._summary["events"]]
    assert names == ["gpu_probe", "first_frame", "sharpness", "complete"]
    assert em._summary["status"] == "complete"


def test_run_verify_computes_sharpness_from_frame_when_value_absent():
    def render(em):
        em.first_frame(2.0)
        frame = (np.arange(256, dtype=np.float64).reshape(16, 16) * 7) % 255
        return verify.RenderOutcome(output_key="k", clip_bytes=1, first_frame_array=frame)
    em = verify.run_verify(RecordingStore(), run_id="r", render=render, probe=_pass_facts, clock=_clock())
    sharp = [e for e in em._summary["events"] if e["event"] == "sharpness"][0]
    assert sharp["payload"]["value"] > 0.0


def test_run_verify_probe_crash_records_error_and_skips_render():
    calls = {"render": 0}
    def render(em):
        calls["render"] += 1
        return verify.RenderOutcome("k", 1)
    def bad_probe():
        raise RuntimeError("no cuda")
    em = verify.run_verify(RecordingStore(), run_id="r", render=render, probe=bad_probe, clock=_clock())
    assert em._summary["status"] == "error"
    assert em._summary["error"]["stage"] == "gpu_probe"
    assert calls["render"] == 0                            # render never runs if the probe dies


def test_run_verify_render_crash_records_error():
    def render(em):
        raise RuntimeError("kernel OOM")
    em = verify.run_verify(RecordingStore(), run_id="r", render=render, probe=_pass_facts, clock=_clock())
    assert em._summary["status"] == "error"
    assert em._summary["error"]["stage"] == "render"


# --------------------------------------------------------------------------- pod render seam (registration)

def test_pod_draft_render_registers_pipeline_before_handling(monkeypatch):
    """Regression lock for the S1 watched-pod bug: `python -m vivijure_backend.verify` is a DIFFERENT
    entrypoint than the worker, so calling the inner harness handler directly left the GPU pipeline
    registry EMPTY and the render died with "no GPU Pipeline registered". The seam now routes through
    `worker.handler`, which registers the per-job pipeline first. This proves registration happens
    before the render runs, on CPU, by stubbing the deferred GPU/R2/clip pieces."""
    from pathlib import Path
    from vivijure_backend import worker as wmod
    from vivijure_backend.harness import handler as hmod
    from vivijure_backend.harness import pipeline_registry
    from vivijure_backend.harness import r2 as r2mod

    monkeypatch.setattr(pipeline_registry, "_PIPELINE", None)   # start with an empty registry
    sentinel = object()
    monkeypatch.setattr(wmod, "build_pipeline", lambda req: sentinel)  # no ModelServer/torch

    seen = {}
    def fake_harness_handler(job):
        # the whole point: by the time the render runs, worker.handler must have registered a pipeline
        seen["registered"] = pipeline_registry.get_pipeline()
        return {"output_key": "renders/verify/full.mp4", "clip_bytes": 500_000}
    monkeypatch.setattr(hmod, "handler", fake_harness_handler)

    class FakeR2:
        def __init__(self, *a, **k):
            pass
        def get_file(self, key, dest):
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            Path(dest).write_bytes(b"clip-bytes")
            return dest
    monkeypatch.setattr(r2mod, "R2", FakeR2)
    monkeypatch.setattr(r2mod, "R2Config", type("C", (), {"from_env": staticmethod(lambda e: object())}))
    monkeypatch.setattr(verify, "_first_frame_array", lambda p: None)
    monkeypatch.setattr(verify, "_first_frame_seconds", lambda store, project, job_id: 3.0)

    em = verify.VerifyEmitter(None, "run-x", clock=_clock())
    env = {"VJ_VERIFY_BUNDLE_KEY": "bundles/x.tgz", "VJ_VERIFY_PROJECT": "verify"}
    outcome = verify._pod_draft_render(em, env)

    assert seen["registered"] is sentinel                      # pipeline WAS registered before render
    assert outcome.output_key == "renders/verify/full.mp4"
    assert outcome.clip_bytes == len(b"clip-bytes")            # measured from the pulled clip


def test_pod_draft_render_requires_bundle_key():
    em = verify.VerifyEmitter(None, "run-x", clock=_clock())
    with pytest.raises(RuntimeError, match="VJ_VERIFY_BUNDLE_KEY"):
        verify._pod_draft_render(em, {})


# --------------------------------------------------------------------------- main() gate

def test_main_is_noop_without_flag(capsys):
    assert verify.main(env={}) == 0
    out = capsys.readouterr().out
    assert "verify_skipped" in out


def _fatal_payload(out):
    return json.loads(out.split("@event verify_fatal ", 1)[1].splitlines()[0])


def test_main_requires_run_id_when_armed_and_fails_loud(capsys):
    # armed but no run id: exit 2 AND a structured verify_fatal on stdout (never a silent empty prefix)
    assert verify.main(env={"VJ_VERIFY": "1"}) == 2
    out = capsys.readouterr().out
    assert "@event verify_fatal " in out
    payload = _fatal_payload(out)
    assert payload["stage"] == "config"
    assert payload["missing"] == ["VJ_VERIFY_RUN_ID"]


def test_main_bad_r2_config_fails_loud_with_missing_names(capsys):
    # armed, run id present, but the R2_* env names are missing/misnamed (the F17 class): the store
    # build raises BEFORE run_verify -- main must emit verify_fatal naming the MISSING env vars to
    # stdout, exit 1, and NOT hang silently. store=None forces the real R2Config.from_env path.
    code = verify.main(env={"VJ_VERIFY": "1", "VJ_VERIFY_RUN_ID": "run-abc"})  # no R2_* keys
    assert code == 1
    out = capsys.readouterr().out
    assert "@event verify_fatal " in out
    payload = _fatal_payload(out)
    assert payload["stage"] == "r2_config"
    # every R2_* name is absent here, so all four are reported, machine-readable (not a prose message)
    assert set(payload["missing"]) == set(verify.R2_ENV_NAMES)


def test_main_bad_r2_config_reports_only_the_missing_name(capsys):
    # exactly one name missing -> the missing list is precisely that one (structured, actionable)
    env = {"VJ_VERIFY": "1", "VJ_VERIFY_RUN_ID": "run-abc",
           "R2_ENDPOINT": "x", "R2_ACCESS_KEY_ID": "x", "R2_SECRET_ACCESS_KEY": "x"}  # R2_BUCKET absent
    assert verify.main(env=env) == 1
    payload = _fatal_payload(capsys.readouterr().out)
    assert payload["missing"] == ["R2_BUCKET"]


def test_main_runs_injected_render_and_reports_status():
    store = RecordingStore()
    env = {"VJ_VERIFY": "1", "VJ_VERIFY_RUN_ID": "run-abc", "VJ_SHARPNESS_BASELINE": "100"}
    code = verify.main(env=env, store=store, render=_good_render())
    assert code == 0
    assert json.loads(store.objects[keys.verify_summary_key("run-abc")])["status"] == "complete"

    # a failing render -> non-zero exit, error status on the channel
    store2 = RecordingStore()
    def bad(em):
        raise RuntimeError("boom")
    assert verify.main(env=env, store=store2, render=bad) == 1


# --------------------------------------------------------------------------- cross-module contract

def test_channel_feeds_runpod_verify_evaluate_and_passes():
    """THE load-bearing test: a run_verify channel, read back with events_from_summary, is exactly
    what the harness's evaluate() reads -- a full PASS with no prose parsing anywhere."""
    rv = _load_runpod_verify()
    store = RecordingStore()
    verify.run_verify(store, run_id="run-abc", render=_good_render(seconds=5.0, value=100.0),
                      probe=_pass_facts, baseline=100.0, clock=_clock())

    raw = store.objects[keys.verify_summary_key("run-abc")]
    events = verify.events_from_summary(raw)               # -> list[(name, payload)]
    result = rv.evaluate(events, rv.VerifyConfig(image="ghcr.io/x:test"))
    assert result.passed, result.reasons
    assert result.checks["gpu_visible"] and result.checks["render_complete"]
    assert result.checks["sharpness_parity"] and result.checks["first_frame_in_time"]


def test_channel_surfaces_a_real_failure_to_evaluate():
    rv = _load_runpod_verify()
    store = RecordingStore()
    # a blurry render (value well under baseline) must FAIL the harness sharpness gate honestly
    verify.run_verify(store, run_id="run-abc", render=_good_render(seconds=5.0, value=50.0),
                      probe=_pass_facts, baseline=100.0, clock=_clock())
    events = verify.events_from_summary(store.objects[keys.verify_summary_key("run-abc")])
    result = rv.evaluate(events, rv.VerifyConfig(image="ghcr.io/x:test"))
    assert not result.passed
    assert not result.checks["sharpness_parity"]
