"""CPU unit tests for the RunPod pod verify harness (deploy/runpod_verify.py). Drives the WHOLE flow
against a FakePodClient + an injected clock: no GPU, no network, no live pod. Asserts the @event
contract, every named gate, the PASS=delete / FAIL=stop teardown, the hard-TTL auto-stop, and the
no-available-GPU no-spend abort."""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "deploy" / "runpod_verify.py"
    spec = importlib.util.spec_from_file_location("runpod_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register so dataclass string annotations resolve at exec time
    spec.loader.exec_module(mod)
    return mod


rv = _load()


# ----------------------------------------------------------------- fake pod client

class FakePodClient:
    """Records every lifecycle call; replays a canned @event log. Single concurrency by construction
    (one pod). `gpus` and `log_lines` are injected per test."""

    def __init__(self, log_lines, gpus=None):
        self._log = list(log_lines)
        self._gpus = gpus if gpus is not None else [
            {"id": "H200", "displayName": "NVIDIA H200", "available": True}]
        self.calls = []
        self.created = 0

    def list_gpu_types(self):
        self.calls.append(("list_gpu_types",))
        return self._gpus

    def create_pod(self, **kw):
        self.created += 1
        assert self.created == 1, "harness must spin exactly one pod (single concurrency)"
        self.calls.append(("create_pod", kw))
        return {"id": "pod-xyz"}

    def get_pod(self, pod_id):
        self.calls.append(("get_pod", pod_id))
        return {"id": pod_id, "desiredStatus": "RUNNING"}

    def read_logs(self, pod_id):
        self.calls.append(("read_logs", pod_id))
        return self._log

    def stop_pod(self, pod_id):
        self.calls.append(("stop_pod", pod_id))
        return {"id": pod_id, "desiredStatus": "EXITED"}

    def delete_pod(self, pod_id):
        self.calls.append(("delete_pod", pod_id))
        return {"id": pod_id, "deleted": True}


def _pass_log():
    return [
        rv.emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                           "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                           "device_name": "NVIDIA H200"}),
        '@event mirror_skipped {"reason": "baked"}',
        "Loading transformer ... (prose the parser must ignore)",
        '@event first_frame {"seconds": 41.0}',
        '@event sharpness {"value": 0.99, "baseline": 1.0}',
        '@event complete {"output_key": "renders/verify/full.mp4", "clip_bytes": 1048576}',
    ]


def _clock():
    """A monotonic stub that advances 1s per call."""
    t = {"v": 0.0}
    def now():
        t["v"] += 1.0
        return t["v"]
    return now


# ----------------------------------------------------------------- @event parsing

def test_parse_events_ignores_prose_and_bad_json():
    lines = ['@event a {"x": 1}', "just a log line", "@event b {bad json", '@event c {}']
    evs = rv.parse_events(lines)
    assert evs == [("a", {"x": 1}), ("c", {})]


def test_find_event_returns_last_payload():
    evs = [("first_frame", {"seconds": 9.0}), ("first_frame", {"seconds": 41.0})]
    assert rv.find_event(evs, "first_frame") == {"seconds": 41.0}
    assert rv.find_event(evs, "missing") is None


def test_emit_gpu_probe_roundtrips_through_parser():
    facts = {"torch_cuda": True, "kernel_ok": True, "vj_baked": True, "weights_on_disk": True,
             "vram_free_gb": 120.0, "vram_total_gb": 141.0, "device_name": "NVIDIA H200"}
    line = rv.emit_gpu_probe(facts)
    (name, payload), = rv.parse_events([line])
    assert name == "gpu_probe" and payload["kernel_ok"] is True and payload["vj_baked"] is True


# ----------------------------------------------------------------- evaluate gates

def test_evaluate_pass_all_gates_green():
    cfg = rv.VerifyConfig(image="img:test")
    res = rv.evaluate(rv.parse_events(_pass_log()), cfg)
    assert res.passed is True
    assert all(res.checks.values())
    assert res.metrics["output_key"] == "renders/verify/full.mp4"


def test_evaluate_fails_when_baked_image_ran_r2_mirror():
    # The .vj-baked early-return did NOT fire: a heavy mirror_complete leg appears -> FAIL.
    log = _pass_log() + ['@event mirror_complete {"total_seconds": 360.0, "total_bytes": 1}']
    res = rv.evaluate(rv.parse_events(log), rv.VerifyConfig(image="img:test"))
    assert res.passed is False
    assert res.checks["baked_no_r2_pull"] is False
    assert any("R2 mirror leg" in r for r in res.reasons)


def test_evaluate_fails_on_kernel_not_loaded():
    # torch sees the GPU but the cu128 kernel did not load on this card (sm_120 class).
    bad = rv.emit_gpu_probe({"torch_cuda": True, "kernel_ok": False, "vj_baked": True,
                             "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                             "device_name": "NVIDIA RTX PRO 6000"})
    log = [bad] + _pass_log()[1:]
    res = rv.evaluate(rv.parse_events(log), rv.VerifyConfig(image="img:test"))
    assert res.passed is False
    assert res.checks["kernel_ok"] is False
    assert any("kernel" in r for r in res.reasons)


def test_evaluate_fails_on_sharpness_below_threshold():
    log = [ln for ln in _pass_log() if not ln.startswith('@event sharpness')]
    log.append('@event sharpness {"value": 0.80, "baseline": 1.0}')
    res = rv.evaluate(rv.parse_events(log), rv.VerifyConfig(image="img:test", min_sharpness_ratio=0.97))
    assert res.passed is False
    assert res.checks["sharpness_parity"] is False


def test_evaluate_fails_on_empty_clip():
    log = [ln for ln in _pass_log() if not ln.startswith('@event complete')]
    log.append('@event complete {"output_key": "renders/verify/full.mp4", "clip_bytes": 0}')
    res = rv.evaluate(rv.parse_events(log), rv.VerifyConfig(image="img:test"))
    assert res.passed is False
    assert res.checks["clip_written"] is False


def test_evaluate_fails_on_slow_first_frame():
    log = [ln for ln in _pass_log() if not ln.startswith('@event first_frame')]
    log.append('@event first_frame {"seconds": 999.0}')
    res = rv.evaluate(rv.parse_events(log), rv.VerifyConfig(image="img:test", max_first_frame_seconds=300))
    assert res.passed is False
    assert res.checks["first_frame_in_time"] is False


# ----------------------------------------------------------------- cost + gpu selection

def test_cost_estimate_conservative_and_positive():
    # 30 min on H200 at 3.99/hr ~= 1.995
    assert rv.cost_estimate_usd("NVIDIA H200", 1800) == pytest.approx(1.995, abs=1e-3)
    assert rv.cost_estimate_usd("unknown-card", 3600) == pytest.approx(4.0, abs=1e-6)  # default high


def test_pick_gpu_prefers_most_capable_available():
    avail = [{"id": "h2", "displayName": "NVIDIA H200", "available": True},
             {"id": "b2", "displayName": "NVIDIA B200", "available": True}]
    assert rv.pick_gpu_type(avail, "i2v") == "h2"  # H200 ranks before B200 in the i2v pref order


def test_pick_gpu_none_when_tier_unavailable():
    avail = [{"id": "l4", "displayName": "NVIDIA L4", "available": True}]
    assert rv.pick_gpu_type(avail, "i2v") is None  # no H200-class -> caller must not spin


# ----------------------------------------------------------------- orchestration + teardown

def test_run_verify_pass_deletes_pod_and_signals_promote():
    client = FakePodClient(_pass_log())
    cfg = rv.VerifyConfig(image="ghcr.io/skyphusion-labs/vivijure-backend:bf16", registry_auth_id="ra-1")
    report = rv.run_verify(client, cfg, clock=_clock())
    assert report["passed"] is True
    assert report["signal"] == "promote"
    assert report["teardown"] == "deleted"
    names = [c[0] for c in client.calls]
    assert "delete_pod" in names and "stop_pod" not in names
    # registry auth + TTL were passed to create_pod (no dashboard, bounded)
    create = next(c for c in client.calls if c[0] == "create_pod")[1]
    assert create["registry_auth_id"] == "ra-1"
    assert create["ttl_seconds"] == cfg.ttl_seconds
    assert report["cost_estimate_usd"] >= 0.0


def test_run_verify_fail_stops_pod_and_emits_debug_handle():
    log = _pass_log() + ['@event mirror_complete {"total_seconds": 400.0}']  # baked but pulled R2 -> FAIL
    client = FakePodClient(log)
    report = rv.run_verify(client, rv.VerifyConfig(image="img:test"), clock=_clock())
    assert report["passed"] is False
    assert report["signal"] == "hold"
    assert report["teardown"] == "stopped"           # STOP not delete: disk preserved for debug
    names = [c[0] for c in client.calls]
    assert "stop_pod" in names and "delete_pod" not in names
    assert report["debug_handle"]["pod_id"] == "pod-xyz"
    assert "start-pod" in report["debug_handle"]["resume"]


def test_run_verify_no_available_gpu_does_not_spin():
    client = FakePodClient(_pass_log(), gpus=[{"id": "l4", "displayName": "NVIDIA L4", "available": True}])
    report = rv.run_verify(client, rv.VerifyConfig(image="img:test", tier="i2v"), clock=_clock())
    assert report["passed"] is False
    assert report["spun"] is False
    assert client.created == 0                        # NO pod created -> no spend
    assert all(c[0] != "create_pod" for c in client.calls)


def test_run_verify_hard_ttl_stops_a_hung_render():
    # Logs never reach @event complete; the TTL must fire, fail the run, and stop (not leak) the pod.
    hung = [rv.emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                               "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                               "device_name": "NVIDIA H200"})]
    client = FakePodClient(hung)
    cfg = rv.VerifyConfig(image="img:test", ttl_seconds=3)  # clock advances 1s/call -> trips fast
    report = rv.run_verify(client, cfg, clock=_clock())
    assert report["passed"] is False
    assert report["timed_out"] is True
    assert report["teardown"] == "stopped"
    assert any("hard TTL" in r for r in report["reasons"])


def test_live_client_is_a_gated_unimplemented_seam():
    # Importing/using the module must never be able to fire a real pod: the live client raises.
    with pytest.raises(NotImplementedError):
        rv.RunpodMcpPodClient().create_pod(image="x", gpu_type_id="y", env={},
                                           registry_auth_id=None, ttl_seconds=1)


def test_main_dry_run_passes_offline():
    # The CLI default path is a mocked dry-run: exit 0, no GPU, no network.
    assert rv.main(["--image", "img:dry"]) == 0
