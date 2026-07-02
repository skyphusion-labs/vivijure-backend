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
    # 30 min on H200 at the conservative-high secure rate 4.39/hr ~= 2.195
    assert rv.cost_estimate_usd("NVIDIA H200", 1800) == pytest.approx(2.195, abs=1e-3)
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


def test_run_verify_fail_deletes_pod_and_captures_evidence():
    log = _pass_log() + ['@event mirror_complete {"total_seconds": 400.0}']  # baked but pulled R2 -> FAIL
    client = FakePodClient(log)
    report = rv.run_verify(client, rv.VerifyConfig(image="img:test"), clock=_clock())
    assert report["passed"] is False
    assert report["signal"] == "hold"
    assert report["teardown"] == "deleted"           # DELETE on every path: no stopped pad left billing
    names = [c[0] for c in client.calls]
    assert "delete_pod" in names and "stop_pod" not in names
    assert report["pod_logs_tail"]                    # FAIL evidence captured BEFORE delete (artifact)


def test_run_verify_no_available_gpu_does_not_spin():
    client = FakePodClient(_pass_log(), gpus=[{"id": "l4", "displayName": "NVIDIA L4", "available": True}])
    report = rv.run_verify(client, rv.VerifyConfig(image="img:test", tier="i2v"), clock=_clock())
    assert report["passed"] is False
    assert report["spun"] is False
    assert client.created == 0                        # NO pod created -> no spend
    assert all(c[0] != "create_pod" for c in client.calls)


def test_run_verify_hard_ttl_deletes_a_hung_render():
    # Logs never reach @event complete; the TTL must fire, fail the run, and stop (not leak) the pod.
    hung = [rv.emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                               "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                               "device_name": "NVIDIA H200"})]
    client = FakePodClient(hung)
    cfg = rv.VerifyConfig(image="img:test", ttl_seconds=3)  # clock advances 1s/call -> trips fast
    report = rv.run_verify(client, cfg, clock=_clock())
    assert report["passed"] is False
    assert report["timed_out"] is True
    assert report["teardown"] == "deleted"
    assert any("hard TTL" in r for r in report["reasons"])


def test_live_client_never_fires_without_creds(monkeypatch):
    # Importing the module never touches the SDK (lazy import -- proven by every other test running with
    # no `runpod` installed). Constructing the live client with no injected sdk AND no key fails CLOSED
    # (ModuleNotFoundError if the SDK is absent, RuntimeError for the missing key if present), so it can
    # never silently spin a real pod by accident.
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(Exception):
        rv.RunpodSdkPodClient()


def test_main_dry_run_passes_offline():
    # The CLI default path is a mocked dry-run: exit 0, no GPU, no network.
    assert rv.main(["--image", "img:dry"]) == 0


# ----------------------------------------------------------------- live SDK client (RunpodSdkPodClient)
# These drive the live client against a FAKE `runpod` SDK object injected via `sdk=` -- no runpod import,
# no network, no pod. They prove the SECURE-only + up/down/list plumbing and the honest read_logs seam.

class FakeSdk:
    """Records every SDK call; returns canned shapes. Injected into RunpodSdkPodClient(sdk=...)."""

    def __init__(self, gpus=None, detail=None, pods=None):
        # get_gpus() = the REAL minimal list shape: id/displayName(short)/memoryInGb, NO cloud fields.
        self._gpus = gpus if gpus is not None else [
            {"id": "NVIDIA H200", "displayName": "H200", "memoryInGb": 141},
            {"id": "NVIDIA L4", "displayName": "L4", "memoryInGb": 24},
            {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}]
        # get_gpu(id) = the detail shape carrying secureCloud (H200 + L4 secure; 4090 community-only).
        self._detail = detail if detail is not None else {
            "NVIDIA H200": {"id": "NVIDIA H200", "secureCloud": True},
            "NVIDIA L4": {"id": "NVIDIA L4", "secureCloud": True},
            "NVIDIA GeForce RTX 4090": {"id": "NVIDIA GeForce RTX 4090", "secureCloud": False}}
        self._pods = pods if pods is not None else []
        self.calls = []

    def get_gpus(self):
        self.calls.append(("get_gpus",))
        return self._gpus

    def get_gpu(self, gid):
        self.calls.append(("get_gpu", gid))
        d = self._detail.get(gid)
        if d is None:
            raise ValueError("No GPU found with the specified ID")
        return d

    def create_pod(self, **kw):
        self.calls.append(("create_pod", kw))
        return {"id": "pod-live-1"}

    def get_pod(self, pod_id):
        self.calls.append(("get_pod", pod_id))
        return {"id": pod_id, "desiredStatus": "RUNNING"}

    def stop_pod(self, pod_id):
        self.calls.append(("stop_pod", pod_id))
        return {"id": pod_id}

    def terminate_pod(self, pod_id):
        self.calls.append(("terminate_pod", pod_id))
        return {"id": pod_id}

    def get_pods(self):
        self.calls.append(("get_pods",))
        return self._pods


def test_sdk_client_list_gpu_types_marks_only_secure_available():
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk)
    gpus = client.list_gpu_types()
    # displayName is the canonical id (what GPU_TIERS/pick_gpu_type match on), available from get_gpu.
    by = {g["id"]: g["available"] for g in gpus}
    assert by["NVIDIA H200"] is True
    assert by["NVIDIA L4"] is True
    assert by["NVIDIA GeForce RTX 4090"] is False  # community-only SKU is never "available"
    assert all(g["displayName"] == g["id"] for g in gpus)
    # i2v picks the secure H200; base skips the community-only 4090 and lands on the secure L4.
    assert rv.pick_gpu_type(gpus, "i2v") == "NVIDIA H200"
    assert rv.pick_gpu_type(gpus, "base") == "NVIDIA L4"


def test_sdk_client_create_pod_forces_secure_cloud():
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk, container_disk_gb=500)
    out = client.create_pod(image="ghcr.io/x:1", gpu_type_id="NVIDIA H200",
                            env={"VJ_VERIFY": "1"}, registry_auth_id=None, ttl_seconds=1800)
    assert out == {"id": "pod-live-1"}
    (_, kw), = [c for c in sdk.calls if c[0] == "create_pod"]
    assert kw["cloud_type"] == "SECURE"          # NEVER COMMUNITY
    assert kw["image_name"] == "ghcr.io/x:1"
    assert kw["container_disk_in_gb"] == 500
    assert kw["env"] == {"VJ_VERIFY": "1"}


def test_sdk_client_create_pod_rejects_registry_auth_id():
    client = rv.RunpodSdkPodClient(sdk=FakeSdk())
    with pytest.raises(ValueError):
        client.create_pod(image="ghcr.io/x:1", gpu_type_id="NVIDIA H200", env={},
                          registry_auth_id="ra-123", ttl_seconds=1800)


def test_sdk_client_read_logs_never_raises_and_defaults_empty():
    client = rv.RunpodSdkPodClient(sdk=FakeSdk())
    assert client.read_logs("pod-live-1") == []           # null fetcher: honest degrade
    def boom(_pid):
        raise RuntimeError("log channel down")
    client2 = rv.RunpodSdkPodClient(sdk=FakeSdk(), log_fetcher=boom)
    assert client2.read_logs("pod-live-1") == []          # a fetch fault must never leak the pod


def test_sdk_client_stop_and_delete_map_to_sdk_verbs():
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk)
    client.stop_pod("p1")
    client.delete_pod("p1")
    names = [c[0] for c in sdk.calls]
    assert "stop_pod" in names and "terminate_pod" in names  # delete => terminate (irreversible)


def test_sdk_client_list_live_pod_ids_tolerates_shapes():
    assert rv.RunpodSdkPodClient(sdk=FakeSdk(pods=[{"id": "a"}, {"id": "b"}])).list_live_pod_ids() == ["a", "b"]
    assert rv.RunpodSdkPodClient(sdk=FakeSdk(pods={"pods": [{"id": "c"}]})).list_live_pod_ids() == ["c"]
    assert rv.RunpodSdkPodClient(sdk=FakeSdk(pods=[])).list_live_pod_ids() == []


def test_sdk_client_drives_run_verify_end_to_end_pass():
    # Inject a log_fetcher that replays a PASS @event stream: the whole harness runs against the live
    # client shape with no network, and PASS => the pod is TERMINATED (full teardown, no leak).
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk, log_fetcher=lambda _pid: _pass_log())
    cfg = rv.VerifyConfig(image="ghcr.io/x:1")
    report = rv.run_verify(client, cfg, clock=_clock(), evaluator=rv.evaluate)
    assert report["passed"] is True
    assert report["signal"] == "promote"
    assert report["teardown"] == "deleted"
    assert any(c[0] == "terminate_pod" for c in sdk.calls)  # PASS path deletes (terminates) the pod


def test_sdk_client_missing_log_channel_fails_closed_and_tears_down():
    # No @event stream (null fetcher) => never reaches @event complete => TTL FAIL, and the pod is
    # DELETED (list-confirmed zero), never left stopped-but-billing.
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk)  # default null fetcher
    cfg = rv.VerifyConfig(image="ghcr.io/x:1", ttl_seconds=3)
    report = rv.run_verify(client, cfg, clock=_clock(), evaluator=rv.evaluate)
    assert report["passed"] is False
    assert report["teardown"] == "deleted"
    assert any(c[0] == "terminate_pod" for c in sdk.calls)


def test_mcp_alias_points_at_sdk_client():
    assert rv.RunpodMcpPodClient is rv.RunpodSdkPodClient


def test_gpu_tier_ids_all_have_a_real_price():
    # The tier ids and the price table must never drift apart: every GPU_TIERS id resolves a REAL price
    # in GPU_HOURLY_USD (not the conservative 4.0/hr default), so a spend estimate never quotes a wrong
    # rate for a listed card. Regression guard: the "NVIDIA RTX 4090" -> "NVIDIA GeForce RTX 4090"
    # rename once left the price row behind.
    for tier, ids in rv.GPU_TIERS.items():
        for gid in ids:
            assert gid in rv.GPU_HOURLY_USD, f"{gid} ({tier} tier) has no GPU_HOURLY_USD price row"


def test_cost_estimate_uses_real_rate_for_geforce_4090():
    # The corrected canonical id resolves its real 0.69/hr rate, not the 4.0/hr fallback.
    assert rv.cost_estimate_usd("NVIDIA GeForce RTX 4090", 3600.0) == 0.69


# ================================================================= S1 live channel + promote (S7)


class _ConfirmClient(FakePodClient):
    """FakePodClient that also models list_live_pod_ids so run_verify's teardown-confirm can assert a
    DELETED pod is actually gone from the live set."""

    def __init__(self, log_lines, gpus=None):
        super().__init__(log_lines, gpus=gpus)
        self._deleted = set()

    def delete_pod(self, pod_id):
        self._deleted.add(pod_id)
        return super().delete_pod(pod_id)

    def list_live_pod_ids(self):
        self.calls.append(("list_live_pod_ids",))
        return [] if "pod-xyz" in self._deleted else ["pod-xyz"]


def test_run_verify_uses_injected_event_reader_not_logs_and_confirms_teardown():
    # Simulate the R2 summary channel: first two polls empty (pod still writing), third terminal.
    passing = rv.parse_events(_pass_log())
    seq = [([], None), ([], None), (passing, "complete")]
    calls = {"n": 0}

    def reader():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    client = _ConfirmClient(_pass_log())
    report = rv.run_verify(client, rv.VerifyConfig(image="ghcr.io/x:1"), clock=_clock(),
                           event_reader=reader)
    assert report["passed"] is True
    assert report["signal"] == "promote" and report["teardown"] == "deleted"
    assert report["teardown_confirmed"] is True          # deleted pod confirmed gone (list-confirm zero)
    assert report["live_pods_after"] == 0
    # The R2 channel was used; the pod-log channel (read_logs) was NEVER touched.
    assert "read_logs" not in [c[0] for c in client.calls]


def test_run_verify_error_status_fails_and_deletes_pod():
    probe = rv.parse_events([rv.emit_gpu_probe(
        {"torch_cuda": True, "kernel_ok": True, "vj_baked": True, "weights_on_disk": True,
         "vram_free_gb": 120.0, "vram_total_gb": 141.0, "device_name": "NVIDIA H200"})])

    def reader():
        return probe, "error"

    client = FakePodClient(_pass_log())
    report = rv.run_verify(client, rv.VerifyConfig(image="img:1"), clock=_clock(), event_reader=reader)
    assert report["passed"] is False
    assert report["teardown"] == "deleted"               # FAIL still ends at delete + list-confirm zero
    assert any("status=error" in r for r in report["reasons"])


def test_run_verify_calls_on_pod_created_with_id():
    seen = []
    client = FakePodClient(_pass_log())
    rv.run_verify(client, rv.VerifyConfig(image="img:1"), clock=_clock(),
                  on_pod_created=seen.append)
    assert seen == ["pod-xyz"]                            # id captured for the always-run stop backstop


def test_promote_image_repins_the_endpoints_template_not_the_endpoint():
    calls = []

    def transport(url, *, method="PATCH", headers, payload=None):
        calls.append((method, url, payload))
        if method == "GET" and "/endpoints/" in url:
            return {"id": "t9wcvlxh8rc5la", "templateId": "tpl-1"}
        if method == "PATCH" and "/templates/" in url:
            return {}
        if method == "GET" and "/templates/" in url:
            return {"imageName": "ghcr.io/x:2"}  # read-back: repin took
        raise AssertionError("unexpected promote call: %s %s" % (method, url))

    out = rv.promote_image("ghcr.io/x:2", endpoint_id="t9wcvlxh8rc5la", api_key="k-1",
                           transport=transport)
    # resolved the endpoint's template, then PATCHed the TEMPLATE image -- never the endpoint (that 400s)
    assert ("GET", "https://rest.runpod.io/v1/endpoints/t9wcvlxh8rc5la", None) in calls
    patch = [c for c in calls if c[0] == "PATCH"][0]
    assert "/templates/tpl-1" in patch[1]
    assert patch[2] == {"imageName": "ghcr.io/x:2"}
    assert not any(m == "PATCH" and "/endpoints/" in u for m, u, _ in calls)
    assert out["image"] == "ghcr.io/x:2" and out["template_id"] == "tpl-1"
    assert out["imageName"] == "ghcr.io/x:2"


def test_promote_image_raises_on_readback_mismatch():
    def transport(url, *, method="PATCH", headers, payload=None):
        if method == "GET" and "/endpoints/" in url:
            return {"templateId": "tpl-1"}
        if method == "PATCH":
            return {}
        if method == "GET" and "/templates/" in url:
            return {"imageName": "ghcr.io/x:OLD"}  # repin silently did not take
        raise AssertionError("unexpected call")

    with pytest.raises(RuntimeError):
        rv.promote_image("ghcr.io/x:2", endpoint_id="e", api_key="k-1", transport=transport)


def test_promote_image_requires_key(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        rv.promote_image("ghcr.io/x:2", transport=lambda *a, **k: {})


def test_r2_summary_event_reader_reads_the_frozen_contract(monkeypatch):
    # Round-trip through the REAL emitter so the reader is proven against the true wire contract, not a
    # hand-rolled summary shape.
    import vivijure_backend.harness.r2 as r2mod
    from vivijure_backend import verify as vmod

    class _FakeStore:
        def __init__(self):
            self.blobs = {}

        def put_bytes(self, data, key, content_type=None):
            self.blobs[key] = data

        def get_bytes(self, key):
            return self.blobs[key]

    store = _FakeStore()
    em = vmod.VerifyEmitter(store, "runX", clock=lambda: 1.0)
    em.gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True, "weights_on_disk": True,
                  "vram_free_gb": 120.0, "vram_total_gb": 141.0, "device_name": "NVIDIA H200"})
    em.complete("renders/verify/x.mp4", 1048576)

    monkeypatch.setattr(r2mod.R2Config, "from_env", classmethod(lambda cls, env=None: object()))
    monkeypatch.setattr(r2mod, "R2", lambda cfg: store)
    reader = rv.r2_summary_event_reader({"R2_ENDPOINT": "e"}, "runX", prefix="verify")
    events, terminal = reader()
    assert terminal == "complete"
    names = [n for n, _ in events]
    assert "gpu_probe" in names and "complete" in names


def test_r2_summary_event_reader_keeps_polling_until_written(monkeypatch):
    import vivijure_backend.harness.r2 as r2mod

    class _EmptyStore:
        def get_bytes(self, key):
            raise KeyError(key)  # not written yet

    monkeypatch.setattr(r2mod.R2Config, "from_env", classmethod(lambda cls, env=None: object()))
    monkeypatch.setattr(r2mod, "R2", lambda cfg: _EmptyStore())
    reader = rv.r2_summary_event_reader({"R2_ENDPOINT": "e"}, "runY")
    events, terminal = reader()
    assert events == [] and terminal is None             # honest "keep waiting", never a crash


def test_github_env_writer_records_pod_id(tmp_path, monkeypatch):
    envfile = tmp_path / "gh_env"
    monkeypatch.setenv("GITHUB_ENV", str(envfile))
    rv._github_env_writer("VJ_VERIFY_POD_ID")("pod-abc")
    assert "VJ_VERIFY_POD_ID=pod-abc" in envfile.read_text()


def test_reap_pod_stops_then_deletes_and_confirms_zero():
    class _ReapClient:
        def __init__(self):
            self.calls = []
            self._live = ["pod-9"]

        def stop_pod(self, pid):
            self.calls.append(("stop_pod", pid))

        def delete_pod(self, pid):
            self.calls.append(("delete_pod", pid))
            self._live = [x for x in self._live if x != pid]

        def list_live_pod_ids(self):
            return list(self._live)

    c = _ReapClient()
    gone = rv.reap_pod("pod-9", client=c)
    assert gone is True
    assert [x[0] for x in c.calls] == ["stop_pod", "delete_pod"]   # stop first (fast halt), then delete


def test_reap_pod_flags_still_live_loudly():
    class _StuckClient:
        def stop_pod(self, pid):
            pass

        def delete_pod(self, pid):
            pass

        def list_live_pod_ids(self):
            return ["pod-9"]  # delete did not take -> still live

    assert rv.reap_pod("pod-9", client=_StuckClient()) is False   # caller fails the backstop step


def test_clean_key_strips_whitespace_and_matched_quotes():
    assert rv._clean_key("  rpa_abc  ") == "rpa_abc"
    assert rv._clean_key('"rpa_abc"') == "rpa_abc"
    assert rv._clean_key("'rpa_abc'") == "rpa_abc"
    assert rv._clean_key('"rpa_abc"\n') == "rpa_abc"        # quoted .env line + trailing newline
    assert rv._clean_key("rpa_abc") == "rpa_abc"            # already clean: unchanged
    assert rv._clean_key('"rpa_abc') == '"rpa_abc'          # unmatched quote: left alone
    assert rv._clean_key(None) == ""


def test_key_shape_never_echoes_the_value(capsys, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", '"rpa_TOPSECRETVALUE42"')
    rc = rv.main(["--key-shape"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "TOPSECRETVALUE42" not in out                    # the VALUE body is never echoed
    assert "cleaned_matches_rpa=yes" in out                 # matched quotes stripped -> valid shape
    assert "starts_with_quote=yes" in out                   # the mangle is diagnosed, not the value


def test_verify_config_pod_command_runs_the_verify_entrypoint():
    cmd = rv.VerifyConfig(image="ghcr.io/x:1").pod_command
    assert cmd.endswith("-m vivijure_backend.verify")     # RUN verify, not the default worker CMD
    assert "conda run" in cmd and "vivijure" in cmd        # inside the image conda env


def test_run_verify_passes_pod_command_to_create_pod():
    client = FakePodClient(_pass_log())
    cfg = rv.VerifyConfig(image="ghcr.io/x:1")
    rv.run_verify(client, cfg, clock=_clock())
    create = next(c for c in client.calls if c[0] == "create_pod")[1]
    assert create["command"] == cfg.pod_command           # the verify entrypoint is wired to the pod


def test_sdk_client_create_pod_sets_command_as_docker_args():
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk)
    client.create_pod(image="ghcr.io/x:1", gpu_type_id="NVIDIA H200", env={}, registry_auth_id=None,
                      ttl_seconds=60, command="conda run -n vivijure python -m vivijure_backend.verify")
    (_, kw), = [c for c in sdk.calls if c[0] == "create_pod"]
    assert kw["docker_args"] == "conda run -n vivijure python -m vivijure_backend.verify"


def test_sdk_client_create_pod_omits_docker_args_without_command():
    sdk = FakeSdk()
    client = rv.RunpodSdkPodClient(sdk=sdk)
    client.create_pod(image="ghcr.io/x:1", gpu_type_id="NVIDIA H200", env={}, registry_auth_id=None,
                      ttl_seconds=60)  # no command
    (_, kw), = [c for c in sdk.calls if c[0] == "create_pod"]
    assert "docker_args" not in kw                         # unset => the SDK default, no accidental empty


def test_run_verify_sleeps_between_polls():
    # The live path MUST wait between R2 summary polls (a real render needs minutes); a no-op sleep
    # burns all 600 polls in milliseconds and gives up before the pod boots. Prove the loop calls the
    # injected poll_sleep between non-terminal polls.
    passing = rv.parse_events(_pass_log())
    seq = [([], None), ([], None), (passing, "complete")]
    calls = {"n": 0}

    def reader():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    slept = []
    client = FakePodClient(_pass_log())
    rv.run_verify(client, rv.VerifyConfig(image="ghcr.io/x:1"), clock=_clock(),
                  poll_sleep=slept.append, event_reader=reader)
    assert len(slept) >= 2                                 # slept through the two empty polls


def test_run_verify_records_pod_state_log_for_pull_visibility():
    # A client whose pod reports PROVISIONING then RUNNING (with runtime) -> the report captures the
    # transition so image-pull/boot time is measurable, and pod_ever_running is True.
    class _StatefulClient(FakePodClient):
        def __init__(self, log_lines):
            super().__init__(log_lines)
            self._n = 0

        def get_pod(self, pod_id):
            self._n += 1
            if self._n <= 1:
                return {"id": pod_id, "desiredStatus": "PROVISIONING"}
            return {"id": pod_id, "desiredStatus": "RUNNING", "runtime": {"uptimeInSeconds": 3}}

    # Reader stays empty for 2 polls (pod still booting), then completes.
    passing = rv.parse_events(_pass_log())
    seq = [([], None), ([], None), (passing, "complete")]
    calls = {"n": 0}

    def reader():
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    client = _StatefulClient(_pass_log())
    report = rv.run_verify(client, rv.VerifyConfig(image="ghcr.io/x:1"), clock=_clock(),
                           event_reader=reader)
    assert report["pod_ever_running"] is True
    statuses = [s["status"] for s in report["pod_state_log"]]
    assert "PROVISIONING" in statuses and "RUNNING" in statuses   # the pull->run transition is recorded


def test_ttl_default_has_cold_pull_headroom():
    assert rv.VerifyConfig(image="ghcr.io/x:1").ttl_seconds >= 3000   # room for an ~87GB cold bake pull


def test_max_polls_follows_ttl_not_a_fixed_600_cap():
    # With a large TTL and a channel that never reaches terminal, the poll loop must be able to poll
    # well past the old fixed 600 cap (600 x 5s = 3000s would cut a 5400s-TTL run short). Prove the
    # loop polls > 600 times before giving up, i.e. max_polls now scales with ttl_seconds.
    calls = {"n": 0}

    def reader():
        calls["n"] += 1
        return [], None  # never terminal

    def clock():
        clock.t += 1.0
        return clock.t
    clock.t = 0.0

    client = FakePodClient(_pass_log())
    cfg = rv.VerifyConfig(image="ghcr.io/x:1", ttl_seconds=5400)
    report = rv.run_verify(client, cfg, clock=clock, poll_sleep=lambda _s: None, event_reader=reader)
    assert report["passed"] is False
    assert report["timed_out"] is True
    assert calls["n"] > 600            # polled past the old fixed cap; TTL is the real ceiling now
    assert report["teardown"] == "deleted"


def test_report_records_pod_env_key_names():
    client = FakePodClient(_pass_log())
    cfg = rv.VerifyConfig(image="ghcr.io/x:1")
    report = rv.run_verify(client, cfg, clock=_clock())
    assert report["pod_env_keys"] == sorted(cfg.env.keys())   # NAMES only, artifact-diagnosable
    assert "VJ_VERIFY" in report["pod_env_keys"]


def test_clean_key_generalizes_to_any_env_value():
    # The generalized cleaner strips a matched quote pair off ANY value (not just the API key), which
    # is what protects every injected pod-env secret from the quote-wrap landing class.
    assert rv._clean_key('"' + "some-r2-secret" + '"') == "some-r2-secret"
    assert rv._clean_key("  some-r2-secret\n") == "some-r2-secret"
    assert rv._clean_key("already-clean") == "already-clean"


def test_key_shape_reports_r2_edges_without_values(capsys, monkeypatch):
    quoted = chr(34) + "R2SECRETBODY" + chr(34)          # a quote-wrapped R2 secret value
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_clean123")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", quoted)
    monkeypatch.setenv("R2_ENDPOINT", "https://x.r2.cloudflarestorage.com")
    rc = rv.main(["--key-shape"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "R2SECRETBODY" not in out                          # value never echoed
    assert "R2_SECRET_ACCESS_KEY: present=yes" in out
    assert "R2_SECRET_ACCESS_KEY:" in out and "starts_quote=yes ends_quote=yes" in out
