"""Unit tests for the de-risk driver's CPU-only logic (no torch, no GPU).

Three GPU-free surfaces are exercised here:
  - the arch-gate set logic (which base targets the cu128 wheel must carry),
  - build_render_inputs, the render() prologue that constructs the contract objects + config, and
  - the RenderPlan API render() consumes (make_plan signature + plan property/method shapes).
The actual probe/render (kernel load, weight load, real i2v) need the baked image + a CUDA device,
so they are proven on the pod, not in CI. Importing the module is import-light (the heavy
vivijure_backend imports are deferred inside build_render_inputs/probe/render)."""
import importlib.util
import json
import socket
from pathlib import Path

import pytest

# Load deploy/vj_derisk.py directly (not a package import).
_SPEC = importlib.util.spec_from_file_location(
    "vj_derisk", Path(__file__).resolve().parents[1] / "deploy" / "vj_derisk.py")
vj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vj)


def test_arch_gate_requires_the_three_base_targets():
    assert vj.ARCH_GATE == ("sm_90", "sm_100", "sm_120")


def test_missing_arches_passes_a_full_cu128_list():
    # A representative torch 2.7 + cu128 arch list: the three base targets present (+ bonus arches/PTX).
    full = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert vj.missing_arches(full) == []


def test_missing_arches_reports_each_absent_base_target():
    assert vj.missing_arches(["sm_80", "sm_90"]) == ["sm_100", "sm_120"]
    assert vj.missing_arches(["sm_90", "sm_100"]) == ["sm_120"]
    assert vj.missing_arches([]) == ["sm_90", "sm_100", "sm_120"]


def test_accelerated_variant_does_not_satisfy_a_base_target():
    # sm_120a (accelerated variant) is BONUS, never a substitute for the base sm_120 target: a kernel
    # built only against the 'a' variant can still trip a real forward (the #15 runtime backstop).
    assert "sm_120" in vj.missing_arches(["sm_90", "sm_100", "sm_120a"])


def test_missing_arches_is_order_independent():
    assert vj.missing_arches(["sm_120", "sm_100", "sm_90"]) == []


def test_build_render_inputs_constructs_against_the_real_contract(tmp_path):
    # Regression guard: render() once passed an unsupported Scene kwarg (duration_seconds), which only
    # blew up ON THE POD at derisk_fail stage=render_portrait because nothing in CI exercised the
    # construction path. build_render_inputs is GPU-free, so the whole prologue is provable here against
    # the REAL vivijure_backend contract. Both aspects, so a bad ASPECTS entry is caught too.
    for aspect, (w, h) in vj.ASPECTS.items():
        req, sb, cast, bundle, cfg, work, gw, gh = vj.build_render_inputs(
            aspect, "final", tmp_path, frames=25, i2v_steps=8, kf_steps=20)
        assert (gw, gh) == (w, h)
        # The Scene duration hint the contract actually supports is target_seconds, NOT duration_seconds.
        assert sb.scenes[0].target_seconds == 2.0
        assert sb.scenes[0].id == "shot_01"
        # Storyboard has its OWN distinct duration_seconds field (this one IS valid on Storyboard).
        assert sb.duration_seconds == 2.0
        assert req.config is cfg
        assert work.is_dir() and bundle.root.is_dir()


def test_scene_contract_rejects_duration_seconds():
    # Pin the exact mismatch that caused the pod failure: Scene does not accept duration_seconds.
    from vivijure_backend.contract import Scene
    with pytest.raises(TypeError):
        Scene(prompt="x", duration_seconds=2.0)


def test_render_plan_api_matches_driver_usage(tmp_path):
    # render() walks the CPU path build_render_inputs -> make_plan(req, sb) -> reads the plan. Two more
    # driver/contract mismatches lurked PAST the prologue crash, each a separate GPU-spend re-fire crash:
    #   - make_plan was called with a cast= kwarg that plan() does not accept, and
    #   - keyframes_to_generate / shots_to_animate are @property (read without parens), not methods.
    # Pin the exact API shapes render() depends on so a contract drift fails in CI, not on a $/hr pod.
    from vivijure_backend.orchestrator import plan as make_plan
    req, sb, _cast, _bundle, _cfg, _work, _gw, _gh = vj.build_render_inputs(
        "portrait", "final", tmp_path, frames=25, i2v_steps=8, kf_steps=20)
    rplan = make_plan(req, sb)                              # signature: positional (request, storyboard)
    assert isinstance(rplan.keyframes_to_generate, int)    # @property, NOT callable
    assert isinstance(rplan.shots_to_animate, int)         # @property, NOT callable
    assert isinstance(rplan.summary(), str)                # summary() IS a method
    assert rplan.keyframes_to_generate == 1 and rplan.shots_to_animate == 1


# --------------------------------------------------------------------------- egress guard (#245)
# The userspace egress guard (deploy/vj_derisk.py) was merged (#161) but DORMANT: DERISK_EGRESS_LOCK was
# set nowhere, so maybe_install_egress_guard always no-opped and NO test exercised the controls. These
# tests pin the guard behavior so it can never silently regress to dormant again. Every test is hermetic:
# an external connect is refused by the guard BEFORE any socket I/O (a routable-looking IP, or a patched
# getaddrinfo), so nothing here touches the real network.


def _parse(text, name):
    prefix = "@event " + name + " "
    return [json.loads(line[len(prefix):]) for line in text.splitlines() if line.startswith(prefix)]


@pytest.fixture
def guard_reset():
    """Save/restore the process-global socket methods the guard monkeypatches + the install flag, so an
    installed guard never leaks into another test."""
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex
    prev = vj._EGRESS["on"]
    vj._EGRESS["on"] = False
    try:
        yield
    finally:
        socket.socket.connect = orig_connect
        socket.socket.connect_ex = orig_connect_ex
        vj._EGRESS["on"] = prev


def _fake_getaddrinfo(real):
    """getaddrinfo stub: resolve the two phone-home hosts to a routable-looking external IP so the guard
    blocks them AT CONNECT (proving the GUARD blocks, not a DNS failure); everything else stays real, so
    loopback still resolves for the positive control."""
    def gai(host, port, *a, **k):
        if host in ("huggingface.co", "github.com"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]
        return real(host, port, *a, **k)
    return gai


class _Dummy:
    def close(self):
        pass


def _fake_create_connection(real):
    """create_connection stub for the guard-OFF discriminator: the phone-home hosts appear REACHABLE
    (return a dummy), loopback stays real. Proves the controls actually distinguish reachable from blocked
    (not a rubber stamp that always reports blocked)."""
    def cc(address, *a, **k):
        if address[0] in ("huggingface.co", "github.com"):
            return _Dummy()
        return real(address, *a, **k)
    return cc


def test_egress_allowed_af_unix_passes():
    assert vj._egress_allowed(socket.AF_UNIX, "/run/x.sock") == (True, "af_unix")


def test_egress_allowed_loopback_variants_pass():
    for addr in vj._LOOPBACK:
        assert vj._egress_allowed(socket.AF_INET, (addr, 443)) == (True, "loopback")


def test_egress_allowed_blocks_every_external_host():
    for host in ("huggingface.co", "github.com", "pypi.org", "8.8.8.8",
                 "objects.r2.cloudflarestorage.com"):
        assert vj._egress_allowed(socket.AF_INET, (host, 443)) == (False, "blocked")


def test_egress_allowed_non_inet_address_passes():
    # A non-tuple / empty address is not an inet egress target, so the guard does not block it.
    assert vj._egress_allowed(socket.AF_INET, None) == (True, "non_inet")
    assert vj._egress_allowed(socket.AF_INET, ()) == (True, "non_inet")


def test_flag_on_reads_env(monkeypatch):
    for v in ("1", "true", "TRUE", "Yes", "on", "  On  "):
        monkeypatch.setenv("DERISK_EGRESS_LOCK", v)
        assert vj._flag_on("DERISK_EGRESS_LOCK") is True
    for v in ("", "0", "false", "no", "off", "nope"):
        monkeypatch.setenv("DERISK_EGRESS_LOCK", v)
        assert vj._flag_on("DERISK_EGRESS_LOCK") is False
    monkeypatch.delenv("DERISK_EGRESS_LOCK", raising=False)
    assert vj._flag_on("DERISK_EGRESS_LOCK") is False


def test_install_egress_guard_blocks_external_connect(guard_reset):
    vj._install_egress_guard()
    # A routable-looking IP: the guard refuses BEFORE any socket I/O, so this never touches the network.
    with pytest.raises(OSError, match="egress_guard"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("93.184.216.34", 443))
    with pytest.raises(OSError, match="egress_guard"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(("93.184.216.34", 443))


def test_install_egress_guard_allows_loopback(guard_reset):
    vj._install_egress_guard()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        socket.create_connection(srv.getsockname(), timeout=5).close()   # loopback: guard allows
    finally:
        srv.close()


def test_install_egress_guard_is_idempotent(guard_reset):
    vj._install_egress_guard()
    first = socket.socket.connect
    vj._install_egress_guard()
    assert socket.socket.connect is first   # second install is a no-op, not a double-wrap
    assert vj._EGRESS["on"] is True


def test_egress_controls_pass_when_guard_blocks(guard_reset, monkeypatch, capsys):
    monkeypatch.setattr(vj.socket, "getaddrinfo", _fake_getaddrinfo(socket.getaddrinfo))
    vj._install_egress_guard()
    assert vj._egress_controls() is True
    text = capsys.readouterr().out
    proven = _parse(text, "egress_guard_proven")
    assert proven[-1] == {"control": "negative", "hf_blocked": True,
                          "github_blocked": True, "ok": True}
    sane = _parse(text, "egress_guard_sane")
    assert sane[-1]["loopback_ok"] is True and sane[-1]["ok"] is True


def test_egress_controls_fail_when_egress_reachable(guard_reset, monkeypatch, capsys):
    # Guard NOT installed and the phone-home hosts are reachable -> controls MUST report NOT proven.
    monkeypatch.setattr(vj.socket, "create_connection",
                        _fake_create_connection(socket.create_connection))
    assert vj._egress_controls() is False
    proven = _parse(capsys.readouterr().out, "egress_guard_proven")
    assert proven[-1]["ok"] is False and proven[-1]["hf_blocked"] is False


def test_maybe_lock_egress_noop_when_flag_off(guard_reset, monkeypatch):
    monkeypatch.delenv("DERISK_EGRESS_LOCK", raising=False)
    before = socket.socket.connect
    assert vj._maybe_lock_egress() == 0
    assert socket.socket.connect is before   # a baseline (unlocked) run is byte-identical: no guard
    assert vj._EGRESS["on"] is False


def test_maybe_lock_egress_installs_and_passes_when_on(guard_reset, monkeypatch):
    monkeypatch.setenv("DERISK_EGRESS_LOCK", "1")
    monkeypatch.setattr(vj.socket, "getaddrinfo", _fake_getaddrinfo(socket.getaddrinfo))
    assert vj._maybe_lock_egress() == 0
    assert vj._EGRESS["on"] is True


def test_maybe_lock_egress_fails_render_when_controls_fail(guard_reset, monkeypatch, capsys):
    # Lock requested but egress is reachable -> the render MUST fail loud (return 1 + derisk_fail), never
    # silently proceed. A lock that cannot prove itself is a failure, not a pass (the honest-fail line).
    monkeypatch.setenv("DERISK_EGRESS_LOCK", "1")
    monkeypatch.setattr(vj.socket, "create_connection",
                        _fake_create_connection(socket.create_connection))
    assert vj._maybe_lock_egress() == 1
    fails = _parse(capsys.readouterr().out, "derisk_fail")
    assert any(f.get("stage") == "egress_guard_inactive" for f in fails)


def test_guard_probe_returns_zero_when_proven(guard_reset, monkeypatch, capsys):
    monkeypatch.setattr(vj.socket, "getaddrinfo", _fake_getaddrinfo(socket.getaddrinfo))
    assert vj.guard_probe() == 0
    text = capsys.readouterr().out
    assert _parse(text, "egress_guard_installed")
    assert _parse(text, "egress_guard_proven")[-1]["ok"] is True
    assert _parse(text, "guard_probe")[-1]["proven"] is True
