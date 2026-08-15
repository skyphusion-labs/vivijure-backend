"""The finish_clip wall-clock guard (#422): the budget, where it is checked, and what it returns.

Every test here is about one of the three ways a guard like this fails while LOOKING correct.

  - IT DOES NOT FIRE. A pre-existing broad except-Exception on the path swallows the expiry into a
    fallback value, and the invocation stays unbounded behind a guard that is present, tested and
    inert. Two tests cover it, one of which drives the REAL handler in finish._restore_frame rather
    than a stand-in that could not have succeeded anyway.
  - IT FIRES BY RAISING. A raise leaves no structured output, classifies deterministic in
    vivijure-core and fails the WHOLE render after the GPU spend is banked, which is strictly worse
    than the hang it replaced. Covered end to end through harness.handler.handler, the one catch
    site, with the expiry coming from a REAL check rather than a stubbed raise.
  - IT CANNOT FIRE. A default at or above the platform ceiling is decoration. Both bounds are
    executable assertions here, so a later bump cannot quietly break either.

The blocking seams (a stalled encoder, a stalled mux) run REAL subprocesses. A faked Popen would
prove the decision and not the mechanism, and the mechanism is the entire point: the watchdog exists
because a blocked write never returns to the next check.
"""
import shutil
import subprocess
import time
import types
from pathlib import Path

import pytest

from vivijure_backend import assemble as assemble_mod
from vivijure_backend import finish
from vivijure_backend.finish import (
    DEFAULT_MAX_SECONDS,
    Deadline,
    FinishDeadlineExceeded,
    FinishParams,
    max_finish_seconds,
)
from vivijure_backend.harness import handler as handler_mod

# vivijure-core src/film-model.ts, read at 1efaae3aac0886e8ef4cc7607c9e58082f9038cb. Restated here
# so the arithmetic below is executable; the citation is what makes them checkable against core.
CORE_FINISH_STEP_MAX_ATTEMPTS = 3
CORE_PHASE_HARD_DEADLINE_SECONDS = 5400
# RunPod queue-endpoint documented execution-timeout default (serverless/endpoints/
# endpoint-configurations: default 600s, range 5s..7 days).
RUNPOD_DOCUMENTED_EXECUTION_TIMEOUT_SECONDS = 600

_HAVE_SHELL_TOOLS = shutil.which("sleep") is not None and shutil.which("dd") is not None


class _Frame:
    """Minimal stand-in for a decoded HxWx3 frame: _encode_uniform only reads .shape."""
    shape = (2, 2, 3)


class _Clock:
    """A monotonic stand-in that advances a fixed step on every read, so a test can place the expiry
    at a CHOSEN check instead of racing a real timer. Deadline.start consumes one read, and each
    check consumes exactly one more, so the k-th check sees elapsed == k*step."""

    def __init__(self, step: float = 1.0):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


# ------------------------------------------------------- the budget, and whether it can ever fire

def test_the_default_can_actually_fire_against_the_platform_ceiling():
    # The lesson the musetalk lane paid for, as an assertion. If RunPod enforces its documented
    # 600s execution timeout on this endpoint, a guard at or above 600 could NEVER fire, and a
    # platform kill is a FAILED envelope with no structured output, i.e. a failed render rather
    # than a degrade. What timeout: 0 means on this endpoint is NOT settled (docs/finish-deadline
    # .md); strictly under 600 fires under BOTH readings, which is what makes the open question
    # harmless instead of load-bearing.
    assert DEFAULT_MAX_SECONDS < RUNPOD_DOCUMENTED_EXECUTION_TIMEOUT_SECONDS


def test_the_default_leaves_the_core_phase_floor_where_it_is():
    # vivijure-core retries a finish step 3 times and a retry moves attempts, never the idx progress
    # marker, so the guard costs up to 3*G of wall clock with no marker movement. Core RAISES the
    # phase deadline to 3*G when that exceeds its 5400s floor, so a bigger default here silently
    # extends every film phase containing this door. Staying under keeps the floor binding.
    assert CORE_FINISH_STEP_MAX_ATTEMPTS * DEFAULT_MAX_SECONDS < CORE_PHASE_HARD_DEADLINE_SECONDS


@pytest.mark.parametrize("raw", [None, "", "   ", "junk", "0", "-1", "-0.5"])
def test_no_env_value_can_turn_the_guard_off(raw):
    # There is deliberately NO value that disables the guard: junk, empty, zero and negatives all
    # resolve to the default. A guard an operator can switch off by typo is the defect this closes.
    env = {} if raw is None else {"VJ_FINISH_MAX_SECONDS": raw}
    assert max_finish_seconds(env) == float(DEFAULT_MAX_SECONDS)


def test_a_valid_override_is_honoured():
    # Probed with a NON-DEFAULT value on purpose: on the default, honoured and substituted are
    # byte-identical and this would pass either way.
    assert max_finish_seconds({"VJ_FINISH_MAX_SECONDS": "137"}) == 137.0


def test_the_reason_fits_the_consumer_120_character_cap():
    # vivijure-cf modules/_shared/finish-soft-degrade.ts degradeReason slices the detail to 120
    # characters, so the guard name, the stage and both numbers have to survive inside the first
    # 120 or the degrade arrives unattributable.
    exc = FinishDeadlineExceeded("interpolator_load", 1234.5, 420.0)
    assert len(exc.reason) <= 120
    assert "finish_clip deadline" in exc.reason
    assert "interpolator_load" in exc.reason
    assert "1234.5s" in exc.reason and "420s" in exc.reason


# ------------------------------------------- the guard cannot be swallowed by a broad handler

def test_the_expiry_is_not_an_exception_subclass():
    # The structural half. Three broad except-Exception handlers already sit on this path
    # (finish._restore_frame, finish._tick, finish._probe_nvenc) and each is CORRECT for its own
    # job. Deriving from BaseException means none of them can ever eat the expiry, instead of every
    # future handler on this path having to remember not to.
    assert issubclass(FinishDeadlineExceeded, BaseException)
    assert not issubclass(FinishDeadlineExceeded, Exception)


def test_a_broad_except_exception_does_not_swallow_the_expiry():
    expired = Deadline(budget_seconds=0.0, started_at=0.0)
    swallowed = False
    try:
        try:
            expired.check("probe")
        except Exception:  # noqa: BLE001 -- the entire point of the test
            swallowed = True
    except FinishDeadlineExceeded:
        pass
    assert swallowed is False


def test_the_real_restore_frame_handler_cannot_swallow_the_expiry():
    # The REAL refusal path, not a stand-in: finish._restore_frame catches Exception and returns the
    # frame untouched, which is right for a bad frame and would be a silent unbounded invocation for
    # an expiry. The expiry is raised from INSIDE the restorer so the PRE-EXISTING handler is the
    # thing under test.
    class _Expiring:
        def restore(self, frame, fidelity=None, only_faces=None):
            raise FinishDeadlineExceeded("face_restore", 1.0, 0.5)

    with pytest.raises(FinishDeadlineExceeded):
        finish._restore_frame(_Expiring(), object(), FinishParams())


def test_the_real_restore_frame_handler_still_swallows_an_ordinary_error():
    # CONTROL for the test above. Without it, a _restore_frame that stopped swallowing ANYTHING
    # would pass that test for the wrong reason and the per-frame best-effort policy would be gone.
    class _Boom:
        def restore(self, frame, fidelity=None, only_faces=None):
            raise RuntimeError("detector missed")

    sentinel = object()
    assert finish._restore_frame(_Boom(), sentinel, FinishParams()) is sentinel


# --------------------------------------------------- WHERE the guard is checked (the N of M table)

class _Interp:
    def interpolate(self, a, b):
        return ("mid", a, b)


class _PassRestorer:
    def restore(self, frame, fidelity=None, only_faces=None):
        return frame


class _Server:
    def frame_interpolator(self):
        return _Interp()

    def face_restorer(self, backend=None):
        return _PassRestorer()


class _EncodeRec:
    """Consumes the streamed generator (so the interpolate checks actually run) without ffmpeg."""

    def __init__(self):
        self.frames = None

    def __call__(self, frames, out_path, fps, deadline=None):
        self.frames = list(frames)


class _patched_modules:
    def __init__(self, mapping):
        self.mapping = mapping
        self.saved = {}

    def __enter__(self):
        import sys
        for name, mod in self.mapping.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *a):
        import sys
        for name, prev in self.saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        return False


def _fake_imageio(frames_in):
    v3 = types.ModuleType("imageio.v3")
    v3.immeta = lambda *a, **k: {"fps": 16}
    v3.imiter = lambda *a, **k: iter(frames_in)
    parent = types.ModuleType("imageio")
    parent.v3 = v3
    return {"imageio": parent, "imageio.v3": v3}


# The k-th deadline check in a 3-frame interpolate+face-restore run, in order. This IS the coverage
# table for finish_clip, expressed as a test: if a check is added, removed or reordered, this list
# stops matching and whoever moved it has to restate the coverage rather than let it drift.
_CHECK_ORDER = [
    (1, "start"),
    (2, "decode"),
    (5, "restorer_load"),
    (6, "face_restore"),
    (9, "interpolator_load"),
    (10, "interpolate"),
]


@pytest.mark.parametrize("kth,stage", _CHECK_ORDER)
def test_every_compute_stage_of_finish_clip_is_inside_the_budget(kth, stage, tmp_path, monkeypatch):
    # A budget of k - 0.5 expires at exactly the k-th check under the stepping clock, so each case
    # names the stage it lands on. A PARTIAL guard is worse than none, and this is what makes the
    # coverage claim checkable instead of a sentence in a PR body.
    monkeypatch.setattr(finish, "time", types.SimpleNamespace(monotonic=_Clock(1.0)))
    monkeypatch.setattr(finish, "_encode_uniform", _EncodeRec())
    monkeypatch.setattr(finish, "_source_has_audio", lambda p, deadline=None: False)
    params = FinishParams(interpolate=True, factor=2, face_restore=True)
    deadline = Deadline.start(kth - 0.5)
    with _patched_modules(_fake_imageio([_Frame() for _ in range(3)])):
        with pytest.raises(FinishDeadlineExceeded) as got:
            finish.finish_clip("s1", tmp_path / "in.mp4", tmp_path / "out.mp4",
                               _Server(), params=params, deadline=deadline)
    assert got.value.stage == stage


def test_finish_clip_without_a_deadline_is_never_interrupted(tmp_path, monkeypatch):
    # CONTROL, and #422 D. Same clock that expires every case above at its first check; the only
    # difference is deadline=None, so every check is skipped and the pass completes. This is what
    # keeps render / i2v_clip / train_lora provably untouched.
    monkeypatch.setattr(finish, "time", types.SimpleNamespace(monotonic=_Clock(1000.0)))
    rec = _EncodeRec()
    monkeypatch.setattr(finish, "_encode_uniform", rec)
    monkeypatch.setattr(finish, "_source_has_audio", lambda p, deadline=None: False)
    params = FinishParams(interpolate=True, factor=2, face_restore=True)
    with _patched_modules(_fake_imageio([_Frame() for _ in range(3)])):
        res = finish.finish_clip("s1", tmp_path / "in.mp4", tmp_path / "out.mp4",
                                 _Server(), params=params)
    assert res.frames_out == 5 and res.interpolated is True
    assert len(rec.frames) == 5


# ------------------------------------------- the blocking seams, against REAL subprocesses

@pytest.mark.skipif(not _HAVE_SHELL_TOOLS, reason="needs sleep and dd on PATH")
def test_the_watchdog_kills_a_stalled_encoder_and_reports_the_expiry(tmp_path, monkeypatch):
    # REAL Popen, REAL pipe, REAL watchdog. sleep never reads its stdin and never exits inside the
    # budget, which is exactly the shape of the hang this guard exists for: a per-frame check
    # cannot help, because proc.wait never returns to one. A faked Popen would prove the decision
    # and skip the mechanism.
    monkeypatch.setattr(finish, "_nvenc_available", lambda: False)
    monkeypatch.setattr(finish, "_encoder_argv", lambda *a, **k: ["sleep", "30"])
    monkeypatch.setattr(finish, "_frame_bytes", lambda fr: b"x")
    started = time.monotonic()
    with pytest.raises(FinishDeadlineExceeded) as got:
        finish._encode_uniform(iter([_Frame(), _Frame()]), tmp_path / "o.mp4", 16,
                               deadline=Deadline.start(0.4))
    assert got.value.stage == "encode"
    # THE assertion that makes this test able to go red. Without the watchdog, proc.wait simply
    # blocks until sleep exits on its own after 30s and the post-wait check raises anyway, so the
    # exception alone proves NOTHING about the mechanism -- only that the clock moved. The wall
    # time is what separates a killed encoder from one nobody interrupted.
    assert time.monotonic() - started < 5.0


@pytest.mark.skipif(not _HAVE_SHELL_TOOLS, reason="needs sleep and dd on PATH")
def test_a_healthy_encode_under_a_generous_budget_still_returns(tmp_path, monkeypatch):
    # POSITIVE CONTROL for the test above: same real seam, same real subprocess, only the budget
    # differs. Without it a watchdog that fired unconditionally would pass the stall test.
    monkeypatch.setattr(finish, "_nvenc_available", lambda: False)
    monkeypatch.setattr(finish, "_encoder_argv",
                        lambda *a, **k: ["dd", "of=/dev/null", "status=none"])
    monkeypatch.setattr(finish, "_frame_bytes", lambda fr: b"x")
    started = time.monotonic()
    finish._encode_uniform(iter([_Frame(), _Frame()]), tmp_path / "o.mp4", 16,
                          deadline=Deadline.start(30.0))
    # It returned because dd finished, not because a watchdog fired at 30s.
    assert time.monotonic() - started < 5.0


@pytest.mark.skipif(not _HAVE_SHELL_TOOLS, reason="needs sleep and dd on PATH")
def test_the_mux_subprocess_is_bounded_by_the_remaining_budget(tmp_path, monkeypatch):
    # The other blocking subprocess. A check BEFORE the call cannot help if ffmpeg itself never
    # returns, so the remaining budget is handed to subprocess.run, which kills the child.
    monkeypatch.setattr(finish, "_mux_audio_argv", lambda *a, **k: ["sleep", "30"])
    started = time.monotonic()
    with pytest.raises(FinishDeadlineExceeded) as got:
        finish._mux_audio(tmp_path / "v.mp4", tmp_path / "a.mp4", tmp_path / "o.mp4",
                          deadline=Deadline.start(0.4))
    assert got.value.stage == "mux_audio"
    # Same reason as the encode watchdog: without the timeout kwarg this still raises, 30 seconds
    # later, because the check AFTER the call sees an expired budget. Wall time is the mechanism.
    assert time.monotonic() - started < 5.0


def test_the_mux_gets_no_timeout_at_all_without_a_deadline(tmp_path, monkeypatch):
    # #422 D control: the render path never passes a deadline and must reach subprocess.run with no
    # timeout kwarg, byte-identical to before.
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(finish, "_mux_audio_argv", lambda *a, **k: ["true"])
    monkeypatch.setattr(subprocess, "run", fake_run)
    finish._mux_audio(tmp_path / "v", tmp_path / "a", tmp_path / "o")
    assert "timeout" not in seen


def test_source_has_audio_hands_the_probe_the_remaining_budget(tmp_path, monkeypatch):
    seen = {}

    def fake_probe(path, *, timeout=None):
        seen["timeout"] = timeout
        return True

    monkeypatch.setattr(assemble_mod, "probe_has_audio", fake_probe)
    assert finish._source_has_audio(tmp_path / "x.mp4", deadline=Deadline.start(50.0)) is True
    assert seen["timeout"] is not None and 0 < seen["timeout"] <= 50.0


def test_source_has_audio_leaves_the_probe_untouched_without_a_deadline(tmp_path, monkeypatch):
    # #422 D control: probe_has_audio keeps its no-timeout default, so assemble and the render path
    # are unchanged.
    seen = {}

    def fake_probe(path, **kw):
        seen.update(kw)
        return False

    monkeypatch.setattr(assemble_mod, "probe_has_audio", fake_probe)
    assert finish._source_has_audio(tmp_path / "x.mp4") is False
    assert seen == {}


def test_a_probe_timeout_reports_the_expiry_and_never_returns_false(tmp_path, monkeypatch):
    # A false negative here silently drops the dialogue track (#240), so a probe that ran out of
    # budget must NOT degrade to no-audio. The stepping clock lets the first check pass and the
    # second one fire, which is the real ordering.
    def boom(path, *, timeout=None):
        raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=timeout or 0)

    monkeypatch.setattr(assemble_mod, "probe_has_audio", boom)
    monkeypatch.setattr(finish, "time", types.SimpleNamespace(monotonic=_Clock(1.0)))
    with pytest.raises(FinishDeadlineExceeded):
        finish._source_has_audio(tmp_path / "x.mp4", deadline=Deadline.start(1.5))


def test_probe_has_audio_default_keeps_the_render_path_unchanged(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return types.SimpleNamespace(returncode=0, stdout="audio", stderr="")

    monkeypatch.setattr(assemble_mod.subprocess, "run", fake_run)
    assert assemble_mod.probe_has_audio(tmp_path / "x.mp4") is True
    assert seen.get("timeout") is None


# ------------------------------------------- the ONE catch site: return data, never raise

def _stub_handler_deps(monkeypatch):
    """Neutralize everything in handler below the deadline, so the expiry path is what is measured.
    Nothing about the guard itself is stubbed: the env, the Deadline, the check and the catch are
    all real, and the expiry comes from a real check rather than a planted raise."""
    from vivijure_backend.harness import job_done_diag, models_mirror, pipeline_registry
    from vivijure_backend.harness.r2 import R2, R2Config

    monkeypatch.setattr(models_mirror, "ensure_models", lambda *a, **k: False)
    monkeypatch.setattr(models_mirror, "start_i2v_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(job_done_diag, "register", lambda *a, **k: None)
    monkeypatch.setattr(handler_mod, "_runpod_progress_hook", lambda job: None)
    monkeypatch.setattr(R2Config, "from_payload_or_env",
                        classmethod(lambda cls, payload, env=None: R2Config("a", "b", "c", "d")))
    monkeypatch.setattr(R2, "__init__", lambda self, config: None)


def test_an_expiry_returns_a_structured_soft_degrade_and_never_raises(monkeypatch):
    # THE point of the whole change. A raise here leaves no structured output, classifies
    # deterministic in vivijure-core and fails the WHOLE render after the GPU spend is banked,
    # which is strictly worse than the unbounded hang the phase ceiling at least recovers. So the
    # assertion is on the RETURNED dict, and the shape is the consumer contract:
    #   ok: False        the entire discriminator (softDegradeInCompletedOutput)
    #   detail           the first key degradeReason reads, sliced to 120 there
    #   no top-level error, which RunPod would lift into a FAILED envelope
    _stub_handler_deps(monkeypatch)
    monkeypatch.setenv("VJ_FINISH_MAX_SECONDS", "0.000001")
    out = handler_mod.handler({"id": "j1", "input": {"action": "finish_clip", "project": "p"}})
    assert set(out) == {"ok", "detail"}
    assert out["ok"] is False
    assert "error" not in out
    assert out["detail"].startswith("finish_clip deadline")
    assert len(out["detail"]) <= 120


def test_the_guard_does_not_leak_outside_finish_clip(monkeypatch):
    # CONTROL for the test above and the #422 D assertion. The SAME impossible budget, the only
    # difference being the action. If the guard leaked, every other action on this worker would be
    # one env var away from returning ok:false, and nothing downstream distinguishes that from a
    # real degrade.
    _stub_handler_deps(monkeypatch)
    monkeypatch.setenv("VJ_FINISH_MAX_SECONDS", "0.000001")
    monkeypatch.setattr(handler_mod, "run_job", lambda payload, **kw: {"ok": True, "ran": True})
    out = handler_mod.handler({"id": "j2", "input": {"action": "render", "project": "p"}})
    assert out == {"ok": True, "ran": True}


def test_run_finish_job_forwards_the_budget_into_finish_clip(tmp_path, monkeypatch):
    # The seam BETWEEN the two files. A deadline that stopped at run_finish_job would leave the
    # entire compute path unguarded while every other test here still passed.
    seen = {}

    class _Store:
        def get_file(self, key, dest):
            Path(dest).write_bytes(b"MP4")

        def put_file(self, path, key, **kw):
            return key

        def put_bytes(self, data, key, **kw):
            return key

    def spy_finish(shot_id, in_path, out_path, server, params=None, deadline=None):
        seen["deadline"] = deadline
        Path(out_path).write_bytes(b"MP4")
        return types.SimpleNamespace(interpolated=True, face_restored=False,
                                     out_fps=32, frames_out=9)

    monkeypatch.setattr(finish, "finish_clip", spy_finish)
    monkeypatch.setattr("vivijure_backend.models.ModelServer", lambda *a, **k: object())
    budget = Deadline.start(99.0)
    handler_mod.run_finish_job(
        {"action": "finish_clip", "project": "neon", "shot_id": "s1",
         "clip_key": "renders/neon/clips/s1_i2v.mp4", "config": {}},
        store=_Store(), workdir=tmp_path, deadline=budget)
    assert seen["deadline"] is budget
