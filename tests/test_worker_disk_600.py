"""vivijure-backend#600: a worker whose disk cannot hold the job fails that one job by number and is replaced.

Measured on 2026-11-06: one worker filled its disk (a 29 GiB stills image plus a ~114 GiB Wan pull on an 80 GB disk)
and then failed every job it took within seconds on ENOSPC. The job workdir was already removed per job; the model
cache, which persists across jobs on purpose, is what outgrew the disk. So: a pull that cannot fit fails before it
writes, a disk below a floor fails a job before it starts, and either (or any ENOSPC) RETURNS the error with RunPod's
`refresh_worker` so the SDK stops the pod. Each job's started / complete / error event carries the free disk.
"""
import errno
import json
import subprocess

import pytest

from vivijure_backend.harness import disk, models_mirror
from vivijure_backend.harness import handler as handler_mod
from vivijure_backend.harness.disk import GIB, DiskSpaceError
from vivijure_backend.harness.progress import ProgressEmitter

# ------------------------------------------------------------------ disk.py


def test_free_bytes_measures_a_path_that_does_not_exist_yet_at_its_parent(tmp_path):
    assert disk.free_bytes(tmp_path / "not" / "yet") == disk.free_bytes(tmp_path) > 0


def test_free_gb_is_none_when_it_cannot_measure(monkeypatch):
    def boom(_p):
        raise OSError("no such filesystem")
    monkeypatch.setattr(disk, "free_bytes", boom)
    assert disk.free_gb("/x") is None


@pytest.mark.parametrize("env,want_gb", [({}, 4.0), ({"VJ_MIN_FREE_GB": "10"}, 10.0), ({"VJ_MIN_FREE_GB": "0"}, 0.0),
                                         ({"VJ_MIN_FREE_GB": "-3"}, 0.0), ({"VJ_MIN_FREE_GB": "lots"}, 4.0), (None, 4.0)])
def test_the_floor(env, want_gb):
    assert disk.min_free_bytes(env) == int(want_gb * GIB)


def _free(monkeypatch, gib):
    monkeypatch.setattr(disk, "free_bytes", lambda _p: int(gib * GIB))


def test_the_job_floor_fails_below_and_passes_at_or_above(monkeypatch):
    _free(monkeypatch, 3.9)
    with pytest.raises(DiskSpaceError, match=r"3\.9 GiB free at /w, below the 4\.0 GiB"):
        disk.check_job_floor("/w")
    _free(monkeypatch, 4.0)
    disk.check_job_floor("/w")


def test_a_pull_must_leave_the_floor_free(monkeypatch):
    _free(monkeypatch, 120)
    disk.check_pull_fits("wan", int(116 * GIB), "/m")  # 116 + 4 = 120: fits exactly
    with pytest.raises(DiskSpaceError, match=r"'wan' does not fit: needs 117\.0 GiB plus 4\.0 GiB .* 120\.0 GiB free"):
        disk.check_pull_fits("wan", int(117 * GIB), "/m")


def _enospc():
    return OSError(errno.ENOSPC, "No space left on device")


def test_is_disk_failure_reads_the_chain():
    assert disk.is_disk_failure(DiskSpaceError("x"))
    assert disk.is_disk_failure(_enospc())
    wrapped = RuntimeError("download failed")
    wrapped.__cause__ = _enospc()
    assert disk.is_disk_failure(wrapped)
    implicit = RuntimeError("during handling")
    implicit.__context__ = _enospc()
    assert disk.is_disk_failure(implicit)
    assert not disk.is_disk_failure(OSError(errno.EACCES, "denied"))
    assert not disk.is_disk_failure(RuntimeError("other"))
    assert not disk.is_disk_failure(None)
    a, b = RuntimeError("a"), RuntimeError("b")
    a.__cause__, b.__cause__ = b, a  # a cycle is not walked forever
    assert not disk.is_disk_failure(a)


# ------------------------------------------------------------------ the mirror


def test_size_cmd_measures_exactly_what_the_copy_copies(tmp_path):
    copy = models_mirror.mirror_cmd("r2:b/models/hf-cache", tmp_path, skip_repos=("models--Wan",),
                                    filters=["+ hub/models--A/**", "- **"])
    size = models_mirror.size_cmd_for(copy)
    assert size[:4] == ["rclone", "size", "--json", "--links"]
    assert size[-1] == "r2:b/models/hf-cache" and str(tmp_path) not in size
    pairs = [(size[i], size[i + 1]) for i in range(4, len(size) - 1, 2)]
    assert ("--exclude", "hub/models--Wan/**") in pairs
    assert ("--filter", "+ hub/models--A/**") in pairs and ("--filter", "- **") in pairs
    assert ("--exclude", models_mirror._INCOMPLETE_GLOB) in pairs
    assert "--transfers" not in size and "copy" not in size


def test_size_cmd_refuses_what_is_not_a_copy():
    with pytest.raises(ValueError):
        models_mirror.size_cmd_for(["rclone", "sync", "a", "b"])


class _Ran:
    def __init__(self, rc, out):
        self.returncode, self.stdout = rc, out


# The real function, captured at import: conftest stubs models_mirror.remote_bytes per test, after collection.
_REAL_REMOTE_BYTES = models_mirror.remote_bytes


@pytest.mark.parametrize("ran,want", [(_Ran(0, json.dumps({"count": 3, "bytes": 123})), 123),
                                      (_Ran(1, ""), None), (_Ran(0, "not json"), None), (_Ran(0, "{}"), None)])
def test_remote_bytes_says_none_when_it_cannot_say(monkeypatch, ran, want):
    monkeypatch.setattr(models_mirror.subprocess, "run", lambda *a, **k: ran)
    assert _REAL_REMOTE_BYTES(["rclone", "size"], {}) == want


def test_remote_bytes_is_none_on_a_timeout_or_no_rclone(monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("rclone", 120)

    def missing(*a, **k):
        raise FileNotFoundError("rclone")
    for fake in (timeout, missing):
        monkeypatch.setattr(models_mirror.subprocess, "run", fake)
        assert _REAL_REMOTE_BYTES(["rclone", "size"], {}) is None


def _copy(tmp_path):
    return models_mirror.mirror_cmd("r2:b/models/x", tmp_path / "dst")


def test_an_unmeasured_pull_goes_ahead_and_says_so(tmp_path, monkeypatch):
    logs = []
    _free(monkeypatch, 0)  # would fail anything measured
    models_mirror.guard_pull("x", _copy(tmp_path), {}, {}, logs.append)
    assert any("could not measure 'x'" in m for m in logs)


def test_a_pull_that_does_not_fit_fails_before_it_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(models_mirror, "remote_bytes", lambda cmd, env: int(114 * GIB))
    _free(monkeypatch, 51)
    with pytest.raises(DiskSpaceError, match=r"'x' does not fit: needs 114\.0 GiB"):
        models_mirror.guard_pull("x", _copy(tmp_path), {}, {}, lambda m: None)


def test_bytes_already_on_disk_are_not_counted_again(tmp_path, monkeypatch):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "part").write_bytes(b"\0" * 1000)
    monkeypatch.setattr(models_mirror, "remote_bytes", lambda cmd, env: 1000 + 10)
    seen = {}
    monkeypatch.setattr(disk, "check_pull_fits", lambda leg, needed, dest, env=None: seen.update(needed=needed))
    models_mirror.guard_pull("x", _copy(tmp_path), {}, {}, lambda m: None)
    assert seen == {"needed": 10}


_R2 = {"R2_ACCESS_KEY_ID": "AKIAFAKE", "R2_SECRET_ACCESS_KEY": "fake-secret",
       "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com", "R2_BUCKET": "vivijure"}


def _mirror_fakes(monkeypatch):
    calls = []
    monkeypatch.setattr(models_mirror.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(models_mirror, "_run_rclone", lambda argv, env, deadline=None, **k: calls.append(argv) or 0)
    monkeypatch.setattr(models_mirror, "_reconstruct_symlinks", lambda *a, **k: 0)
    monkeypatch.setattr(models_mirror, "_jitter_seconds", lambda *_: 0.0)
    monkeypatch.setattr(models_mirror, "remote_bytes", lambda cmd, env: int(500 * GIB))
    _free(monkeypatch, 80)
    return calls


def test_the_cold_mirror_is_guarded_before_its_first_copy(tmp_path, monkeypatch):
    calls = _mirror_fakes(monkeypatch)
    env = {**_R2, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}
    with pytest.raises(DiskSpaceError):
        models_mirror.ensure_models(env=env, log=lambda *_: None)
    assert calls == []


def test_the_lazy_i2v_pull_is_guarded_before_its_first_copy(tmp_path, monkeypatch):
    calls = _mirror_fakes(monkeypatch)
    monkeypatch.setattr(models_mirror, "is_baked", lambda e: False)
    env = {**_R2, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}
    with pytest.raises(DiskSpaceError):
        models_mirror.ensure_i2v_models(env=env, log=lambda *_: None)
    assert calls == []


def test_a_hub_repo_pull_is_guarded_before_its_copy(tmp_path, monkeypatch):
    calls = _mirror_fakes(monkeypatch)
    monkeypatch.setattr(models_mirror, "is_baked", lambda e: False)
    env = {**_R2, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}
    with pytest.raises(DiskSpaceError):
        models_mirror.ensure_hub_repos(("models--x--y",), env=env, log=lambda *_: None)
    assert calls == []


def test_a_pull_that_fits_runs(tmp_path, monkeypatch):
    calls = _mirror_fakes(monkeypatch)
    monkeypatch.setattr(models_mirror, "is_baked", lambda e: False)
    monkeypatch.setattr(models_mirror, "remote_bytes", lambda cmd, env: int(1 * GIB))
    env = {**_R2, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}
    assert models_mirror.ensure_hub_repos(("models--x--y",), env=env, log=lambda *_: None) is True
    assert len(calls) == 1 and calls[0][:2] == ["rclone", "copy"]


# ------------------------------------------------------------------ the handler


def _stub_store(monkeypatch):
    from vivijure_backend.harness import job_done_diag
    from vivijure_backend.harness.r2 import R2, R2Config
    monkeypatch.setattr(job_done_diag, "register", lambda *a, **k: None)
    monkeypatch.setattr(handler_mod, "_runpod_progress_hook", lambda job: None)
    monkeypatch.setattr(R2Config, "from_payload_or_env",
                        classmethod(lambda cls, payload, env=None: R2Config("a", "b", "c", "d")))
    monkeypatch.setattr(R2, "__init__", lambda self, config: None)


def _render(monkeypatch, *, mirror=lambda **k: False, run=None):
    from vivijure_backend.harness import pipeline_registry
    _stub_store(monkeypatch)
    monkeypatch.setattr(models_mirror, "ensure_models", mirror)
    monkeypatch.setattr(models_mirror, "start_i2v_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda *a, **k: object())
    ran = []
    monkeypatch.setattr(handler_mod, "run_job", run or (lambda payload, **kw: ran.append(1) or {"ran": True}))
    errors = []
    monkeypatch.setattr(handler_mod.ProgressEmitter, "error",
                        lambda self, stage, e, **f: errors.append((stage, str(e))))
    return ran, errors


RENDER = {"id": "j", "input": {"action": "render", "project": "p"}}


def test_a_pull_that_does_not_fit_returns_the_error_and_replaces_the_worker(monkeypatch):
    def mirror(**k):
        raise DiskSpaceError("model pull 'hf-cache' does not fit: needs 114.0 GiB")
    ran, errors = _render(monkeypatch, mirror=mirror)
    got = handler_mod.handler(RENDER)
    assert got == {"error": "model pull 'hf-cache' does not fit: needs 114.0 GiB; this worker is being replaced",
                   "refresh_worker": True}
    assert ran == []
    assert ("disk", "model pull 'hf-cache' does not fit: needs 114.0 GiB") in errors


def test_a_disk_below_the_floor_fails_before_the_job_starts(monkeypatch):
    ran, _ = _render(monkeypatch)
    _free(monkeypatch, 1)
    got = handler_mod.handler(RENDER)
    assert got["refresh_worker"] is True and "below the 4.0 GiB a job needs to start" in got["error"]
    assert ran == []


def test_an_enospc_inside_the_job_returns_the_error_and_replaces_the_worker(monkeypatch):
    def run(payload, **kw):
        raise _enospc()
    _render(monkeypatch, run=run)
    _free(monkeypatch, 50)
    got = handler_mod.handler(RENDER)
    assert got["refresh_worker"] is True and got["error"].startswith("worker disk full: ")
    assert "No space left on device" in got["error"] and got["error"].endswith("this worker is being replaced")


def test_any_other_failure_still_raises(monkeypatch):
    def run(payload, **kw):
        raise RuntimeError("a real bug")
    _render(monkeypatch, run=run)
    _free(monkeypatch, 50)
    with pytest.raises(RuntimeError, match="a real bug"):
        handler_mod.handler(RENDER)


def test_a_healthy_job_is_untouched(monkeypatch):
    _render(monkeypatch)
    _free(monkeypatch, 50)
    assert handler_mod.handler(RENDER) == {"ran": True}


# ------------------------------------------------------------------ the free-disk figure


def _emitter(tmp_path):
    lines = []
    return ProgressEmitter(None, "p", "j", log=lines.append, disk_path=str(tmp_path)), lines


@pytest.mark.parametrize("event", ["started", "complete", "error"])
def test_start_and_end_events_carry_the_free_disk(tmp_path, monkeypatch, event):
    _free(monkeypatch, 42.04)
    em, _ = _emitter(tmp_path)
    if event == "complete":
        em.complete()
    elif event == "error":
        em.error("render", "boom")
    else:
        em.emit("started")
    assert em._events[-1]["disk_free_gb"] == 42.0


def test_other_events_do_not(tmp_path, monkeypatch):
    _free(monkeypatch, 42)
    em, _ = _emitter(tmp_path)
    em.emit("keyframe_done", shot="s")
    assert "disk_free_gb" not in em._events[-1]


def test_an_unmeasurable_disk_reads_none_never_zero(tmp_path, monkeypatch):
    def boom(_p):
        raise OSError("gone")
    monkeypatch.setattr(disk, "free_bytes", boom)
    em, _ = _emitter(tmp_path)
    em.emit("started")
    assert em._events[-1]["disk_free_gb"] is None


def test_the_default_disk_path_is_the_job_temp_root():
    import tempfile
    assert ProgressEmitter(None, "p", "j")._disk_path == tempfile.gettempdir()
