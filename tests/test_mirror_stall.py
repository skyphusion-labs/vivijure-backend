"""A model pull ends on a STALL, not only on its budget, and reports progress while it runs (#566).

The budgets in finish.py bound a whole pull, so they cannot tell a slow pull from a hung one. The stall
check reads rclone's own stats lines. The parser is held to lines captured from a real rclone 1.75 run,
and the kill tests run a real child process, so the watchdog, the pipe and the kill are the shipped ones.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from vivijure_backend import finish
from vivijure_backend.finish import (
    DEFAULT_MIRROR_STALL_SECONDS,
    MIN_MIRROR_STALL_SECONDS,
    MIRROR_STALL_SECONDS_ENV,
    Deadline,
    FinishDeadlineExceeded,
    MirrorStalled,
    max_mirror_stall_seconds,
)
from vivijure_backend.harness import handler as handler_mod
from vivijure_backend.harness import models_mirror as mm

CREDS = {"R2_ACCESS_KEY_ID": "AK", "R2_SECRET_ACCESS_KEY": "SK",
         "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com", "R2_BUCKET": "vivijure"}

# Captured verbatim from `rclone copy --disable Copy --stats 1s --stats-one-line -v --bwlimit 8M`,
# rclone v1.75.2, 2026-11-06. Do not tidy these: the spacing is what rclone prints.
REAL_STATS = [
    "2026/11/06 06:59:44 INFO  :     8.027 MiB / 24.000 MiB, 33%, 0 B/s, ETA -",
    "2026/11/06 06:59:45 INFO  :    16.027 MiB / 24.000 MiB, 67%, 8.027 MiB/s, ETA 0s",
    "2026/11/06 06:59:46 INFO  :    24.000 MiB / 24.000 MiB, 100%, 8.014 MiB/s, ETA 0s",
]
REAL_OTHER = [
    "2026/11/06 06:59:43 INFO  : Starting bandwidth limiter at 8Mi Byte/s",
    "2026/11/06 06:59:43 INFO  : s.txt: Copied (new)",
    "2026/11/06 06:59:46 ERROR : Attempt 1/3 failed with 1 errors and: directory not found",
]


def _rclone(tmp_path: Path, monkeypatch, script: str) -> None:
    """A real executable named rclone on PATH. `exec` in a script makes the stall the killed process."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "rclone"
    exe.write_text("#!/bin/sh\n" + script + "\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def _stats(mib: float) -> str:
    return f'echo "2026/11/06 06:59:45 INFO  :    {mib:.3f} MiB / 24.000 MiB, 50%, 1 MiB/s, ETA 1s"'


# ------------------------------------------------------------------ the parser, on real lines

def test_the_parser_reads_every_real_stats_line():
    mib = 1 << 20
    assert [mm.parse_rclone_stats(line) for line in REAL_STATS] == [
        (int(8.027 * mib), 24 * mib), (int(16.027 * mib), 24 * mib), (24 * mib, 24 * mib)]


def test_the_parser_ignores_every_other_real_line():
    assert [mm.parse_rclone_stats(line) for line in REAL_OTHER] == [None, None, None]


@pytest.mark.parametrize("line, want", [
    ("INFO  :         0 B / 0 B, -, 0 B/s, ETA -", (0, 0)),
    ("INFO  :    1.500 GiB / 24.425 GiB, 6%, 9.1 MiB/s, ETA 41m", (int(1.5 * (1 << 30)), int(24.425 * (1 << 30)))),
    ("INFO  :   512 KiB / 2 TiB, 0%, 1 KiB/s, ETA 1y", (512 << 10, 2 << 40)),
])
def test_the_parser_reads_each_unit_and_an_empty_total(line, want):
    assert mm.parse_rclone_stats(line) == want


# ------------------------------------------------------------------ the stall window

def test_the_stall_window_defaults_and_cannot_be_turned_off():
    assert max_mirror_stall_seconds({}) == DEFAULT_MIRROR_STALL_SECONDS
    for junk in ("", "x", "0", "-5", "nan"):
        assert max_mirror_stall_seconds({MIRROR_STALL_SECONDS_ENV: junk}) == DEFAULT_MIRROR_STALL_SECONDS


def test_a_window_shorter_than_two_stats_lines_is_raised_to_the_floor():
    assert MIN_MIRROR_STALL_SECONDS == 2 * finish.MIRROR_STATS_SECONDS
    assert max_mirror_stall_seconds({MIRROR_STALL_SECONDS_ENV: "30"}) == MIN_MIRROR_STALL_SECONDS
    assert max_mirror_stall_seconds({MIRROR_STALL_SECONDS_ENV: "900"}) == 900.0


def test_rclone_prints_stats_as_often_as_the_floor_assumes():
    argv = mm.mirror_cmd("r2:b/models/x", Path("/tmp/x"))
    assert argv[argv.index("--stats") + 1] == f"{finish.MIRROR_STATS_SECONDS}s"
    assert "--stats-one-line" in argv and "-v" in argv


def test_a_stall_routes_like_an_expired_budget_and_names_itself():
    e = MirrorStalled("model_mirror", 312.4, 300, 12_340_000_000)
    assert isinstance(e, FinishDeadlineExceeded)
    assert e.reason == str(e) and len(e.reason) <= 120
    assert "stalled" in e.reason and "300s" in e.reason and "12.34 GB" in e.reason


# ------------------------------------------------------------------ a real rclone, really killed

def test_a_pull_that_stops_moving_is_killed_as_stalled(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, _stats(5) + "; exec sleep 60")
    started = time.monotonic()
    with pytest.raises(MirrorStalled) as hit:
        mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=0.4)
    assert time.monotonic() - started < 20
    assert hit.value.moved_bytes == 5 << 20 and hit.value.stage == "model_mirror"


def test_an_rclone_that_never_prints_a_stats_line_is_a_stall(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, "exec sleep 60")
    with pytest.raises(MirrorStalled) as hit:
        mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=0.4)
    assert hit.value.moved_bytes == 0


def test_stats_lines_that_keep_arriving_with_the_same_bytes_are_still_a_stall(tmp_path, monkeypatch):
    # The read loop keeps running here, so this is the case a line-arrival check would miss. The loop is
    # bounded and then exits 0, so a regression fails this test in seconds instead of hanging the suite.
    _rclone(tmp_path, monkeypatch,
            "i=0; while [ $i -lt 100 ]; do " + _stats(7) + "; sleep 0.05; i=$((i+1)); done; exit 0")
    with pytest.raises(MirrorStalled) as hit:
        mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=0.4)
    assert hit.value.moved_bytes == 7 << 20


def test_a_slow_pull_that_keeps_moving_is_left_alone(tmp_path, monkeypatch):
    # Moves every 0.1 s for about 1.5 s, against a 0.5 s window: three windows long, never idle for one.
    lines = "; ".join(_stats(i) + "; sleep 0.1" for i in range(1, 16))
    _rclone(tmp_path, monkeypatch, lines)
    seen = []
    assert mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=0.5,
                          on_stats=lambda moved, planned: seen.append(moved)) == 0
    assert seen == [i << 20 for i in range(1, 16)]


def test_a_shorter_budget_still_wins_as_a_budget(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, "exec sleep 60")
    with pytest.raises(FinishDeadlineExceeded) as hit:
        mm._run_rclone(["rclone", "copy"], dict(os.environ), Deadline.start(0.3), stall_seconds=30)
    assert not isinstance(hit.value, MirrorStalled)


def test_a_failing_progress_report_never_touches_the_pull(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, _stats(1) + "; " + _stats(2))

    def boom(moved, planned):
        raise RuntimeError("progress sink down")

    assert mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=5, on_stats=boom) == 0


def test_an_ordinary_failure_is_still_an_ordinary_failure(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, _stats(1) + "; exit 3")
    with pytest.raises(subprocess.CalledProcessError):
        mm._run_rclone(["rclone", "copy"], dict(os.environ), stall_seconds=5)


def test_the_default_window_is_the_one_the_environment_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "max_mirror_stall_seconds", lambda: 0.4)
    _rclone(tmp_path, monkeypatch, "exec sleep 60")
    with pytest.raises(MirrorStalled):
        mm._run_rclone(["rclone", "copy"], dict(os.environ))


# ------------------------------------------------------------------ every pull path uses it

def _cold_env(tmp_path: Path) -> dict:
    return {**CREDS, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}


def test_a_cold_pull_reports_each_stats_line_by_leg(tmp_path, monkeypatch):
    def fake(argv, env, deadline=None, *, on_stats=None, **_kw):
        on_stats(3, 9)
        return 0

    monkeypatch.setattr(mm.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(mm, "_run_rclone", fake)
    monkeypatch.setattr(mm, "_reconstruct_symlinks", lambda *a, **k: 0)
    monkeypatch.setattr(mm, "_jitter_seconds", lambda *_: 0.0)
    events: list[dict] = []
    assert mm.ensure_models(env=_cold_env(tmp_path), log=lambda *_: None, on_progress=events.append) is True
    assert [e["leg"] for e in events][:2] == ["hf-cache", "antelopev2"]
    assert all(e["moved_bytes"] == 3 and e["planned_bytes"] == 9 and e["leg_seconds"] >= 0 for e in events)


def _stalls(*a, **k):
    raise MirrorStalled("model_mirror", 301.0, 300, 1)


def test_the_lazy_i2v_pull_ends_on_a_stall(tmp_path, monkeypatch):
    # A no-op rclone on PATH, so a pull that bypasses _run_rclone succeeds at once and fails this test fast.
    _rclone(tmp_path, monkeypatch, "exit 0")
    monkeypatch.setattr(mm.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(mm, "_run_rclone", _stalls)
    with pytest.raises(MirrorStalled):
        mm.ensure_i2v_models(env=_cold_env(tmp_path), log=lambda *_: None)
    assert not (tmp_path / mm.I2V_SENTINEL).exists()


def test_a_hub_repo_pull_ends_on_a_stall(tmp_path, monkeypatch):
    _rclone(tmp_path, monkeypatch, "exit 0")  # see the i2v test above
    monkeypatch.setattr(mm.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(mm, "_run_rclone", _stalls)
    with pytest.raises(MirrorStalled):
        mm.ensure_hub_repos(("models--xinsir--controlnet-canny-sdxl-1.0",), env=_cold_env(tmp_path),
                            log=lambda *_: None)


def test_a_stalled_volume_fill_falls_back_and_releases_its_lock(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    vol.mkdir()
    monkeypatch.setattr(mm, "ensure_models", _stalls)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert mm._self_preload_volume(e, "1", log=lambda *_: None) is mm.VolumeState.INCOMPLETE
    assert not (vol / mm._PRELOAD_LOCK).exists()
    assert "HF_HOME" not in e


# ------------------------------------------------------------------ the handler

def _stub_store(monkeypatch):
    from vivijure_backend.harness import job_done_diag
    from vivijure_backend.harness.r2 import R2, R2Config
    monkeypatch.setattr(job_done_diag, "register", lambda *a, **k: None)
    monkeypatch.setattr(handler_mod, "_runpod_progress_hook", lambda job: None)
    monkeypatch.setattr(R2Config, "from_payload_or_env",
                        classmethod(lambda cls, payload, env=None: R2Config("a", "b", "c", "d")))
    monkeypatch.setattr(R2, "__init__", lambda self, config: None)


def _render_with(monkeypatch, mirror):
    from vivijure_backend.harness import models_mirror, pipeline_registry
    _stub_store(monkeypatch)
    monkeypatch.setattr(models_mirror, "ensure_models", mirror)
    monkeypatch.setattr(models_mirror, "start_i2v_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda *a, **k: object())
    ran: list[int] = []
    monkeypatch.setattr(handler_mod, "run_job", lambda payload, **kw: ran.append(1) or {"ran": True})
    return ran


RENDER = {"id": "j", "input": {"action": "render", "project": "p"}}


def test_a_stalled_pull_fails_a_render_by_its_own_name(monkeypatch):
    ran = _render_with(monkeypatch, _stalls)
    monkeypatch.setattr(handler_mod, "_finish_deadline_degrade",
                        lambda *a, **k: pytest.fail("a render has nothing to pass through"))
    errors = []
    monkeypatch.setattr(handler_mod.ProgressEmitter, "error",
                        lambda self, stage, e: errors.append((stage, str(e))))
    with pytest.raises(RuntimeError, match=r"^model pull stalled: no bytes for 300s"):
        handler_mod.handler(RENDER)
    assert ran == []
    assert errors and errors[0][0] == "mirror" and "stalled" in errors[0][1]


def test_a_stalled_finish_pull_soft_degrades_like_its_budget(monkeypatch):
    _stub_store(monkeypatch)
    from vivijure_backend.harness import models_mirror
    monkeypatch.setattr(models_mirror, "ensure_models", _stalls)
    got = handler_mod.handler({"id": "j", "input": {"action": "finish_clip", "project": "p",
                                                    "config": {"interpolate": True, "face_restore": False}}})
    assert got["ok"] is False and "stalled" in got["detail"]


def test_the_handler_turns_pull_progress_into_events(monkeypatch):
    def mirror(**k):
        k["on_progress"]({"leg": "hf-cache", "moved_bytes": 5, "planned_bytes": 9, "leg_seconds": 1.0})
        return True

    _render_with(monkeypatch, mirror)
    emitted = []
    monkeypatch.setattr(handler_mod.ProgressEmitter, "emit",
                        lambda self, event, **fields: emitted.append((event, fields)))
    assert handler_mod.handler(RENDER) == {"ran": True}
    assert ("mirror_progress", {"leg": "hf-cache", "moved_bytes": 5, "planned_bytes": 9,
                                "leg_seconds": 1.0}) in emitted
