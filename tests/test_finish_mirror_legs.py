"""A finish job mirrors only what its step loads, and a finished mirror never costs the job (#553).

Measured on a cold worker: a RIFE step spent 464.7 s on the 48.7 GiB hf-cache leg it never opens, then
a check on the 420 s budget threw the ready worker away. rclone is a stub in the pull tests, except
where a stalled rclone is the subject: that one is a real process, so the kill is real.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from vivijure_backend import finish
from vivijure_backend.finish import (
    FACEXLIB_LEG,
    FINISH_LEGS,
    GFPGAN_LEG,
    RIFE_LEG,
    Deadline,
    FinishDeadlineExceeded,
    FinishParams,
    job_mirror_legs,
    max_mirror_seconds,
    mirror_legs,
)
from vivijure_backend.harness import handler as handler_mod
from vivijure_backend.harness import models_mirror as mm
from vivijure_backend.models import ModelServer

CREDS = {"R2_ACCESS_KEY_ID": "AK", "R2_SECRET_ACCESS_KEY": "SK",
         "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com", "R2_BUCKET": "vivijure"}


def _leg_of(argv: list[str]) -> str:
    """The R2 source's last segment: `r2:vivijure/models/rife` -> `rife`."""
    return next(a for a in argv if a.startswith("r2:")).rsplit("/", 1)[-1]


@pytest.fixture
def pulls(tmp_path, monkeypatch):
    """Stub rclone. Returns (env, pulled), where `pulled` lists each leg the mirror asked rclone for."""
    pulled: list[str] = []

    def fake(argv, env, deadline=None):
        pulled.append(_leg_of(argv))
        return 0

    monkeypatch.setattr(mm.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(mm, "_run_rclone", fake)
    monkeypatch.setattr(mm, "_reconstruct_symlinks", lambda *a, **k: 0)
    monkeypatch.setattr(mm, "_jitter_seconds", lambda *_: 0.0)
    return {**CREDS, "HF_HOME": str(tmp_path / "hf-cache"), "VJ_MODELS_ROOT": str(tmp_path)}, pulled


def _finish(env, legs, **kw):
    return mm.ensure_models(env=env, log=kw.pop("log", lambda *_: None), action="finish_clip",
                            finish_legs=legs, **kw)


# ------------------------------------------------------------ which legs a finish step loads

@pytest.mark.parametrize("params, legs", [
    (FinishParams(interpolate=True), (RIFE_LEG,)),
    (FinishParams(interpolate=True, factor=1), (RIFE_LEG,)),  # finish_clip still loads RIFE at 1x
    (FinishParams(face_restore=True), (GFPGAN_LEG, FACEXLIB_LEG)),
    (FinishParams(face_restore=True, face_restore_backend="codeformer"), (FACEXLIB_LEG,)),
    (FinishParams(face_restore=True, face_restore_backend="CodeFormer"), (FACEXLIB_LEG,)),
    (FinishParams(interpolate=True, face_restore=True), FINISH_LEGS),
    (FinishParams(), ()),
])
def test_mirror_legs_are_the_ones_the_step_loads(params, legs):
    assert mirror_legs(params) == legs


@pytest.mark.parametrize("config, legs", [
    (None, (RIFE_LEG,)),  # run_finish_job defaults interpolate on, so the mirror has to as well
    ({"interpolate": True, "interpolation_factor": 2, "face_restore": False}, (RIFE_LEG,)),  # the cf door
    ({"interpolate": False, "face_restore": True}, (GFPGAN_LEG, FACEXLIB_LEG)),
    ({"interpolate": False, "face_restore": "none"}, ()),
])
def test_job_mirror_legs_reads_the_job_as_the_job_runs_it(config, legs):
    assert job_mirror_legs({"config": config}) == legs


@pytest.mark.parametrize("config", [{"interpolation_factor": "junk"}, ["not", "a", "dict"]])
def test_a_config_that_will_fail_the_job_pulls_nothing(config):
    # NOT a fall back to the full mirror: this job fails on its config by name and loads no model.
    assert job_mirror_legs({"config": config}) == ()


def test_each_leg_is_the_directory_the_real_loader_opens(tmp_path, monkeypatch):
    # The un-stubbable half of the table above: the real ModelServer picks the paths.
    seen = {}
    monkeypatch.setenv("VJ_MODELS_ROOT", str(tmp_path))
    from vivijure_backend import models
    monkeypatch.setattr(models, "_RifeInterpolator", lambda p: seen.setdefault("rife", p))
    monkeypatch.setattr(models, "_GfpganRestorer", lambda p, root: seen.setdefault("gfpgan", p))
    server = ModelServer()
    server.frame_interpolator()
    server.face_restorer("gfpgan")
    assert Path(seen["rife"]).parent == tmp_path / RIFE_LEG
    assert Path(seen["gfpgan"]).parent == tmp_path / GFPGAN_LEG


def test_the_facexlib_leg_is_the_directory_the_shim_links_from(tmp_path, monkeypatch):
    from vivijure_backend import models
    (tmp_path / FACEXLIB_LEG).mkdir()
    (tmp_path / FACEXLIB_LEG / models._FACEXLIB_WEIGHTS[0]).write_bytes(b"w")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    models._ensure_facexlib_offline(str(tmp_path))
    assert (cwd / "gfpgan" / "weights" / models._FACEXLIB_WEIGHTS[0]).read_bytes() == b"w"


# ------------------------------------------------------------ what the mirror pulls, and when

def test_a_cold_rife_job_pulls_only_the_rife_leg(pulls, tmp_path):
    env, pulled = pulls
    lines = []
    assert _finish(env, (RIFE_LEG,), log=lines.append) is True
    assert pulled == [RIFE_LEG]
    assert not (tmp_path / mm.SENTINEL).exists()  # the rest of the base set is still owed
    # The mirror-timing record is unchanged in shape.
    stats = mm.last_mirror_stats()
    assert list(stats["legs"]) == [RIFE_LEG] and "wall_seconds" in stats
    assert isinstance(stats["total_bytes"], int)
    assert any(line.startswith("@event mirror_complete") for line in lines)


def test_a_warm_worker_does_not_copy_the_leg_again(pulls):
    env, pulled = pulls
    _finish(env, (RIFE_LEG,))
    pulled.clear()
    assert _finish(env, (RIFE_LEG,)) is False
    assert pulled == []
    assert mm.last_mirror_stats() is None  # a warm job reports no pull, never the previous job's


def test_a_later_finish_job_pulls_only_the_legs_still_missing(pulls):
    env, pulled = pulls
    _finish(env, (RIFE_LEG,))
    pulled.clear()
    assert _finish(env, FINISH_LEGS) is True
    assert pulled == [GFPGAN_LEG, FACEXLIB_LEG]


def test_a_later_render_still_pulls_the_rest_and_leaves_the_leg_alone(pulls, tmp_path):
    env, pulled = pulls
    _finish(env, (RIFE_LEG,))
    pulled.clear()
    assert mm.ensure_models(env=env, log=lambda *_: None, action="render") is True
    assert pulled == ["hf-cache", "antelopev2", GFPGAN_LEG, FACEXLIB_LEG]
    assert (tmp_path / mm.SENTINEL).exists()
    pulled.clear()
    assert _finish(env, FINISH_LEGS) is False  # now warm for every finish step
    assert pulled == []


def test_a_later_stills_job_still_pulls_its_own_set(pulls, monkeypatch):
    env, pulled = pulls
    _finish(env, (RIFE_LEG,))
    pulled.clear()
    monkeypatch.setattr(mm.stills_manifest, "prune_dangling_links", lambda *a, **k: 0)
    monkeypatch.setattr(mm.stills_manifest, "verify", lambda *a, **k: None)
    assert mm.ensure_models(env=env, log=lambda *_: None, action="preview") is True
    assert pulled == ["hf-cache", "antelopev2"]


def test_a_finish_job_on_a_fully_mirrored_worker_pulls_nothing(pulls, tmp_path):
    env, pulled = pulls
    (tmp_path / mm.SENTINEL).write_text(mm._DEFAULT_MODEL_VERSION + "\n")
    assert _finish(env, FINISH_LEGS) is False
    assert pulled == []


def test_a_model_version_bump_pulls_the_leg_again(pulls):
    env, pulled = pulls
    _finish(env, (RIFE_LEG,))
    pulled.clear()
    assert _finish({**env, "VJ_MODEL_VERSION": "2"}, (RIFE_LEG,)) is True
    assert pulled == [RIFE_LEG]


def test_a_step_that_loads_nothing_pulls_nothing(pulls):
    env, pulled = pulls
    lines = []
    assert _finish(env, (), log=lines.append) is False
    assert pulled == []
    assert any("finish_no_legs" in line for line in lines)


def test_a_baked_worker_still_skips_before_any_leg(pulls, tmp_path):
    env, pulled = pulls
    (tmp_path / mm.BAKED_SENTINEL).write_text("baked\n")
    assert _finish(env, FINISH_LEGS) is False
    assert pulled == []


def test_a_failed_leg_writes_no_sentinel(pulls, tmp_path, monkeypatch):
    import subprocess
    env, _ = pulls

    def boom(argv, env, deadline=None):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(mm, "_run_rclone", boom)
    with pytest.raises(subprocess.CalledProcessError):
        _finish(env, (RIFE_LEG,))
    assert not (tmp_path / mm.FINISH_LEG_SENTINEL.format(RIFE_LEG)).exists()


def test_a_finish_job_does_not_fill_a_network_volume(tmp_path, monkeypatch):
    # The fill is a pull of the whole corpus, which is the defect again, and no budget can end it.
    vol = tmp_path / "vol"
    vol.mkdir()
    monkeypatch.setattr(mm, "_self_preload_volume",
                        lambda *a, **k: pytest.fail("a finish job started a volume fill"))
    e = {"VJ_VOLUME_ROOT": str(vol), "VJ_VOLUME_KIND": "network", "VJ_VOLUME_SELF_PRELOAD": "1"}
    assert mm._resolve_volume(e, "1", lambda *_: None, action="finish_clip") is mm.VolumeState.INCOMPLETE


# ------------------------------------------------------------ the real rclone, on a local remote

@pytest.mark.skipif(shutil.which("rclone") is None, reason="needs the rclone binary (CI installs it)")
def test_real_rclone_lands_only_the_leg_then_the_rest(tmp_path, monkeypatch):
    # The stubs above prove which legs were ASKED for. This proves what lands on disk, through the real
    # argv and the real binary, with only the remote's type swapped from s3 to local.
    bucket = tmp_path / "bucket"
    for rel in ("rife/flownet.pkl", "GFPGANv1.4/GFPGANv1.4.pth", "facexlib/detection_Resnet50_Final.pth",
                "antelopev2/glintr100.onnx", "hf-cache/hub/models--o--r/blobs/big"):
        f = bucket / "models" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"w")
    monkeypatch.setattr(mm, "rclone_env", lambda e: {**os.environ, "RCLONE_CONFIG_R2_TYPE": "local"})
    monkeypatch.setattr(mm, "_jitter_seconds", lambda *_: 0.0)
    root = tmp_path / "worker"
    env = {**CREDS, "R2_BUCKET": str(bucket), "VJ_MODELS_ROOT": str(root), "HF_HOME": str(root / "hf-cache")}

    assert _finish(env, (RIFE_LEG,), deadline=Deadline.start(60)) is True
    assert (root / "rife" / "flownet.pkl").is_file()
    assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ["hf-cache", "rife"]
    assert not (root / "hf-cache" / "hub").exists()

    (root / "rife" / "flownet.pkl").write_bytes(b"marker")  # a re-copy would overwrite this
    assert _finish(env, (RIFE_LEG,), deadline=Deadline.start(60)) is False
    assert mm.ensure_models(env=env, log=lambda *_: None, action="render") is True
    assert (root / "rife" / "flownet.pkl").read_bytes() == b"marker"
    assert (root / "hf-cache" / "hub" / "models--o--r" / "blobs" / "big").is_file()
    assert (root / "GFPGANv1.4" / "GFPGANv1.4.pth").is_file()
    assert (root / mm.SENTINEL).is_file()


# ------------------------------------------------------------ a true hang still ends

def _real_rclone_script(tmp_path, monkeypatch, script: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "rclone"
    exe.write_text("#!/bin/sh\n" + script + "\n")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def test_a_stalled_rclone_is_killed_at_the_budget(tmp_path, monkeypatch):
    # `exec` so the stall IS the process the watchdog kills, as real rclone is.
    _real_rclone_script(tmp_path, monkeypatch, "exec sleep 60")
    started = time.monotonic()
    with pytest.raises(FinishDeadlineExceeded) as hit:
        mm._run_rclone(["rclone", "copy"], dict(os.environ), Deadline.start(0.3))
    assert time.monotonic() - started < 20
    assert hit.value.stage == "model_mirror"


def test_an_ordinary_rclone_failure_stays_an_ordinary_failure(tmp_path, monkeypatch):
    import subprocess
    _real_rclone_script(tmp_path, monkeypatch, "exit 3")
    with pytest.raises(subprocess.CalledProcessError):
        mm._run_rclone(["rclone", "copy"], dict(os.environ), Deadline.start(30))


def test_a_healthy_rclone_under_a_budget_returns(tmp_path, monkeypatch):
    _real_rclone_script(tmp_path, monkeypatch, 'echo "ERROR : Attempt 1/3 failed"')
    assert mm._run_rclone(["rclone", "copy"], dict(os.environ), Deadline.start(30)) == 1


# ------------------------------------------------------------ the handler, end to end

def _stub_store(monkeypatch):
    from vivijure_backend.harness import job_done_diag
    from vivijure_backend.harness.r2 import R2, R2Config
    monkeypatch.setattr(job_done_diag, "register", lambda *a, **k: None)
    monkeypatch.setattr(handler_mod, "_runpod_progress_hook", lambda job: None)
    monkeypatch.setattr(R2Config, "from_payload_or_env",
                        classmethod(lambda cls, payload, env=None: R2Config("a", "b", "c", "d")))
    monkeypatch.setattr(R2, "__init__", lambda self, config: None)


JOB = {"id": "j", "input": {"action": "finish_clip", "project": "p",
                            "config": {"interpolate": True, "face_restore": False}}}


def test_a_mirror_longer_than_the_work_budget_still_runs_the_step(monkeypatch):
    # THE measured defect, at its own scale: the mirror finishes after the work budget's length. The
    # step has to run, with its whole budget still ahead of it.
    from vivijure_backend.harness import models_mirror
    _stub_store(monkeypatch)
    monkeypatch.setenv("VJ_FINISH_MAX_SECONDS", "0.2")
    monkeypatch.setattr(models_mirror, "ensure_models", lambda **k: time.sleep(0.3) or True)
    seen = {}

    def run(payload, **kw):
        seen.update(kw)
        return {"ran": True}

    monkeypatch.setattr(handler_mod, "run_finish_job", run)
    assert handler_mod.handler(JOB) == {"ran": True}
    assert seen["deadline"].elapsed() < 0.2


def test_the_handler_hands_the_mirror_the_jobs_legs_and_a_budget(monkeypatch):
    from vivijure_backend.harness import models_mirror
    _stub_store(monkeypatch)
    seen = {}
    monkeypatch.setattr(models_mirror, "ensure_models", lambda **k: seen.update(k) or False)
    monkeypatch.setattr(handler_mod, "run_finish_job", lambda payload, **kw: {"ran": True})
    handler_mod.handler(JOB)
    assert seen["finish_legs"] == (RIFE_LEG,)
    assert seen["deadline"].budget_seconds == finish.DEFAULT_MIRROR_MAX_SECONDS


def test_other_actions_keep_the_full_mirror_with_no_budget(monkeypatch):
    from vivijure_backend.harness import models_mirror, pipeline_registry
    _stub_store(monkeypatch)
    seen = {}
    monkeypatch.setattr(models_mirror, "ensure_models", lambda **k: seen.update(k) or False)
    monkeypatch.setattr(models_mirror, "start_i2v_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(handler_mod, "run_job", lambda payload, **kw: {"ran": True})
    handler_mod.handler({"id": "j", "input": {"action": "render", "project": "p"}})
    assert seen["finish_legs"] is None and seen["deadline"] is None


def test_a_hung_mirror_ends_as_a_soft_degrade_with_its_record(tmp_path, monkeypatch):
    # Through the real ensure_models and a real stalled process. Nothing below the store is stubbed.
    _stub_store(monkeypatch)
    _real_rclone_script(tmp_path, monkeypatch, "exec sleep 60")
    for key, value in CREDS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("VJ_MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("VJ_FINISH_MIRROR_MAX_SECONDS", "0.3")
    monkeypatch.delenv("VJ_VOLUME_ROOT", raising=False)
    emitted = []
    real_emit = handler_mod.ProgressEmitter.emit
    monkeypatch.setattr(handler_mod.ProgressEmitter, "emit",
                        lambda self, event, **f: (emitted.append(event), real_emit(self, event, **f))[1])
    monkeypatch.setattr(handler_mod, "run_finish_job",
                        lambda *a, **k: pytest.fail("a hung mirror must not reach the step"))
    out = handler_mod.handler(JOB)
    assert out["ok"] is False and "error" not in out
    assert "stage model_mirror" in out["detail"]
    assert "mirror_failed" in emitted
    assert not (tmp_path / mm.FINISH_LEG_SENTINEL.format(RIFE_LEG)).exists()


# ------------------------------------------------------------ the mirror budget's own rules

@pytest.mark.parametrize("raw", [None, "", "   ", "junk", "0", "-1"])
def test_no_env_value_can_turn_the_mirror_bound_off(raw):
    env = {} if raw is None else {finish.MIRROR_MAX_SECONDS_ENV: raw}
    assert max_mirror_seconds(env) == finish.DEFAULT_MIRROR_MAX_SECONDS


def test_a_valid_mirror_override_is_honoured():
    assert max_mirror_seconds({finish.MIRROR_MAX_SECONDS_ENV: "45"}) == 45.0
