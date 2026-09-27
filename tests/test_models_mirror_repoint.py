"""ensure_models must repoint the ENVIRONMENT it was asked to use when it finds a preloaded volume.

`mirror_env` hands ensure_models a COPY; the repoint used to land on the copy only, so the deferred
torch/diffusers loads (which read HF_HOME / VJ_MODELS_ROOT at call time) never saw the volume."""
import os

from vivijure_backend.harness.models_mirror import (
    I2V_SENTINEL,
    SENTINEL,
    _DEFAULT_MODEL_VERSION,
    ensure_models,
)


def _volume(root):
    (root / "hf-cache").mkdir(parents=True)
    (root / SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
    (root / I2V_SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
    return root


def test_ensure_models_repoints_os_environ_on_a_preloaded_volume(tmp_path, monkeypatch):
    vol = _volume(tmp_path / "vol")
    monkeypatch.setenv("VJ_VOLUME_ROOT", str(vol))
    monkeypatch.setenv("VJ_MODELS_ROOT", str(tmp_path / "not-baked"))   # not baked: no sentinel there
    monkeypatch.setenv("HF_HOME", str(tmp_path / "old-hf"))
    assert ensure_models(log=lambda *_: None) is False                   # volume skip, no R2 pull
    assert os.environ["VJ_MODELS_ROOT"] == str(vol)
    assert os.environ["HF_HOME"] == str(vol / "hf-cache")


def test_ensure_models_repoints_the_callers_env_dict(tmp_path):
    vol = _volume(tmp_path / "vol")
    env = {"VJ_VOLUME_ROOT": str(vol), "VJ_MODELS_ROOT": str(tmp_path / "not-baked"),
           "HF_HOME": str(tmp_path / "old-hf")}
    assert ensure_models(env=env, log=lambda *_: None) is False
    assert env["VJ_MODELS_ROOT"] == str(vol)
    assert env["HF_HOME"] == str(vol / "hf-cache")


def test_ensure_models_leaves_the_environment_alone_when_no_volume_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("VJ_VOLUME_ROOT", str(tmp_path / "not-mounted"))
    monkeypatch.setenv("VJ_MODELS_ROOT", str(tmp_path / "not-baked"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "old-hf"))
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("MODELS_R2_ACCESS_KEY_ID", raising=False)
    ensure_models(log=lambda *_: None)       # falls through to the no-creds skip
    assert os.environ["VJ_MODELS_ROOT"] == str(tmp_path / "not-baked")
    assert os.environ["HF_HOME"] == str(tmp_path / "old-hf")


# ------------- readiness is per-JOB, but the repoint is process-global (backend#453)

def _keyframe_only_volume(root):
    """The honest post-2026-09-26 corpus: base sentinel, no i2v sentinel."""
    (root / "hf-cache").mkdir(parents=True)
    (root / SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
    return root


def test_a_keyframe_only_volume_is_read_by_a_job_that_runs_no_i2v_stage(tmp_path):
    vol = _keyframe_only_volume(tmp_path / "vol")
    env = {"VJ_VOLUME_ROOT": str(vol), "VJ_MODELS_ROOT": str(tmp_path / "opt-models"),
           "HF_HOME": str(tmp_path / "opt-models" / "hf-cache")}
    assert ensure_models(env=env, log=lambda *_: None, action="preview") is False
    assert env["VJ_MODELS_ROOT"] == str(vol)
    assert env["HF_HOME"] == str(vol / "hf-cache")


def test_a_later_i2v_job_does_not_inherit_the_keyframe_only_repoint(tmp_path):
    """Readiness is a per-job question now, but the repoint is process-global (it lands in
    os.environ so the deferred torch/diffusers loads see it). A preview job may legitimately read a
    keyframe-only volume; the very next render job on the same WARM worker must get the endpoint's
    own roots back. Without the undo it would find the volume's base sentinel current, skip the R2
    pull, and then try to mirror Wan ONTO the shared volume at i2v time -- exactly the failure the
    wholesale-degrade guard exists to prevent."""
    vol = _keyframe_only_volume(tmp_path / "vol")
    opt = tmp_path / "opt-models"
    env = {"VJ_VOLUME_ROOT": str(vol), "VJ_MODELS_ROOT": str(opt),
           "HF_HOME": str(opt / "hf-cache")}
    assert ensure_models(env=env, log=lambda *_: None, action="preview") is False
    assert env["VJ_MODELS_ROOT"] == str(vol)          # job 1 reads off the volume

    # job 2, same warm worker, this time an action that CAN reach the Wan stage
    assert ensure_models(env=env, log=lambda *_: None, action="render") is False
    assert env["VJ_MODELS_ROOT"] == str(opt)          # repoint taken back
    assert env["HF_HOME"] == str(opt / "hf-cache")


def test_the_undo_removes_the_roots_the_endpoint_never_set(tmp_path):
    # Endpoint set no roots at all (every reader falls back to its documented /opt/models default).
    # The undo must DELETE the repointed keys rather than leave the volume in place.
    vol = _keyframe_only_volume(tmp_path / "vol")
    env = {"VJ_VOLUME_ROOT": str(vol)}
    assert ensure_models(env=env, log=lambda *_: None, action="finish_clip") is False
    assert env["VJ_MODELS_ROOT"] == str(vol)
    assert ensure_models(env=env, log=lambda *_: None, action="i2v_clip") is False
    assert "VJ_MODELS_ROOT" not in env and "HF_HOME" not in env
