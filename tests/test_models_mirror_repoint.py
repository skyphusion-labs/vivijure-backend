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
