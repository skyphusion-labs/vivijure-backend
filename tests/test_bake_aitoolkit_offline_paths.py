"""Bake-time stable dirs + ai-toolkit hub-id scrub (Conrad: bake it, don't just remap)."""
from pathlib import Path

import pytest

from vivijure_backend import wan_lora_train as W


def test_default_wan_base_path_prefers_env(monkeypatch, tmp_path):
    monkeypatch.setenv(W.WAN_BASE_PATH_ENV, str(tmp_path / "wan-base"))
    assert W.default_wan_base_path() == str(tmp_path / "wan-base")


def test_default_wan_base_path_falls_back_to_hub_id(monkeypatch):
    monkeypatch.delenv(W.WAN_BASE_PATH_ENV, raising=False)
    # no bake dir on a laptop CPU checkout
    monkeypatch.setattr(W, "DEFAULT_WAN_BASE_PATH", "/nonexistent/wan-base")
    assert W.default_wan_base_path() == W.DEFAULT_WAN_BASE_REPO


def test_wan_train_runtime_ready_prefers_env_python(monkeypatch, tmp_path):
    py = tmp_path / "aitoolkit-py"
    py.write_text("#!/usr/bin/env python3\n")
    py.chmod(0o755)
    monkeypatch.setenv("VIVIJURE_AITOOLKIT_PYTHON", str(py))
    assert W.wan_train_runtime_ready() is True


def test_wan_train_runtime_ready_false_on_plain_cpu_box(monkeypatch):
    monkeypatch.delenv("VIVIJURE_AITOOLKIT_PYTHON", raising=False)
    monkeypatch.delenv("VIVIJURE_WAN_BASE_PATH", raising=False)
    assert W.wan_train_runtime_ready() is False


def test_wan_lora_train_config_default_uses_default_wan_base_path(monkeypatch, tmp_path):
    base = tmp_path / "wan-base"
    base.mkdir()
    monkeypatch.setenv(W.WAN_BASE_PATH_ENV, str(base))
    cfg = W.WanLoraTrainConfig()
    assert cfg.base_repo == str(base)


def test_bake_aitoolkit_offline_paths_materialize_and_patch(tmp_path, monkeypatch):
    # Import the deploy script as a module (not on package path); exec via path load.
    import importlib.util
    import sys
    import types

    script = Path(__file__).resolve().parents[1] / "deploy" / "bake_aitoolkit_offline_paths.py"
    spec = importlib.util.spec_from_file_location("bake_aitoolkit_offline_paths", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    snaps = {}
    for key, repo in mod.REPOS.items():
        s = tmp_path / "snaps" / key
        s.mkdir(parents=True)
        (s / "config.json").write_text("{}")
        snaps[repo] = s

    monkeypatch.setenv("VIVIJURE_AITOOLKIT_DIR", str(tmp_path / "ai-toolkit"))
    monkeypatch.setenv("VIVIJURE_AITOOLKIT_WEIGHTS", str(tmp_path / "stable"))
    # re-bind module paths after env
    mod.AITOOLKIT_DIR = Path(tmp_path / "ai-toolkit")
    mod.STABLE_ROOT = Path(tmp_path / "stable")

    wan21 = mod.AITOOLKIT_DIR / "toolkit/models/wan21"
    wan22 = mod.AITOOLKIT_DIR / "extensions_built_in/diffusion_models/wan22"
    wan21.mkdir(parents=True)
    wan22.mkdir(parents=True)
    (wan21 / "wan21.py").write_text('te_path = "ai-toolkit/umt5_xxl_encoder"\n')
    (wan22 / "wan22_14b_model.py").write_text('_wan_vae_path = "ai-toolkit/wan2.1-vae"\n')

    def fake_snap(repo, local_files_only=True):
        assert local_files_only is True
        return str(snaps[repo])

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert mod.main() == 0
    marker = mod.STABLE_ROOT / "BAKE_OFFLINE.json"
    assert marker.is_file()
    assert "umt5" in (wan21 / "wan21.py").read_text()
    assert '"ai-toolkit/umt5_xxl_encoder"' not in (wan21 / "wan21.py").read_text()
    assert '"ai-toolkit/wan2.1-vae"' not in (wan22 / "wan22_14b_model.py").read_text()
    # stable symlinks exist
    assert (mod.STABLE_ROOT / "wan-base").is_dir()
