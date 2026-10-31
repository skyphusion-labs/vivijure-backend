"""An anime job gets its checkpoint on a worker that does not carry it, and only that job pulls it.

The seam that is not stubbed: a real executable named rclone on PATH, so `_run_rclone`, the argv,
`_reconstruct_symlinks` and the on-disk checks all run for real against an R2-shaped tree. The fake
diffusers refuses any source that is not a directory holding the unet, which is what
`local_files_only=True` does on a cache miss.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vivijure_backend.config import KeyframeConfig
from vivijure_backend.harness import models_mirror as mm
from vivijure_backend.models import (
    DEFAULT_SPECS,
    KEYFRAME_ANIME_REPO,
    ModelRole,
    ModelServer,
    missing_keyframe_weights_error,
)

ANIME_DIR = "models--cagliostrolab--animagine-xl-4.0"
PHOTOREAL = DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id
REV = "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96"
ROOT = Path(__file__).resolve().parents[1]


def _r2_repo(dst: Path, components=("unet", "vae", "text_encoder", "text_encoder_2")) -> None:
    """The repo as the R2 mirror stores it: blobs, refs/main, and .rclonelink markers."""
    (dst / "blobs").mkdir(parents=True)
    (dst / "refs").mkdir()
    (dst / "refs" / "main").write_text(REV)
    snap = dst / "snapshots" / REV
    snap.mkdir(parents=True)
    (dst / "blobs" / "index").write_text("{}")
    (snap / "model_index.json.rclonelink").write_text("../../blobs/index")
    for c in components:
        (snap / c).mkdir()
        (dst / "blobs" / c).write_text("weights")
        (snap / c / "model.safetensors.rclonelink").write_text(f"../../../blobs/{c}")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A cold worker with R2 creds, and a real rclone executable that records and copies."""
    root = tmp_path / "models"
    root.mkdir()
    mirror = tmp_path / "r2"
    _r2_repo(mirror / ANIME_DIR)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "rclone-calls"
    exe = bin_dir / "rclone"
    exe.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> "{calls}"\n'
        'for a in "$@"; do src="$dst"; dst="$a"; done\n'
        'if [ -n "$FAKE_RCLONE_EXIT" ]; then exit "$FAKE_RCLONE_EXIT"; fi\n'
        f'mkdir -p "$dst" && cp -R "{mirror}/{ANIME_DIR}/." "$dst"\n'
    )
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("VJ_MODELS_ROOT", str(root))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("VJ_MODEL_VERSION", raising=False)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("R2_ENDPOINT", "https://r2.example")
    monkeypatch.setattr(mm, "_PRISTINE_ROOTS", None)
    return types.SimpleNamespace(root=root, mirror=mirror, calls=calls, tmp=tmp_path)


def _pulls(world) -> list[str]:
    return world.calls.read_text().splitlines() if world.calls.exists() else []


def _snapshot(root: Path) -> Path:
    return root / "hf-cache" / "hub" / ANIME_DIR / "snapshots" / REV


def test_a_cold_anime_job_pulls_exactly_its_repo_and_gets_a_loadable_directory(world):
    source = mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)

    assert source == str(_snapshot(world.root))
    pulls = _pulls(world)
    assert len(pulls) == 1
    assert f"r2:vivijure/models/hf-cache/hub/{ANIME_DIR} " in pulls[0]
    # The rclone markers became real links, so a load reads the weight and not the marker.
    unet = Path(source) / "unet" / "model.safetensors"
    assert unet.is_symlink() and unet.read_text() == "weights"


def test_a_photoreal_job_never_pulls_it(world):
    assert mm.ensure_keyframe_base(PHOTOREAL, log=lambda *_: None) == PHOTOREAL
    assert _pulls(world) == []
    assert not (world.root / "hf-cache" / "hub" / ANIME_DIR).exists()


def test_a_warm_worker_does_not_pull_it_twice(world):
    first = mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    assert mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None) == first
    assert len(_pulls(world)) == 1


def test_a_failed_pull_raises_and_the_partial_copy_is_not_trusted(world, monkeypatch):
    # The first pull leaves weights on disk and then fails, as a pull that dies partway does.
    real = mm._run_rclone

    def dies_after_copying(cmd, env, deadline=None):
        real(cmd, env)
        raise subprocess.CalledProcessError(3, cmd)

    monkeypatch.setattr(mm, "_run_rclone", dies_after_copying)
    with pytest.raises(subprocess.CalledProcessError):
        mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    monkeypatch.setattr(mm, "_run_rclone", real)

    mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    assert len(_pulls(world)) == 2


def test_a_real_nonzero_rclone_exit_raises(world, monkeypatch):
    monkeypatch.setenv("FAKE_RCLONE_EXIT", "3")
    with pytest.raises(subprocess.CalledProcessError):
        mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)


def test_a_mirror_missing_a_component_fails_by_name(world):
    import shutil
    shutil.rmtree(world.mirror / ANIME_DIR)
    _r2_repo(world.mirror / ANIME_DIR, components=("unet", "vae", "text_encoder"))
    with pytest.raises(RuntimeError, match="text_encoder_2"):
        mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    # And it is not recorded as complete.
    with pytest.raises(RuntimeError, match="text_encoder_2"):
        mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    assert len(_pulls(world)) == 2


def test_no_credentials_is_a_loud_failure_naming_the_repo(world, monkeypatch):
    monkeypatch.delenv("R2_ACCESS_KEY_ID")
    with pytest.raises(RuntimeError, match=KEYFRAME_ANIME_REPO):
        mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    assert _pulls(world) == []


def test_a_baked_image_without_the_repo_still_pulls(world):
    (world.root / mm.BAKED_SENTINEL).write_text("x")
    source = mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)
    assert source == str(_snapshot(world.root))
    assert len(_pulls(world)) == 1


def test_weights_already_on_disk_are_used_without_a_pull(world):
    # A bake that keeps the repo, or a preloaded volume: real links, no sentinel from this code.
    hub = world.root / "hf-cache" / "hub" / ANIME_DIR
    _r2_repo(hub)
    mm._reconstruct_symlinks(hub, lambda *_: None)
    assert mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None) == str(_snapshot(world.root))
    assert _pulls(world) == []


def test_a_read_only_volume_is_never_written_the_pull_lands_on_local_disk(world, monkeypatch):
    volume = world.tmp / "volume"
    (volume / "hf-cache" / "hub").mkdir(parents=True)
    monkeypatch.setattr(mm, "_PRISTINE_ROOTS", {"VJ_MODELS_ROOT": str(world.root), "HF_HOME": None})
    monkeypatch.setenv("VJ_MODELS_ROOT", str(volume))
    monkeypatch.setenv("HF_HOME", str(volume / "hf-cache"))

    source = mm.ensure_keyframe_base(KEYFRAME_ANIME_REPO, log=lambda *_: None)

    assert source == str(_snapshot(world.root))
    assert not (volume / "hf-cache" / "hub" / ANIME_DIR).exists()
    assert not list(volume.glob(".vj-*"))


# ---- the load sites ---------------------------------------------------------------------------

class _LoadReached(Exception):
    """Ends a load once the source has been recorded."""


def _fake_from_pretrained(seen: list):
    def from_pretrained(source, **kw):
        seen.append(source)
        if source != PHOTOREAL and not (Path(source) / "unet").is_dir():
            raise OSError("not in the local cache")
        raise _LoadReached
    return from_pretrained


def _server(monkeypatch, seen, repo):
    fake = types.ModuleType("diffusers")
    for name in ("StableDiffusionXLControlNetPipeline", "StableDiffusionXLPipeline"):
        setattr(fake, name, types.SimpleNamespace(from_pretrained=_fake_from_pretrained(seen)))
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = MagicMock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    server = ModelServer(specs={ModelRole.KEYFRAME_BASE: dataclasses.replace(
        DEFAULT_SPECS[ModelRole.KEYFRAME_BASE], repo_id=repo)})
    monkeypatch.setattr(ModelServer, "_controlnet", lambda self, role: "cn")
    return server


def test_an_anime_config_resolves_to_a_repo_the_lazy_path_owns():
    # The chain from the studio's style to the pull: if these two ever disagree, anime fails live.
    base = KeyframeConfig.from_dict({"style": "anime"}).base_model
    assert "models--" + base.replace("/", "--") in mm.KEYFRAME_LAZY_REPOS


def test_keyframe_pipeline_anime_loads_the_pulled_snapshot(world, monkeypatch):
    seen: list = []
    server = _server(monkeypatch, seen, KEYFRAME_ANIME_REPO)
    with pytest.raises(_LoadReached):
        server.keyframe_pipeline()
    assert seen == [str(_snapshot(world.root))]


def test_keyframe_pipeline_photoreal_loads_by_repo_id_with_no_pull(world, monkeypatch):
    seen: list = []
    server = _server(monkeypatch, seen, PHOTOREAL)
    with pytest.raises(_LoadReached):
        server.keyframe_pipeline()
    assert seen == [PHOTOREAL]
    assert _pulls(world) == []


def test_a_failed_pull_fails_the_job_and_nothing_loads(world, monkeypatch):
    monkeypatch.setenv("FAKE_RCLONE_EXIT", "3")
    seen: list = []
    server = _server(monkeypatch, seen, KEYFRAME_ANIME_REPO)
    with pytest.raises(subprocess.CalledProcessError):
        server.keyframe_pipeline()
    assert seen == []  # no load at all, so nothing can fall back to the photoreal base


def test_instantid_pipeline_anime_loads_the_pulled_snapshot(world, monkeypatch):
    seen: list = []
    server = _server(monkeypatch, seen, KEYFRAME_ANIME_REPO)
    with pytest.raises(_LoadReached):
        server.instantid_pipeline()
    assert seen == [str(_snapshot(world.root))]


def test_lora_train_on_the_anime_base_loads_the_snapshot(world, monkeypatch, tmp_path):
    from vivijure_backend import lora_train
    seen: list = []
    fake = MagicMock()
    fake.AutoencoderKL.from_pretrained = _fake_from_pretrained(seen)
    monkeypatch.setitem(sys.modules, "diffusers", fake)
    for mod in ("diffusers.utils", "torch", "torch.nn", "torch.nn.functional", "peft", "peft.utils",
                "transformers"):
        monkeypatch.setitem(sys.modules, mod, MagicMock())
    char = types.SimpleNamespace(slot="A", name="A", ref_paths=[tmp_path / "ref.png"])
    with pytest.raises(_LoadReached):
        lora_train.train_slot(char, tmp_path / "out", base_repo=KEYFRAME_ANIME_REPO)
    assert seen == [str(_snapshot(world.root))]


def test_the_missing_weights_message_no_longer_tells_an_operator_to_rebake():
    assert "pulled on demand" in str(missing_keyframe_weights_error(KEYFRAME_ANIME_REPO))


# ---- runtime and bake agree -------------------------------------------------------------------

def test_the_cold_pull_and_the_bake_both_leave_it_to_the_lazy_path():
    manifest = json.loads((ROOT / "deploy" / "bake-manifest.json").read_text())
    drop = {r["repo"]: r["reason"] for r in manifest["drop_repos"]}
    keep = {r["repo"] for r in manifest["keep_repos"]}
    for repo in mm.KEYFRAME_LAZY_REPOS:
        assert repo in mm.DEFAULT_SKIP_REPOS
        assert repo in drop and repo not in keep
        assert "ensure_keyframe_base" in drop[repo]  # names the owner, not "never loaded"
