"""The stills list against its sources of truth, and the filter against the real rclone."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from vivijure_backend.harness import models_mirror as mm
from vivijure_backend.harness import stills_manifest as sm
from vivijure_backend.models import DEFAULT_SPECS, ModelRole

ROOT = Path(__file__).resolve().parent.parent
BIG = "a" * 64


def test_revisions_are_the_ones_the_bake_manifest_pins():
    keep = {r["repo"]: r["rev"] for r in json.loads((ROOT / "deploy/bake-manifest.json").read_text())["keep_repos"]}
    for r in sm.load_manifest():
        assert keep[r["dir"]] == r["revision"], r["dir"]


@pytest.mark.parametrize("role", [ModelRole.KEYFRAME_BASE, ModelRole.KEYFRAME_FEWSTEP, ModelRole.IP_ADAPTER,
                                  ModelRole.CONTROLNET_POSE, ModelRole.CONTROLNET_CANNY, ModelRole.INSTANTID])
def test_every_role_a_stills_job_loads_is_listed(role):
    assert DEFAULT_SPECS[role].repo_id in {r["repo_id"] for r in sm.load_manifest()}


def _files(repo_id):
    return {f["path"] for r in sm.load_manifest() if r["repo_id"] == repo_id for f in r["files"]}


def test_the_named_files_are_the_ones_the_loaders_open():
    few = DEFAULT_SPECS[ModelRole.KEYFRAME_FEWSTEP]
    assert few.weight_name in _files(few.repo_id)
    keyframe_src = (ROOT / "src/vivijure_backend/keyframe.py").read_text()
    models_src = (ROOT / "src/vivijure_backend/models.py").read_text()
    assert '"ip-adapter_sdxl.safetensors"' in keyframe_src
    assert "sdxl_models/ip-adapter_sdxl.safetensors" in _files(DEFAULT_SPECS[ModelRole.IP_ADAPTER].repo_id)
    assert '"ip-adapter.bin"' in models_src
    assert "ip-adapter.bin" in _files(DEFAULT_SPECS[ModelRole.INSTANTID].repo_id)


def test_lora_training_needs_are_covered():
    base = _files(DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id)
    for folder in ("unet/", "vae/", "text_encoder/", "text_encoder_2/"):
        assert any(p.startswith(folder) for p in base), folder


def test_the_set_is_the_measured_size():
    assert 25.7e9 < sm.total_bytes() < 25.8e9


REPO = {"dir": "models--o--r", "repo_id": "o/r", "revision": "rev1", "files": [
    {"path": "unet/m.safetensors", "blob": BIG, "bytes": 5}]}


def _seed(src: Path) -> None:
    repo = src / "hub" / "models--o--r"
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("rev1")
    (repo / "blobs" / BIG).write_bytes(b"12345")
    (repo / "blobs" / ("b" * 64)).write_bytes(b"F" * 9)  # the fp16 twin
    (repo / "blobs" / ("c" * 64)).write_bytes(b"P" * 9)  # a sample png
    (repo / "blobs" / ("d" * 40)).write_text("{}")       # a config, git-stored
    snap = repo / "snapshots" / "rev1"
    (snap / "unet").mkdir(parents=True)
    (snap / "unet" / "m.safetensors").symlink_to(f"../../../blobs/{BIG}")
    (snap / "unet" / "m.fp16.safetensors").symlink_to(f"../../../blobs/{'b' * 64}")
    (snap / "example.png").symlink_to(f"../../blobs/{'c' * 64}")
    (snap / "unet" / "config.json").symlink_to(f"../../../blobs/{'d' * 40}")
    old = repo / "snapshots" / "rev0"
    old.mkdir()
    (old / "config.json").symlink_to(f"../../blobs/{'d' * 40}")
    other = src / "hub" / "models--wan--huge" / "blobs"
    other.mkdir(parents=True)
    (other / ("e" * 64)).write_bytes(b"W")


@pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone is not installed")
def test_the_real_rclone_pulls_only_the_listed_set(tmp_path):
    """The un-stubbed reader: if a rule is wrong, this is the test that sees what lands on disk."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _seed(src)
    subprocess.run(mm.mirror_cmd(str(src), dst, filters=sm.rclone_filters([REPO])), check=True,
                   env={**os.environ, "RCLONE_CONFIG": os.devnull})
    repo = dst / "hub" / "models--o--r"
    assert (repo / "blobs" / BIG).read_bytes() == b"12345"
    assert (repo / "blobs" / ("d" * 40)).exists()
    assert (repo / "refs" / "main").read_text() == "rev1"
    assert not (repo / "blobs" / ("b" * 64)).exists()
    assert not (repo / "blobs" / ("c" * 64)).exists()
    assert not (repo / "snapshots" / "rev0").exists()
    assert not (dst / "hub" / "models--wan--huge").exists()
    # The filter itself leaves these behind. Prune is the backstop for what no suffix names.
    snap = repo / "snapshots" / "rev1"
    assert not list(snap.rglob("m.fp16.safetensors*"))
    assert not list(snap.rglob("example.png*"))
    mm._reconstruct_symlinks(dst, print)
    sm.prune_dangling_links(dst / "hub", print)
    assert not (repo / "snapshots" / "rev1" / "unet" / "m.fp16.safetensors").is_symlink()
    assert (repo / "snapshots" / "rev1" / "unet" / "config.json").read_text() == "{}"
    sm.verify(dst, [REPO])


def _tree(hf: Path, repo: dict, *, size: int = 5) -> None:
    repo_dir = hf / "hub" / repo["dir"]
    (repo_dir / "blobs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "blobs" / BIG).write_bytes(b"x" * size)
    link = repo_dir / "snapshots" / repo["revision"] / "unet" / "m.safetensors"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(f"../../../blobs/{BIG}")


def test_verify_names_a_missing_file(tmp_path):
    with pytest.raises(sm.StillsIncomplete, match="o/r/unet/m.safetensors"):
        sm.verify(tmp_path, [REPO])


def test_verify_refuses_a_short_file(tmp_path):
    _tree(tmp_path, REPO, size=4)
    with pytest.raises(sm.StillsIncomplete, match="4 bytes"):
        sm.verify(tmp_path, [REPO])


def test_prune_leaves_a_live_link(tmp_path):
    (tmp_path / "t").write_text("x")
    (tmp_path / "live").symlink_to("t")
    (tmp_path / "dead").symlink_to("nope")
    assert sm.prune_dangling_links(tmp_path, print) == 1
    assert (tmp_path / "live").exists()


def _env(tmp_path, **extra):
    return {"R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s", "R2_ENDPOINT": "https://e",
            "R2_BUCKET": "b", "HF_HOME": str(tmp_path / "hf"), "VJ_MODELS_ROOT": str(tmp_path / "m"),
            "VJ_VOLUME_ROOT": str(tmp_path / "unmounted"), "VJ_MIRROR_JITTER_MAX_S": "0", **extra}


@pytest.fixture
def fake_rclone(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mm.shutil, "which", lambda n: "/x/" + n)
    monkeypatch.setattr(mm, "_jitter_seconds", lambda e: 0)
    monkeypatch.setattr(mm, "_run_rclone", lambda cmd, env: calls.append(cmd) or 0)
    return calls


def _legs(calls):
    return [c[-2].rsplit("/", 1)[1] for c in calls]


def test_a_preview_pulls_only_the_stills_set_and_writes_its_own_sentinel(tmp_path, fake_rclone, monkeypatch):
    monkeypatch.setattr(sm, "load_manifest", lambda: [REPO])
    env = _env(tmp_path)
    _tree(tmp_path / "hf", REPO)
    dead = tmp_path / "hf" / "hub" / REPO["dir"] / "snapshots" / "rev1" / "unet" / "m.xml"
    dead.symlink_to("../../../blobs/not-pulled")
    assert mm.ensure_models(env=env, log=lambda m: None, action="preview") is True
    assert not dead.is_symlink()  # a link to a blob outside the set does not survive the pull
    assert _legs(fake_rclone) == ["hf-cache", "antelopev2"]
    assert "--filter" in fake_rclone[0]
    assert (tmp_path / "m" / mm.STILLS_SENTINEL).exists()
    assert not (tmp_path / "m" / mm.SENTINEL).exists()


def test_a_render_keeps_the_full_mirror(tmp_path, fake_rclone):
    assert mm.ensure_models(env=_env(tmp_path), log=lambda m: None, action="render") is True
    assert _legs(fake_rclone) == ["hf-cache", "antelopev2", "rife", "GFPGANv1.4", "facexlib"]
    assert "--filter" not in fake_rclone[0]
    assert (tmp_path / "m" / mm.SENTINEL).exists()


@pytest.mark.parametrize("action", [None, "finalize", "i2v_clip", "nonsense"])
def test_every_other_action_keeps_the_full_mirror(action):
    assert mm.wants_stills_only(action, {}) is False


@pytest.mark.parametrize("action", ["preview", "regen_shot", "train_lora"])
def test_the_stills_actions(action):
    assert mm.wants_stills_only(action, {}) is True
    assert mm.wants_stills_only(action, {"VJ_STILLS_MIRROR": "0"}) is False


def test_a_missing_file_fails_loud_and_writes_no_sentinel(tmp_path, fake_rclone, monkeypatch):
    monkeypatch.setattr(sm, "load_manifest", lambda: [REPO])
    with pytest.raises(sm.StillsIncomplete):
        mm.ensure_models(env=_env(tmp_path), log=lambda m: None, action="preview")
    assert not (tmp_path / "m" / mm.STILLS_SENTINEL).exists()


def test_a_warm_stills_worker_pulls_nothing_but_a_render_after_it_pulls_the_rest(tmp_path, fake_rclone, monkeypatch):
    monkeypatch.setattr(sm, "load_manifest", lambda: [REPO])
    env = _env(tmp_path)
    _tree(tmp_path / "hf", REPO)
    mm.ensure_models(env=env, log=lambda m: None, action="preview")
    fake_rclone.clear()
    assert mm.ensure_models(env=env, log=lambda m: None, action="preview") is False
    assert fake_rclone == []
    assert mm.ensure_models(env=env, log=lambda m: None, action="render") is True
    assert "rife" in _legs(fake_rclone)
