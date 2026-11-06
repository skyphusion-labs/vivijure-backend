"""The stills weights base image (#564): what it may hold, how it is staged, and how a worker trusts it.

The image itself is built on the bake box. These tests hold its Dockerfile, its runbook, the stage and
check commands, and the runtime check of the stills sentinel, so a base that would lie fails here first.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vivijure_backend.finish import FINISH_LEGS
from vivijure_backend.harness import models_mirror as mm
from vivijure_backend.harness import stills_manifest as sm

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "deploy" / "stills-weights.Dockerfile"
RUNBOOK = ROOT / "docs" / "stills-weights-base.md"
BIG = "a" * 64
REPO = {"dir": "models--o--r", "repo_id": "o/r", "revision": "rev1", "files": [
    {"path": "unet/m.safetensors", "blob": BIG, "bytes": 5}]}


def _instructions() -> list[str]:
    return [ln.strip() for ln in DOCKERFILE.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


# ------------------------------------------------------------------ the Dockerfile

def test_the_base_builds_on_a_named_toolchain_with_no_default():
    lines = _instructions()
    assert lines[0] == "ARG TOOLCHAIN_REF"
    assert lines[1] == "FROM ${TOOLCHAIN_REF}"


def test_the_base_only_copies_bins_into_the_models_root():
    copies = [ln for ln in _instructions()[2:]]
    assert copies == [f"COPY bin-{i:02d}/ /opt/models/" for i in range(len(copies))]
    assert len(copies) >= 1


def test_the_base_never_claims_the_full_set():
    # .vj-baked ends every pull for every action; this image holds only the stills set and the legs.
    assert not any("vj-baked" in ln or ln.startswith("RUN") for ln in _instructions())


def test_the_runbook_packs_as_many_bins_as_the_dockerfile_copies():
    copies = len(_instructions()) - 2
    bins = re.findall(r"--bins (\d+)", RUNBOOK.read_text())
    assert bins and all(int(b) == copies for b in bins)


def _ffd_fits(sizes: list[int], bins: int, ceiling: int) -> bool:
    room = [0] * bins
    for sz in sorted(sizes, reverse=True):
        for i in range(bins):
            if room[i] + sz < ceiling:
                room[i] += sz
                break
        else:
            return False
    return True


def test_the_stills_set_fits_the_bins_under_the_runbook_ceiling():
    import sys
    sys.path.insert(0, str(ROOT / "deploy"))
    import bake_layers

    ceiling_gib = float(re.search(r"--ceiling-gb ([0-9.]+)", RUNBOOK.read_text()).group(1))
    ceiling = int(ceiling_gib * 1024**3)
    sizes = [f["bytes"] for r in sm.load_manifest() for f in r["files"]]
    assert ceiling < bake_layers.GHCR_LAYER_LIMIT_BYTES
    assert max(sizes) < ceiling, "the largest stills file no longer fits one layer"
    assert _ffd_fits(sizes, len(_instructions()) - 2, ceiling)


def test_the_runbook_floor_sits_below_the_set():
    floor_gib = float(re.search(r"--min-gb ([0-9.]+)", RUNBOOK.read_text()).group(1))
    assert floor_gib * 1024**3 < sm.total_bytes()


# ------------------------------------------------------------------ check_stills

def _complete(tmp_path: Path, version: str = "1") -> dict:
    hf = tmp_path / "hf-cache"
    repo_dir = hf / "hub" / REPO["dir"]
    (repo_dir / "blobs").mkdir(parents=True)
    (repo_dir / "blobs" / BIG).write_bytes(b"x" * 5)
    link = repo_dir / "snapshots" / "rev1" / "unet" / "m.safetensors"
    link.parent.mkdir(parents=True)
    link.symlink_to(f"../../../blobs/{BIG}")
    (tmp_path / mm.STILLS_SENTINEL).write_text(version + "\n")
    for leg in FINISH_LEGS:
        (tmp_path / mm.FINISH_LEG_SENTINEL.format(leg)).write_text(version + "\n")
    (tmp_path / "antelopev2").mkdir()
    (tmp_path / "antelopev2" / "glintr100.onnx").write_bytes(b"o")
    return {"VJ_MODELS_ROOT": str(tmp_path), "HF_HOME": str(hf), "VJ_MODEL_VERSION": version}


@pytest.fixture
def one_repo(monkeypatch):
    monkeypatch.setattr(sm, "load_manifest", lambda: [REPO])


def test_a_complete_tree_has_no_problems(tmp_path, one_repo):
    assert mm.check_stills(_complete(tmp_path)) == []


def test_a_baked_marker_is_a_problem(tmp_path, one_repo):
    env = _complete(tmp_path)
    (tmp_path / mm.BAKED_SENTINEL).write_text("BAKED\n")
    assert any(mm.BAKED_SENTINEL in p for p in mm.check_stills(env))


def test_a_sentinel_at_another_model_version_is_a_problem(tmp_path, one_repo):
    env = _complete(tmp_path, version="1")
    env["VJ_MODEL_VERSION"] = "2"
    assert any(mm.STILLS_SENTINEL in p for p in mm.check_stills(env))


def test_a_short_file_is_a_problem(tmp_path, one_repo):
    env = _complete(tmp_path)
    (Path(env["HF_HOME"]) / "hub" / REPO["dir"] / "blobs" / BIG).write_bytes(b"x" * 4)
    assert any("4 bytes" in p for p in mm.check_stills(env))


@pytest.mark.parametrize("leg", FINISH_LEGS)
def test_a_missing_finish_leg_is_a_problem(tmp_path, one_repo, leg):
    env = _complete(tmp_path)
    (tmp_path / mm.FINISH_LEG_SENTINEL.format(leg)).unlink()
    assert any(leg in p for p in mm.check_stills(env))


def test_an_empty_antelope_is_a_problem(tmp_path, one_repo):
    env = _complete(tmp_path)
    (tmp_path / "antelopev2" / "glintr100.onnx").unlink()
    assert any("antelopev2" in p for p in mm.check_stills(env))


def test_the_command_exits_by_the_check(tmp_path, one_repo, monkeypatch):
    env = _complete(tmp_path)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert mm._stills_main(["--check-stills"]) == 0
    (tmp_path / mm.STILLS_SENTINEL).unlink()
    assert mm._stills_main(["--check-stills"]) == 1


# ------------------------------------------------------------------ stage_stills

def test_stage_runs_the_workers_own_pull_for_stills_and_every_leg(tmp_path, one_repo, monkeypatch):
    env = _complete(tmp_path)
    env["VJ_VOLUME_ROOT"] = "/runpod-volume"
    monkeypatch.setattr(mm, "DEFAULT_VOLUME_ROOT", str(tmp_path / "no-volume"))
    calls = []
    monkeypatch.setattr(mm, "ensure_models", lambda **k: calls.append(k) or True)
    assert mm.stage_stills(env, log=lambda *_: None) == []
    assert [(c["action"], c.get("finish_legs")) for c in calls] == [("preview", None), ("finish_clip", FINISH_LEGS)]
    assert all("VJ_VOLUME_ROOT" not in c["env"] for c in calls)


def test_stage_reports_what_the_pull_left_incomplete(tmp_path, one_repo, monkeypatch):
    env = _complete(tmp_path)
    (tmp_path / mm.STILLS_SENTINEL).unlink()
    monkeypatch.setattr(mm, "DEFAULT_VOLUME_ROOT", str(tmp_path / "no-volume"))
    monkeypatch.setattr(mm, "ensure_models", lambda **k: True)
    assert mm.stage_stills(env, log=lambda *_: None)


def test_stage_refuses_a_host_with_a_volume(tmp_path, one_repo, monkeypatch):
    (tmp_path / "vol").mkdir()
    monkeypatch.setattr(mm, "DEFAULT_VOLUME_ROOT", str(tmp_path / "vol"))
    monkeypatch.setattr(mm, "ensure_models", lambda **k: pytest.fail("must not pull"))
    assert mm.stage_stills(_complete(tmp_path), log=lambda *_: None)


# ------------------------------------------------------------------ the worker trusts the sentinel only with its files

def _pull_env(tmp_path: Path) -> dict:
    env = _complete(tmp_path)
    env.update({"R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s", "R2_ENDPOINT": "https://e",
                "R2_BUCKET": "b", "VJ_VOLUME_ROOT": str(tmp_path / "unmounted")})
    return env


@pytest.fixture
def rclone(monkeypatch):
    calls = []
    monkeypatch.setattr(mm.shutil, "which", lambda n: "/x/" + n)
    monkeypatch.setattr(mm, "_jitter_seconds", lambda e: 0)
    monkeypatch.setattr(mm, "_run_rclone", lambda cmd, env, *a, **k: calls.append(cmd) or 0)
    return calls


def test_a_baked_stills_tree_serves_a_stills_job_with_no_pull(tmp_path, one_repo, rclone):
    lines: list[str] = []
    assert mm.ensure_models(env=_pull_env(tmp_path), log=lines.append, action="preview") is False
    assert rclone == []
    assert any('"warm_stills"' in ln for ln in lines)


def test_a_sentinel_without_its_files_pulls_again(tmp_path, one_repo, rclone, monkeypatch):
    env = _pull_env(tmp_path)
    blob = Path(env["HF_HOME"]) / "hub" / REPO["dir"] / "blobs" / BIG
    blob.unlink()
    monkeypatch.setattr(mm, "_reconstruct_symlinks", lambda *a, **k: blob.write_bytes(b"x" * 5) or 0)
    lines: list[str] = []
    assert mm.ensure_models(env=env, log=lines.append, action="preview") is True
    assert rclone, "a stills sentinel without its files must not skip the pull"
    assert any("pulling it again" in ln for ln in lines)


def test_a_baked_stills_tree_still_pulls_for_a_render(tmp_path, one_repo, rclone):
    assert mm.ensure_models(env=_pull_env(tmp_path), log=lambda *_: None, action="render") is True
    assert rclone
