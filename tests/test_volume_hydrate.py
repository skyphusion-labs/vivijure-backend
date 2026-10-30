"""A real filesystem, no fake store: the tree these tests read is the tree a loader would read."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vivijure_backend import load_events
from vivijure_backend.harness import volume_hydrate as vh

REV = "abc123"
REPO = "models--SG161222--RealVisXL_V5.0"


class Events(list):
    def __call__(self, event, **fields):
        self.append((event, fields))

    def named(self, name):
        return [f for e, f in self if e == name]


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _volume(tmp_path: Path) -> Path:
    hf = tmp_path / "vol" / "hf-cache"
    snap = hf / "hub" / REPO / "snapshots" / REV
    _write(hf / "hub" / REPO / "refs" / "main", REV.encode())
    _write(snap / "model_index.json", b"{}")
    _write(snap / "unet" / "config.json", b"{}")
    _write(snap / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors", b"U" * 300)
    _write(snap / "unet" / "diffusion_pytorch_model-00002-of-00002.safetensors", b"V" * 50)
    _write(snap / "unet" / "diffusion_pytorch_model.fp16.safetensors", b"F" * 100)
    _write(snap / "text_encoder" / "model.safetensors", b"T" * 40)
    _write(snap / "RealVisXL_V5.0_fp32.safetensors", b"R" * 500)
    _write(hf / "hub" / "models--Wan-AI--Other" / "snapshots" / REV / "w.safetensors", b"W")
    return hf


def _hydrate(tmp_path, events, **kw):
    hf = _volume(tmp_path)
    local = tmp_path / "local" / "hf-cache"
    kw.setdefault("free_bytes", lambda p: 10 ** 15)
    return hf, local, vh.hydrate(hf, local, events, chunk_bytes=64, **kw)


def _local_snap(local):
    return local / "hub" / REPO / "snapshots" / REV


def test_only_the_files_a_loader_opens_become_real_and_the_rest_stay_reachable(tmp_path):
    ev = Events()
    _, local, complete = _hydrate(tmp_path, ev)
    snap = _local_snap(local)
    assert complete
    shard = snap / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    assert not shard.is_symlink() and shard.read_bytes() == b"U" * 300
    assert not (snap / "text_encoder" / "model.safetensors").is_symlink()
    for not_read in (snap / "unet" / "diffusion_pytorch_model.fp16.safetensors",
                     snap / "RealVisXL_V5.0_fp32.safetensors"):
        assert not_read.is_symlink() and not_read.stat().st_size > 0
    assert (snap / "model_index.json").is_file() and not (snap / "model_index.json").is_symlink()
    assert (local / "hub" / REPO / "refs" / "main").read_text() == REV
    other = local / "hub" / "models--Wan-AI--Other"
    assert other.is_symlink() and (other / "snapshots" / REV / "w.safetensors").read_bytes() == b"W"


def test_no_part_file_survives(tmp_path):
    _, local, _ = _hydrate(tmp_path, Events())
    assert list(local.rglob("*.part")) == []


def test_each_file_reports_its_own_rate(tmp_path):
    ev = Events()
    _hydrate(tmp_path, ev)
    names = {f["file"].rsplit("/", 1)[1] for f in ev.named("hydrate_file")}
    assert "diffusion_pytorch_model-00001-of-00002.safetensors" in names
    assert ev.named("hydrate_start")[0]["bytes"] == 390
    assert ev.named("hydrate_done")[0]["complete"] is True


def test_a_time_cap_leaves_a_complete_tree(tmp_path):
    ev = Events()
    _, local, complete = _hydrate(tmp_path, ev, max_seconds=-1)
    shard = _local_snap(local) / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    assert not complete
    assert shard.is_symlink() and shard.read_bytes() == b"U" * 300
    assert ev.named("hydrate_done")[0]["reason"] == "time_cap"
    assert not (local / vh.SENTINEL).exists()
    assert list(local.rglob("*.part")) == []


def test_a_full_disk_leaves_a_complete_tree(tmp_path):
    ev = Events()
    _, local, complete = _hydrate(tmp_path, ev, free_bytes=lambda p: 0)
    assert not complete and ev.named("hydrate_done")[0]["reason"] == "disk"
    assert (_local_snap(local) / "text_encoder" / "model.safetensors").read_bytes() == b"T" * 40


def test_a_read_error_leaves_the_symlink_in_place(tmp_path, monkeypatch):
    ev = Events()

    def boom(*a):
        raise OSError("transport endpoint is not connected")

    monkeypatch.setattr(vh.os, "pread", boom)
    _, local, complete = _hydrate(tmp_path, ev)
    assert not complete and ev.named("hydrate_done")[0]["reason"].startswith("read_error")
    monkeypatch.undo()
    assert (_local_snap(local) / "unet" / "diffusion_pytorch_model-00001-of-00002.safetensors").read_bytes() == b"U" * 300


def test_a_warm_worker_copies_nothing(tmp_path):
    _hydrate(tmp_path, Events())
    ev = Events()
    assert vh.hydrate(tmp_path / "vol" / "hf-cache", tmp_path / "local" / "hf-cache", ev)
    assert ev.named("hydrate_file") == []
    assert ev.named("hydrate_done")[0]["reason"] == "warm"


def test_a_partial_run_resumes_without_recopying(tmp_path):
    hf, local, _ = _hydrate(tmp_path, Events(), max_seconds=-1)
    ev = Events()
    assert vh.hydrate(hf, local, ev, chunk_bytes=64, free_bytes=lambda p: 10 ** 15)
    assert ev.named("hydrate_done")[0]["reason"] is None
    ev2 = Events()
    (local / vh.SENTINEL).unlink()
    assert vh.hydrate(hf, local, ev2, chunk_bytes=64, free_bytes=lambda p: 10 ** 15)
    assert ev2.named("hydrate_file") == []


def _env(tmp_path, **extra):
    hf = _volume(tmp_path)
    return {"VJ_VOLUME_ROOT": str(hf.parent), "VJ_VOLUME_KIND": "global", "HF_HOME": str(hf),
            "VJ_HYDRATE_ROOT": str(tmp_path / "local"), **extra}


def test_maybe_hydrate_repoints_only_when_the_tree_is_built(tmp_path):
    e = _env(tmp_path)
    assert vh.maybe_hydrate(Events(), e) == "local"
    assert e["HF_HOME"] == str(tmp_path / "local" / "hf-cache")


def test_a_second_job_on_a_warm_worker_stays_local(tmp_path):
    e = _env(tmp_path)
    vh.maybe_hydrate(Events(), e)
    e["HF_HOME"] = str(tmp_path / "vol" / "hf-cache")  # the mirror repoints at the volume every job
    assert vh.maybe_hydrate(Events(), e) == "local"
    assert e["HF_HOME"] == str(tmp_path / "local" / "hf-cache")


def test_a_build_failure_keeps_the_job_on_the_volume(tmp_path, monkeypatch):
    e = _env(tmp_path)
    before = e["HF_HOME"]
    monkeypatch.setattr(vh, "build_tree", lambda *a: (_ for _ in ()).throw(PermissionError("no")))
    ev = Events()
    assert vh.maybe_hydrate(ev, e) == "failed"
    assert e["HF_HOME"] == before
    assert ev.named("hydrate_failed")[0]["error"] == "PermissionError"


def test_an_already_imported_hub_refuses_the_repoint(tmp_path, monkeypatch):
    import sys
    import types
    fake = types.ModuleType("huggingface_hub.constants")
    fake.HF_HUB_CACHE = str(tmp_path / "vol" / "hf-cache" / "hub")
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", fake)
    e = _env(tmp_path)
    before = e["HF_HOME"]
    ev = Events()
    assert vh.maybe_hydrate(ev, e) == "too_late"
    assert e["HF_HOME"] == before and ev.named("hydrate_skipped")


@pytest.mark.parametrize("env,want", [
    ({"VJ_VOLUME_KIND": "global"}, True),
    ({"VJ_VOLUME_KIND": "network"}, False),
    ({}, False),
    ({"VJ_VOLUME_KIND": "global", "VJ_HYDRATE": "0"}, False),
    ({"VJ_HYDRATE": "1"}, True),
])
def test_hydrate_enabled(env, want):
    assert vh.hydrate_enabled(env) is want


def test_maybe_hydrate_leaves_a_job_with_no_volume_alone():
    e = {"VJ_VOLUME_KIND": "global", "HF_HOME": "/opt/models/hf-cache"}
    assert vh.maybe_hydrate(Events(), e) == "no_volume"
    assert e["HF_HOME"] == "/opt/models/hf-cache"


def test_the_real_hub_resolves_files_through_the_tree(tmp_path):
    """The one un-stubbed reader: huggingface_hub's own offline lookup, through copies and symlinks."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import try_to_load_from_cache

    _, local, _ = _hydrate(tmp_path, Events())
    cache = str(local / "hub")
    repo = "SG161222/RealVisXL_V5.0"
    copied = try_to_load_from_cache(repo, "unet/diffusion_pytorch_model-00001-of-00002.safetensors", cache_dir=cache)
    linked = try_to_load_from_cache(repo, "RealVisXL_V5.0_fp32.safetensors", cache_dir=cache)
    assert Path(copied).read_bytes() == b"U" * 300
    assert Path(linked).read_bytes() == b"R" * 500


class Rec:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def test_timed_load_reports_start_and_elapsed():
    rec = Rec()
    load_events.set_emitter(rec)
    try:
        with load_events.timed_load("unet"):
            pass
    finally:
        load_events.set_emitter(None)
    assert [e for e, _ in rec.events] == ["load_start", "load_done"]
    assert rec.events[1][1]["name"] == "unet" and rec.events[1][1]["elapsed_s"] >= 0


def test_timed_load_reports_a_failure_and_reraises():
    rec = Rec()
    load_events.set_emitter(rec)
    try:
        with pytest.raises(OSError):
            with load_events.timed_load("unet"):
                raise OSError("missing")
    finally:
        load_events.set_emitter(None)
    assert rec.events[-1][0] == "load_failed" and rec.events[-1][1]["error"] == "OSError"


def test_timed_load_heartbeats_during_a_long_load():
    import time
    rec = Rec()
    load_events.set_emitter(rec)
    try:
        with load_events.timed_load("unet", heartbeat_seconds=0.01):
            time.sleep(0.1)
    finally:
        load_events.set_emitter(None)
    assert any(e == "load_heartbeat" for e, _ in rec.events)


def test_a_broken_emitter_never_reaches_the_load():
    class Broken:
        def emit(self, *a, **k):
            raise RuntimeError("r2 down")

    load_events.set_emitter(Broken())
    try:
        with load_events.timed_load("unet"):
            value = 1
    finally:
        load_events.set_emitter(None)
    assert value == 1
