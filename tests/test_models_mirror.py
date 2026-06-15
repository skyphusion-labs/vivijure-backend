"""The .rclonelink -> symlink reconstruction, tested on CPU (no R2, no rclone). rclone --links
leaves `<name>.rclonelink` marker files on download; the HF cache needs real symlinks."""
from pathlib import Path

import json

from vivijure_backend.harness import models_mirror
from vivijure_backend.harness.models_mirror import (
    DEFAULT_SKIP_REPOS,
    HF_OFFLINE_STUBS,
    I2V_LAZY_REPOS,
    I2V_SENTINEL,
    _DEFAULT_MODEL_VERSION,
    _PRELOAD_LOCK,
    SENTINEL,
    _acquire_volume_lock,
    _dir_bytes,
    _jitter_seconds,
    _mirror_event,
    _reconstruct_symlinks,
    _resolve_volume,
    _self_preload_volume,
    _skip_event,
    _truthy,
    ensure_i2v_models,
    mirror_cmd,
    start_i2v_prefetch,
    write_no_exist_stubs,
)


# --------------------------------------------------- preloaded network volume (#55 Phase C)

def _seed_volume(root: Path, version: str, *, base: bool = True, i2v: bool = True) -> Path:
    """A preloaded-volume layout: hf-cache dir + the requested sentinels at `version`. A FULLY
    preloaded volume has both; pass base/i2v to simulate a partial preload."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "hf-cache").mkdir(parents=True, exist_ok=True)
    if base:
        (root / SENTINEL).write_text(version + "\n")
    if i2v:
        (root / I2V_SENTINEL).write_text(version + "\n")
    return root


def test_resolve_volume_repoints_and_skips_on_a_fully_preloaded_volume(tmp_path):
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION)  # both sentinels
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is True
    # repointed at the volume so the deferred torch/diffusers loads read from it
    assert e["VJ_MODELS_ROOT"] == str(vol)
    assert e["HF_HOME"] == str(vol / "hf-cache")


def test_resolve_volume_requires_BOTH_sentinels_base_only_falls_back(tmp_path):
    # Partial preload: base sentinel written but i2v missing. Must NOT repoint, else a standalone
    # i2v_clip would try to mirror Wan onto the read-only volume and fail (Mackaye's catch).
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION, base=True, i2v=False)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert "HF_HOME" not in e  # falls back to R2 wholesale


def test_resolve_volume_i2v_only_also_falls_back(tmp_path):
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION, base=False, i2v=True)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert "HF_HOME" not in e


def test_resolve_volume_noop_when_unset(tmp_path):
    e = {}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert "VJ_MODELS_ROOT" not in e and "HF_HOME" not in e  # untouched -> R2 fallback runs


def test_resolve_volume_falls_back_on_version_mismatch(tmp_path):
    vol = _seed_volume(tmp_path / "vol", "1")  # both sentinels at v1
    e = {"VJ_VOLUME_ROOT": str(vol)}
    # want v2 but the volume carries v1 (e.g. mid-refresh): do NOT use it, do NOT repoint
    assert _resolve_volume(e, "2", log=lambda *_: None) is False
    assert "HF_HOME" not in e


def test_resolve_volume_falls_back_when_sentinel_absent(tmp_path):
    vol = tmp_path / "vol"
    (vol / "hf-cache").mkdir(parents=True)            # mounted but never preloaded (no sentinel)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert "HF_HOME" not in e


def test_resolve_volume_emits_volume_skip_event(tmp_path):
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION)  # both sentinels
    msgs: list[str] = []
    _resolve_volume({"VJ_VOLUME_ROOT": str(vol)}, _DEFAULT_MODEL_VERSION, log=msgs.append)
    assert any('"reason": "volume"' in m for m in msgs)


# --------------------------------------------------- self-preloading volumes (#55 Phase D)

def test_truthy():
    for v in ("1", "true", "TRUE", "Yes", "on"):
        assert _truthy(v)
    for v in ("0", "false", "", "no", "off", None):
        assert not _truthy(v)


def test_acquire_lock_basic_and_contention(tmp_path):
    lk = tmp_path / _PRELOAD_LOCK
    assert _acquire_volume_lock(lk, log=lambda *_: None) is True   # first wins
    assert lk.exists()
    assert _acquire_volume_lock(lk, log=lambda *_: None) is False  # fresh lock held -> loser


def test_acquire_lock_takes_over_stale(tmp_path):
    import os, time
    lk = tmp_path / _PRELOAD_LOCK
    lk.write_text("old")
    old = time.time() - 7200  # 2h, well past the 1h TTL
    os.utime(lk, (old, old))
    assert _acquire_volume_lock(lk, log=lambda *_: None) is True   # stale -> taken over


def test_resolve_volume_no_self_preload_when_flag_unset(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(models_mirror, "_self_preload_volume", lambda *a: called.append(1) or True)
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION, base=False, i2v=False)
    e = {"VJ_VOLUME_ROOT": str(vol)}                # flag NOT set
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert not called                              # default stays read-only, no self-preload


def test_resolve_volume_routes_to_self_preload_when_enabled(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(models_mirror, "_self_preload_volume",
                        lambda e, v, l: called.append(1) or True)
    vol = _seed_volume(tmp_path / "vol", _DEFAULT_MODEL_VERSION, base=False, i2v=False)
    e = {"VJ_VOLUME_ROOT": str(vol), "VJ_VOLUME_SELF_PRELOAD": "1"}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is True
    assert called                                  # routed to self-preload


def test_self_preload_lock_loser_falls_back_without_mirroring(tmp_path, monkeypatch):
    import time
    vol = tmp_path / "vol"
    vol.mkdir()
    (vol / _PRELOAD_LOCK).write_text(str(time.time()))  # a live writer already holds the lock
    def boom(**k):
        raise AssertionError("lock loser must not mirror")
    monkeypatch.setattr(models_mirror, "ensure_models", boom)
    monkeypatch.setattr(models_mirror, "ensure_i2v_models", boom)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _self_preload_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert "HF_HOME" not in e


def test_self_preload_winner_fills_repoints_and_releases_lock(tmp_path, monkeypatch):
    vol = tmp_path / "vol"
    (vol / "hf-cache").mkdir(parents=True)
    def fake_base(env=None, log=None, **k):
        (Path(env["VJ_MODELS_ROOT"]) / SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
        return True
    def fake_i2v(env=None, log=None, **k):
        (Path(env["VJ_MODELS_ROOT"]) / I2V_SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
        return True
    monkeypatch.setattr(models_mirror, "ensure_models", fake_base)
    monkeypatch.setattr(models_mirror, "ensure_i2v_models", fake_i2v)
    e = {"VJ_VOLUME_ROOT": str(vol)}
    assert _self_preload_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is True
    assert e["VJ_MODELS_ROOT"] == str(vol)
    assert e["HF_HOME"] == str(vol / "hf-cache")
    assert (vol / SENTINEL).exists() and (vol / I2V_SENTINEL).exists()
    assert not (vol / _PRELOAD_LOCK).exists()      # lock released after fill


# --------------------------------------------------- R2-egress jitter knob (#55 Phase C)

def test_jitter_off_by_default():
    assert _jitter_seconds({}) == 0.0
    assert _jitter_seconds({"VJ_MIRROR_JITTER_SEC": "0"}) == 0.0


def test_jitter_within_ceiling():
    for _ in range(50):
        assert 0.0 <= _jitter_seconds({"VJ_MIRROR_JITTER_SEC": "10"}) <= 10.0


def test_jitter_ignores_garbage():
    assert _jitter_seconds({"VJ_MIRROR_JITTER_SEC": "not-a-number"}) == 0.0


# ------------------------------------------------------ lazy i2v split (cold-start weight trim)

def test_cold_start_skips_heavy_i2v_and_dead_repos():
    # The cold-start pull must exclude the lazy i2v model and the two stray SDXL repos so a
    # keyframe/preview worker does not pull ~120GB + ~90GB of dead weight it never loads.
    for repo in ("models--Wan-AI--Wan2.2-I2V-A14B-Diffusers",
                 "models--stabilityai--stable-diffusion-xl-base-1.0",
                 "models--stabilityai--sdxl-turbo"):
        assert repo in DEFAULT_SKIP_REPOS
    cmd = mirror_cmd(Path("/x/conf"), "r2:b/models/hf-cache", Path("/dst"), skip_repos=DEFAULT_SKIP_REPOS)
    for repo in DEFAULT_SKIP_REPOS:
        assert f"hub/{repo}/**" in cmd          # each skip repo becomes an rclone --exclude


def test_every_lazy_repo_is_cold_start_skipped():
    # Invariant: anything the lazy path owns must be excluded from the cold-start pull, so it is
    # never double-pulled and never missed. (Both Wan I2V and the Lightning distill, now that
    # Lightning is seeded in R2 and would otherwise be pulled eagerly.)
    for repo in I2V_LAZY_REPOS:
        assert repo in DEFAULT_SKIP_REPOS


def test_ensure_i2v_skips_when_sentinel_present(tmp_path):
    (tmp_path / I2V_SENTINEL).write_text(_DEFAULT_MODEL_VERSION + "\n")
    env = {"VJ_MODELS_ROOT": str(tmp_path), "R2_ACCESS_KEY_ID": "x"}
    assert ensure_i2v_models(env=env, log=lambda *_: None) is False  # warm: no pull


def test_ensure_i2v_skips_when_no_r2_creds(tmp_path):
    env = {"VJ_MODELS_ROOT": str(tmp_path)}  # no R2 creds -> weights assumed pre-provisioned
    assert ensure_i2v_models(env=env, log=lambda *_: None) is False


# --------------------------------------------------------- eager i2v prefetch (perf #1)

def test_mirror_cmd_includes_multi_thread_flags():
    cmd = mirror_cmd(Path("/x/conf"), "r2:b/src", Path("/dst"))
    assert "--multi-thread-streams" in cmd
    assert "8" in cmd
    assert "--multi-thread-cutoff" in cmd
    assert "100M" in cmd


def test_start_i2v_prefetch_skips_warm(tmp_path, monkeypatch):
    monkeypatch.setattr(models_mirror, "_i2v_prefetch_thread", None)
    (tmp_path / I2V_SENTINEL).write_text("ok\n")
    env = {"VJ_MODELS_ROOT": str(tmp_path), "R2_ACCESS_KEY_ID": "x"}
    assert start_i2v_prefetch(env=env, log=lambda *_: None) is None


def test_start_i2v_prefetch_skips_no_creds(tmp_path, monkeypatch):
    monkeypatch.setattr(models_mirror, "_i2v_prefetch_thread", None)
    env = {"VJ_MODELS_ROOT": str(tmp_path)}  # no R2 creds
    assert start_i2v_prefetch(env=env, log=lambda *_: None) is None


def test_ensure_i2v_joins_prefetch_thread(tmp_path, monkeypatch):
    # Fake thread: is_alive()=True so the join branch fires; join() writes the sentinel.
    sentinel = tmp_path / I2V_SENTINEL
    joined = []

    class _FakeThread:
        def is_alive(self): return True
        def join(self):
            joined.append(True)
            sentinel.write_text(_DEFAULT_MODEL_VERSION + "\n")

    monkeypatch.setattr(models_mirror, "_i2v_prefetch_thread", _FakeThread())
    env = {"VJ_MODELS_ROOT": str(tmp_path)}
    result = ensure_i2v_models(env=env, log=lambda *_: None)
    assert joined, "ensure_i2v_models did not join the prefetch thread"
    assert result is False  # sentinel written by join -> skipped


def test_ensure_i2v_no_self_join_when_called_from_prefetch_thread(tmp_path, monkeypatch):
    # When ensure_i2v_models is called from within the prefetch thread itself (via
    # start_i2v_prefetch._pull), _i2v_prefetch_thread IS threading.current_thread().
    # Without the guard, join() raises RuntimeError("cannot join current thread").
    import threading
    monkeypatch.setattr(models_mirror, "_i2v_prefetch_thread", threading.current_thread())
    # No R2 creds -> returns False via "no creds" path; the point is no RuntimeError.
    env = {"VJ_MODELS_ROOT": str(tmp_path)}
    result = ensure_i2v_models(env=env, log=lambda *_: None)
    assert result is False


def test_reconstructs_symlink_from_marker(tmp_path):
    # mimic an HF-cache layout: a blob + a snapshot dir whose file is an .rclonelink marker
    (tmp_path / "blobs").mkdir()
    blob = tmp_path / "blobs" / "deadbeef"
    blob.write_text("weights")
    snap = tmp_path / "snapshots" / "rev" / "tokenizer"
    snap.mkdir(parents=True)
    marker = snap / "tokenizer_config.json.rclonelink"
    marker.write_text("../../../blobs/deadbeef")  # relative link target, as rclone stores it

    n = _reconstruct_symlinks(tmp_path, log=lambda *_: None)

    link = snap / "tokenizer_config.json"
    assert n == 1
    assert link.is_symlink()
    assert not marker.exists()                       # marker consumed
    assert link.read_text() == "weights"             # resolves through to the blob


def test_idempotent_and_quiet_when_no_markers(tmp_path):
    (tmp_path / "f.json").write_text("{}")
    assert _reconstruct_symlinks(tmp_path, log=lambda *_: None) == 0


# --------------------------------------------------------- cold-start telemetry (issue #55)

def _parse_event(line):
    # "@event <name> <json>" -> (name, payload dict)
    assert line.startswith("@event ")
    _, name, blob = line.split(" ", 2)
    return name, json.loads(blob)


def test_mirror_event_reports_per_leg_and_total_timing():
    legs = [("hf-cache", 120.0), ("antelopev2", 8.0), ("rife", 1.0), ("GFPGANv1.4", 3.0)]
    name, p = _parse_event(_mirror_event(legs, total_bytes=50_000_000_000, cold=True, model_version="1"))
    assert name == "mirror_complete"
    assert p["cold"] is True
    assert p["model_version"] == "1"
    assert p["total_seconds"] == 132.0          # sum of legs
    assert p["total_bytes"] == 50_000_000_000
    assert p["legs"]["hf-cache"] == 120.0
    # throughput = bytes / 1e6 / total_seconds (MB/s), derived not assumed
    assert p["throughput_mbps"] == round(50_000_000_000 / 1e6 / 132.0, 1)


def test_mirror_event_custom_name_for_i2v():
    name, p = _parse_event(_mirror_event([("wan", 600.0)], 120_000_000_000, cold=True,
                                         model_version="1", event="i2v_mirror_complete"))
    assert name == "i2v_mirror_complete"
    assert p["legs"]["wan"] == 600.0


def test_mirror_event_zero_time_no_divide_by_zero():
    name, p = _parse_event(_mirror_event([], total_bytes=0, cold=True, model_version="1"))
    assert p["total_seconds"] == 0.0
    assert p["throughput_mbps"] == 0.0          # guarded, not a ZeroDivisionError


def test_skip_event_carries_reason():
    name, p = _parse_event(_skip_event("warm"))
    assert name == "mirror_skipped"
    assert p["reason"] == "warm"
    name, p = _parse_event(_skip_event("no_creds", event="i2v_mirror_skipped"))
    assert name == "i2v_mirror_skipped"
    assert p["reason"] == "no_creds"


def test_dir_bytes_counts_real_files_not_symlinks(tmp_path):
    # one real blob + a snapshot symlink pointing at it: the blob counts once, the symlink never.
    (tmp_path / "blobs").mkdir()
    blob = tmp_path / "blobs" / "deadbeef"
    blob.write_bytes(b"x" * 1000)
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob)
    assert _dir_bytes(tmp_path) == 1000          # symlink not double-counted


# --------------------------------------------------------- HF offline .no_exist stub writer

def test_write_no_exist_stubs_creates_empty_files(tmp_path):
    # Simulate a post-snapshot_download HF cache with refs/main populated.
    cache_dir = tmp_path / "models--Org--Repo"
    (cache_dir / "refs").mkdir(parents=True)
    (cache_dir / "refs" / "main").write_text("abc123deadbeef\n")

    stubs = [("models--Org--Repo", "subfolder/weights.index.json")]
    written = write_no_exist_stubs(tmp_path, stubs, log=lambda *_: None)

    assert len(written) == 1
    stub = cache_dir / ".no_exist" / "abc123deadbeef" / "subfolder" / "weights.index.json"
    assert stub.exists()
    assert stub.read_text() == ""


def test_write_no_exist_stubs_skips_missing_refs(tmp_path):
    # If refs/main doesn't exist (snapshot_download failed), skip with a warning; no crash.
    stubs = [("models--Missing--Repo", "some/file.json")]
    written = write_no_exist_stubs(tmp_path, stubs, log=lambda *_: None)
    assert written == []


def test_write_no_exist_stubs_idempotent(tmp_path):
    cache_dir = tmp_path / "models--X--Y"
    (cache_dir / "refs").mkdir(parents=True)
    (cache_dir / "refs" / "main").write_text("rev1\n")
    stubs = [("models--X--Y", "a/b.json")]
    write_no_exist_stubs(tmp_path, stubs, log=lambda *_: None)
    write_no_exist_stubs(tmp_path, stubs, log=lambda *_: None)  # second call: no error
    assert (cache_dir / ".no_exist" / "rev1" / "a" / "b.json").exists()


def test_hf_offline_stubs_covers_known_probes():
    paths = {p for _, p in HF_OFFLINE_STUBS}
    # Probe 1: VAE shard-index (diffusers checks for sharded weights; single-file VAE has none)
    assert "vae/diffusion_pytorch_model.safetensors.index.json" in paths
    # Probe 2: ControlNet shard-index
    assert "diffusion_pytorch_model.safetensors.index.json" in paths
    # Probe 3: IP-Adapter image_encoder PEFT adapter_config (IP-Adapter is not a PEFT model)
    assert "sdxl_models/image_encoder/adapter_config.json" in paths
    assert len(HF_OFFLINE_STUBS) == 3  # probe 4 fixed in lora_train.py; update if more added


def test_overwrites_a_stale_nonsymlink(tmp_path):
    # if a plain file already sits where the symlink should go, the marker still wins
    (tmp_path / "x").write_text("real")
    (tmp_path / "x.json").write_text("stale")
    (tmp_path / "x.json.rclonelink").write_text("x")
    _reconstruct_symlinks(tmp_path, log=lambda *_: None)
    assert (tmp_path / "x.json").is_symlink()
    assert (tmp_path / "x.json").read_text() == "real"
