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
    BAKED_SENTINEL,
    is_baked,
    ensure_models,
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
    rclone_env,
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


def test_resolve_volume_unmounted_volume_falls_back_no_crash(tmp_path, monkeypatch):
    # VJ_VOLUME_ROOT set + self-preload on, but the path is NOT mounted: must fall back cleanly,
    # never attempt the lock (which would os.open a missing dir and crash the worker).
    called = []
    monkeypatch.setattr(models_mirror, "_self_preload_volume", lambda *a: called.append(1) or True)
    e = {"VJ_VOLUME_ROOT": str(tmp_path / "not-mounted"), "VJ_VOLUME_SELF_PRELOAD": "1"}
    assert _resolve_volume(e, _DEFAULT_MODEL_VERSION, log=lambda *_: None) is False
    assert not called
    assert "HF_HOME" not in e


def test_acquire_lock_missing_dir_returns_false(tmp_path):
    # lock path under a non-existent dir -> OSError(ENOENT) must be swallowed, not raised
    assert _acquire_volume_lock(tmp_path / "nope" / _PRELOAD_LOCK, log=lambda *_: None) is False


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
    cmd = mirror_cmd("r2:b/models/hf-cache", Path("/dst"), skip_repos=DEFAULT_SKIP_REPOS)
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
    cmd = mirror_cmd("r2:b/src", Path("/dst"))
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



# --------------------------------------------------------------------------- baked-image early-return

def _baked_root(tmp_path):
    """A models root that looks BAKED: the .vj-baked marker present, NO R2 creds, NO volume."""
    root = tmp_path / "opt-models"
    root.mkdir(parents=True, exist_ok=True)
    (root / BAKED_SENTINEL).write_text("")
    return root


def test_is_baked_true_when_marker_present(tmp_path):
    root = _baked_root(tmp_path)
    assert is_baked({"VJ_MODELS_ROOT": str(root)}) is True


def test_is_baked_false_when_marker_absent(tmp_path):
    root = tmp_path / "opt-models"
    root.mkdir()
    assert is_baked({"VJ_MODELS_ROOT": str(root)}) is False


def test_ensure_models_baked_skips_everything(tmp_path):
    # A baked worker MUST early-return (no pull) even with a volume set + no R2 creds: the weights
    # are in the image. We set VJ_VOLUME_ROOT to a bogus path to prove the volume path is not taken
    # (if it were, _resolve_volume would run before our early-return).
    root = _baked_root(tmp_path)
    e = {"VJ_MODELS_ROOT": str(root), "HF_HOME": str(root / "hf-cache"),
         "VJ_VOLUME_ROOT": str(tmp_path / "nonexistent-vol")}
    assert ensure_models(env=e) is False  # skipped, no raise (no rclone, no R2 needed)


def test_ensure_i2v_models_baked_skips_pull(tmp_path):
    root = _baked_root(tmp_path)
    e = {"VJ_MODELS_ROOT": str(root), "HF_HOME": str(root / "hf-cache")}
    assert ensure_i2v_models(env=e) is False  # baked fp8 i2v already present; no R2 pull


def test_not_baked_without_creds_does_not_falsely_skip_as_baked(tmp_path):
    # Negative: absent the marker, a no-creds worker takes the EXISTING no_creds path, not the baked
    # one -- baked must be a distinct, marker-gated decision.
    root = tmp_path / "opt-models"
    (root / "hf-cache" / "hub").mkdir(parents=True)
    (root / "hf-cache" / "hub" / "something").write_text("x")
    e = {"VJ_MODELS_ROOT": str(root), "HF_HOME": str(root / "hf-cache")}
    assert is_baked(e) is False
    assert ensure_models(env=e) is False  # no_creds skip, but NOT via the baked branch


# ------------------------------------------------------- repo_in_hf_cache (offline presence gate)

def test_repo_in_hf_cache_true_only_with_a_populated_snapshot(tmp_path):
    env = {"HF_HOME": str(tmp_path / "hf-cache")}
    repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    cache = tmp_path / "hf-cache" / "hub" / "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers"
    assert models_mirror.repo_in_hf_cache(repo, env) is False          # absent entirely
    (cache / "snapshots").mkdir(parents=True)
    assert models_mirror.repo_in_hf_cache(repo, env) is False          # empty snapshots/
    snap = cache / "snapshots" / "abc123"
    snap.mkdir()
    assert models_mirror.repo_in_hf_cache(repo, env) is False          # empty snapshot hash dir
    (snap / "model_index.json").write_text("{}")
    assert models_mirror.repo_in_hf_cache(repo, env) is True           # populated -> loadable


def test_repo_in_hf_cache_unbaked_fp8_repo_is_false(tmp_path):
    # The exact de-risk failure: bf16 repo present, -fp8 repo absent.
    env = {"HF_HOME": str(tmp_path / "hf-cache")}
    bf16 = (tmp_path / "hf-cache" / "hub" / "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers"
            / "snapshots" / "h")
    bf16.mkdir(parents=True)
    (bf16 / "model_index.json").write_text("{}")
    assert models_mirror.repo_in_hf_cache("Wan-AI/Wan2.2-I2V-A14B-Diffusers", env) is True
    assert models_mirror.repo_in_hf_cache("Wan-AI/Wan2.2-I2V-A14B-Diffusers-fp8", env) is False


# --------------------------------------- R2 secret never on disk (py/clear-text-storage, CodeQL #1)

_R2_CREDS = {"R2_ACCESS_KEY_ID": "AKIAFAKE", "R2_SECRET_ACCESS_KEY": "s3cr3t-NEVER-on-disk",
             "R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com", "R2_BUCKET": "vivijure"}


def _drive_cold_pull(tmp_path, monkeypatch):
    """Run ensure_models' cold path with a fake rclone (subprocess.run captured), returning the list
    of (argv, kwargs) calls. Not baked, no volume, no sentinel, creds present -> the mirror legs run."""
    import subprocess as _sp
    calls = []

    def fake_run(argv, *a, **kw):
        calls.append((argv, kw))
        class _CP:  # minimal CompletedProcess stand-in
            returncode = 0
        return _CP()

    monkeypatch.setattr(models_mirror.shutil, "which", lambda _: "/usr/bin/rclone")
    monkeypatch.setattr(models_mirror.subprocess, "run", fake_run)
    monkeypatch.setattr(models_mirror, "_reconstruct_symlinks", lambda *a, **k: 0)
    monkeypatch.setattr(models_mirror, "_jitter_seconds", lambda *_: 0.0)
    hf = tmp_path / "hf-cache"
    env = {**_R2_CREDS, "HF_HOME": str(hf), "VJ_MODELS_ROOT": str(tmp_path)}
    ran = ensure_models(env=env, log=lambda *_: None)
    return ran, calls


def test_cold_pull_passes_secret_in_env_never_in_argv(tmp_path, monkeypatch):
    ran, calls = _drive_cold_pull(tmp_path, monkeypatch)
    assert ran is True and calls, "cold pull did not run the mirror legs"
    secret = _R2_CREDS["R2_SECRET_ACCESS_KEY"]
    for argv, kw in calls:
        # the secret rides in the child env...
        assert kw.get("env", {}).get("RCLONE_CONFIG_R2_SECRET_ACCESS_KEY") == secret
        # ...and NEVER in the command line (argv) or any flag
        assert not any(secret in str(tok) for tok in argv), "secret leaked into rclone argv"
        assert "--config" not in argv, "an on-disk rclone.conf path was passed"
        assert argv[0] == "rclone" and "copy" in argv


def test_cold_pull_writes_no_secret_bearing_file_anywhere(tmp_path, monkeypatch):
    _drive_cold_pull(tmp_path, monkeypatch)
    secret = _R2_CREDS["R2_SECRET_ACCESS_KEY"].encode()
    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert secret not in f.read_bytes(), f"secret written to disk at {f}"
    # and explicitly: no rclone.conf was created (the old clear-text-on-disk path is gone)
    assert not list(tmp_path.rglob("rclone.conf"))


def test_cold_pull_resolves_the_r2_remote_from_env(tmp_path, monkeypatch):
    _, calls = _drive_cold_pull(tmp_path, monkeypatch)
    for argv, kw in calls:
        src = argv[-2]
        assert src.startswith("r2:"), f"leg src {src!r} does not use the r2: remote"
        # the remote name in argv matches the RCLONE_CONFIG_R2_* env that configures it
        assert kw["env"]["RCLONE_CONFIG_R2_TYPE"] == "s3"


def test_cold_pull_legs_are_behavior_equivalent(tmp_path, monkeypatch):
    # Same remote + same legs as before the env-var change: hf-cache + antelopev2 + the three finish
    # dirs (rife, GFPGANv1.4, facexlib). Guards that the secret-removal did not drop or reorder a leg.
    _, calls = _drive_cold_pull(tmp_path, monkeypatch)
    leg_srcs = [argv[-2] for argv, _ in calls]
    assert leg_srcs == [
        "r2:vivijure/models/hf-cache",
        "r2:vivijure/models/antelopev2",
        "r2:vivijure/models/rife",
        "r2:vivijure/models/GFPGANv1.4",
        "r2:vivijure/models/facexlib",
    ]
    # the hf-cache leg still carries the lazy/dead-repo excludes (behavior preserved)
    hf_argv = calls[0][0]
    for repo in DEFAULT_SKIP_REPOS:
        assert f"hub/{repo}/**" in hf_argv
