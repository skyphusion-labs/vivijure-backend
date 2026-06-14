"""The .rclonelink -> symlink reconstruction, tested on CPU (no R2, no rclone). rclone --links
leaves `<name>.rclonelink` marker files on download; the HF cache needs real symlinks."""
from pathlib import Path

from vivijure_backend.harness import models_mirror
from vivijure_backend.harness.models_mirror import (
    DEFAULT_SKIP_REPOS,
    HF_OFFLINE_STUBS,
    I2V_LAZY_REPOS,
    I2V_SENTINEL,
    _DEFAULT_MODEL_VERSION,
    _reconstruct_symlinks,
    ensure_i2v_models,
    mirror_cmd,
    start_i2v_prefetch,
    write_no_exist_stubs,
)


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
