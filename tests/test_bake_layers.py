"""Guards for the bake layer tooling -- specifically the #4 empty-bake defenses: bin-pack's lower
floor and `assert-weights` (which gates the .vj-baked sentinel in the Dockerfile). CPU-only, no R2,
no docker. Shard sizes are faked with sparse files (truncate) so the suite stays fast and tiny.

bake_layers.py lives in deploy/ (standalone, off the src/ pythonpath), so it is imported by path."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import bake_layers  # noqa: E402


def _mkfile(path: Path, size: int) -> None:
    """Create a (sparse) file of `size` bytes; st_size reports `size` without writing the bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if size:
            f.truncate(size)


MB = 1024**2
GB = 1024**3


# --------------------------------------------------------------------------- assert-weights

def test_assert_weights_empty_dir_fails(tmp_path):
    """The exact #4 shape: a model root with nothing in it must never pass."""
    with pytest.raises(SystemExit):
        bake_layers.assert_weights(tmp_path, min_gb=1.0)


def test_assert_weights_missing_root_fails(tmp_path):
    with pytest.raises(SystemExit):
        bake_layers.assert_weights(tmp_path / "nope", min_gb=0.0)


def test_assert_weights_stub_tree_fails_no_real_shard(tmp_path):
    """Config/stub files only: total can clear a tiny byte floor, but there is no real shard.
    This is the empty-bake's actual disk shape (lots of small JSON, largest file < 10 MB)."""
    for i in range(20):
        _mkfile(tmp_path / "hf-cache" / f"config_{i}.json", 8 * 1024)
    with pytest.raises(SystemExit):
        bake_layers.assert_weights(tmp_path, min_gb=0.0, min_shard_bytes=1 * MB)


def test_assert_weights_under_byte_floor_fails(tmp_path):
    """A single real-ish shard, but nowhere near the byte floor -> under-staged seed."""
    _mkfile(tmp_path / "transformer" / "shard.safetensors", 2 * MB)
    with pytest.raises(SystemExit):
        bake_layers.assert_weights(tmp_path, min_gb=1.0, min_shard_bytes=1 * MB)


def test_assert_weights_passes_real_set(tmp_path):
    """Byte floor cleared AND a real shard present -> pass, with an informative summary."""
    _mkfile(tmp_path / "transformer" / "shard-1.safetensors", 3 * MB)
    _mkfile(tmp_path / "vae" / "diffusion_pytorch_model.safetensors", 1 * MB)
    _mkfile(tmp_path / "hf-cache" / "config.json", 4 * 1024)
    summary = bake_layers.assert_weights(tmp_path, min_gb=0.002, min_shard_bytes=1 * MB)
    assert summary["files"] == 3
    assert summary["shards_over_min"] == 2
    assert summary["largest_gb"] > 0  # the 3 MB shard registers (exact value is rounded to 3 dp)


def test_floor_for_resolution():
    assert bake_layers._floor_for("bf16", None) == bake_layers.WEIGHT_FLOOR_GB["bf16"]
    assert bake_layers._floor_for("fp8", None) == bake_layers.WEIGHT_FLOOR_GB["fp8"]
    assert bake_layers._floor_for(None, None) == bake_layers.DEFAULT_WEIGHT_FLOOR_GB
    assert bake_layers._floor_for("bf16", 7.5) == 7.5  # explicit --min-gb wins


# --------------------------------------------------------------------------- bin-pack floor

def test_bin_pack_empty_seed_fails(tmp_path):
    """An empty staged seed must die at bin-pack (the default 0.5 GB floor), not produce zero-byte
    bins that build into a hollow image."""
    src = tmp_path / "_seed"
    src.mkdir()
    with pytest.raises(SystemExit):
        bake_layers.bin_pack(src, tmp_path / "bins", bins=4,
                             ceiling=int(9.0 * GB), min_gb=0.5)


def test_bin_pack_under_floor_fails(tmp_path):
    src = tmp_path / "_seed"
    _mkfile(src / "a.safetensors", 5 * MB)
    with pytest.raises(SystemExit):
        bake_layers.bin_pack(src, tmp_path / "bins", bins=4,
                             ceiling=int(9.0 * GB), min_gb=1.0)


def test_bin_pack_succeeds_above_floor(tmp_path):
    """Above the floor: packs, writes the plan, creates every bin dir."""
    src = tmp_path / "_seed"
    _mkfile(src / "transformer" / "a.safetensors", 6 * MB)
    _mkfile(src / "vae" / "b.safetensors", 2 * MB)
    out = tmp_path / "bins"
    plan = bake_layers.bin_pack(src, out, bins=4, ceiling=int(9.0 * GB), min_gb=0.001)
    assert plan["total_gb"] > 0 and plan["per_bin_files"] and sum(plan["per_bin_files"]) == 2
    assert (out / "bake-bins.json").is_file()
    for i in range(4):
        assert (out / f"bin-{i:02d}").is_dir()


# ------------------------------------------------------------------- assert-no-tree-cache (#206)

def _tree_listing(hub: Path, repo: str, commit: str) -> Path:
    """Write a fake hf_hub tree-cache listing at <hub>/<repo>/trees/<commit>.json."""
    p = hub / repo / "trees" / f"{commit}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"format_version": 1, "files": {}}')
    return p


def test_assert_no_tree_cache_clean_passes(tmp_path):
    """No trees/ anywhere -> the gate passes (the scrubbed, shippable state)."""
    _mkfile(tmp_path / "hf-cache" / "hub" / "models--x--y" / "snapshots" / "abc" / "config.json", 512)
    summary = bake_layers.assert_no_tree_cache(tmp_path)
    assert summary["tree_listings"] == 0


def test_assert_no_tree_cache_survivor_fails(tmp_path):
    """A surviving tree listing (the #206 landmine) must fail the bake, not ship."""
    hub = tmp_path / "hf-cache" / "hub"
    _tree_listing(hub, "models--SG161222--RealVisXL_V5.0", "ac93e0dd")
    with pytest.raises(SystemExit):
        bake_layers.assert_no_tree_cache(tmp_path)


# --------------------------------------------------------------------------- write_manifest


def _bins_with(tmp_path, files):
    """Build a bins-root (bin-00.. ) holding {relpath: bytes}, via the real bin_pack."""
    src = tmp_path / "_seed"
    for rel, data in files.items():
        pth = src / rel
        pth.parent.mkdir(parents=True, exist_ok=True)
        pth.write_bytes(data)
    out = tmp_path / "bins"
    bake_layers.bin_pack(src, out, bins=4, ceiling=int(9.0 * GB), min_gb=0.0)
    return out


def test_write_manifest_roundtrip(tmp_path):
    files = {
        "hf-cache/hub/models--X/blobs/aaa": b"first-shard-bytes",
        "facexlib/detection.pth": b"finish-weight-bytes",
    }
    bins = _bins_with(tmp_path, files)
    out = bins / "weights-manifest.sha256"
    n = bake_layers.write_manifest(bins, out)
    assert n == len(files)
    lines = out.read_text().splitlines()
    got = {path: sha for sha, _sep, path in (ln.partition("  ") for ln in lines)}
    # union-keyed: the bin-NN/ prefix is stripped, keyed by the runtime (VJ_MODELS_ROOT) path
    assert set(got) == set(files)
    for rel, data in files.items():
        assert got[rel] == hashlib.sha256(data).hexdigest()
    # sha256sum -c wire format: 64 hex chars then exactly two spaces then the path
    for ln in lines:
        assert ln[64:66] == "  "
    # sorted for a stable, reviewable diff
    paths = [ln.partition("  ")[2] for ln in lines]
    assert paths == sorted(paths)


def test_write_manifest_excludes_symlinks_and_self(tmp_path):
    src = tmp_path / "_seed"
    blob = src / "hf-cache/hub/models--X/blobs/aaa"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"real-shard")
    snap = src / "hf-cache/hub/models--X/snapshots/rev"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "model.safetensors").symlink_to("../../blobs/aaa")
    bins = tmp_path / "bins"
    bake_layers.bin_pack(src, bins, bins=4, ceiling=int(9.0 * GB), min_gb=0.0)
    out = bins / "weights-manifest.sha256"
    n = bake_layers.write_manifest(bins, out)
    # exactly one regular file; the symlink is structure (not content), and the manifest skips itself
    assert n == 1
    paths = [ln.partition("  ")[2] for ln in out.read_text().splitlines()]
    assert paths == ["hf-cache/hub/models--X/blobs/aaa"]
