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

def _bin_of(out: Path) -> dict:
    """{relpath: bin dir name} for every regular file the pack placed."""
    return {str(p.relative_to(b)): b.name
            for b in sorted(out.iterdir()) if b.is_dir()
            for p in b.rglob("*") if p.is_file() and not p.is_symlink()}


def test_bin_pack_is_independent_of_walk_order(tmp_path, monkeypatch):
    """Equal-size files must pack the same way whatever order the filesystem lists them in (#564,
    fc#2620). Three 1 MB files with a 2.5 MB ceiling: two fit bin-00 and the third goes to bin-01.
    With a size-only sort key the third is whichever the walk listed last, so the same staged set
    packs differently on two hosts and an unchanged bin rebuilds to a different layer."""
    src = tmp_path / "src"
    names = ["a.bin", "b.bin", "c.bin"]
    for n in names:
        _mkfile(src / n, 1 * MB)
    placed = []
    for order in (names, list(reversed(names))):
        monkeypatch.setattr(bake_layers, "_walk_regular_files",
                            lambda root, order=order: ((Path(n), (root / n).stat().st_size) for n in order))
        out = tmp_path / f"bins-{len(placed)}"
        bake_layers.bin_pack(src, out, bins=2, ceiling=int(2.5 * MB), min_gb=0.0)
        placed.append(_bin_of(out))
    assert placed[0] == placed[1]


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


# --------------------------------------------------------------------------- layer-tar (#564, fc#2620)
# stills-base.yml appends each bin as ONE reproducible tar with a pinned crane. An unchanged bin must
# rebuild to the same tar, so the same diff_id and, through the same crane, the same layer digest.

import os  # noqa: E402
import tarfile  # noqa: E402

EPOCH = 1762900000


def _bin_tree(root: Path, order: str, mtime: int, file_mode: int) -> Path:
    """The same content, created in `order` ("fwd"/"rev") with the given mtimes and modes."""
    blob = bytes(range(256)) * 40
    steps = [
        lambda: _write(root / "hub" / "blobs" / "aaa", blob),
        lambda: _write(root / "facexlib" / "b.pth", b"small\n"),
        lambda: _link(root / "hub" / "snap" / "model.safetensors", "../blobs/aaa"),
        lambda: _write(root / "zz" / "run.sh", b"#!/bin/sh\n", mode=0o750),
    ]
    for step in (steps if order == "fwd" else steps[::-1]):
        step()
    for p in root.rglob("*"):
        if not p.is_symlink():
            os.utime(p, (mtime, mtime))
            if p.is_file() and p.name != "run.sh":
                p.chmod(file_mode)
    return root


def _write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _link(path: Path, target: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def _tar(tmp_path: Path, src: Path, name: str, epoch: int = EPOCH) -> tuple[dict, bytes]:
    out = tmp_path / name
    r = bake_layers.layer_tar(src, out, epoch, log=lambda *_: None)
    return r, out.read_bytes()


def test_layer_tar_same_input_gives_the_same_bytes_and_diff_id(tmp_path):
    """Same content, built in a different order, with different mtimes and non-exec modes: the same
    tar, byte for byte, and the diff_id is that tar's sha256."""
    a = _bin_tree(tmp_path / "a", "fwd", 1_600_000_000, 0o644)
    b = _bin_tree(tmp_path / "b", "rev", 1_700_000_000, 0o600)
    ra, ta = _tar(tmp_path, a, "a.tar")
    rb, tb = _tar(tmp_path, b, "b.tar")
    assert ta == tb
    assert ra == rb
    assert ra["diff_id"] == "sha256:" + hashlib.sha256(ta).hexdigest()
    assert ra["bytes"] == len(ta)


def test_layer_tar_entries_are_sorted_rootless_and_fixed(tmp_path):
    src = _bin_tree(tmp_path / "a", "rev", 1_600_000_000, 0o600)
    # A hardlink inside the bin (bin_pack hardlinks into the stage) is still a plain file entry.
    os.link(src / "facexlib" / "b.pth", src / "facexlib" / "b2.pth")
    _, data = _tar(tmp_path, src, "a.tar")
    with tarfile.open(tmp_path / "a.tar") as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert names[:2] == ["opt", "opt/models"]
        assert names[2:] == sorted(names[2:])
        assert all(n.startswith("opt/models/") for n in names[2:])
        assert all((m.uid, m.gid, m.uname, m.gname, m.mtime) == (0, 0, "", "", EPOCH) for m in members)
        by = {m.name: m for m in members}
        assert by["opt/models/hub"].isdir() and by["opt/models/hub"].mode == 0o755
        assert by["opt/models/facexlib/b.pth"].isreg() and by["opt/models/facexlib/b.pth"].mode == 0o644
        assert by["opt/models/facexlib/b2.pth"].isreg()
        assert tar.extractfile("opt/models/facexlib/b2.pth").read() == b"small\n"
        assert by["opt/models/zz/run.sh"].mode == 0o755
        link = by["opt/models/hub/snap/model.safetensors"]
        assert link.issym() and link.linkname == "../blobs/aaa" and link.mode == 0o777
    assert data


def test_layer_tar_changes_with_the_content_and_the_epoch(tmp_path):
    a = _bin_tree(tmp_path / "a", "fwd", 1_600_000_000, 0o644)
    r0, _ = _tar(tmp_path, a, "0.tar")
    r1, _ = _tar(tmp_path, a, "1.tar", epoch=EPOCH + 1)
    (a / "facexlib" / "b.pth").write_bytes(b"smalL\n")
    r2, _ = _tar(tmp_path, a, "2.tar")
    assert len({r0["diff_id"], r1["diff_id"], r2["diff_id"]}) == 3


def test_layer_tar_of_an_empty_bin_holds_only_the_models_root(tmp_path):
    (tmp_path / "empty").mkdir()
    _tar(tmp_path, tmp_path / "empty", "e.tar")
    with tarfile.open(tmp_path / "e.tar") as tar:
        assert tar.getnames() == ["opt", "opt/models"]


def test_layer_tar_refuses_a_missing_bin_and_a_special_file(tmp_path):
    with pytest.raises(SystemExit):
        bake_layers.layer_tar(tmp_path / "nope", tmp_path / "x.tar", EPOCH, log=lambda *_: None)
    src = tmp_path / "fifo"
    src.mkdir()
    os.mkfifo(src / "pipe")
    with pytest.raises(SystemExit):
        bake_layers.layer_tar(src, tmp_path / "y.tar", EPOCH, log=lambda *_: None)


def test_layer_tar_cli_prints_only_the_diff_id(tmp_path, capsys, monkeypatch):
    src = _bin_tree(tmp_path / "a", "fwd", 1_600_000_000, 0o644)
    out = tmp_path / "cli.tar"
    monkeypatch.setattr(sys, "argv", ["bake_layers.py", "layer-tar", "--bin", str(src), "--out", str(out),
                                      "--epoch", str(EPOCH)])
    bake_layers.main()
    assert capsys.readouterr().out == "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest() + "\n"


# --------------------------------------------------------------------------- assert-appended

BASE_CFG = {
    "config": {"Entrypoint": ["/opt/nvidia/nvidia_entrypoint.sh"], "Env": ["PATH=/usr/bin", "A=1"],
               "Labels": {"maintainer": "NVIDIA", "org.opencontainers.image.version": "22.04"}},
    "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "1" * 64, "sha256:" + "2" * 64]},
}
TARS = ["sha256:" + "a" * 64, "sha256:" + "b" * 64]
NEW_LABELS = {"org.opencontainers.image.revision": "c0ffee", "org.skyphusion.vivijure.model-version": "1"}


def _candidate(**over) -> dict:
    import copy
    c = copy.deepcopy(BASE_CFG)
    c["config"]["Labels"].update(NEW_LABELS)
    c["rootfs"]["diff_ids"] += TARS
    for k, v in over.items():
        if k == "diff_ids":
            c["rootfs"]["diff_ids"] = v
        else:
            c["config"][k] = v
    return c


def test_assert_appended_passes_the_toolchain_plus_the_tars():
    assert bake_layers.assert_appended(BASE_CFG, _candidate(), TARS, NEW_LABELS, log=lambda *_: None) == {
        "base_layers": 2, "appended": 2}


@pytest.mark.parametrize("cand", [
    _candidate(Entrypoint=["/bin/sh"]),                                         # entrypoint changed
    _candidate(Env=["PATH=/usr/bin"]),                                          # env lost
    _candidate(Labels={**NEW_LABELS}),                                          # base labels lost
    _candidate(Labels={**BASE_CFG["config"]["Labels"], **NEW_LABELS,
                       "org.skyphusion.vivijure.model-version": "2"}),          # label at a wrong value
    _candidate(diff_ids=BASE_CFG["rootfs"]["diff_ids"] + TARS[::-1]),           # tars out of order
    _candidate(diff_ids=BASE_CFG["rootfs"]["diff_ids"] + TARS[:1]),             # a tar missing
    _candidate(diff_ids=["sha256:" + "9" * 64] + BASE_CFG["rootfs"]["diff_ids"][1:] + TARS),  # base changed
])
def test_assert_appended_refuses_anything_else(cand):
    with pytest.raises(SystemExit):
        bake_layers.assert_appended(BASE_CFG, cand, TARS, NEW_LABELS, log=lambda *_: None)


def test_assert_appended_refuses_a_base_with_no_layers():
    empty = {"config": BASE_CFG["config"], "rootfs": {"type": "layers", "diff_ids": []}}
    cand = {"config": _candidate()["config"], "rootfs": {"type": "layers", "diff_ids": TARS}}
    with pytest.raises(SystemExit):
        bake_layers.assert_appended(empty, cand, TARS, NEW_LABELS, log=lambda *_: None)
