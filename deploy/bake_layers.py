#!/usr/bin/env python3
"""Bake-image layer tooling: bin-pack a staged weight seed into <10GB layers, and verify a built
image's layers stay under the GHCR per-layer ceiling.

WHY THIS EXISTS
---------------
The baked worker image carries the curated model set in the image itself (datacenter-agnostic, no
R2 cold-pull, no network-volume DC pinning). GHCR rejects an oversized layer: the per-layer ceiling
is 10 GB. A single `COPY seed/ /opt/models` would make ONE layer summing the whole set (tens to
~120 GB) -- rejected. Per-FILE COPY would make one layer per file, but the curated set is ~hundreds
of files and an image has a hard layer-count limit (~125), so that breaks too.

So we BIN-PACK: distribute the seed's files into a bounded number of bins, each bin < the ceiling,
each bin copied by ONE `COPY` in the Dockerfile -> one layer per bin, every layer < 10 GB, layer
count = number of bins (small). Each bin mirrors the file's relative path under the seed root, and
every bin is COPYed to the same `VJ_MODELS_ROOT`, so the union reconstructs the seed tree exactly
(cross-bin symlinks resolve because all bins land under one root).

This file is STANDALONE (no `vivijure_backend` import) so it runs in CI before `COPY src/` and does
not depend on the package being importable. CPU-only; no GPU, no model load.

SUBCOMMANDS
-----------
  bin  --src <staged-seed> --out <bins-dir> [--bins N] [--ceiling-gb 9.0] [--precision fp8|bf16] [--min-gb F]
        First-fit-decreasing pack the seed at <staged-seed> into <bins-dir>/bin-00 .. bin-(N-1).
        HARD GATE (upper): a single regular file >= the GHCR limit (10 GB) is fatal -- it can
        never fit a layer, so the seed curation must reshard it (e.g. a single-file checkpoint ->
        diffusers sharded safetensors). HARD GATE (lower): the packed total must be >= the floor
        (--min-gb, or the --precision default, else 0.5 GB), so an EMPTY/under-staged seed dies here
        rather than producing zero-byte bins that build into a hollow image (the #4 empty-bake bug).
        Files are hardlinked into bins when possible (no second copy of the bytes on the same
        filesystem), else copied. Symlinks are recreated as symlinks (tiny; HF-cache snapshot links).
        Writes <bins-dir>/bake-bins.json (the plan + per-bin bytes).

  assert-weights  --root <model-root> [--precision fp8|bf16] [--min-gb F] [--min-shard-gb 1.0]
        Hard-fail unless <model-root> holds a REAL weight set: total regular-file bytes >= the floor
        AND at least one shard >= --min-shard-gb. This GATES the `.vj-baked` sentinel write in the
        Dockerfile (a weightless image can never be stamped baked) and doubles as a post-build CI
        check. Stdlib-only: one implementation, both call sites. Catches the #4 empty-bake bug at the
        source (is_baked() trusts the sentinel; the sentinel must not lie).

  verify-image  --image <ref> [--ceiling-gb 10.0]
        Post-build gate: read the built image's per-layer sizes (`docker history`) and FAIL if any
        layer is >= the ceiling. This is the authoritative check (the bin step is the prediction;
        this confirms the registry will accept the push). Run AFTER build, BEFORE push.

  assert-shared-diff-ids  --image <ref> --base <ref> [--min-shared-gb 90]
        Post-build gate for deps-overlay (and optional full-rebuild regression): fail unless
        <image> shares most RootFS layer digests with <base>. Catches the t2 failure mode where
        `COPY --from=seed` re-emitted ~101 GB of near-identical weight layers under new digests.

  bins-needed  --src <staged-seed> [--ceiling-gb 9.0]
        Print the minimum bin count for the staged seed (capacity planning; the Dockerfile fixes a
        generous bin count and empty bins are harmless zero-byte layers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Sibling reader for the facexlib finish-leg pins (single source of truth = deploy/bake-manifest.json).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from facexlib_pins import load_facexlib_pins, verify_file  # noqa: E402

# GHCR hard per-layer ceiling. A layer at or above this is rejected by the registry.
GHCR_LAYER_LIMIT_BYTES = 10 * 1024**3

# Minimum on-disk weight footprint (sum of regular-file bytes under the model root) for a bake we
# will TRUST enough to stamp `.vj-baked`. A real curated set is ~50-60 GB (fp8) / ~80-90 GB (bf16);
# these floors sit well below the real size (so a legitimately re-curated seed is not rejected) but
# far above a config-only stub tree. The empty-bake bug (#4) shipped ~34 MB of HF config + .no_exist
# stubs with ZERO weight shards, yet still wrote the sentinel; any of these floors kills that.
WEIGHT_FLOOR_GB = {"fp8": 40.0, "bf16": 60.0}
DEFAULT_WEIGHT_FLOOR_GB = 40.0

# A genuine bake also has at least one LARGE shard. The stub tree's largest file was <10 MB, so a
# pile of small JSON could never reach the byte floor by accident -- but we require a real shard too
# as a second, independent signal that these are model weights, not config.
MIN_SHARD_BYTES = 1 * 1024**3


def _walk_regular_files(root: Path):
    """Yield (relpath, size_bytes) for every REGULAR file under root (symlinks handled separately)."""
    for p in root.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        yield p.relative_to(root), p.stat().st_size


def _walk_symlinks(root: Path):
    """Yield relpath for every symlink under root (recreated as-is in bin-00; they are tiny)."""
    for p in root.rglob("*"):
        if p.is_symlink():
            yield p.relative_to(root)


def _place(src_file: Path, dst_file: Path) -> None:
    """Hardlink src_file -> dst_file (no second copy of the bytes); fall back to copy across devices."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_file, dst_file)
    except OSError:
        shutil.copy2(src_file, dst_file)


def bin_pack(src: Path, out: Path, bins: int, ceiling: int, min_gb: float = 0.5, log=print) -> dict:
    """First-fit-decreasing pack src's regular files into `bins` bin dirs under `out`, each < ceiling.

    HARD GATE (upper): any single regular file >= GHCR_LAYER_LIMIT_BYTES is fatal (cannot fit a layer).
    HARD GATE (lower): the packed total must be >= `min_gb`, so an EMPTY or under-staged seed dies here
    instead of silently producing zero-byte bins that build into a hollow image (the #4 empty-bake bug;
    `rclone copy` of an empty R2 prefix exits 0). Returns the plan dict (also written to bake-bins.json)."""
    if ceiling >= GHCR_LAYER_LIMIT_BYTES:
        raise SystemExit("ceiling must be < 10 GB (the GHCR limit); pick e.g. 9.0 GB for headroom")
    files = sorted(_walk_regular_files(src), key=lambda rs: rs[1], reverse=True)
    over = [(str(r), sz) for r, sz in files if sz >= GHCR_LAYER_LIMIT_BYTES]
    if over:
        for rel, sz in over:
            log(f"FATAL: seed file {rel} is {sz/1024**3:.2f} GB >= 10 GB GHCR layer limit -- "
                "reshard it in the seed curation (no layer can hold it).")
        raise SystemExit(2)

    out.mkdir(parents=True, exist_ok=True)
    bin_bytes = [0] * bins
    bin_counts = [0] * bins
    for rel, sz in files:
        placed = False
        for i in range(bins):
            if bin_bytes[i] + sz < ceiling:
                _place(src / rel, out / f"bin-{i:02d}" / rel)
                bin_bytes[i] += sz
                bin_counts[i] += 1
                placed = True
                break
        if not placed:
            raise SystemExit(
                f"cannot fit {rel} ({sz/1024**3:.2f} GB) into {bins} bins at ceiling "
                f"{ceiling/1024**3:.2f} GB -- raise --bins or the seed grew; "
                f"current packed total {sum(bin_bytes)/1024**3:.1f} GB")

    packed = sum(bin_bytes)
    if packed < int(min_gb * 1024**3):
        raise SystemExit(
            f"bake_layers: packed seed is {packed/1024**3:.3f} GB, below the {min_gb:.1f} GB floor -- "
            f"the staged seed at {src} is EMPTY or under-staged. Refusing to bin-pack a hollow bake "
            f"(the #4 empty-bake bug). Re-stage r2:vivijure/bake-seed-<precision>/ and rebuild.")

    # Symlinks (HF-cache snapshot links) all go to bin-00; they are bytes-free path entries and must
    # land under the same root as their blob targets (every bin COPYs to VJ_MODELS_ROOT, so they do).
    sym = 0
    for rel in _walk_symlinks(src):
        target = os.readlink(src / rel)
        link = out / "bin-00" / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            os.symlink(target, link)
            sym += 1

    # Pre-create every bin dir so the Dockerfile's fixed COPY list never references a missing path
    # (an empty bin = a harmless zero-byte layer).
    for i in range(bins):
        (out / f"bin-{i:02d}").mkdir(parents=True, exist_ok=True)

    plan = {
        "ceiling_gb": round(ceiling / 1024**3, 3),
        "ghcr_limit_gb": round(GHCR_LAYER_LIMIT_BYTES / 1024**3, 3),
        "bins": bins,
        "symlinks": sym,
        "total_gb": round(sum(bin_bytes) / 1024**3, 3),
        "per_bin_gb": [round(b / 1024**3, 3) for b in bin_bytes],
        "per_bin_files": bin_counts,
        "max_bin_gb": round(max(bin_bytes) / 1024**3, 3) if bins else 0.0,
    }
    (out / "bake-bins.json").write_text(json.dumps(plan, indent=2) + "\n")
    log(f"bake_layers: packed {sum(bin_counts)} files + {sym} symlinks into {bins} bins; "
        f"total {plan['total_gb']:.1f} GB; largest bin {plan['max_bin_gb']:.2f} GB "
        f"(ceiling {plan['ceiling_gb']:.1f} GB, GHCR limit 10 GB). OK.")
    return plan


def assert_weights(root: Path, min_gb: float, min_shard_bytes: int = MIN_SHARD_BYTES,
                   log=print) -> dict:
    """Hard-fail unless `root` holds a REAL baked weight set, defending the `.vj-baked` sentinel from
    ever being written over an empty/stub tree (the #4 empty-bake bug). Two independent signals must
    both hold: (1) total regular-file bytes >= `min_gb`; (2) at least one shard >= `min_shard_bytes`.
    Returns a summary dict; raises SystemExit on failure. Stdlib-only so it runs inside the image
    build (gating the sentinel write) AND as a post-build CI step -- one implementation, two call
    sites."""
    if not root.is_dir():
        raise SystemExit(f"bake_layers: assert-weights FATAL -- model root {root} does not exist.")
    total = largest = nfiles = big = 0
    for _rel, sz in _walk_regular_files(root):
        total += sz
        nfiles += 1
        largest = max(largest, sz)
        if sz >= min_shard_bytes:
            big += 1
    summary = {
        "root": str(root), "files": nfiles,
        "total_gb": round(total / 1024**3, 3),
        "largest_gb": round(largest / 1024**3, 3),
        "shards_over_min": big,
        "min_gb": min_gb,
        "min_shard_gb": round(min_shard_bytes / 1024**3, 3),
    }
    if total < int(min_gb * 1024**3):
        raise SystemExit(
            f"bake_layers: assert-weights FATAL -- {root} holds {summary['total_gb']:.3f} GB of "
            f"regular files across {nfiles} file(s), below the {min_gb:.1f} GB floor. The weight seed "
            f"is EMPTY or under-staged; a .vj-baked sentinel must NEVER be written over a stub tree "
            f"(the #4 empty-bake bug). Re-stage r2:vivijure/bake-seed-<precision>/ and rebuild.")
    if big < 1:
        raise SystemExit(
            f"bake_layers: assert-weights FATAL -- {root} has {summary['total_gb']:.3f} GB across "
            f"{nfiles} file(s) but NO single file >= {summary['min_shard_gb']:.1f} GB. That looks like "
            f"config/stub files, not model shards. Re-stage the R2 bake seed and rebuild.")
    log(f"bake_layers: assert-weights OK -- {summary['total_gb']:.1f} GB across {nfiles} files, "
        f"largest {summary['largest_gb']:.2f} GB, {big} shard(s) >= {summary['min_shard_gb']:.1f} GB "
        f"(floor {min_gb:.1f} GB).")
    return summary


def assert_finish_shas(root: Path, manifest_path=None, log=print) -> dict:
    """Hard-fail unless every pinned facexlib finish-leg weight under <root>/facexlib matches its
    manifest sha256 (and exact size) EXACTLY. A baked artifact we make public reproducibility claims
    about cannot rest on a size-band + magic-byte gate -- that would accept a corrupted-but-plausible
    or substituted .pth. This asserts the bytes are the pinned bytes, at image build time, before the
    .vj-baked sentinel is trusted. Reads the SAME manifest pins the staging upload and the de-risk
    runtime probe use, so the three cannot drift. Returns a summary dict; raises SystemExit on any
    mismatch."""
    fx = root / "facexlib"
    try:
        pins = load_facexlib_pins(manifest_path)
    except (OSError, ValueError) as exc:
        raise SystemExit("bake_layers: assert-finish-shas FATAL -- cannot load facexlib pins: " + str(exc))
    checked = []
    for pin in pins:
        try:
            verify_file(fx / pin["name"], pin)
        except ValueError as exc:
            raise SystemExit(
                "bake_layers: assert-finish-shas FATAL -- " + str(exc) + ". The baked facexlib weight"
                " does not match its pinned sha256 (a corrupted / substituted / under-staged file)."
                " Re-stage r2:vivijure/models/facexlib via deploy/stage_facexlib_to_r2.py and rebuild.")
        checked.append(pin["name"])
    log("bake_layers: assert-finish-shas OK -- " + str(len(checked)) +
        " facexlib weight(s) match the manifest sha256 pin (" + ", ".join(checked) + ").")
    return {"facexlib_dir": str(fx), "verified": checked}


def assert_no_tree_cache(root, log=print):
    """Hard-fail if any huggingface_hub tree-cache listing (`<repo>/trees/<commit>.json`) survives
    under `root` (the HF hub). A baked tree listing makes hf_hub 1.x offline load hard-fail
    (IncompleteSnapshotError) against our deliberately-curated model subset -- the exact #206 break:
    the listing names siblings we correctly omit (e.g. RealVisXL_V5.0 root single-file checkpoints,
    one 13.8 GB, over the 10 GB layer ceiling), and the render-time completeness check demands them.
    deploy/bake_hf_configs.py scrubs these after the online config bake; THIS gate re-asserts, at
    BAKE, that none survived, so the class fails at build, never at the first prod job. Stdlib-only;
    scans `<root>/**/trees/*.json` so it is robust to the HF_HOME layout. Returns a summary; raises
    SystemExit on any survivor."""
    root = Path(root)
    stray = sorted(str(q.relative_to(root)) for q in root.rglob("trees/*.json"))
    if stray:
        sample = ", ".join(stray[:5]) + (f", ... ({len(stray) - 5} more)" if len(stray) > 5 else "")
        raise SystemExit(
            f"bake_layers: assert-no-tree-cache FATAL -- {len(stray)} huggingface_hub tree listing(s) "
            f"survived under {root} ({sample}). A baked tree listing makes offline load hard-fail with "
            f"IncompleteSnapshotError against the curated model subset (#206). deploy/bake_hf_configs.py "
            f"must scrub every <repo>/trees/ after the online config bake; re-run it or drop these.")
    log(f"bake_layers: assert-no-tree-cache OK -- no tree listing survives under {root}.")
    return {"root": str(root), "tree_listings": 0}


def _floor_for(precision: str | None, explicit_min_gb: float | None) -> float:
    """Resolve the byte floor: an explicit --min-gb wins; else the precision default; else the global
    default. Keeps the Dockerfile/CI call sites declarative (pass --precision, get the right floor)."""
    if explicit_min_gb is not None:
        return explicit_min_gb
    if precision:
        return WEIGHT_FLOOR_GB.get(precision, DEFAULT_WEIGHT_FLOOR_GB)
    return DEFAULT_WEIGHT_FLOOR_GB


def verify_image(image: str, ceiling: int, log=print) -> int:
    """Post-build gate: assert every layer of `image` is < ceiling via `docker history`."""
    out = subprocess.run(
        ["docker", "history", "--no-trunc", "--format", "{{.Size}}\t{{.CreatedBy}}", image],
        check=True, capture_output=True, text=True).stdout
    bad = []
    for line in out.splitlines():
        if not line.strip():
            continue
        size_str = line.split("\t", 1)[0].strip()
        nbytes = _human_to_bytes(size_str)
        if nbytes >= ceiling:
            bad.append((size_str, line.split("\t", 1)[-1][:80]))
    if bad:
        for sz, what in bad:
            log(f"FATAL: image layer {sz} >= ceiling -- {what}")
        log(f"bake_layers: {len(bad)} layer(s) over the {ceiling/1024**3:.0f} GB ceiling; "
            "GHCR would reject this push.")
        return 1
    log(f"bake_layers: all layers of {image} are under the {ceiling/1024**3:.0f} GB ceiling. OK.")
    return 0


def _rootfs_diff_ids(image: str) -> list[str]:
    """Uncompressed layer digests from a local docker image (RootFS.Layers)."""
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image],
        check=True, capture_output=True, text=True).stdout.strip()
    ids = json.loads(out)
    if not isinstance(ids, list) or not ids:
        raise SystemExit(f"bake_layers: no RootFS.Layers on image {image}")
    return ids


def assert_shared_diff_ids(
    image: str,
    base: str,
    min_shared_gb: float,
    log=print,
) -> int:
    """Fail unless `image` shares enough RootFS layer digests with `base`.

    Deps-overlay runtimes MUST inherit the base's weight layers by blob identity. A full
    FROM-cuda rebuild that re-COPY --from=seed can emit NEW weight digests for near-identical
    bytes (t2 vs t1, 2026-07-15: ~101 GB miss) and break RunPod cold pulls. This gate catches
    that before push when an overlay/base ref is supplied.
    """
    a = set(_rootfs_diff_ids(image))
    b = set(_rootfs_diff_ids(base))
    shared = a & b
    # docker history sizes are human; approximate shared mass via inspect Size is too coarse.
    # Require a high shared *count* of layers AND that shared is most of the base's layers.
    if not b:
        log(f"FATAL: base image {base} has no RootFS layers")
        return 1
    shared_frac = len(shared) / len(b)
    # Weight-heavy images: base has ~50+ layers; overlay should share nearly all of them.
    # min_shared_gb is documented for operators; we enforce via fraction + absolute count
    # because uncompressed size per diff_id is not in `docker image inspect` without history.
    min_count = max(20, int(0.7 * len(b)))
    if len(shared) < min_count or shared_frac < 0.7:
        log(
            f"FATAL: {image} shares only {len(shared)}/{len(b)} RootFS layers with {base} "
            f"({shared_frac:.0%}; need >= {min_count} and >= 70%). "
            f"Weight layers were likely re-emitted -- use deploy/runtime-overlay.Dockerfile "
            f"for deps-only bumps, or fix COPY determinism before a full rebuild. "
            f"(operator floor was {min_shared_gb} GB shared compressed blobs on GHCR.)"
        )
        only_new = len(a - b)
        log(f"bake_layers: new-only layers={only_new} base-only={len(b - a)}")
        return 1
    log(
        f"bake_layers: assert-shared-diff-ids OK -- {image} shares {len(shared)}/{len(b)} "
        f"RootFS layers with {base} ({shared_frac:.0%}; new-only={len(a - b)})."
    )
    return 0


def _human_to_bytes(s: str) -> float:
    """Parse a docker-history size like '9.31GB', '512MB', '0B' into bytes."""
    s = s.strip()
    units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12,
             "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4}
    for u in sorted(units, key=len, reverse=True):
        if s.upper().endswith(u):
            try:
                return float(s[: -len(u)]) * units[u]
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def reconstruct_symlinks(root: Path, log=print) -> int:
    """Turn every `*.rclonelink` marker under root into the real symlink it names (its content is the
    link target). rclone --links stores HF-cache symlinks as `<name>.rclonelink` text files and recent
    rclone does NOT translate them back on download, so the staged seed carries markers, not links;
    baking the markers verbatim would break the offline HF cache. This mirrors
    harness/models_mirror._reconstruct_symlinks (the same convention the runtime R2 mirror uses), run
    once over the staged seed BEFORE bin-packing. Idempotent. Returns the number rebuilt."""
    n = 0
    for marker in root.rglob("*.rclonelink"):
        target = marker.read_text().strip()
        link = marker.with_suffix("")  # drop the .rclonelink extension
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
            marker.unlink()
            n += 1
        except OSError as exc:
            log(f"bake_layers: could not rebuild symlink {link} -> {target} ({exc})")
    log(f"bake_layers: reconstructed {n} HF-cache symlink(s) from .rclonelink markers under {root}.")
    return n


def _sha256_file(path: Path) -> str:
    """Stream-hash a file with sha256 (constant memory; handles multi-GB shards)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(bins_root: Path, out: Path, log=print) -> int:
    """Emit a `sha256sum -c`-compatible manifest of every weight file across the bins, keyed by the
    UNION path (the bin-NN/ prefix stripped), so the CONSUMING image can verify after its per-bin
    COPY --from lands them all under one root:

        cd /opt/models && sha256sum -c weights-manifest.sha256

    Each bin-NN/ mirrors the file path relative to VJ_MODELS_ROOT (bin_pack invariant) and bin_pack
    PARTITIONS the seed, so a union path appears in exactly one bin -- stripping the leading bin-NN/
    yields a unique, deterministic key. Regular files only: symlinks are structure, not content, and
    `sha256sum -c` would follow them to a blob already covered by its own entry. Sorted for a stable,
    reviewable diff. This is the source-of-truth checksum the weights-base image carries; a byte
    mismatch in a consumer build fails LOUD. Returns the number of files recorded."""
    entries = []
    for bin_dir in sorted(bins_root.glob("bin-*")):
        if not bin_dir.is_dir():
            continue
        for rel, _sz in _walk_regular_files(bin_dir):
            union = rel.as_posix()  # path relative to the bin == path relative to VJ_MODELS_ROOT
            if union == out.name:
                continue
            entries.append((union, _sha256_file(bin_dir / rel)))
    entries.sort(key=lambda e: e[0])
    # sha256sum -c format: "<hex>  <path>" (two spaces); relative paths resolve against the
    # verifier cwd (VJ_MODELS_ROOT).
    out.write_text("".join(f"{sha}  {path}\n" for path, sha in entries))
    log(f"bake_layers: wrote {len(entries)} weight file checksum(s) to {out} "
        f"(union-keyed sha256; verify with `sha256sum -c` under VJ_MODELS_ROOT).")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake-image layer bin-packer + per-layer GHCR gate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_r = sub.add_parser("reconstruct-symlinks", help="rebuild HF-cache symlinks from .rclonelink markers")
    p_r.add_argument("--root", required=True, type=Path)

    p_bin = sub.add_parser("bin", help="bin-pack a staged seed into <ceiling layers")
    p_bin.add_argument("--src", required=True, type=Path)
    p_bin.add_argument("--out", required=True, type=Path)
    p_bin.add_argument("--bins", type=int, default=24)
    p_bin.add_argument("--ceiling-gb", type=float, default=9.0)
    p_bin.add_argument("--precision", choices=sorted(WEIGHT_FLOOR_GB), default=None,
                       help="select the empty-seed floor (fp8/bf16); overridden by --min-gb")
    p_bin.add_argument("--min-gb", type=float, default=None,
                       help="hard lower bound on packed bytes; default 0.5 GB (or the precision floor)")

    p_aw = sub.add_parser("assert-weights",
                          help="hard-fail unless the model root holds a real weight set (gates .vj-baked)")
    p_aw.add_argument("--root", required=True, type=Path)
    p_aw.add_argument("--precision", choices=sorted(WEIGHT_FLOOR_GB), default=None,
                      help="select the byte floor (fp8/bf16); overridden by --min-gb")
    p_aw.add_argument("--min-gb", type=float, default=None,
                      help="byte floor in GB; default = the precision floor or the global default")
    p_aw.add_argument("--min-shard-gb", type=float, default=MIN_SHARD_BYTES / 1024**3,
                      help="require at least one regular file this large (real shard, not config)")

    p_fx = sub.add_parser("assert-finish-shas",
                          help="hard-fail unless baked facexlib weights match the manifest sha256 pin")
    p_fx.add_argument("--root", required=True, type=Path,
                      help="model root; <root>/facexlib must hold the pinned files")
    p_fx.add_argument("--manifest", type=Path, default=None,
                      help="bake-manifest.json with the facexlib pins (default: alongside this script)")

    p_tc = sub.add_parser("assert-no-tree-cache",
                          help="hard-fail if any hf_hub tree-cache listing survives (the #206 gate)")
    p_tc.add_argument("--root", required=True, type=Path,
                      help="models root (or HF hub root); scanned recursively for */trees/*.json")

    p_v = sub.add_parser("verify-image", help="assert built image layers < ceiling")
    p_v.add_argument("--image", required=True)
    p_v.add_argument("--ceiling-gb", type=float, default=10.0)

    p_s = sub.add_parser(
        "assert-shared-diff-ids",
        help="assert image shares most RootFS layers with a base (deps-overlay / weight-dedup gate)",
    )
    p_s.add_argument("--image", required=True)
    p_s.add_argument("--base", required=True, help="known-good runtime (e.g. runtime-1-bf16-t1@digest)")
    p_s.add_argument(
        "--min-shared-gb",
        type=float,
        default=90.0,
        help="operator floor (documented); gate enforces shared RootFS layer count/fraction",
    )

    p_n = sub.add_parser("bins-needed", help="print min bin count for a staged seed")
    p_n.add_argument("--src", required=True, type=Path)
    p_n.add_argument("--ceiling-gb", type=float, default=9.0)

    p_m = sub.add_parser("manifest",
                         help="write a union-keyed sha256sum -c manifest of the bins (weights-base gate)")
    p_m.add_argument("--bins-root", required=True, type=Path,
                     help="the bins dir (deploy/seed-bins) holding bin-NN/ subtrees")
    p_m.add_argument("--out", required=True, type=Path, help="manifest output path")

    args = ap.parse_args()
    if args.cmd == "reconstruct-symlinks":
        reconstruct_symlinks(args.root)
    elif args.cmd == "bin":
        floor = _floor_for(args.precision, args.min_gb) if (args.precision or args.min_gb is not None) else 0.5
        bin_pack(args.src, args.out, args.bins, int(args.ceiling_gb * 1024**3), min_gb=floor)
    elif args.cmd == "assert-weights":
        assert_weights(args.root, _floor_for(args.precision, args.min_gb),
                       int(args.min_shard_gb * 1024**3))
    elif args.cmd == "assert-finish-shas":
        assert_finish_shas(args.root, args.manifest)
    elif args.cmd == "assert-no-tree-cache":
        assert_no_tree_cache(args.root)
    elif args.cmd == "verify-image":
        sys.exit(verify_image(args.image, int(args.ceiling_gb * 1024**3)))
    elif args.cmd == "assert-shared-diff-ids":
        sys.exit(assert_shared_diff_ids(args.image, args.base, args.min_shared_gb))
    elif args.cmd == "bins-needed":
        ceiling = int(args.ceiling_gb * 1024**3)
        total = sum(sz for _, sz in _walk_regular_files(args.src))
        floor = -(-total // ceiling)
        print(f"total {total/1024**3:.1f} GB; >= {floor} bins at ceiling {args.ceiling_gb} GB "
              f"(add headroom; the Dockerfile fixes a generous bin count, empty bins are free).")
    elif args.cmd == "manifest":
        write_manifest(args.bins_root, args.out)


if __name__ == "__main__":
    main()
