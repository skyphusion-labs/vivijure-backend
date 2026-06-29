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
  bin  --src <staged-seed> --out <bins-dir> [--bins N] [--ceiling-gb 9.0]
        First-fit-decreasing pack the seed at <staged-seed> into <bins-dir>/bin-00 .. bin-(N-1).
        HARD GATE (pre-build): a single regular file >= the GHCR limit (10 GB) is fatal -- it can
        never fit a layer, so the seed curation must reshard it (e.g. a single-file checkpoint ->
        diffusers sharded safetensors). Files are hardlinked into bins when possible (no second
        copy of the bytes on the same filesystem), else copied. Symlinks are recreated as symlinks
        (tiny; HF-cache snapshot links). Writes <bins-dir>/bake-bins.json (the plan + per-bin bytes).

  verify-image  --image <ref> [--ceiling-gb 10.0]
        Post-build gate: read the built image's per-layer sizes (`docker history`) and FAIL if any
        layer is >= the ceiling. This is the authoritative check (the bin step is the prediction;
        this confirms the registry will accept the push). Run AFTER build, BEFORE push.

  bins-needed  --src <staged-seed> [--ceiling-gb 9.0]
        Print the minimum bin count for the staged seed (capacity planning; the Dockerfile fixes a
        generous bin count and empty bins are harmless zero-byte layers).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# GHCR hard per-layer ceiling. A layer at or above this is rejected by the registry.
GHCR_LAYER_LIMIT_BYTES = 10 * 1024**3


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


def bin_pack(src: Path, out: Path, bins: int, ceiling: int, log=print) -> dict:
    """First-fit-decreasing pack src's regular files into `bins` bin dirs under `out`, each < ceiling.

    HARD GATE: any single regular file >= GHCR_LAYER_LIMIT_BYTES is fatal (cannot fit a layer).
    Returns the plan dict (also written to out/bake-bins.json)."""
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

    p_v = sub.add_parser("verify-image", help="assert built image layers < ceiling")
    p_v.add_argument("--image", required=True)
    p_v.add_argument("--ceiling-gb", type=float, default=10.0)

    p_n = sub.add_parser("bins-needed", help="print min bin count for a staged seed")
    p_n.add_argument("--src", required=True, type=Path)
    p_n.add_argument("--ceiling-gb", type=float, default=9.0)

    args = ap.parse_args()
    if args.cmd == "reconstruct-symlinks":
        reconstruct_symlinks(args.root)
    elif args.cmd == "bin":
        bin_pack(args.src, args.out, args.bins, int(args.ceiling_gb * 1024**3))
    elif args.cmd == "verify-image":
        sys.exit(verify_image(args.image, int(args.ceiling_gb * 1024**3)))
    elif args.cmd == "bins-needed":
        ceiling = int(args.ceiling_gb * 1024**3)
        total = sum(sz for _, sz in _walk_regular_files(args.src))
        floor = -(-total // ceiling)
        print(f"total {total/1024**3:.1f} GB; >= {floor} bins at ceiling {args.ceiling_gb} GB "
              f"(add headroom; the Dockerfile fixes a generous bin count, empty bins are free).")


if __name__ == "__main__":
    main()
