"""The exact weight files a stills job loads, and how to pull only those from the R2 mirror.

`stills_manifest.json` is the one list. The mirror filter, the post-pull check and the volume hydrate
all read it, and tests/test_stills_mirror.py holds it to `models.DEFAULT_SPECS` and to the revisions
pinned in deploy/bake-manifest.json. Regenerate it with scripts/gen_stills_manifest.py.

A hub cache names a weight by its content hash under `blobs/`, with a link from `snapshots/<rev>/`.
So pulling one file means naming its blob. Everything small in a listed repo (configs, tokenizers,
refs) still comes, because the loaders read those too and the exact set is not worth listing.

Fp32, as loaded today. An fp16 or bf16 set is a separate, quality-checked change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

_MANIFEST_PATH = Path(__file__).with_name("stills_manifest.json")

# A link to any of these in a listed repo is left behind unless the manifest names it. These are the
# flax, onnx, openvino, fp16 and sample-image duplicates that make a whole repo 40 GB.
_UNLISTED_LINK_SUFFIXES = ("safetensors", "bin", "ckpt", "pth", "pt", "msgpack", "onnx", "onnx_data",
                           "xml", "h5", "gguf", "png", "jpg", "jpeg", "webp", "gif", "mp4")
# A large file is stored by its 64-hex LFS hash; a small git file by a 40-hex hash.
_LFS_BLOB = "{{[0-9a-f]{64}}}"


class StillsIncomplete(RuntimeError):
    """A stills pull finished without a file a loader will open. Fails the job at the mirror, naming
    the file, instead of as an offline Hugging Face error deep in a load."""


def load_manifest() -> list[dict]:
    return json.loads(_MANIFEST_PATH.read_text())["repos"]


def total_bytes(repos: list[dict] | None = None) -> int:
    return sum(f["bytes"] for r in (repos or load_manifest()) for f in r["files"])


def rclone_filters(repos: list[dict] | None = None) -> list[str]:
    """`--filter` rules, in order. Rclone stops at the first match, so listed files come first."""
    rules: list[str] = []
    for r in repos or load_manifest():
        base = f"hub/{r['dir']}"
        snap = f"{base}/snapshots/{r['revision']}"
        for f in r["files"]:
            rules.append(f"+ {base}/blobs/{f['blob']}")
            rules.append(f"+ {snap}/{f['path']}.rclonelink")
        rules.append(f"- {base}/blobs/{_LFS_BLOB}")
        rules.append(f"- {snap}/**.{{{','.join(_UNLISTED_LINK_SUFFIXES)}}}.rclonelink")
        rules.append(f"+ {snap}/**")
        rules.append(f"- {base}/snapshots/**")
        rules.append(f"+ {base}/**")
    rules.append("- **")
    return rules


def prune_dangling_links(hub: Path, log: Callable[[str], None] = print) -> int:
    """Drop links to blobs this set did not pull, so no reader sees a file that cannot open."""
    removed = 0
    for root, _dirs, files in os.walk(hub):
        for name in files:
            p = Path(root) / name
            if p.is_symlink() and not p.exists():
                p.unlink()
                removed += 1
    if removed:
        log(f"models_mirror: removed {removed} links to files outside the stills set")
    return removed


def verify(hf_home: Path, repos: list[dict] | None = None) -> None:
    """Raise `StillsIncomplete` unless every listed file resolves at its exact size."""
    bad: list[str] = []
    for r in repos or load_manifest():
        snap = hf_home / "hub" / r["dir"] / "snapshots" / r["revision"]
        for f in r["files"]:
            p = snap / f["path"]
            try:
                size = p.stat().st_size
            except OSError:
                bad.append(f"{r['repo_id']}/{f['path']} (missing)")
                continue
            if size != f["bytes"]:
                bad.append(f"{r['repo_id']}/{f['path']} ({size} bytes, want {f['bytes']})")
    if bad:
        raise StillsIncomplete("stills mirror is missing files a loader will open: " + "; ".join(bad))
