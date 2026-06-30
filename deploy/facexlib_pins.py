#!/usr/bin/env python3
"""Single source of truth for the facexlib finish-leg weight pins.

The pins live in deploy/bake-manifest.json (the change-controlled bake record); this module is the
ONE reader so every consumer -- the R2 staging upload (stage_facexlib_to_r2.py), the build-time bake
gate (bake_layers.py), and the de-risk runtime probe (vj_derisk.py reads the baked copy of the same
manifest) -- asserts against the SAME bytes and cannot drift to a private literal. Stdlib-only so it
runs inside the image build and on jello alike.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Repo-relative default; callers inside the image pass the baked copy explicitly.
_DEFAULT_MANIFEST = Path(__file__).resolve().parent / "bake-manifest.json"


def load_facexlib_pins(manifest_path=None):
    """Return the facexlib finish-dir file pins (name/url/size/sha256) from the manifest.

    Raises ValueError if the manifest has no facexlib finish-dir entry or it carries no pinned files,
    so a malformed/unpinned manifest fails loud rather than silently skipping the integrity gate."""
    mp = Path(manifest_path) if manifest_path else _DEFAULT_MANIFEST
    manifest = json.loads(Path(mp).read_text())
    for entry in manifest.get("finish_dirs", []):
        if entry.get("dir") == "facexlib":
            files = entry.get("files") or []
            if not files:
                raise ValueError(str(mp) + ": facexlib finish-dir entry has no pinned files")
            for f in files:
                miss = [k for k in ("name", "url", "size", "sha256") if not f.get(k)]
                if miss:
                    name = f.get("name", "?")
                    raise ValueError(str(mp) + ": facexlib pin " + name + " missing " + str(miss))
            return files
    raise ValueError(str(mp) + ": no facexlib entry under finish_dirs")


def sha256_of(path, _chunk=1 << 20):
    """Streaming sha256 of a file (chunked so a ~100 MB weight does not load whole into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_file(path, pin):
    """Hard-assert a single file matches its manifest pin (size THEN sha256). Raises ValueError."""
    p = Path(path)
    name = pin["name"]
    if not p.is_file():
        raise ValueError(name + ": missing at " + str(p))
    size = p.stat().st_size
    if size != pin["size"]:
        raise ValueError(name + ": size " + str(size) + " != pinned " + str(pin["size"]) + " (" + str(p) + ")")
    got = sha256_of(p)
    if got != pin["sha256"]:
        raise ValueError(name + ": sha256 " + got + " != pinned " + pin["sha256"] + " (" + str(p) + ")")
