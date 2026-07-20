#!/usr/bin/env python3
"""Bake-time: materialize stable local dirs for Wan train weights + patch ai-toolkit hub IDs.

Conrad ruling (cf#29 D2c): true-offline means the train image embeds every asset ai-toolkit
loads, and ai-toolkit must NOT keep HuggingFace hub IDs that trigger model_info() under
HF_HUB_OFFLINE=1. Runtime path remapping alone is insufficient.

This script runs INSIDE the train image build AFTER the HF-cache bins are COPYed:

  1. snapshot_download(local_files_only=True) for the three baked repos
  2. Symlink them to stable absolute paths under /opt/models/aitoolkit/
  3. Rewrite ai-toolkit's hardcoded UMT5 + VAE hub-id strings to those absolute paths
  4. Write a marker JSON the offline smoke asserts on

Fail loud if any repo is missing from the bake (LocalEntryNotFoundError).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AITOOLKIT_DIR = Path(os.environ.get("VIVIJURE_AITOOLKIT_DIR", "/opt/ai-toolkit"))
STABLE_ROOT = Path(os.environ.get("VIVIJURE_AITOOLKIT_WEIGHTS", "/opt/models/aitoolkit"))

REPOS = {
    "wan-base": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
    "umt5": "ai-toolkit/umt5_xxl_encoder",
    "vae": "ai-toolkit/wan2.1-vae",
}

# (relative path under ai-toolkit checkout, hub id as it appears in source, stable key)
HUB_ID_PATCHES = (
    ("toolkit/models/wan21/wan21.py", "ai-toolkit/umt5_xxl_encoder", "umt5"),
    ("extensions_built_in/diffusion_models/wan22/wan22_14b_model.py", "ai-toolkit/wan2.1-vae", "vae"),
)


def _symlink_stable(key: str, snapshot: Path) -> Path:
    STABLE_ROOT.mkdir(parents=True, exist_ok=True)
    dest = STABLE_ROOT / key
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            # refuse to rm a real directory tree; rebuilds should replace the symlink only
            raise RuntimeError(f"stable path {dest} exists and is not a symlink; refuse to clobber")
    dest.symlink_to(snapshot.resolve(), target_is_directory=True)
    return dest


def materialize() -> dict[str, str]:
    from huggingface_hub import snapshot_download  # local: bake image has it; CPU tests stub it

    resolved: dict[str, str] = {}
    for key, repo in REPOS.items():
        snap = Path(snapshot_download(repo, local_files_only=True))
        if not snap.is_dir():
            raise FileNotFoundError(f"snapshot for {repo} is not a directory: {snap}")
        dest = _symlink_stable(key, snap)
        resolved[key] = str(dest)
        print(f"stable {key}: {repo} -> {dest} (-> {snap})")
    return resolved


def patch_aitoolkit(resolved: dict[str, str]) -> None:
    for rel, hub_id, key in HUB_ID_PATCHES:
        path = AITOOLKIT_DIR / rel
        if not path.is_file():
            raise FileNotFoundError(f"ai-toolkit file missing for bake patch: {path}")
        text = path.read_text(encoding="utf-8")
        needle = f'"{hub_id}"'
        if needle not in text:
            if resolved[key] in text:
                print(f"already patched: {rel}")
                continue
            raise RuntimeError(
                f"bake patch: {rel} has neither hub id {hub_id!r} nor stable path "
                f"{resolved[key]!r}; ai-toolkit pin may have drifted"
            )
        path.write_text(text.replace(needle, repr(resolved[key])), encoding="utf-8")
        print(f"patched {rel}: {hub_id} -> {resolved[key]}")


def write_marker(resolved: dict[str, str]) -> None:
    marker = STABLE_ROOT / "BAKE_OFFLINE.json"
    marker.write_text(
        json.dumps(
            {
                "repos": REPOS,
                "stable": resolved,
                "aitoolkit_dir": str(AITOOLKIT_DIR),
                "rule": "bake-it-dont-just-remap",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {marker}")


def assert_no_hub_ids_remain() -> None:
    for rel, hub_id, _key in HUB_ID_PATCHES:
        text = (AITOOLKIT_DIR / rel).read_text(encoding="utf-8")
        if f'"{hub_id}"' in text:
            raise RuntimeError(f"hub id {hub_id!r} still present in {rel} after bake patch")


def main() -> int:
    if not AITOOLKIT_DIR.is_dir():
        print(f"error: ai-toolkit checkout missing at {AITOOLKIT_DIR}", file=sys.stderr)
        return 2
    resolved = materialize()
    patch_aitoolkit(resolved)
    assert_no_hub_ids_remain()
    write_marker(resolved)
    print("bake_aitoolkit_offline_paths: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
