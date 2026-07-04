"""Guards for the config-bake tree-cache scrub (#206). bake_hf_configs.py defers its huggingface_hub
import into bake_configs(), so scrub_tree_cache is importable + testable stdlib-only. bake_hf_configs
lives in deploy/ (off the src/ pythonpath), imported by path like bake_layers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import bake_hf_configs  # noqa: E402


def _tree(hub: Path, repo: str, commit: str) -> Path:
    p = hub / repo / "trees" / f"{commit}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"format_version": 1, "files": {"a.safetensors": {"size": 1}}}')
    return p


def test_scrub_removes_all_tree_listings(tmp_path):
    """Every <repo>/trees/*.json the online snapshot_download wrote must be removed, and the now-empty
    trees/ dir dropped, so the baked layer carries no tree cache (the #206 fix)."""
    hub = tmp_path / "hub"
    _tree(hub, "models--SG161222--RealVisXL_V5.0", "ac93e0dd")
    _tree(hub, "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers", "596658fd")
    # a snapshot/config that MUST be preserved (the scrub is surgical to trees/)
    keep = hub / "models--SG161222--RealVisXL_V5.0" / "snapshots" / "ac93e0dd" / "model_index.json"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("{}")

    removed = bake_hf_configs.scrub_tree_cache(hub)
    assert removed == 2
    assert not list(hub.rglob("trees/*.json"))
    assert not list(hub.rglob("*/trees"))  # empty dirs dropped
    assert keep.is_file()  # non-tree content untouched


def test_scrub_idempotent_and_empty_safe(tmp_path):
    """No trees/ present -> zero removed, no error (idempotent re-run / clean cache)."""
    hub = tmp_path / "hub"
    (hub / "models--x--y").mkdir(parents=True)
    assert bake_hf_configs.scrub_tree_cache(hub) == 0
