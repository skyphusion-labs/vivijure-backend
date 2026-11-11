"""fleet-chezmoi#2085: RunPod REST v1 (its own host) retired 2026-11-15. No shipped script or module in this
repo may name it, so a new call cannot slip back in. Tests that assert its absence are not scanned."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_HOST = "rest." + "runpod.io"  # split so this file does not match itself
TREES = ["src", "deploy", "scripts"]
FILES = ["deploy.sh"]


def _scanned():
    out = [ROOT / f for f in FILES]
    for t in TREES:
        out += [p for p in (ROOT / t).rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".mjs", ".js", ".ts")]
    return out


def test_control_the_scan_reads_the_files_that_used_v1():
    names = {p.relative_to(ROOT).as_posix() for p in _scanned()}
    assert {"deploy.sh", "deploy/runpod_verify.py", "scripts/runpod_image_bake.py"} <= names
    assert V1_HOST in ("https://" + V1_HOST + "/v1")


def test_no_shipped_file_names_the_v1_host():
    hits = [p.relative_to(ROOT).as_posix() for p in _scanned() if V1_HOST in p.read_text(errors="replace")]
    assert hits == []
