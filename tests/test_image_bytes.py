"""deploy/image_bytes.py (#636): the compressed size of the image the smoke pulls, from the manifest.

The probe (INSPECT) is a stub that answers from a table of raw manifests. No network, no docker."""
import json
import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "image_bytes.py"
REPO = "ghcr.io/skyphusion-labs/vivijure-backend"
TOP = REPO + "@sha256:" + "a" * 64
AMD = "sha256:" + "b" * 64
ARM = "sha256:" + "c" * 64

STUB = """#!/usr/bin/env python3
import json, os, sys
table = json.load(open(os.environ["MANIFESTS"]))
ref = sys.argv[-1]
if ref not in table:
    sys.stderr.write("not found\\n"); sys.exit(1)
print(json.dumps(table[ref]))
"""


def run(tmp_path, manifests, img=TOP):
    stub = tmp_path / "inspect"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "m.json").write_text(json.dumps(manifests))
    env = {"PATH": os.environ["PATH"], "INSPECT": str(stub), "MANIFESTS": str(tmp_path / "m.json"), "IMG": img}
    return subprocess.run(["python3", "-I", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)


def image(*sizes, config=100):
    return {"mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"size": config}, "layers": [{"size": s} for s in sizes]}


def test_a_single_manifest_sums_config_and_layers(tmp_path):
    r = run(tmp_path, {TOP: image(1000, 2000, 30_000_000_000)})
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == 100 + 1000 + 2000 + 30_000_000_000


def test_an_index_picks_linux_amd64(tmp_path):
    index = {"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": [
        {"digest": ARM, "platform": {"os": "linux", "architecture": "arm64"}},
        {"digest": AMD, "platform": {"os": "linux", "architecture": "amd64"}},
    ]}
    r = run(tmp_path, {TOP: index, f"{REPO}@{AMD}": image(5, config=1), f"{REPO}@{ARM}": image(999)})
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) == 6


def test_an_index_without_amd64_fails(tmp_path):
    index = {"mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [{"digest": ARM, "platform": {"os": "linux", "architecture": "arm64"}}]}
    r = run(tmp_path, {TOP: index})
    assert r.returncode == 1 and "no linux/amd64" in r.stderr


def test_a_manifest_with_no_layers_is_never_size_zero(tmp_path):
    r = run(tmp_path, {TOP: {"mediaType": "application/vnd.oci.image.manifest.v1+json", "config": {"size": 1}, "layers": []}})
    assert r.returncode == 1 and "no layers" in r.stderr


def test_a_tag_without_a_digest_is_refused(tmp_path):
    r = run(tmp_path, {}, img=REPO + ":1.1.19")
    assert r.returncode == 1 and "digest ref" in r.stderr
