"""deploy/thin-image-resolve.sh: the inputs of thin-image.yml are checked, fail closed, before any build.

The registry probe (INSPECT) is a stub that reads a table: each ref either exists, is "not found", or
fails with another error (the registry cannot tell). No network, no docker."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "thin-image-resolve.sh"
IMAGE = "ghcr.io/skyphusion-labs/vivijure-backend"
BASE = IMAGE + ":stills-t1-m1@sha256:" + "6d0e19bd18bdba3318e510fcc0055d9d75c1a1353df918795dc92be0d70740b6"

STUB = """#!/bin/sh
# stub inspect: the last argument is the ref; the table says what the registry answers
ref=$(for a in "$@"; do :; done; echo "$a")
echo "$ref" >> "$PROBE_LOG"
line=$(grep -F "$ref " "$PROBE_TABLE" | head -1)
case "${line##* }" in
  exists) echo "Name: $ref"; exit 0 ;;
  error) echo "ERROR: failed to authorize: 401 Unauthorized" >&2; exit 1 ;;
  *) echo "ERROR: $ref: not found" >&2; exit 1 ;;
esac
"""


def run(tmp_path, table, **env):
    stub = tmp_path / "inspect"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "table").write_text("".join(f"{ref} {answer}\n" for ref, answer in table.items()))
    log = tmp_path / "probe.log"
    log.write_text("")
    out = tmp_path / "out"
    out.write_text("")
    e = {"PATH": os.environ["PATH"], "IMAGE": IMAGE, "INSPECT": str(stub),
         "PROBE_TABLE": str(tmp_path / "table"), "PROBE_LOG": str(log), "GITHUB_OUTPUT": str(out)}
    e.update(env)
    r = subprocess.run(["bash", str(SCRIPT)], env=e, capture_output=True, text=True, timeout=30)
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines() if "=" in line)
    return r, outputs, log.read_text().splitlines()


def test_a_prebuilt_digest_builds_no_toolchain(tmp_path):
    r, o, probed = run(tmp_path, {BASE: "exists"}, VER="1.1.19", REF=BASE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert o == {"version": "1.1.19", "mode": "prebuilt", "toolchain_ref": BASE}
    assert probed == [BASE, IMAGE + ":1.1.19"]


def test_an_existing_version_tag_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {BASE: "exists", IMAGE + ":1.1.18": "exists"}, VER="1.1.18", REF=BASE)
    assert r.returncode == 1
    assert "already exists on GHCR" in r.stdout
    assert o == {}


def test_a_registry_that_cannot_tell_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {BASE: "exists", IMAGE + ":1.1.19": "error"}, VER="1.1.19", REF=BASE)
    assert r.returncode == 1
    assert "cannot tell" in r.stdout
    assert o == {}


@pytest.mark.parametrize("ref", [
    IMAGE + ":stills-t1-m1",                                    # a tag, not a digest
    IMAGE + "@sha256:" + "6d0e19bd" * 4,                        # a short digest
    "docker.io/library/ubuntu@sha256:" + "a" * 64,              # another registry
    "ghcr.io/someone-else/x@sha256:" + "a" * 64,                # another owner
    IMAGE + "@sha256:" + "A" * 64,                              # upper-case hex
    BASE + " --push",                                           # trailing words
])
def test_a_bad_ref_shape_is_refused_before_any_probe(tmp_path, ref):
    r, o, probed = run(tmp_path, {}, VER="1.1.19", REF=ref)
    assert r.returncode == 1
    assert "must be a full digest" in r.stdout
    assert probed == [] and o == {}


def test_a_ref_that_is_not_on_ghcr_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {}, VER="1.1.19", REF=BASE)
    assert r.returncode == 1
    assert "cannot be read on GHCR" in r.stdout
    assert o == {}


@pytest.mark.parametrize("ver", ["1.1", "v1.1.19", "1.1.19-rc1", ""])
def test_a_bad_version_is_refused(tmp_path, ver):
    r, o, probed = run(tmp_path, {BASE: "exists"}, VER=ver, REF=BASE)
    assert r.returncode == 1 and "version must be X.Y.Z" in r.stdout
    assert probed == [] and o == {}


def test_without_a_ref_the_toolchain_tag_path_is_kept(tmp_path):
    r, o, _ = run(tmp_path, {}, VER="1.1.19", TOOL="t2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert o == {"version": "1.1.19", "mode": "build", "toolchain_ref": IMAGE + ":toolchain-t2"}


def test_without_a_ref_a_bad_toolchain_tag_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {}, VER="1.1.19", TOOL="latest")
    assert r.returncode == 1 and "toolchain_tag must be t<N>" in r.stdout
    assert o == {}


def test_the_workflow_uses_the_resolver_and_skips_the_toolchain_with_a_ref():
    wf = (ROOT / ".github" / "workflows" / "thin-image.yml").read_text()
    assert "deploy/thin-image-resolve.sh" in wf
    assert "toolchain_ref:" in wf
    assert "if: steps.cfg.outputs.mode == 'build'" in wf
    assert "TOOLCHAIN_REF=${{ steps.cfg.outputs.toolchain_ref }}" in wf
    assert "steps.thin.outputs.digest" in wf
