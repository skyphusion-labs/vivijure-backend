"""deploy/stills-base-resolve.sh (#564, fleet-chezmoi#2620): the inputs of stills-base.yml are checked,
fail closed, before anything is staged, built or pushed. The workflow pushes only a candidate tag.

The registry probe (INSPECT) is a stub that reads a table: each ref either exists, is "not found", or
fails with another error (the registry cannot tell). No network, no docker."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "stills-base-resolve.sh"
IMAGE = "ghcr.io/skyphusion-labs/vivijure-backend"
TOOL = IMAGE + ":toolchain-t1@sha256:" + "456b417b" * 8

STUB = """#!/bin/sh
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


def test_a_free_candidate_tag_resolves(tmp_path):
    r, o, probed = run(tmp_path, {TOOL: "exists"}, TAG="stills-t1-m1-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 0, r.stdout + r.stderr
    assert o == {"tag": "stills-t1-m1-rc1", "toolchain_ref": TOOL, "model_version": "1"}
    assert probed == [TOOL, IMAGE + ":stills-t1-m1-rc1"]


@pytest.mark.parametrize("tag", [
    "stills-t1-m1",          # the LIVE base under 1.1.19: never written here
    "stills-t1-m2",          # any final-shaped tag
    "1.1.19",                # the live worker version
    "stills-t1-m1-rc",       # no candidate number
    "stills-t1-m1-rc1 ",     # trailing space
    "",
])
def test_a_tag_that_is_not_a_candidate_is_refused_before_any_probe(tmp_path, tag):
    r, o, probed = run(tmp_path, {TOOL: "exists"}, TAG=tag, MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "tag must be a candidate" in r.stdout
    assert o == {}
    assert probed == []


def test_an_existing_candidate_tag_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {TOOL: "exists", IMAGE + ":stills-t1-m1-rc1": "exists"},
                  TAG="stills-t1-m1-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "already exists on GHCR" in r.stdout
    assert o == {}


def test_a_registry_that_cannot_tell_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {TOOL: "exists", IMAGE + ":stills-t1-m1-rc1": "error"},
                  TAG="stills-t1-m1-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "cannot tell" in r.stdout
    assert o == {}


def test_an_unreadable_toolchain_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {}, TAG="stills-t1-m1-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "cannot be read on GHCR" in r.stdout
    assert o == {}


@pytest.mark.parametrize("ref", [
    IMAGE + ":toolchain-t1",                                   # a tag, not a digest
    IMAGE + ":stills-t1-m1@sha256:" + "6d0e19bd" * 8,          # not a toolchain
    "ghcr.io/someone-else/vivijure-backend:toolchain-t1@sha256:" + "a" * 64,
])
def test_a_toolchain_ref_that_is_not_a_toolchain_digest_is_refused(tmp_path, ref):
    r, o, probed = run(tmp_path, {ref: "exists"}, TAG="stills-t1-m1-rc1", MODEL_VERSION="1", REF=ref)
    assert r.returncode == 1
    assert "toolchain_ref must be a full digest" in r.stdout
    assert probed == []


def test_a_tag_that_names_another_toolchain_is_refused(tmp_path):
    r, o, probed = run(tmp_path, {TOOL: "exists"}, TAG="stills-t2-m1-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "names toolchain t2" in r.stdout
    assert probed == []


def test_a_tag_that_names_another_model_version_is_refused(tmp_path):
    r, o, probed = run(tmp_path, {TOOL: "exists"}, TAG="stills-t1-m2-rc1", MODEL_VERSION="1", REF=TOOL)
    assert r.returncode == 1
    assert "model version m2" in r.stdout
    assert probed == []
