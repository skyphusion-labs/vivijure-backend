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
    assert o == {"version": "1.1.19", "mode": "prebuilt", "toolchain_ref": BASE, "smoke_ref": "", "smoke_base": BASE}
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
    assert o == {"version": "1.1.19", "mode": "build", "toolchain_ref": IMAGE + ":toolchain-t2", "smoke_ref": "", "smoke_base": ""}


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


# --- #636: smoke only, on an existing digest, nothing built or pushed ---

WORKER = IMAGE + ":1.1.19@sha256:" + "e" * 64


def test_smoke_only_on_an_existing_worker_digest(tmp_path):
    r, o, probed = run(tmp_path, {WORKER: "exists", IMAGE + ":1.1.19": "exists"}, VER="1.1.19", SMOKE=WORKER)
    assert r.returncode == 0, r.stdout + r.stderr
    assert o == {"mode": "smoke", "smoke_ref": WORKER, "smoke_base": ""}
    # The version tag existing is fine here: nothing is pushed, so nothing is overwritten.
    assert probed == [WORKER]


def test_smoke_only_refuses_another_image(tmp_path):
    other = "ghcr.io/skyphusion-labs/vivijure-cf@sha256:" + "e" * 64
    r, o, probed = run(tmp_path, {other: "exists"}, VER="1.1.19", SMOKE=other)
    assert r.returncode == 1 and "smoke_ref must be a full digest of" in r.stdout
    assert o == {} and probed == []


def test_smoke_only_refuses_a_tag_without_a_digest(tmp_path):
    r, o, probed = run(tmp_path, {}, VER="1.1.19", SMOKE=IMAGE + ":1.1.19")
    assert r.returncode == 1 and o == {} and probed == []


def test_smoke_only_refuses_a_digest_not_on_ghcr(tmp_path):
    r, o, _ = run(tmp_path, {WORKER: "missing"}, VER="1.1.19", SMOKE=WORKER)
    assert r.returncode == 1 and "cannot be read on GHCR" in r.stdout and o == {}


def test_smoke_only_with_a_toolchain_ref_is_refused(tmp_path):
    r, o, probed = run(tmp_path, {WORKER: "exists", BASE: "exists"}, VER="1.1.20", SMOKE=WORKER, REF=BASE)
    assert r.returncode == 1 and "together" in r.stdout and o == {} and probed == []


def test_the_workflow_moves_docker_to_mnt_checks_room_and_skips_the_build_in_smoke_mode():
    wf = (ROOT / ".github" / "workflows" / "thin-image.yml").read_text()
    assert "smoke_ref:" in wf and "SMOKE: ${{ inputs.smoke_ref }}" in wf
    thin = wf.split("- name: Build and push thin worker", 1)[1].split("- name:", 1)[0]
    assert "if: steps.cfg.outputs.mode != 'smoke'" in thin
    pushed = wf.split("- name: Pushed digest", 1)[1].split("- name:", 1)[0]
    assert "if: steps.cfg.outputs.mode != 'smoke'" in pushed
    room = wf.split("- name: Move Docker storage to /mnt", 1)[1].split("- name: CPU import smoke", 1)[0]
    assert '"data-root"]="/mnt/docker"' in room
    assert "deploy/smoke_image.py plan" in room
    assert "cannot hold the image" in room
    smoke = wf.split("- name: CPU import smoke", 1)[1]
    assert "${{ steps.room.outputs.img }}" in smoke
    assert "deploy/smoke_image.py stream | docker load" in smoke
    assert "docker pull" not in smoke
    # The smoke must come after the move, never before it.
    assert wf.index("- name: Move Docker storage to /mnt") < wf.index("- name: CPU import smoke")


# --- #645: smoke a worker on the stills base without its weight layers ---

def test_smoke_only_with_a_base_passes_it_on(tmp_path):
    r, o, probed = run(tmp_path, {WORKER: "exists", BASE: "exists"}, SMOKE=WORKER, SMOKE_BASE=BASE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert o == {"mode": "smoke", "smoke_ref": WORKER, "smoke_base": BASE}
    assert probed == [WORKER, BASE]


@pytest.mark.parametrize("ref", [IMAGE + ":stills-t1-m1", "docker.io/library/ubuntu@sha256:" + "a" * 64, BASE + " x"])
def test_smoke_base_must_be_a_full_digest(tmp_path, ref):
    r, o, probed = run(tmp_path, {WORKER: "exists"}, SMOKE=WORKER, SMOKE_BASE=ref)
    assert r.returncode == 1 and "smoke_base must be a full digest" in r.stdout
    assert o == {} and probed == [WORKER]


def test_smoke_base_not_on_ghcr_is_refused(tmp_path):
    r, o, _ = run(tmp_path, {WORKER: "exists"}, SMOKE=WORKER, SMOKE_BASE=BASE)
    assert r.returncode == 1 and "smoke_base" in r.stdout and "cannot be read on GHCR" in r.stdout and o == {}


def test_smoke_base_without_smoke_ref_is_refused(tmp_path):
    r, o, probed = run(tmp_path, {BASE: "exists"}, VER="1.1.21", REF=BASE, SMOKE_BASE=BASE)
    assert r.returncode == 1 and "only for a smoke_ref run" in r.stdout
    assert o == {} and probed == []


def test_the_main_only_check_runs_after_the_resolve_and_skips_only_a_smoke_run():
    wf = (ROOT / ".github" / "workflows" / "thin-image.yml").read_text()
    assert wf.index("- name: Resolve and check inputs") < wf.index("- name: Assert commit is on main")
    assert wf.index("- name: Assert commit is on main") < wf.index("- name: Build and push toolchain")
    main = wf.split("- name: Assert commit is on main", 1)[1].split("- name:", 1)[0]
    assert "if: steps.cfg.outputs.mode != 'smoke'" in main
    assert "git merge-base --is-ancestor" in main


def test_the_smoke_loads_the_planned_image_and_checks_both_pipe_statuses():
    wf = (ROOT / ".github" / "workflows" / "thin-image.yml").read_text()
    assert "smoke_base:" in wf and "SMOKE_BASE: ${{ inputs.smoke_base }}" in wf
    room = wf.split("- name: Move Docker storage to /mnt", 1)[1].split("- name: CPU import smoke", 1)[0]
    assert 'BASE="${SMOKE_BASE}"' in room
    # run 43315326937 measured the load at 2.53 times the compressed size; the guard must sit above it
    assert "need=$(( bytes * 3 ))" in room and "bytes * 5 / 2" not in room
    smoke = wf.split("- name: CPU import smoke", 1)[1].split("- name:", 1)[0]
    assert 'st=("${PIPESTATUS[@]}")' in smoke
    assert '"${st[0]}" = 0' in smoke and '"${st[1]}" = 0' in smoke
    assert ".RootFS.Layers" in smoke and 'p["diff_ids"]' in smoke
    assert 'docker run --rm "${TAG}"' in smoke
