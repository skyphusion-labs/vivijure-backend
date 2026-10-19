"""The bake planner never calls RunPod. It only checks the request shape."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runpod_image_bake", ROOT / "scripts" / "runpod_image_bake.py"
)
bake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bake)


def _args(**overrides):
    parser = bake.build_parser()
    base = [
        "plan",
        "--name", "bake-test",
        "--repo", "skyphusion-labs/example",
        "--sha", "a" * 40,
        "--dockerfile", "deploy/runtime.Dockerfile",
        "--destination", "ghcr.io/skyphusion-labs/example:runtime-t1",
        "--destination-immutable", "ghcr.io/skyphusion-labs/example:runtime-t1-1",
        "--disk-gb", "40",
        "--ghcr-user", "skyphusion",
        "--ghcr-token", "not-a-real-token",
    ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        base.extend([flag, value])
    return parser.parse_args(base)


def test_body_is_a_cpu_pod_with_no_volume():
    body = bake.create_body(_args())
    assert body["cpu"] == {"id": "cpu3c", "vcpuCount": 4}
    assert "gpu" not in body
    assert "mounts" not in body
    assert body["cloud"] == "SECURE"
    assert body["disk"] == 40
    assert body["image"] == bake.BUILDER_IMAGE
    assert body["ports"] == []
    command = json.loads(body["args"])
    assert command["entrypoint"] == ["/busybox/sh", "-c"]
    assert "BAKE_OK digest=" in command["cmd"][0]
    assert "$GHCR_TOKEN" in command["cmd"][0]
    assert "not-a-real-token" not in command["cmd"][0]


def test_plan_view_hides_the_registry_token():
    shown = json.dumps(bake.public_view(bake.create_body(_args())))
    assert "not-a-real-token" not in shown
    assert '"GHCR_TOKEN": "SET"' in shown


def test_disk_cap_rejects_the_old_fat_image_size():
    try:
        bake.disk_gb("400")
    except Exception as exc:  # argparse.ArgumentTypeError
        assert "160" in str(exc)
    else:
        raise AssertionError("400 GB disk was accepted")


def test_marker_reads_the_digest_and_a_failure():
    digest = "sha256:" + ("ab" * 32)
    assert bake.marker("noise\nBAKE_OK digest=%s\n" % digest) == digest
    assert bake.marker("BAKE_FAIL") == "FAIL"
    assert bake.marker("still pulling") is None
    assert bake.marker("BAKE_OK digest=sha256:zz") is None
