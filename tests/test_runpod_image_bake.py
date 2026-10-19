"""The bake planner never calls RunPod. These tests check shapes, not builds.

Every test here guards something that was measured wrong on 2026-10-19:
a /v2 base that redirects to the docs site, a urllib User-Agent the RunPod
edge bans, a request body in field names the API does not accept, and a
computeType default of GPU that turns an incomplete body into a $2.49 per
hour pod instead of an error.
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "runpod_image_bake", ROOT / "scripts" / "runpod_image_bake.py"
)
bake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bake)

DIGEST = "sha256:" + ("ab" * 32)


def _no_calls(*a, **kw):
    raise AssertionError("the network was called when it should not have been")


def _args(**overrides):
    command = overrides.pop("command", "plan")
    parser = bake.build_parser()
    base = [
        command,
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
        base.extend(["--" + key.replace("_", "-"), value])
    return parser.parse_args(base)


# ----------------------------------------------------------------- the endpoint

def test_the_api_base_is_v1():
    """/v2 is not an API path. It 301s to docs.runpod.io, so every call failed."""
    assert bake.API == "https://rest.runpod.io/v1"


def test_a_redirect_is_never_followed():
    handler = bake._NoRedirect()
    assert handler.redirect_request(None, None, 301, "", {}, "https://docs.example") is None


def test_every_request_names_the_client():
    """urllib's default UA gets a Cloudflare 1010 from the RunPod edge."""
    seen = {}

    def fake(method, url, headers, body=None):
        seen.update(headers)
        return 200, "{}", {}

    original = bake._http
    bake._http = fake
    try:
        bake._request("GET", "/pods", "tok")
    finally:
        bake._http = original
    assert seen["User-Agent"] == bake.USER_AGENT
    assert "Python-urllib" not in seen["User-Agent"]


# --------------------------------------------------------------- the pod body

def test_body_is_a_cpu_pod_in_live_field_names():
    body = bake.create_body(_args())
    assert body["computeType"] == "CPU"
    assert body["cloudType"] == "SECURE"
    assert body["cpuFlavorIds"] == [bake.CPU_ID]
    assert body["vcpuCount"] == bake.VCPU_COUNT
    assert body["imageName"] == bake.BUILDER_IMAGE
    assert body["containerDiskInGb"] == 40
    assert body["ports"] == []
    assert body["dockerEntrypoint"] == ["/busybox/sh", "-c"]
    assert "BAKE_OK digest=" in body["dockerStartCmd"][0]
    assert "not-a-real-token" not in body["dockerStartCmd"][0]
    assert "$GHCR_TOKEN" in body["dockerStartCmd"][0]


def test_body_uses_no_field_the_api_rejects():
    """These five keys were in the first version. The API refuses all of them."""
    body = bake.create_body(_args())
    for dead in ("disk", "args", "cloud", "cpu", "image"):
        assert dead not in body, "%r is not in the v1 input schema" % dead


def test_body_never_lets_the_gpu_default_through():
    """computeType defaults to GPU, so omitting it rents a $2.49 per hour pod."""
    for body in (bake.create_body(_args()), bake.selftest_body("n")):
        assert body.get("computeType") == "CPU"
        assert "gpuTypeIds" not in body
        assert "gpuCount" not in body


def test_no_volume_is_attached_so_terminate_ends_the_bill():
    """volumeInGb defaults to 20, which is a billed pod volume."""
    for body in (bake.create_body(_args()), bake.selftest_body("n")):
        assert body["volumeInGb"] == 0
        assert "networkVolumeId" not in body


def test_selftest_body_is_the_cheapest_pod_and_carries_no_registry_token():
    body = bake.selftest_body("bake-selftest")
    assert body["imageName"] == bake.SELFTEST_IMAGE
    assert body["vcpuCount"] == 2
    assert body["containerDiskInGb"] == bake.DISK_MIN
    assert body["env"] == {}


def test_plan_view_hides_the_registry_token():
    shown = json.dumps(bake.public_view(bake.create_body(_args())))
    assert "not-a-real-token" not in shown
    assert '"GHCR_TOKEN": "SET"' in shown


def test_scrub_removes_a_secret_from_an_error_body():
    body = 'bad key "supersecretvalue" rejected'
    assert "supersecretvalue" not in bake.scrub(body, "supersecretvalue")
    # A short string is left alone: scrubbing "x" would redact half the message.
    assert bake.scrub("a b c", "b") == "a b c"


def test_disk_cap_rejects_the_old_fat_image_size():
    try:
        bake.disk_gb("400")
    except Exception as exc:  # argparse.ArgumentTypeError
        assert "160" in str(exc)
    else:
        raise AssertionError("400 GB disk was accepted")


# ------------------------------------------------------------ the digest channel

def test_split_ref_reads_a_ghcr_reference():
    assert bake.split_ref("ghcr.io/skyphusion-labs/x:runtime-t1") == (
        "skyphusion-labs/x", "runtime-t1")
    assert bake.split_ref("ghcr.io/a/b/c:1-bf16-t2-99-1") == ("a/b/c", "1-bf16-t2-99-1")


def test_split_ref_refuses_everything_that_is_not_one():
    """The repo half goes straight into a GHCR URL path, so the grammar is the guard."""
    bad = (
        "ghcr.io/owner/name",                 # no tag
        "notaref",                            # no host
        "ghcr.io/o/n:a/b",                    # slash in the tag
        "evil.com/o/n:t",                     # foreign host
        "https://ghcr.io/o/n:t",              # host not at the start
        "ghcr.io.evil.com/o/n:t",             # host is a prefix of another host
        "x-ghcr.io/o/n:t",                    # host preceded by something
        "ghcr.io/o/../../n:t",                # dot segments
        "ghcr.io/o/n%2f..:t",                 # percent encoding
        "ghcr.io/o/n:t?x=1",                  # query
        "ghcr.io/o/n:t#f",                    # fragment
        "ghcr.io/O/N:t",                      # uppercase repository
        "ghcr.io/o:t",                        # no owner/name pair
        "ghcr.io/o/n:" + "t" * 200,           # tag beyond the length cap
        "",
        None,
    )
    for ref in bad:
        try:
            bake.split_ref(ref)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (ref,))


def test_valid_digest_is_strict():
    assert bake.valid_digest(DIGEST)
    for bad in ("sha256:zz", DIGEST + "a", "md5:" + "a" * 64, "", DIGEST.upper()):
        assert not bake.valid_digest(bad)


def test_ghcr_digest_reads_the_header_and_rejects_a_bad_one(monkeypatch):
    def fake(method, url, headers, body=None):
        if "/token" in url:
            return 200, json.dumps({"token": "t"}), {}
        return 200, "", {"Docker-Content-Digest": DIGEST}

    monkeypatch.setattr(bake, "_http", fake)
    assert bake.ghcr_digest("ghcr.io/o/n:tag", "") == DIGEST

    def junk(method, url, headers, body=None):
        if "/token" in url:
            return 200, json.dumps({"token": "t"}), {}
        return 200, "", {"Docker-Content-Digest": "sha256:nope"}

    monkeypatch.setattr(bake, "_http", junk)
    assert bake.ghcr_digest("ghcr.io/o/n:tag", "") is None


def test_ghcr_digest_is_none_when_the_tag_is_absent(monkeypatch):
    def fake(method, url, headers, body=None):
        if "/token" in url:
            return 200, json.dumps({"token": "t"}), {}
        return 404, "", {}

    monkeypatch.setattr(bake, "_http", fake)
    assert bake.ghcr_digest("ghcr.io/o/n:tag", "") is None


# ----------------------------------------------------------------- the refusals

def test_run_refuses_before_any_request_when_the_key_is_unset(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(bake, "_http", _no_calls)
    assert bake.run(_args(command="run")) == 1


def test_run_refuses_when_the_immutable_tag_already_exists(monkeypatch):
    """A tag that is already there cannot prove THIS run pushed anything."""
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(bake, "ghcr_digest", lambda ref, token: DIGEST)
    monkeypatch.setattr(bake, "_request", _no_calls)
    assert bake.run(_args(command="run")) == 1


def test_reap_refuses_before_any_request_when_the_key_is_unset(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr(bake, "_http", _no_calls)
    args = bake.build_parser().parse_args(["reap", "--pod-id", "abc123"])
    assert bake.reap(args) == 1


def test_a_build_command_still_demands_its_request_fields():
    for argv in (
        ["plan", "--name", "n"],
        ["run", "--name", "n", "--repo", "r/r", "--sha", "s", "--dockerfile", "d",
         "--destination", "x", "--destination-immutable", "y"],
    ):
        try:
            bake.main(argv)
        except SystemExit as exc:
            assert exc.code != 0
        else:
            raise AssertionError("accepted an incomplete build request: %r" % argv)


# ---------------------------------------------------------------- the terminate

def test_terminate_does_not_believe_the_delete_response(monkeypatch):
    """A 200 on DELETE while the pod still reads RUNNING is not a termination."""
    calls = []

    def fake(method, path, token, body=None):
        calls.append((method, path))
        if method == "DELETE":
            return 200, "{}"
        return 200, json.dumps({"desiredStatus": "RUNNING"})

    monkeypatch.setattr(bake, "_request", fake)
    monkeypatch.setattr(bake.time, "sleep", lambda _s: None)
    accepted, gone = bake.terminate("pod-1", "t")
    assert (accepted, gone) == (True, False)
    assert calls[0][0] == "DELETE"
    assert ("GET", "/pods/pod-1") in calls


def test_terminate_confirms_on_a_404(monkeypatch):
    def fake(method, path, token, body=None):
        return (204, "") if method == "DELETE" else (404, "")

    monkeypatch.setattr(bake, "_request", fake)
    assert bake.terminate("pod-1", "t") == (True, True)


def test_terminate_confirms_on_a_terminated_status(monkeypatch):
    def fake(method, path, token, body=None):
        if method == "DELETE":
            return 200, "{}"
        return 200, json.dumps({"desiredStatus": "TERMINATED"})

    monkeypatch.setattr(bake, "_request", fake)
    assert bake.terminate("pod-1", "t") == (True, True)


def test_a_refused_delete_is_reported_as_not_accepted(monkeypatch):
    def fake(method, path, token, body=None):
        if method == "DELETE":
            return 500, "boom"
        return 200, json.dumps({"desiredStatus": "RUNNING"})

    monkeypatch.setattr(bake, "_request", fake)
    monkeypatch.setattr(bake.time, "sleep", lambda _s: None)
    accepted, _gone = bake.terminate("pod-1", "t")
    assert accepted is False
