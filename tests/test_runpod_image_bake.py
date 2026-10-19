"""The bake planner never calls RunPod. These tests check shapes, not builds.

Every test here guards something that was measured wrong on 2026-10-19:
a /v2 base that redirects to the docs site, a urllib User-Agent the RunPod
edge bans, a request body in field names the API does not accept, and a
computeType default of GPU that turns an incomplete body into a $2.49 per
hour pod instead of an error.
"""

import importlib.util
import json
import urllib.error
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


# --------------------------------------------- the two pod-leak windows (strummer)

def _run_args(tmp_path, **over):
    a = _args(command="run")
    a.pod_id_out = str(tmp_path / "pod-id.txt")
    a.pod_name_out = str(tmp_path / "pod-name.txt")
    a.timeout_minutes = 0
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_the_pod_name_is_on_disk_before_the_create_is_attempted(tmp_path, monkeypatch):
    """Ordering is the whole fix: the handle must exist before the thing it backs up."""
    seen = {}

    def fake_create(body, token, secrets):
        seen["name_file_existed"] = (tmp_path / "pod-name.txt").exists()
        seen["contents"] = (tmp_path / "pod-name.txt").read_text().strip()
        return None

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(bake, "ghcr_digest", lambda ref, token: None)
    monkeypatch.setattr(bake, "create_pod", fake_create)
    monkeypatch.setattr(bake, "reap_by_name", lambda name, token: True)
    bake.run(_run_args(tmp_path))
    assert seen["name_file_existed"] is True
    assert seen["contents"] == "bake-test"


def test_a_create_that_raises_still_reaps_by_name(tmp_path, monkeypatch):
    """Window B: _http catches only HTTPError, so a URLError leaves a pod we cannot name."""
    reaped = []

    def boom(body, token, secrets):
        raise urllib.error.URLError("connection reset")

    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(bake, "ghcr_digest", lambda ref, token: None)
    monkeypatch.setattr(bake, "create_pod", boom)
    monkeypatch.setattr(bake, "reap_by_name", lambda name, token: reaped.append(name) or True)
    try:
        bake.run(_run_args(tmp_path))
    except urllib.error.URLError:
        pass
    else:
        raise AssertionError("the URLError should still propagate after the reap")
    assert reaped == ["bake-test"], "a created pod with an unknown id was not reaped"


def test_a_failed_pod_id_write_still_terminates(tmp_path, monkeypatch):
    """Window A: the id write used to sit outside the try, so an OSError leaked the pod."""
    killed = []
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(bake, "ghcr_digest", lambda ref, token: None)
    monkeypatch.setattr(bake, "create_pod", lambda b, t, s: "pod-xyz")
    monkeypatch.setattr(bake, "terminate", lambda pid, tok: killed.append(pid) or (True, True))
    args = _run_args(tmp_path, pod_id_out=str(tmp_path / "no-such-dir" / "pod-id.txt"))
    try:
        bake.run(args)
    except OSError:
        pass
    assert killed == ["pod-xyz"], "the pod was not terminated when the id write failed"


# ------------------------------------------- red when it WAS pushed (strummer)

def test_ghcr_digest_final_treats_404_as_a_real_answer(monkeypatch):
    calls = []
    monkeypatch.setattr(bake, "_ghcr_head", lambda ref, t: calls.append(1) or (404, None))
    assert bake.ghcr_digest_final("ghcr.io/o/n:t", "") == (None, True)
    assert len(calls) == 1, "a 404 is conclusive and must not be retried"


def test_ghcr_digest_final_retries_a_transient_then_succeeds(monkeypatch):
    seq = [(429, None), (503, None), (200, DIGEST)]
    monkeypatch.setattr(bake, "_ghcr_head", lambda ref, t: seq.pop(0))
    monkeypatch.setattr(bake.time, "sleep", lambda _s: None)
    assert bake.ghcr_digest_final("ghcr.io/o/n:t", "") == (DIGEST, True)


def test_ghcr_digest_final_says_it_could_not_tell(monkeypatch):
    """Not answered is not the same as absent, and only one of them may report a failure."""
    monkeypatch.setattr(bake, "_ghcr_head", lambda ref, t: (429, None))
    monkeypatch.setattr(bake.time, "sleep", lambda _s: None)
    assert bake.ghcr_digest_final("ghcr.io/o/n:t", "", attempts=3) == (None, False)


# ------------------------------------------------------- reaping by name

def test_pods_named_parses_a_bare_array(monkeypatch):
    """GET /pods returns a bare array, not an object with a pods key."""
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (
        200, json.dumps([{"id": "a1", "name": "bake-1"},
                         {"id": "b2", "name": "bake-1"},
                         {"noid": 1}])))
    assert bake.pods_named("bake-1", "t") == ["a1", "b2"]


def test_pods_named_checks_each_row_name_itself(monkeypatch):
    """This feeds the only destructive call, so the server's filter is not trusted.

    A 200 with an empty array cannot distinguish exact matching from the parameter
    being ignored while the account holds no other pod. Each of these is a server
    behaviour consistent with what has actually been observed live.
    """
    rows = [
        {"id": "mine", "name": "bake-99"},
        {"id": "longer-run-id", "name": "bake-991"},        # prefix match
        {"id": "contains", "name": "x-bake-99-y"},          # substring match
        {"id": "someone-elses", "name": "nightly-train"},   # parameter ignored
        {"id": "no-name-field"},
    ]
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, json.dumps(rows)))
    assert bake.pods_named("bake-99", "t") == ["mine"]


def test_pods_named_refuses_an_empty_name(monkeypatch):
    """name is required:false upstream, so an empty value would list the account.

    main() rejects reap with neither handle, so this is the layer underneath it and
    is unreachable today. The test exists so a refactor of main() cannot quietly
    remove the floor without anything going red.
    """
    monkeypatch.setattr(bake, "_request", _no_calls)
    assert bake.pods_named("", "t") == []


def test_pods_named_is_empty_on_a_non_200(monkeypatch):
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (500, "boom"))
    assert bake.pods_named("bake-1", "t") == []


def test_pods_named_url_encodes_the_name(monkeypatch):
    seen = {}

    def fake(method, path, token, body=None):
        seen["path"] = path
        return 200, "[]"

    monkeypatch.setattr(bake, "_request", fake)
    bake.pods_named("bake 1&x=2", "t")
    assert " " not in seen["path"] and "&" not in seen["path"].split("name=")[1]


def test_reap_takes_a_name_when_there_is_no_id(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(bake, "reap_by_name", lambda name, token: name == "bake-7")
    args = bake.build_parser().parse_args(["reap", "--name", "bake-7"])
    assert bake.reap(args) == 0


def test_reap_needs_one_of_the_two_handles():
    try:
        bake.main(["reap"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("reap ran with neither an id nor a name")


# ------------------------------- the two-state collapse behind the reap (strummer)

def test_a_scalar_pods_body_does_not_raise(monkeypatch):
    """A bare null, number or boolean is an ordinary empty REST reply.

    Iterating one raises TypeError inside run()'s finally, which REPLACES the
    exception that sent us there, so a create timeout gets reported as a complaint
    about iteration.
    """
    for body in ("null", "5", "true", '"a string"'):
        monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None, _b=body: (200, _b))
        assert bake.pods_named("bake-1", "t") == []
        found = bake.pods_named_detail("bake-1", "t")
        assert found.ids == []
        # A body we cannot read establishes nothing, so it is not an answer.
        assert found.answered is False


def test_an_empty_account_and_an_all_rejected_listing_are_different(monkeypatch):
    """Same shape, different facts, and only one of them can hide a billing pod."""
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, "[]"))
    empty = bake.pods_named_detail("bake-1", "t")
    assert (empty.ids, empty.mismatched, empty.answered) == ([], 0, True)
    assert bake.reap_by_name("bake-1", "t") is True

    rows = json.dumps([{"id": "a", "name": "bake-1-x"}, {"id": "b", "name": "other"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    all_rejected = bake.pods_named_detail("bake-1", "t")
    assert (all_rejected.ids, all_rejected.mismatched, all_rejected.answered) == ([], 2, True)
    assert bake.reap_by_name("bake-1", "t") is False, (
        "rows returned and every one rejected must not report success")


def test_all_rejected_is_annotated_as_the_measurement_it_is(monkeypatch, capsys):
    """Rows back and all rejected can only happen if the server filter is not exact.

    That is the question a live read could not settle without renting a second pod,
    so the first dispatch that trips this is the experiment. It must not be silent.
    """
    rows = json.dumps([{"id": "a", "name": "bake-1-decorated"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    bake.reap_by_name("bake-1", "t")
    out = capsys.readouterr()
    assert "::warning::" in out.out
    assert "not exact" in out.out


def test_a_match_alongside_rejected_rows_still_reaps_the_match(monkeypatch):
    """Rejecting the wrong pods must never stop the right one being reaped."""
    killed = []
    rows = json.dumps([{"id": "wrong", "name": "bake-1-x"}, {"id": "mine", "name": "bake-1"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    monkeypatch.setattr(bake, "terminate", lambda pid, tok: killed.append(pid) or (True, True))
    assert bake.reap_by_name("bake-1", "t") is True
    assert killed == ["mine"]


def test_each_rejection_reason_is_counted_separately(monkeypatch):
    """One counter made four facts share one sentence, and one of them was false.

    A row whose name MATCHES but carries no id is not a filter problem: the pod we
    asked for came back and we cannot address it. It gets its own field and its own
    message, because it leads to a different action than the others.
    """
    rows = json.dumps([
        "bare string",                      # unreadable row
        None,                               # unreadable row
        {"id": "x"},                        # an id but no name: assert no cause
        {"id": "y", "name": 7},             # a non-string name: assert no cause
        {"id": "z", "name": "other"},       # a real different name: filter not exact
        {"name": "bake-1"},                 # OURS, no id
        {"id": None, "name": "bake-1"},     # OURS, unusable id
    ])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    found = bake.pods_named_detail("bake-1", "t")
    assert found.ids == []
    assert found.mismatched == 1
    assert found.matched_no_id == 2
    assert found.unusable == 4
    assert found.answered is True


# ------------- the same inversion, one branch over: an unlistable account ---------

def test_a_failed_listing_is_not_a_successful_reap(monkeypatch, capsys):
    """"No pod is running" is a conclusion a failed listing cannot produce.

    This is the likelier of the two unanswerable cases: a 429 or 5xx on a list
    endpoint is ordinary, while a non-exact server filter is the unlikely case the
    spec argues against. Before this, a 503 returned True and printed that nothing
    was running, and the whole suite permitted either policy.
    """
    for code in (401, 403, 429, 500, 502, 503):
        monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None, _c=code: (_c, "err"))
        found = bake.pods_named_detail("bake-99", "t")
        assert found.answered is False, "HTTP %s was treated as an answer" % code
        assert bake.reap_by_name("bake-99", "t") is False, (
            "HTTP %s reported a successful reap" % code)
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "no pod named" not in out, "printed a conclusion the listing cannot support"


def test_an_unlistable_account_and_an_empty_one_are_different(monkeypatch):
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, "[]"))
    assert bake.reap_by_name("bake-99", "t") is True
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (503, "boom"))
    assert bake.reap_by_name("bake-99", "t") is False


def test_a_non_json_listing_is_not_an_answer(monkeypatch):
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, "<html>nope"))
    assert bake.pods_named_detail("bake-99", "t").answered is False
    assert bake.reap_by_name("bake-99", "t") is False


def test_our_pod_with_no_id_does_not_blame_the_name_filter(monkeypatch, capsys):
    """The one case where the old single message was actively false."""
    rows = json.dumps([{"name": "bake-99"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    assert bake.reap_by_name("bake-99", "t") is False
    out = capsys.readouterr().out
    assert "no usable id" in out
    assert "not exact" not in out, "blamed the filter for an unaddressable pod"


def test_a_different_name_is_reported_as_the_filter_measurement(monkeypatch, capsys):
    rows = json.dumps([{"id": "z", "name": "bake-99-decorated"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    assert bake.reap_by_name("bake-99", "t") is False
    out = capsys.readouterr().out
    assert "not exact" in out
    assert "no usable id" not in out


def test_an_unreadable_row_asserts_no_cause(monkeypatch, capsys):
    rows = json.dumps(["junk", {"id": "x"}])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    assert bake.reap_by_name("bake-99", "t") is False
    out = capsys.readouterr().out
    assert "could not be read" in out
    assert "not exact" not in out
    assert "no usable id" not in out


def test_a_match_is_still_reaped_alongside_every_rejection_kind(monkeypatch):
    """Narrowing must not narrow to nothing: the right pod still gets reaped."""
    killed = []
    rows = json.dumps([
        "junk", {"id": "x"}, {"id": "z", "name": "other"},
        {"name": "bake-99"}, {"id": "mine", "name": "bake-99"},
    ])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    monkeypatch.setattr(bake, "terminate", lambda pid, tok: killed.append(pid) or (True, True))
    assert bake.reap_by_name("bake-99", "t") is True
    assert killed == ["mine"]


# ------------------- never render a pod row: it carries env (strummer) -----------

CANARY = "ghs_CANARY_do_not_print_me"
FAT_ROW = {
    "name": "bake-99", "desiredStatus": "RUNNING", "costPerHr": 0.12,
    "containerDiskInGb": 120,
    "env": {"GHCR_TOKEN": CANARY, "BAKE_REPO": "o/r"},
    "dockerStartCmd": ["sh", "-c", "auth=$(printf %s:%s $GHCR_USER " + CANARY + ")"],
    "machineId": "mach-1", "publicIp": "203.0.113.9",
}


def test_pod_summary_renders_only_the_allowlisted_fields():
    """A Pod response has 34 properties. Allowlist, so a new one cannot leak by default."""
    line = bake.pod_summary(FAT_ROW)
    for leaky in (CANARY, "GHCR_TOKEN", "env", "machineId", "mach-1", "203.0.113.9",
                  "dockerStartCmd"):
        assert leaky not in line, "%r reached the summary" % leaky
    assert "desiredStatus='RUNNING'" in line
    assert "costPerHr=0.12" in line


def test_the_matched_no_id_branch_never_prints_the_row(capsys):
    """That branch fires only for a row whose name is ours, so it is always OUR pod.

    create_body puts GHCR_TOKEN in that pod's env, so the one branch guaranteed to
    be printing our own pod was printing its environment. This repo is public.
    pods_named_detail holds the RunPod token and not the GHCR one, so it cannot
    scrub even in principle; the fix is to render fields, never a row.
    """
    import json as _json
    rows = _json.dumps([{k: v for k, v in FAT_ROW.items() if k != "id"}])
    original = bake._request
    bake._request = lambda m, p, t, b=None: (200, rows)
    try:
        assert bake.reap_by_name("bake-99", "t") is False
    finally:
        bake._request = original
    captured = capsys.readouterr()
    whole = captured.out + captured.err
    assert "no usable id" in whole
    for leaky in (CANARY, "GHCR_TOKEN", "machineId", "203.0.113.9"):
        assert leaky not in whole, "%r reached an Actions log" % leaky


def test_a_nameless_row_with_an_id_names_the_id_only(monkeypatch, capsys):
    rows = json.dumps([dict(FAT_ROW, id="pod-xyz", name=None)])
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, rows))
    found = bake.pods_named_detail("bake-99", "t")
    assert found.unusable == 1
    whole = capsys.readouterr().err
    assert "pod-xyz" in whole
    assert CANARY not in whole


# --------------- the same inversion again, one body SHAPE over (strummer) --------

def test_a_dict_body_with_no_pods_list_is_not_an_answer(monkeypatch):
    """The dict branch exists to survive a shape change.

    Reading a missing key as an empty list is the one way it would survive one by
    silently reporting that nothing is running.
    """
    for body in ('{"items": [{"id": "a", "name": "bake-99"}]}',
                 '{"data": []}', '{}', '{"pods": null}', '{"pods": {"a": 1}}'):
        monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None, _b=body: (200, _b))
        found = bake.pods_named_detail("bake-99", "t")
        assert found.answered is False, "%s was treated as an answer" % body
        assert bake.reap_by_name("bake-99", "t") is False


def test_a_dict_body_with_a_real_pods_list_still_works(monkeypatch):
    body = json.dumps({"pods": [{"id": "mine", "name": "bake-99"}]})
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, body))
    found = bake.pods_named_detail("bake-99", "t")
    assert (found.ids, found.answered) == (["mine"], True)
    body = json.dumps({"pods": []})
    monkeypatch.setattr(bake, "_request", lambda m, p, t, b=None: (200, body))
    assert bake.reap_by_name("bake-99", "t") is True


def test_an_empty_name_is_not_an_answered_question(monkeypatch):
    """Consistency with the rubric this function now applies everywhere else."""
    monkeypatch.setattr(bake, "_request", _no_calls)
    found = bake.pods_named_detail("", "t")
    assert (found.ids, found.answered) == ([], False)
