#!/usr/bin/env python3
"""Rent one RunPod CPU pod, build one image, push it to GHCR, then terminate the pod.

The pod is a public kaniko debug image. It downloads this public repo at one
commit, builds the Dockerfile, and pushes to GHCR. No volume is attached, so
terminate drops the container disk and the bill stops.

cpu3c, 4 vCPU, secure cloud. Catalog price read 2026-10-19: $0.03 per vCPU
per hour, so $0.12 per hour of compute. The create response cost is printed.
Container disk is extra and ends when the pod is terminated.

HOW SUCCESS IS READ. The runner cannot read pod stdout: the REST API has no
logs endpoint (checked against the live OpenAPI document, 2026-10-19; there is
no /pods/{id}/logs and no logs path at all). So the bake is judged by the thing
it is supposed to produce. Each run pushes an immutable tag carrying the run id,
and this script polls GHCR for that tag. The digest it reports is the digest the
registry holds, which is the value being pinned. A log line would have been a
claim about the artifact; this is the artifact.

This file does not start a pod unless you pass run or selftest. plan prints the
request with the registry token removed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

# v1 is the live REST base. https://rest.runpod.io/v2/... is not an API path: it
# 301-redirects to the docs site, so every call failed before a pod was created.
# Measured 2026-10-19: GET /v1/pods is 200, GET /v2/pods is 301 to docs.runpod.io.
API = "https://rest.runpod.io/v1"
GHCR = "https://ghcr.io"

# urllib sends "Python-urllib/3.x" by default and the RunPod edge answers that UA
# with a Cloudflare 1010, "browser_signature_banned". Measured 2026-10-19 with one
# variable changed: UA Python-urllib/3.13 got 403, curl/8.7.1 and this string both
# got 200 on the same GET with the same key. Name the client; do not drop this header.
USER_AGENT = "vivijure-image-bake/1 (+https://github.com/skyphusion-labs/vivijure-backend)"

BUILDER_IMAGE = "gcr.io/kaniko-project/executor:v1.23.2-debug"
CPU_ID = "cpu3c"
VCPU_COUNT = 4
DISK_MIN = 20
DISK_MAX = 160
DIGEST_PREFIX = "sha256:"
DEAD_STATES = ("EXITED", "TERMINATED", "ERROR", "DEAD")

# Busybox sh inside the kaniko debug image. No bash. The registry token stays
# in the environment and in the docker config file. It is not echoed. The final
# line is for a human reading the pod console; nothing in CI depends on it.
POD_SCRIPT = r"""set -eu
mkdir -p /workspace /kaniko/.docker
auth=$(printf '%s:%s' "$GHCR_USER" "$GHCR_TOKEN" | base64 | tr -d '\n')
printf '%s\n' "{\"auths\":{\"ghcr.io\":{\"auth\":\"${auth}\"}}}" > /kaniko/.docker/config.json
unset auth
cd /workspace
wget -O src.tar.gz "https://github.com/${BAKE_REPO}/archive/${BAKE_SHA}.tar.gz"
mkdir -p src
tar -xzf src.tar.gz -C src --strip-components=1
set -- \
  --context dir:///workspace/src \
  --dockerfile "$BAKE_DOCKERFILE" \
  --destination "$BAKE_DEST" \
  --destination "$BAKE_DEST_IMMUTABLE" \
  --digest-file /workspace/digest.txt \
  --cache=false \
  --compressed-caching=false \
  --snapshot-mode=redo
if [ -n "${BAKE_BUILD_ARGS:-}" ]; then
  for pair in $BAKE_BUILD_ARGS; do
    set -- "$@" --build-arg "$pair"
  done
fi
/kaniko/executor "$@"
digest=$(tr -d ' \n\r' < /workspace/digest.txt)
case "$digest" in
  sha256:*) ;;
  *) echo "BAKE_FAIL bad digest file"; exit 1 ;;
esac
printf 'BAKE_OK digest=%s\n' "$digest"
"""

SELFTEST_SCRIPT = "set -eu\necho selftest pod is up\nsleep 60\n"
SELFTEST_IMAGE = "busybox:1.37"


def disk_gb(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("disk_gb must be an integer")
    n = int(value)
    if n < DISK_MIN or n > DISK_MAX:
        raise argparse.ArgumentTypeError(
            "disk_gb must be %d to %d" % (DISK_MIN, DISK_MAX)
        )
    return n


def _cpu_pod(name: str, image: str, disk: int, vcpus: int, script: str, env: dict) -> dict:
    """A CPU pod body in live v1 field names.

    computeType is not optional in practice. Nothing in this schema is required,
    and computeType defaults to GPU, so an incomplete body does not fail: it
    rents a GPU pod. Measured 2026-10-19, a body of only {"name": ...} created a
    $2.49 per hour GPU pod. volumeInGb defaults to 20, which is a billed pod
    volume, so it is pinned to 0 to keep the claim that terminate ends the bill.
    """
    return {
        "name": name,
        "computeType": "CPU",
        "cloudType": "SECURE",
        "cpuFlavorIds": [CPU_ID],
        "vcpuCount": vcpus,
        "imageName": image,
        "containerDiskInGb": disk,
        "volumeInGb": 0,
        "dockerEntrypoint": ["/bin/sh", "-c"],
        "dockerStartCmd": [script],
        "env": env,
        "ports": [],
    }


def create_body(args: argparse.Namespace) -> dict:
    env = {
        "BAKE_REPO": args.repo,
        "BAKE_SHA": args.sha,
        "BAKE_DOCKERFILE": args.dockerfile,
        "BAKE_DEST": args.destination,
        "BAKE_DEST_IMMUTABLE": args.destination_immutable,
        "BAKE_BUILD_ARGS": args.build_arg or "",
        "GHCR_USER": args.ghcr_user,
        "GHCR_TOKEN": args.ghcr_token,
    }
    body = _cpu_pod(
        args.name, BUILDER_IMAGE, args.disk_gb or DISK_MIN, VCPU_COUNT, POD_SCRIPT, env
    )
    # The kaniko debug image carries busybox sh at this path, not /bin/sh.
    body["dockerEntrypoint"] = ["/busybox/sh", "-c"]
    return body


def selftest_body(name: str) -> dict:
    return _cpu_pod(name, SELFTEST_IMAGE, DISK_MIN, 2, SELFTEST_SCRIPT, {})


def public_view(body: dict) -> dict:
    shown = json.loads(json.dumps(body))
    if shown.get("env", {}).get("GHCR_TOKEN"):
        shown["env"]["GHCR_TOKEN"] = "SET"
    return shown


def scrub(text: str, *secrets: str) -> str:
    for secret in secrets:
        if secret and len(secret) > 7:
            text = text.replace(secret, "<REDACTED>")
    return text


def valid_digest(value: str) -> bool:
    if not value.startswith(DIGEST_PREFIX):
        return False
    body = value[len(DIGEST_PREFIX):]
    return len(body) == 64 and all(c in "0123456789abcdef" for c in body)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect. These APIs do not redirect, so one means a wrong path."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _http(method: str, url: str, headers: dict, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with _OPENER.open(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", "replace"), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            return exc.code, "redirect to %s (wrong path: these APIs do not redirect)" % (
                exc.headers.get("location", "?")), dict(exc.headers)
        return exc.code, exc.read().decode("utf-8", "replace"), dict(exc.headers)


def _request(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, str]:
    status, text, _headers = _http(method, API + path, {
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }, body)
    return status, text


# --------------------------------------------------------------------- registry

MANIFEST_ACCEPT = ", ".join((
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
))


# Validate the whole reference, do not sanitize a piece of it. The earlier version
# tested ref.startswith("ghcr.io/") and fell through to accepting anything else,
# so a foreign host parsed clean and its path went into a GHCR URL unescaped.
# CodeQL flagged it as py/incomplete-url-substring-sanitization and was right.
# This grammar is anchored at both ends and admits no dot segment, percent, query
# or fragment, so nothing reaching the URL needs escaping. GHCR repository names
# are lowercase, so an uppercase one is a real error and not a case to tolerate.
REF_RE = re.compile(
    r"^ghcr\.io/"
    r"([a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)+)"
    r":([A-Za-z0-9_][A-Za-z0-9._-]{0,127})$"
)


def split_ref(ref: str) -> tuple[str, str]:
    """ghcr.io/owner/name:tag -> (owner/name, tag). Anything else is refused."""
    match = REF_RE.match(ref or "")
    if not match:
        raise ValueError("not a ghcr.io/<lowercase/repo>:<tag> reference: %r" % ref)
    return match.group(1), match.group(2)


def ghcr_token(repo: str, token: str) -> str:
    """A pull token for one repository. Anonymous when no job token is supplied."""
    url = "%s/token?%s" % (GHCR, urllib.parse.urlencode({
        "service": "ghcr.io", "scope": "repository:%s:pull" % repo}))
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if token:
        basic = base64.b64encode(("x:" + token).encode()).decode()
        headers["Authorization"] = "Basic " + basic
    status, text, _h = _http("GET", url, headers)
    if status != 200:
        return ""
    try:
        return str(json.loads(text).get("token") or "")
    except json.JSONDecodeError:
        return ""


def _ghcr_head(ref: str, token: str) -> tuple[int, str | None]:
    """(status, digest). Status matters: 404 means absent, anything else may mean nothing."""
    repo, tag = split_ref(ref)
    bearer = ghcr_token(repo, token)
    headers = {"Accept": MANIFEST_ACCEPT, "User-Agent": USER_AGENT}
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    status, _text, resp_headers = _http(
        "HEAD", "%s/v2/%s/manifests/%s" % (GHCR, repo, tag), headers)
    if status != 200:
        return status, None
    digest = ""
    for key, value in resp_headers.items():
        if key.lower() == "docker-content-digest":
            digest = value.strip()
            break
    return status, digest if valid_digest(digest) else None


def ghcr_digest(ref: str, token: str) -> str | None:
    """The digest GHCR holds for a tag, or None. For polling, where a retry is free."""
    return _ghcr_head(ref, token)[1]


def ghcr_digest_final(ref: str, token: str, attempts: int = 5) -> tuple[str | None, bool]:
    """The read that decides the verdict, so it must not confuse absent with unreachable.

    Returns (digest, answered). A 404 is a real answer: the tag is not there. Anything
    else (429, 5xx, a token-endpoint hiccup) is not, and this loop issues hundreds of
    manifest reads per bake, so a rate-limit reply is ordinary. Reporting "never pushed"
    off one of those turns a paid, successful build into a red run with no artifact.
    """
    for attempt in range(attempts):
        status, digest = _ghcr_head(ref, token)
        if digest:
            return digest, True
        if status == 404:
            return None, True
        print("registry read for %s was inconclusive: HTTP %s (attempt %d of %d)"
              % (ref, status, attempt + 1, attempts), file=sys.stderr)
        if attempt + 1 < attempts:
            time.sleep(min(5 * (attempt + 1), 20))
    return None, False


# ------------------------------------------------------------------------- pods

def pods_named(name: str, token: str) -> list[str]:
    """Pod ids whose OWN name equals this one. GET /pods?name= returns a bare array.

    The server's filter is asked for and then not trusted, because this feeds the
    only destructive call in the script and a query parameter decides its blast
    radius. The live spec calls the parameter "Filter to Pods with the provided
    name", which reads as exact, and it is almost certainly implemented that way.
    But a 200 with an empty array cannot tell exact matching apart from the
    parameter being ignored when the account holds no other pod, which is the only
    state it has been observed in. So the check that CAN be made locally is made
    locally: every row carries its own name, and comparing it rules out prefix
    matching, substring matching and an ignored parameter at once, now and after
    any change to how the server behaves.

    An empty name would list the account, so it returns nothing rather than
    everything. main() refuses that case too; this is the layer underneath.
    """
    return pods_named_detail(name, token).ids


class PodLookup(NamedTuple):
    """What a pods-by-name listing actually established.

    One rejected count made four different facts print one sentence, and in one
    reachable case that sentence was false, so each fact that leads to a different
    operator action gets its own field and its own message. A message must not
    assert a cause the data cannot support.
    """

    ids: list[str]
    mismatched: int      # a real, different string name: the server filter is not exact
    matched_no_id: int   # OUR name came back with no usable id: we cannot address it
    unusable: int        # unreadable row, or no name to read: assert no cause
    answered: bool       # the listing call itself succeeded


def pods_named_detail(name: str, token: str) -> PodLookup:
    """Ask for pods by name and report what the answer can and cannot establish.

    "Nothing is running", "pods came back and none matched" and "we could not list
    at all" are three different facts with the same shape. Two of them mean we
    cannot say whether an orphan is billing, and the third is the only one that may
    report success. A failed listing establishes nothing about what is running.
    """
    if not name:
        return PodLookup([], 0, 0, 0, True)
    code, payload = _request("GET", "/pods?name=%s" % urllib.parse.quote(name, safe=""), token)
    if code != 200:
        print("could not list pods named %s: HTTP %s" % (name, code), file=sys.stderr)
        return PodLookup([], 0, 0, 0, False)
    try:
        rows = json.loads(payload)
    except json.JSONDecodeError:
        print("pods listing for %s was not JSON" % name, file=sys.stderr)
        return PodLookup([], 0, 0, 0, False)
    if isinstance(rows, dict):
        rows = rows.get("pods") or []
    if not isinstance(rows, list):
        # A bare null, number or boolean is ordinary enough as an empty REST reply,
        # and iterating one raises TypeError. That is loud rather than silent, but
        # inside run()'s finally it REPLACES the exception that sent us there, so a
        # create timeout would be reported as a complaint about iteration.
        print("pods listing for %s was not a list: %r" % (name, type(rows).__name__),
              file=sys.stderr)
        return PodLookup([], 0, 0, 0, False)
    ids: list[str] = []
    mismatched = matched_no_id = unusable = 0
    for row in rows:
        if not isinstance(row, dict):
            unusable += 1
            continue
        row_name = row.get("name")
        if row_name == name:
            if row.get("id"):
                ids.append(str(row["id"]))
            else:
                # The pod we asked for, with no handle to terminate it by. This is
                # the one that must not be described as a filter problem.
                print("pod named %r came back with no usable id: %r" % (name, row),
                      file=sys.stderr)
                matched_no_id += 1
            continue
        if not isinstance(row_name, str):
            # No name to read, so nothing can be concluded about the server's filter.
            unusable += 1
            continue
        print("ignoring pod %s: its name is %r, not %r"
              % (row.get("id"), row_name, name), file=sys.stderr)
        mismatched += 1
    return PodLookup(ids, mismatched, matched_no_id, unusable, True)


def reap_by_name(name: str, token: str) -> bool:
    """Terminate every pod carrying this name. The backstop for an unknown id.

    A create can leave a pod running whose id never reached us: the POST can time
    out or raise after RunPod has made the pod. The NAME is ours and is known before
    the call, so it is the only handle that cannot be outrun by the thing it backs up.

    The name is bake-<run id> and carries no attempt number, so it is NOT unique per
    attempt, and that is deliberate: a re-run after a dead runner shares the orphan's
    name and therefore cleans it up. Adding the attempt would lose that. With the
    name checked client side, the set reaped is exactly "pods named bake-<run id>".
    """
    found = pods_named_detail(name, token)
    ids = found.ids

    # Each of these says what it can establish and nothing more. They are not
    # exclusive, so every applicable one is emitted before the verdict.
    if not found.answered:
        # The likelier of the two unanswerable cases: a 429 or a 5xx on a list
        # endpoint is ordinary, while a non-exact server filter is the unlikely case
        # the spec argues against. "No pod is running" is a conclusion a failed
        # listing cannot produce, so it is not printed here.
        print("::warning::could not list pods named %s, so whether an orphan is "
              "billing is unknown. Check the account." % name)
    if found.matched_no_id:
        print("::warning::%d pod(s) named %s came back with no usable id, so they "
              "cannot be terminated from here. This is not a name-filter problem: "
              "the pod we asked for arrived unaddressable. Check the account."
              % (found.matched_no_id, name))
    if found.mismatched:
        # The free measurement. A row with a real, different name can only come back
        # if the server's name filter is not exact, which is the question a live read
        # could not settle without renting a second pod, so the first dispatch that
        # trips this IS the experiment.
        print("::warning::%d pod(s) came back for name %s under a DIFFERENT name, so "
              "the server's name filter is not exact. The strict check here is what "
              "stopped them being terminated; re-read that assumption."
              % (found.mismatched, name))
    if found.unusable:
        print("::warning::%d row(s) in the listing for %s could not be read, so no "
              "conclusion is drawn from them." % (found.unusable, name))

    if ids:
        ok = True
        for pod_id in ids:
            print("::warning::reaping pod %s found by name %s" % (pod_id, name))
            accepted, gone = terminate(pod_id, token)
            ok = ok and accepted and gone
        return ok
    if not found.answered or found.mismatched or found.matched_no_id or found.unusable:
        # An unconfirmed reap of a billable resource is not a success. This is the
        # same contract the id path already has, where reap returns 0 only when the
        # pod reads back gone.
        return False
    print("no pod named %s is running" % name)
    return True


def terminate(pod_id: str, token: str) -> tuple[bool, bool]:
    """Terminate one pod. Returns (delete_accepted, gone_confirmed).

    The DELETE response is not proof. Only a 404 or a TERMINATED status is.
    Verification failing is a warning, never a reason to fail a good bake:
    failing one would also skip the repin, which throws the paid result away.
    """
    code, payload = _request("DELETE", "/pods/%s" % pod_id, token)
    print("terminate pod id=%s HTTP %s" % (pod_id, code))
    accepted = code in (200, 202, 204, 404)
    if not accepted:
        print(scrub(payload, token)[:300], file=sys.stderr)
    for _ in range(6):
        code, payload = _request("GET", "/pods/%s" % pod_id, token)
        if code == 404:
            print("pod %s is gone" % pod_id)
            return accepted, True
        if code == 200:
            try:
                state = json.loads(payload).get("desiredStatus")
            except json.JSONDecodeError:
                state = None
            if state == "TERMINATED":
                print("pod %s desiredStatus=TERMINATED" % pod_id)
                return accepted, True
        time.sleep(5)
    print("::warning::pod %s did not read back as gone. Check it." % pod_id)
    return accepted, False


def pod_state(pod_id: str, token: str) -> str | None:
    code, payload = _request("GET", "/pods/%s" % pod_id, token)
    if code == 404:
        return "TERMINATED"
    if code != 200:
        return None
    try:
        return json.loads(payload).get("desiredStatus")
    except json.JSONDecodeError:
        return None


def create_pod(body: dict, token: str, secrets: tuple[str, ...]) -> str | None:
    status, raw = _request("POST", "/pods", token, body)
    if status not in (200, 201):
        # A validation error echoes the request, and the request carries
        # GHCR_TOKEN. Actions logs on a public repo are public, so scrub the
        # body rather than printing it raw or withholding it entirely.
        print("create pod failed: HTTP %s" % status, file=sys.stderr)
        print(scrub(raw, *secrets)[:600], file=sys.stderr)
        return None
    try:
        created = json.loads(raw)
    except json.JSONDecodeError:
        print("create pod returned a body that is not JSON", file=sys.stderr)
        return None
    pod_id = str(created.get("id") or "")
    if not pod_id:
        print("create pod returned no id", file=sys.stderr)
        return None
    print("pod id=%s desiredStatus=%s cost_per_hour=%s disk=%sGB" % (
        pod_id, created.get("desiredStatus"), created.get("costPerHr"),
        created.get("containerDiskInGb")))
    return pod_id


def run(args: argparse.Namespace) -> int:
    if not args.ghcr_token:
        args.ghcr_token = os.environ.get("GHCR_TOKEN", "")
    token = os.environ.get("RUNPOD_API_KEY", "")
    if not token:
        print("RUNPOD_API_KEY is unset. Refusing to create a pod.", file=sys.stderr)
        return 1
    if not args.ghcr_token:
        print("GHCR token is unset. Refusing to create a pod.", file=sys.stderr)
        return 1
    try:
        split_ref(args.destination_immutable)
    except ValueError as exc:
        print("bad immutable destination: %s" % exc, file=sys.stderr)
        return 1
    # The immutable tag carries the run id, so a digest found under it was pushed
    # by THIS run. A mutable tag could be an older bake and would read as success.
    if ghcr_digest(args.destination_immutable, args.ghcr_token):
        print("%s already exists. Pick a fresh tag." % args.destination_immutable,
              file=sys.stderr)
        return 1

    # The name goes to disk BEFORE the create. Between RunPod making the pod and this
    # process learning its id there is a window: the POST can time out, or _http can
    # raise a URLError, with the pod already running and its id unknown to us. The name
    # is ours and exists before the call, so writing it first is the only ordering in
    # which the backstop cannot be outrun by the thing it backs up.
    if args.pod_name_out:
        with open(args.pod_name_out, "w", encoding="utf-8") as fh:
            fh.write(args.name + "\n")

    pod_id = None
    digest = None
    answered = True
    deadline = time.time() + args.timeout_minutes * 60
    try:
        pod_id = create_pod(create_body(args), token, (args.ghcr_token, token))
        if not pod_id:
            return 1
        if args.pod_id_out:
            with open(args.pod_id_out, "w", encoding="utf-8") as fh:
                fh.write(pod_id + "\n")
        while time.time() < deadline:
            digest = ghcr_digest(args.destination_immutable, args.ghcr_token)
            if digest:
                print("registry holds %s -> %s" % (args.destination_immutable, digest))
                break
            state = pod_state(pod_id, token)
            if state in DEAD_STATES:
                # The push can land moments before the pod exits, and this read is the
                # verdict, so it retries rather than reading one transient as absence.
                digest, answered = ghcr_digest_final(
                    args.destination_immutable, args.ghcr_token)
                if not digest and answered:
                    print("pod reached %s and the tag was never pushed" % state,
                          file=sys.stderr)
                break
            time.sleep(args.poll_seconds)
        else:
            print("bake timed out after %d minutes" % args.timeout_minutes,
                  file=sys.stderr)
    finally:
        if pod_id:
            accepted, _gone = terminate(pod_id, token)
            if not accepted:
                print("pod %s may still be billed. Terminate it." % pod_id,
                      file=sys.stderr)
        else:
            # No id, so a pod may exist that we cannot address. Ask by name.
            reap_by_name(args.name, token)
    if not digest:
        if not answered:
            print("the registry never gave a conclusive answer for %s. The image may "
                  "have been pushed; check the tag before re-running."
                  % args.destination_immutable, file=sys.stderr)
        print("bake failed", file=sys.stderr)
        return 1
    print("digest=%s" % digest)
    if args.digest_out:
        with open(args.digest_out, "w", encoding="utf-8") as fh:
            fh.write(digest + "\n")
    return 0


def reap(args: argparse.Namespace) -> int:
    """Terminate by id, or by name when no id was ever written. Safe to run twice."""
    token = os.environ.get("RUNPOD_API_KEY", "")
    if not token:
        print("RUNPOD_API_KEY is unset. Cannot reap.", file=sys.stderr)
        return 1
    if args.pod_id:
        accepted, gone = terminate(args.pod_id, token)
        if not accepted:
            print("terminate was refused for pod %s" % args.pod_id, file=sys.stderr)
            return 1
        return 0 if gone else 1
    return 0 if reap_by_name(args.name, token) else 1


def selftest(args: argparse.Namespace) -> int:
    """Prove create, status and terminate on the cheapest CPU pod there is.

    This exercises every RunPod call the bake makes, for about a cent, so a
    schema or edge change is caught without building anything.
    """
    token = os.environ.get("RUNPOD_API_KEY", "")
    if not token:
        print("RUNPOD_API_KEY is unset. Refusing to create a pod.", file=sys.stderr)
        return 1
    pod_id = create_pod(selftest_body(args.name), token, (token,))
    if not pod_id:
        return 1
    seen = []
    try:
        deadline = time.time() + args.timeout_minutes * 60
        while time.time() < deadline:
            state = pod_state(pod_id, token)
            print("  desiredStatus=%s" % state)
            seen.append(state)
            if state == "RUNNING":
                break
            time.sleep(args.poll_seconds)
    finally:
        accepted, gone = terminate(pod_id, token)
    if "RUNNING" not in seen:
        print("pod never reported RUNNING: %r" % seen, file=sys.stderr)
        return 1
    if not (accepted and gone):
        print("terminate did not confirm", file=sys.stderr)
        return 1
    print("selftest ok: created a CPU pod, read its status, terminated it, confirmed gone")
    return 0


def check_tag(args: argparse.Namespace) -> int:
    """Read one tag's digest out of GHCR. The success channel, on its own."""
    digest = ghcr_digest(args.destination, os.environ.get("GHCR_TOKEN", ""))
    if not digest:
        print("no digest for %s" % args.destination, file=sys.stderr)
        return 1
    print("%s -> %s" % (args.destination, digest))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bake one image on a RunPod CPU pod.")
    parser.add_argument("command", choices=("plan", "run", "selftest", "reap", "check-tag"))
    # Only plan and run need the whole build request; main() checks per command.
    parser.add_argument("--name", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--dockerfile", default="")
    parser.add_argument("--destination", default="")
    parser.add_argument("--destination-immutable", default="")
    parser.add_argument("--disk-gb", type=disk_gb, default=0)
    parser.add_argument("--pod-id", default="")
    parser.add_argument("--build-arg", default="")
    parser.add_argument("--ghcr-user", default="")
    parser.add_argument("--ghcr-token", default="")
    parser.add_argument("--digest-out", default="")
    parser.add_argument("--pod-id-out", default="")
    parser.add_argument("--pod-name-out", default="")
    parser.add_argument("--timeout-minutes", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=20)
    return parser


BUILD_REQUIRED = (
    "name", "repo", "sha", "dockerfile", "destination", "destination_immutable",
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("plan", "run"):
        missing = [f for f in BUILD_REQUIRED if not getattr(args, f)]
        if not args.disk_gb:
            missing.append("disk_gb")
        if missing:
            parser.error("%s needs --%s" % (
                args.command, ", --".join(f.replace("_", "-") for f in missing)))
    if args.command == "reap":
        if not args.pod_id and not args.name:
            parser.error("reap needs --pod-id or --name")
        return reap(args)
    if args.command == "selftest":
        if not args.name:
            parser.error("selftest needs --name")
        return selftest(args)
    if args.command == "check-tag":
        if not args.destination:
            parser.error("check-tag needs --destination")
        return check_tag(args)
    if args.command == "plan":
        print(json.dumps(public_view(create_body(args)), indent=2, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
