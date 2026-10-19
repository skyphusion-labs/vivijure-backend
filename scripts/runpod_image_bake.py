#!/usr/bin/env python3
"""Rent one RunPod CPU pod, build one image, push it to GHCR, then terminate the pod.

The pod is a public kaniko debug image. It downloads this public repo at one
commit, builds the Dockerfile, and pushes to GHCR. There is no network volume
and no host disk, so terminate drops the container disk and the bill stops.

cpu3c, 4 vCPU, secure cloud. Catalog price read 2026-10-19: $0.03 per vCPU
per hour, so $0.12 per hour of compute. The create response cost is printed.
Container disk is extra and ends when the pod is terminated.

This file does not start a pod unless you pass run. plan prints the request
with the registry token removed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v2"
BUILDER_IMAGE = "gcr.io/kaniko-project/executor:v1.23.2-debug"
CPU_ID = "cpu3c"
VCPU_COUNT = 4
DISK_MIN = 20
DISK_MAX = 160
DIGEST_PREFIX = "sha256:"

# Busybox sh inside the kaniko debug image. No bash. The registry token stays
# in the environment and in the docker config file. It is not echoed.
POD_SCRIPT = r"""set -eu
trap 'echo BAKE_FAIL' EXIT
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
trap - EXIT
printf 'BAKE_OK digest=%s\n' "$digest"
"""


def disk_gb(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("disk_gb must be an integer")
    n = int(value)
    if n < DISK_MIN or n > DISK_MAX:
        raise argparse.ArgumentTypeError(
            "disk_gb must be %d to %d" % (DISK_MIN, DISK_MAX)
        )
    return n


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
    return {
        "name": args.name,
        "cloud": "SECURE",
        "cpu": {"id": CPU_ID, "vcpuCount": VCPU_COUNT},
        "image": BUILDER_IMAGE,
        "disk": args.disk_gb,
        "args": json.dumps({"entrypoint": ["/busybox/sh", "-c"], "cmd": [POD_SCRIPT]}),
        "env": env,
        "ports": [],
    }


def public_view(body: dict) -> dict:
    shown = json.loads(json.dumps(body))
    token = shown.get("env", {}).get("GHCR_TOKEN")
    if token:
        shown["env"]["GHCR_TOKEN"] = "SET"
    return shown


def marker(text: str) -> str | None:
    """Return a sha256 digest, the string FAIL, or None when the bake is still going."""
    found = None
    for line in text.splitlines():
        line = line.strip()
        if line == "BAKE_FAIL" or line.startswith("BAKE_FAIL "):
            return "FAIL"
        prefix = "BAKE_OK digest="
        if line.startswith(prefix):
            digest = line[len(prefix):].strip()
            if digest.startswith(DIGEST_PREFIX) and len(digest) == len(DIGEST_PREFIX) + 64:
                if all(c in "0123456789abcdef" for c in digest[len(DIGEST_PREFIX):]):
                    found = digest
    return found


def _request(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, str]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _logs_text(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("logs", "log", "output", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
        return json.dumps(parsed)
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("message") or item.get("log") or item))
        return "\n".join(parts)
    return raw


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
    body = create_body(args)
    status, raw = _request("POST", "/pods", token, body)
    if status not in (200, 201):
        print("create pod failed: HTTP %s" % status, file=sys.stderr)
        print(_logs_text(raw)[:500], file=sys.stderr)
        return 1
    created = json.loads(raw)
    pod_id = str(created.get("id") or "")
    if not pod_id:
        print("create pod returned no id", file=sys.stderr)
        return 1
    print("pod id=%s status=%s cost_per_hour=%s" % (
        pod_id, created.get("status"), created.get("cost")))
    if args.pod_id_out:
        with open(args.pod_id_out, "w", encoding="utf-8") as fh:
            fh.write(pod_id + "\n")
    deadline = time.time() + args.timeout_minutes * 60
    result = None
    try:
        while time.time() < deadline:
            code, payload = _request("GET", "/pods/%s/logs" % pod_id, token)
            if code == 200:
                result = marker(_logs_text(payload))
                if result:
                    break
            code, payload = _request("GET", "/pods/%s" % pod_id, token)
            if code == 200:
                try:
                    state = json.loads(payload).get("status")
                except json.JSONDecodeError:
                    state = None
                if state in ("EXITED", "ERROR", "TERMINATED") and result is None:
                    code, payload = _request("GET", "/pods/%s/logs" % pod_id, token)
                    if code == 200:
                        result = marker(_logs_text(payload))
                    if result is None:
                        result = "FAIL"
                    break
            time.sleep(args.poll_seconds)
        else:
            print("bake timed out", file=sys.stderr)
            result = "FAIL"
    finally:
        code, payload = _request("DELETE", "/pods/%s" % pod_id, token)
        print("terminate pod id=%s HTTP %s" % (pod_id, code))
        if code not in (200, 202, 204, 404):
            print(payload[:300], file=sys.stderr)
            print("pod %s may still be billed. Terminate it." % pod_id, file=sys.stderr)
            return 1
    if not result or result == "FAIL":
        print("bake failed", file=sys.stderr)
        return 1
    print("digest=%s" % result)
    if args.digest_out:
        with open(args.digest_out, "w", encoding="utf-8") as fh:
            fh.write(result + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bake one image on a RunPod CPU pod.")
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--dockerfile", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--destination-immutable", required=True)
    parser.add_argument("--disk-gb", type=disk_gb, required=True)
    parser.add_argument("--build-arg", default="")
    parser.add_argument("--ghcr-user", default="")
    parser.add_argument("--ghcr-token", default="")
    parser.add_argument("--digest-out", default="")
    parser.add_argument("--pod-id-out", default="")
    parser.add_argument("--timeout-minutes", type=int, default=90)
    parser.add_argument("--poll-seconds", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        print(json.dumps(public_view(create_body(args)), indent=2, sort_keys=True))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
