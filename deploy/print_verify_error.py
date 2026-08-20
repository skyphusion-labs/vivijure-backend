#!/usr/bin/env python3
"""Print ONLY the R2 verify summary error (stage + message). No secrets. No GPU.

Standalone boto3 so the cheap dump job does not import vivijure_backend (yaml/torch).
Key layout matches harness.keys.verify_summary_key."""
from __future__ import annotations

import json
import os
import sys

import boto3
from botocore.config import Config


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: print_verify_error.py <run_id> [prefix]", file=sys.stderr)
        return 2
    run_id = args[0].strip()
    prefix = (args[1] if len(args) > 1 else "verify").strip() or "verify"
    endpoint = os.environ.get("R2_ENDPOINT") or ""
    key_id = os.environ.get("R2_ACCESS_KEY_ID") or ""
    secret = os.environ.get("R2_SECRET_ACCESS_KEY") or ""
    bucket = os.environ.get("R2_BUCKET") or "vivijure"
    if not (endpoint and key_id and secret):
        print("R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY required", file=sys.stderr)
        return 2
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    obj_key = "%s/%s/summary.json" % (prefix, run_id)
    raw = client.get_object(Bucket=bucket, Key=obj_key)["Body"].read()
    summary = json.loads(raw)
    err = summary.get("error") or {}
    out = {
        "run_id": summary.get("run_id") or run_id,
        "status": summary.get("status"),
        "last_event": summary.get("last_event"),
        "error": {
            "stage": err.get("stage"),
            "message": str(err.get("message") or "")[:500],
        },
        "event_names": [e.get("event") for e in (summary.get("events") or []) if isinstance(e, dict)],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if summary.get("status") else 1


if __name__ == "__main__":
    raise SystemExit(main())
