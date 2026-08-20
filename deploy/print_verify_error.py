#!/usr/bin/env python3
"""Print ONLY the R2 verify summary error (stage + message). No secrets. No GPU."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vivijure_backend.harness import keys
from vivijure_backend.harness.r2 import R2, R2Config


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: print_verify_error.py <run_id>", file=sys.stderr)
        return 2
    run_id = args[0].strip()
    prefix = (args[1] if len(args) > 1 else "verify").strip() or "verify"
    raw = R2(R2Config.from_env(os.environ)).get_bytes(keys.verify_summary_key(run_id, prefix=prefix))
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
