#!/usr/bin/env python3
"""Homelab HTTP entry for finish_clip (RIFE) on LOCAL_FINISH_RIFE_URL."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from runpod_http_serve import run_serve  # noqa: E402
from vivijure_backend.worker import handler  # noqa: E402

if __name__ == "__main__":
    run_serve(
        handler,
        service="vivijure-backend-finish-rife",
        port=int(os.environ.get("PORT", "8010") or "8010"),
    )
