#!/usr/bin/env python3
"""Stage stills onto a mounted global volume.

Point VJ_MODELS_ROOT at the mount. Do not set VJ_VOLUME_ROOT.
See docs/global-volume.md.
"""
from vivijure_backend.harness.models_mirror import _preload_main

if __name__ == "__main__":
    raise SystemExit(_preload_main(["--preload", "--stills", "--flatten"]))
