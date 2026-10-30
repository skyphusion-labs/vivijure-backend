#!/usr/bin/env python3
"""Stage the full model set onto a mounted global volume.

Point VJ_MODELS_ROOT at the mount. Do not set VJ_VOLUME_ROOT.
Do not set VJ_VOLUME_KIND=global on the writer.
See docs/global-volume.md.
"""
from vivijure_backend.harness.models_mirror import _preload_main

if __name__ == "__main__":
    raise SystemExit(_preload_main(["--preload", "--flatten"]))
