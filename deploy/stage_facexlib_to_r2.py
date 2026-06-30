#!/usr/bin/env python3
"""One-time staging of facexlib face-detection + parsing weights into the R2 models hub, so the bake
(stage_bake_seed.py finish_dirs) carries them and the finish leg (GFPGAN / CodeFormer) resolves
OFFLINE instead of load_file_from_url-ing them from github at render time (the finish-leg egress the
sm_120 de-risk surfaced).

FLOW: fetch the two pinned facexlib release assets -> verify each is a real torch checkpoint (NOT a
404 / HTML body) by magic bytes -> HARD-ASSERT exact size + sha256 against the manifest pin
(deploy/bake-manifest.json, the single source of truth) -> upload to r2:vivijure/models/facexlib/.
stage_bake_seed.py then copies them server-side into the seed (bake-seed-bf16/facexlib/), and the
:0.3.3 build bakes them. The build-time bake gate and the de-risk runtime probe assert the SAME
manifest sha, so a corrupted-but-plausible or substituted .pth can never reach the baked image.

CREDS: a SCOPED, THROWAWAY R2 token (revoked right after). Provide it via --env FILE containing
R2_S3_ENDPOINT / R2_S3_ACCESS_KEY_ID / R2_S3_SECRET_ACCESS_KEY (the form the lead drops). Writes are
CONFINED to models/facexlib/. NEVER touch derisk/* -- a live de-risk pod streams its log there.

  python deploy/stage_facexlib_to_r2.py --env ~/vivijure-r2-facexlib.env [--dry-run]

Idempotent: rclone skips same-size objects; re-running re-verifies (size + sha256) and re-uploads
only on mismatch. The pinned bytes match exactly what facexlib own loaders fetch (retinaface_resnet50
detector + parsenet parser) and what the bake/probe assert, so baked == staged == expected.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Sibling helper: the ONE reader of the manifest pins (single source of truth, no private literals).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from facexlib_pins import load_facexlib_pins, verify_file  # noqa: E402

DST = "r2fx:vivijure/models/facexlib"          # the scoped remote, models/facexlib/ ONLY
# Torch checkpoints are either a zip archive (PK\x03\x04) or a legacy pickle (\x80). Reject anything
# that looks like an HTML / text error body so a 404 never gets staged as a .pth (cheap pre-hash gate).
_OK_MAGIC = (b"PK\x03\x04", b"\x80")


def _verify(path: Path, pin: dict) -> str:
    head = path.read_bytes()[:4]
    if not any(head.startswith(m) for m in _OK_MAGIC):
        sys.exit("FAIL " + pin["name"] + ": not a torch checkpoint (head=" + repr(head) + "); a 404/HTML body?")
    # HARD pin: exact size THEN sha256 against the manifest (single source of truth). Raises on drift.
    try:
        verify_file(path, pin)
    except ValueError as exc:
        sys.exit("FAIL " + str(exc))
    print("  ok " + pin["name"] + ": size " + str(pin["size"]) + " sha256 MATCHES pin " + pin["sha256"])
    return pin["sha256"]


def _rclone_conf(env_file: Path, conf_dir: Path) -> Path:
    env = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip(chr(34)).strip(chr(39))
    miss = [k for k in ("R2_S3_ENDPOINT", "R2_S3_ACCESS_KEY_ID", "R2_S3_SECRET_ACCESS_KEY") if not env.get(k)]
    if miss:
        sys.exit("env file missing: " + ", ".join(miss))
    conf = conf_dir / "rclone.conf"
    conf.write_text(
        "[r2fx]\ntype = s3\nprovider = Cloudflare\n"
        "access_key_id = " + env["R2_S3_ACCESS_KEY_ID"] + "\n"
        "secret_access_key = " + env["R2_S3_SECRET_ACCESS_KEY"] + "\n"
        "endpoint = " + env["R2_S3_ENDPOINT"] + "\n"
        "acl = private\nno_check_bucket = true\n"
    )
    conf.chmod(0o600)
    return conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="R2_S3_* env file for the scoped staging token")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pins = load_facexlib_pins()   # from deploy/bake-manifest.json
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        conf = _rclone_conf(Path(args.env).expanduser(), tdp)
        shas = {}
        for pin in pins:
            local = tdp / pin["name"]
            print("[fetch] " + pin["url"])
            if args.dry_run:
                print("  would download + verify (size+sha256 pin) + rclone copyto " + DST + "/" + pin["name"])
                continue
            urllib.request.urlretrieve(pin["url"], local)   # noqa: S310 (pinned github release URL)
            shas[pin["name"]] = _verify(local, pin)
            subprocess.run(
                ["rclone", "--config", str(conf), "copyto", str(local),
                 DST + "/" + pin["name"], "--s3-no-check-bucket", "--stats-one-line"],
                check=True)
            print("  staged -> " + DST + "/" + pin["name"])
        if not args.dry_run:
            print("\n=== staged sha256 (all matched the manifest pin) ===")
            for n, s in shas.items():
                print("  " + n + "  " + s)
            print("\nNext: python deploy/stage_bake_seed.py --skip-existing   # copies models/facexlib -> bake-seed-bf16/facexlib")
            print("then confirm r2:vivijure/bake-seed-bf16/facexlib/ has both files before the :0.3.3 build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
