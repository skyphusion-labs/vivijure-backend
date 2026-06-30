#!/usr/bin/env python3
"""One-time staging of facexlib's face-detection + parsing weights into the R2 models hub, so the
bake (stage_bake_seed.py finish_dirs) carries them and the finish leg (GFPGAN / CodeFormer) resolves
OFFLINE instead of load_file_from_url-ing them from github at render time (the finish-leg egress the
sm_120 de-risk surfaced).

FLOW: fetch the two pinned facexlib release assets -> verify each is a real torch checkpoint (NOT a
404 / HTML body) by magic bytes + size band -> log sha256 -> upload to r2:vivijure/models/facexlib/.
stage_bake_seed.py then copies them server-side into the seed (bake-seed-bf16/facexlib/), and the
:0.3.3 build bakes them. baked_probe asserts both files at $0; the --network=none docker verify
proves the offline resolution end to end.

CREDS: a SCOPED, THROWAWAY R2 token (revoked right after). Provide it via --env FILE containing
R2_S3_ENDPOINT / R2_S3_ACCESS_KEY_ID / R2_S3_SECRET_ACCESS_KEY (the form the lead drops). Writes are
CONFINED to models/facexlib/. NEVER touch derisk/* -- a live de-risk pod streams its log there.

  python deploy/stage_facexlib_to_r2.py --env ~/vivijure-r2-facexlib.env [--dry-run]

Idempotent: rclone skips same-size objects; re-running re-verifies and re-uploads only on mismatch.
The two files are version-pinned to the exact URLs facexlib's own loaders fetch (retinaface_resnet50
detector + parsenet parser), so the baked bytes match what the library expects.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Pinned facexlib release assets (what facexlib.detection / .parsing load_file_from_url fetch).
ASSETS = [
    {
        "name": "detection_Resnet50_Final.pth",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "min_mb": 90, "max_mb": 130,   # ~104 MB
    },
    {
        "name": "parsing_parsenet.pth",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "min_mb": 70, "max_mb": 100,   # ~85 MB
    },
]

DST = "r2fx:vivijure/models/facexlib"          # the scoped remote, models/facexlib/ ONLY
# Torch checkpoints are either a zip archive (PK\x03\x04) or a legacy pickle (\x80). Reject anything
# that looks like an HTML / text error body so a 404 never gets staged as a .pth.
_OK_MAGIC = (b"PK\x03\x04", b"\x80")


def _verify(path: Path, spec: dict) -> str:
    size = path.stat().st_size
    mb = size / 1024 / 1024
    if not (spec["min_mb"] <= mb <= spec["max_mb"]):
        sys.exit(f"FAIL {spec['name']}: size {mb:.1f} MB outside [{spec['min_mb']}, {spec['max_mb']}]")
    head = path.read_bytes()[:4]
    if not any(head.startswith(m) for m in _OK_MAGIC):
        sys.exit(f"FAIL {spec['name']}: not a torch checkpoint (head={head!r}); a 404/HTML body?")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"  ok {spec['name']}: {mb:.1f} MB  sha256={sha}")
    return sha


def _rclone_conf(env_file: Path, conf_dir: Path) -> Path:
    env = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    miss = [k for k in ("R2_S3_ENDPOINT", "R2_S3_ACCESS_KEY_ID", "R2_S3_SECRET_ACCESS_KEY") if not env.get(k)]
    if miss:
        sys.exit(f"env file missing: {', '.join(miss)}")
    conf = conf_dir / "rclone.conf"
    conf.write_text(
        "[r2fx]\ntype = s3\nprovider = Cloudflare\n"
        f"access_key_id = {env['R2_S3_ACCESS_KEY_ID']}\n"
        f"secret_access_key = {env['R2_S3_SECRET_ACCESS_KEY']}\n"
        f"endpoint = {env['R2_S3_ENDPOINT']}\n"
        "acl = private\nno_check_bucket = true\n"
    )
    conf.chmod(0o600)
    return conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="R2_S3_* env file for the scoped staging token")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        conf = _rclone_conf(Path(args.env).expanduser(), tdp)
        shas = {}
        for spec in ASSETS:
            local = tdp / spec["name"]
            print(f"[fetch] {spec['url']}")
            if args.dry_run:
                print(f"  would download + verify + rclone copyto {DST}/{spec['name']}")
                continue
            urllib.request.urlretrieve(spec["url"], local)   # noqa: S310 (pinned github release URL)
            shas[spec["name"]] = _verify(local, spec)
            subprocess.run(
                ["rclone", "--config", str(conf), "copyto", str(local),
                 f"{DST}/{spec['name']}", "--s3-no-check-bucket", "--stats-one-line"],
                check=True)
            print(f"  staged -> {DST}/{spec['name']}")
        if not args.dry_run:
            print("\n=== staged sha256 (record these) ===")
            for n, s in shas.items():
                print(f"  {n}  {s}")
            print("\nNext: add facexlib to bake-manifest finish_dirs (done in the code PR), then")
            print("  python deploy/stage_bake_seed.py --skip-existing   # copies models/facexlib -> bake-seed-bf16/facexlib")
            print("then confirm r2:vivijure/bake-seed-bf16/facexlib/ has both files before the :0.3.3 build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
