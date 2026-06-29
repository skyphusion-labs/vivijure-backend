#!/usr/bin/env python3
"""Stage the bf16 bake seed to R2 from deploy/bake-manifest.json (#13).

Builds r2:vivijure/bake-seed-bf16/ -- the curated, precision-correct weight set the release build
(release.yml -> deploy/Dockerfile) bakes into the image -- from the existing runtime mirror
r2:vivijure/models/. The manifest is the single source of truth (keep + drop + pinned revisions);
this script CONSUMES it, so the seed is reproducible from the repo, not from anyone's memory.

Three move types, by manifest entry:
  - keep repo, no recast  -> server-side R2->R2 copy of the whole repo (no bytes touch this box).
  - keep repo, recast      -> download (rclone -l: blobs + snapshot symlinks), recast every fp32
                              .safetensors blob to bf16 IN PLACE (same blob filename, so the snapshot
                              symlinks still resolve; .fp16 variant files are left untouched), upload
                              (rclone -l). This is the precision the worker loads at (from_pretrained
                              torch_dtype=bfloat16), so it is byte-identical to the runtime load-time
                              cast -- and it halves the fp32 shards so none exceeds the 10 GB GHCR
                              layer limit.
  - keep repo, include_only -> download just the pinned files as REAL files into the snapshot tree
                              (+ refs/main), upload. For LoRA collections (Lightning), not diffusers
                              from_pretrained repos, so file-level trim is safe.
  - finish_dirs            -> server-side R2->R2 copy (antelopev2/ rife/ GFPGANv1.4/).

SECRET HYGIENE: rclone reads the r2: remote from the caller's env-configured rclone (rollins'
per-identity R2 key). This script never reads, prints, or transforms the key value.

Run AS rollins (R2 creds in his login env):
    python3 deploy/stage_bake_seed.py --dry-run     # print the plan + R2 sizes, transfer nothing
    python3 deploy/stage_bake_seed.py               # do it (long: ~140 GB down / ~75 GB up + R2->R2)
    python3 deploy/stage_bake_seed.py --only models--SG161222--RealVisXL_V5.0   # one repo (resume)
Resumable: a repo whose dest already matches (size) is skipped; recast is idempotent (a blob already
bf16 is left as-is). Verify when done:
    python3 deploy/bake_layers.py assert-weights --root <local recast stage> --precision bf16
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "bake-manifest.json"
STAGE = Path(os.environ.get("VJ_BAKE_STAGE", str(Path.home() / "bake-seed-bf16-stage")))

# Ride out R2's intermittent 501 NotImplemented on copy (seen on small objects); rclone retries clear it.
RETRY = ["--retries", "8", "--low-level-retries", "20"]
# Local-only resume markers must never reach the seed (and R2 501s on them otherwise abort the upload).
EXCLUDE_MARKERS = ["--exclude", ".vj-recast-done", "--exclude", ".vj-*"]


def sh(cmd: list[str], *, capture: bool = False) -> str:
    """Run a command; raise on failure. Never logs env (no secret exposure)."""
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, check=True, text=True,
                       capture_output=capture)
    return r.stdout if capture else ""


def rclone_bytes(remote_path: str) -> int:
    try:
        out = subprocess.run(["rclone", "size", remote_path, "--json"],
                             check=True, text=True, capture_output=True).stdout
        return int(json.loads(out)["bytes"])
    except subprocess.CalledProcessError:
        return 0


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def recast_repo_inplace(local_repo: Path) -> dict:
    """Recast every fp32 .safetensors under local_repo/snapshots to bf16, in place (overwrite the
    real blob the snapshot symlink resolves to). .fp16.safetensors files and already-bf16 files are
    left untouched. Idempotent. Returns a summary."""
    import torch  # deferred: only the recast path needs it
    from safetensors.torch import load_file, save_file

    snap = local_repo / "snapshots"
    done = skipped = 0
    seen_blobs: set[str] = set()
    for f in sorted(snap.rglob("*.safetensors")):
        if f.name.endswith(".fp16.safetensors"):
            continue  # alternate-precision variant, never loaded by the no-variant from_pretrained
        blob = Path(os.path.realpath(f))  # the real bytes (symlink target, or f itself)
        if str(blob) in seen_blobs:
            continue
        seen_blobs.add(str(blob))
        tensors = load_file(str(blob))
        if not any(v.dtype == torch.float32 for v in tensors.values()):
            skipped += 1
            continue  # already bf16/fp16 (resume-safe) or non-fp32
        out = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
               for k, v in tensors.items()}
        tmp = blob.with_suffix(blob.suffix + ".bf16.tmp")
        save_file(out, str(tmp))
        os.replace(tmp, blob)
        done += 1
        print(f"    recast {f.relative_to(snap)} ({blob.name[:12]}) fp32->bf16", flush=True)
    return {"recast": done, "left": skipped}


def stage_recast_repo(repo: str, rev: str, src_hub: str, dst_hub: str, dry: bool) -> None:
    local = STAGE / repo
    marker = local / ".vj-recast-done"
    print(f"[recast] {repo}")
    if dry:
        print(f"    would: rclone copy -l {src_hub}/{repo} {local}; recast fp32->bf16; "
              f"rclone copy -l {local} {dst_hub}/{repo}")
        return
    local.mkdir(parents=True, exist_ok=True)
    # Resume: once the local recast is complete (marker present), do NOT re-download. The local
    # blobs are now bf16 (smaller than the R2 fp32), so a plain `rclone copy -l` would see a size
    # mismatch and CLOBBER the recast with fp32 again. The marker is the single source of "done".
    if marker.exists():
        print("    recast already complete (marker present); skipping download + recast.")
    else:
        sh(["rclone", "copy", "-l", f"{src_hub}/{repo}", str(local),
            "--transfers", "16", "--checkers", "16", "--multi-thread-streams", "8",
            "--multi-thread-cutoff", "100M", "--stats", "30s", "--stats-one-line"] + RETRY)
        summary = recast_repo_inplace(local)
        print(f"    recast summary: {summary}")
        marker.write_text("recast fp32->bf16 complete\n")
    # Upload is idempotent/resumable (rclone skips files already in dest); the local marker is excluded.
    sh(["rclone", "copy", "-l", str(local), f"{dst_hub}/{repo}",
        "--transfers", "16", "--checkers", "16", "--stats", "30s", "--stats-one-line"]
       + EXCLUDE_MARKERS + RETRY)


def stage_copy_repo(repo: str, src_hub: str, dst_hub: str, dry: bool) -> None:
    print(f"[copy]   {repo}  (server-side R2->R2)")
    if dry:
        print(f"    would: rclone copy {src_hub}/{repo} {dst_hub}/{repo}")
        return
    sh(["rclone", "copy", f"{src_hub}/{repo}", f"{dst_hub}/{repo}",
        "--transfers", "32", "--checkers", "32", "--stats", "30s", "--stats-one-line"] + RETRY)


def stage_include_only(repo: str, rev: str, files: list[str], src_hub: str, dst_hub: str,
                       dry: bool) -> None:
    """Pull only the pinned files (resolve marker -> blob) into the snapshot tree as REAL files,
    plus refs/main, then upload. For LoRA collections only."""
    print(f"[trim]   {repo}  ({len(files)} pinned file(s))")
    local = STAGE / repo
    if dry:
        for rel in files:
            print(f"    would stage snapshots/{rev}/{rel}")
        print(f"    would copy refs/main; upload {local} -> {dst_hub}/{repo}")
        return
    local.mkdir(parents=True, exist_ok=True)
    # refs/main (HF resolution needs it)
    sh(["rclone", "copy", f"{src_hub}/{repo}/refs", str(local / "refs"), "--stats-one-line"] + RETRY)
    for rel in files:
        marker = f"{src_hub}/{repo}/snapshots/{rev}/{rel}.rclonelink"
        blobrel = subprocess.run(["rclone", "cat", marker], check=True, text=True,
                                 capture_output=True).stdout.strip()
        bhash = blobrel.rsplit("/", 1)[-1]
        dest = local / "snapshots" / rev / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        sh(["rclone", "copyto", f"{src_hub}/{repo}/blobs/{bhash}", str(dest), "--stats-one-line"] + RETRY)
    sh(["rclone", "copy", "-l", str(local), f"{dst_hub}/{repo}", "--stats-one-line"]
       + EXCLUDE_MARKERS + RETRY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="stage just this repo (resume one entry)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a keep repo whose dest size already matches source (resume)")
    args = ap.parse_args()

    m = load_manifest()
    src = m["source"]; dst = m["dest"]; hub = m["hub_subpath"]
    src_hub = f"{src}/{hub}"; dst_hub = f"{dst}/{hub}"
    print(f"manifest: precision={m['precision']} model_version={m['model_version']}")
    print(f"source={src}  dest={dst}\n")

    for e in m["keep_repos"]:
        repo = e["repo"]
        if args.only and repo != args.only:
            continue
        if args.skip_existing and not args.dry_run:
            if rclone_bytes(f"{dst_hub}/{repo}") >= rclone_bytes(f"{src_hub}/{repo}") * 0.5 and \
               "include_only" not in e and not e.get("recast_fp32_to_bf16"):
                print(f"[skip]   {repo} (already in dest)"); continue
        if e.get("recast_fp32_to_bf16"):
            stage_recast_repo(repo, e["rev"], src_hub, dst_hub, args.dry_run)
        elif "include_only" in e:
            stage_include_only(repo, e["rev"], e["include_only"], src_hub, dst_hub, args.dry_run)
        else:
            stage_copy_repo(repo, src_hub, dst_hub, args.dry_run)

    if not args.only:
        for fd in m["finish_dirs"]:
            d = fd["dir"]
            print(f"[copy]   finish/{d}  (server-side R2->R2)")
            if not args.dry_run:
                sh(["rclone", "copy", f"{src}/{d}", f"{dst}/{d}",
                    "--transfers", "32", "--checkers", "32", "--stats-one-line"] + RETRY)

    print("\n=== dest size ===")
    if not args.dry_run:
        total = rclone_bytes(dst)
        print(f"r2 dest total: {total/1024**3:.1f} GiB")
        print("Now verify: python3 deploy/bake_layers.py assert-weights "
              f"--root {STAGE} --precision bf16  (the recast stage has the >60GB of bf16 weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
