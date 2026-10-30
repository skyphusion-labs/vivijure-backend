"""Copy the weights a stills job reads from a global volume to container disk.

WHY. Measured on film-a9939630: from_pretrained read the 9.99 GiB fp32 UNet shard off the global
volume in 404 s, about 26 MB/s, and one job reads the same files twice (LoRA training, then the
keyframe pipeline). Container disk is free and far faster, and a warm worker keeps it.

THE TREE IS ALWAYS COMPLETE. The local hub starts as symlinks into the volume, and a copied file
replaces its symlink atomically. So a time cap, a full disk or a read error leaves a slower job,
never a missing file. A repo with no rule is one symlink. Nothing here can fail a render.

Parallel ranged reads are used because a single mmap stream is what was measured slow. Whether they
scale on this store is not known until a live run, so every file reports its own MB/s.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

CHUNK_BYTES = 64 * 1024 * 1024
DEFAULT_THREADS = 8
DEFAULT_MAX_SECONDS = 600.0
FREE_MARGIN_BYTES = 5 * 1024 ** 3
SMALL_FILE_BYTES = 8 * 1024 * 1024
PROGRESS_SECONDS = 30.0
SENTINEL = ".vj-hydrated"
WEIGHT_SUFFIXES = (".safetensors", ".bin")

_DIFFUSERS_WEIGHT = r"diffusion_pytorch_model(-\d+-of-\d+)?\.safetensors"
_TRANSFORMERS_WEIGHT = r"model(-\d+-of-\d+)?\.safetensors"

# Hydrated first to last, so a time cap keeps the files every SDXL job reads. Each pattern names
# exactly what the loaders open: no `variant=` is passed, so the fp16 files are never read, and a
# repo's root single-file checkpoint is never read. Anything unmatched stays a symlink and still works.
HYDRATE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("models--SG161222--RealVisXL_V5.0", (
        rf"unet/{_DIFFUSERS_WEIGHT}", rf"vae/{_DIFFUSERS_WEIGHT}",
        rf"text_encoder/{_TRANSFORMERS_WEIGHT}", rf"text_encoder_2/{_TRANSFORMERS_WEIGHT}",
    )),
    ("models--xinsir--controlnet-openpose-sdxl-1.0", (_DIFFUSERS_WEIGHT,)),
    ("models--ByteDance--Hyper-SD", (r"Hyper-SDXL-8steps-lora\.safetensors",)),
    ("models--h94--IP-Adapter", (
        r"sdxl_models/ip-adapter_sdxl\.safetensors",
        rf"sdxl_models/image_encoder/{_TRANSFORMERS_WEIGHT}",
    )),
    ("models--xinsir--controlnet-canny-sdxl-1.0", (_DIFFUSERS_WEIGHT,)),
    ("models--InstantX--InstantID", (r"ip-adapter\.bin",)),
)

Emit = Callable[..., None]


def hydrate_enabled(env: dict) -> bool:
    """`VJ_HYDRATE` wins when set. Otherwise only a global volume hydrates: it is the store measured
    slow, and a datacenter network volume has not shown the same read cost."""
    raw = str(env.get("VJ_HYDRATE", "")).strip().lower()
    if raw:
        return raw not in ("0", "false", "no", "off")
    return str(env.get("VJ_VOLUME_KIND", "")).strip().lower() == "global"


def _snapshot_relpath(rel: str) -> str | None:
    """`snapshots/<rev>/a/b` -> `a/b`; None for anything outside a snapshot."""
    parts = rel.split("/")
    if len(parts) < 3 or parts[0] != "snapshots":
        return None
    return "/".join(parts[2:])


def should_hydrate(rel_in_repo: str, patterns: tuple[str, ...]) -> bool:
    inner = _snapshot_relpath(rel_in_repo)
    return inner is not None and any(re.fullmatch(p, inner) for p in patterns)


def _link(target: Path, dest: Path) -> None:
    if dest.is_symlink() or dest.exists():
        return
    os.symlink(target, dest)


def build_tree(volume_hf: Path, local_hf: Path) -> list[tuple[Path, Path, int]]:
    """Make the complete local tree. Returns (source, dest, size) of each file to hydrate, in rule
    order, largest first inside a repo. A file already copied keeps its real file."""
    local_hub = local_hf / "hub"
    local_hub.mkdir(parents=True, exist_ok=True)
    for entry in volume_hf.iterdir():
        if entry.name != "hub":
            _link(entry, local_hf / entry.name)
    rules = dict(HYDRATE_RULES)
    plan: list[tuple[Path, Path, int]] = []
    volume_hub = volume_hf / "hub"
    for repo_dir in sorted(volume_hub.iterdir()):
        if repo_dir.name not in rules or not repo_dir.is_dir():
            _link(repo_dir, local_hub / repo_dir.name)
            continue
        for root, dirs, files in os.walk(repo_dir):
            rel_root = Path(root).relative_to(repo_dir)
            (local_hub / repo_dir.name / rel_root).mkdir(parents=True, exist_ok=True)
            for name in files:
                src = Path(root) / name
                dest = local_hub / repo_dir.name / rel_root / name
                size = src.stat().st_size
                rel = (rel_root / name).as_posix()
                if should_hydrate(rel, rules[repo_dir.name]):
                    _link(src, dest)
                    if dest.is_symlink() or dest.stat().st_size != size:
                        plan.append((src, dest, size))
                elif size <= SMALL_FILE_BYTES and not name.endswith(WEIGHT_SUFFIXES):
                    if not dest.exists():
                        tmp = dest.with_name(dest.name + ".part")
                        shutil.copyfile(src, tmp)
                        os.replace(tmp, dest)
                else:
                    _link(src, dest)
    order = {name: i for i, (name, _) in enumerate(HYDRATE_RULES)}
    plan.sort(key=lambda p: (order[p[1].relative_to(local_hub).parts[0]], -p[2]))
    return plan


class _Stop(Exception):
    """The time cap passed. Not an error: the rest stays on the volume."""


def _copy_file(src: Path, dest: Path, size: int, pool: ThreadPoolExecutor, deadline: float,
               on_bytes: Callable[[int], None], chunk_bytes: int) -> None:
    """Parallel ranged copy to `<dest>.part`, then an atomic replace of the symlink."""
    part = dest.with_name(dest.name + ".part")
    src_fd = os.open(src, os.O_RDONLY)
    dst_fd = os.open(part, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    done = False
    try:
        os.ftruncate(dst_fd, size)

        def one(offset: int) -> None:
            if time.monotonic() > deadline:
                raise _Stop()
            want = min(chunk_bytes, size - offset)
            got = 0
            while got < want:
                data = os.pread(src_fd, want - got, offset + got)
                if not data:
                    raise OSError(f"short read at {offset + got} of {src}")
                os.pwrite(dst_fd, data, offset + got)
                got += len(data)
            on_bytes(want)

        for future in [pool.submit(one, o) for o in range(0, size, chunk_bytes)]:
            future.result()
        done = True
    finally:
        os.close(src_fd)
        os.close(dst_fd)
        if not done:
            try:
                part.unlink()
            except OSError:
                pass
    if part.stat().st_size != size:
        part.unlink()
        raise OSError(f"size mismatch copying {src}")
    os.replace(part, dest)


def hydrate(volume_hf: Path, local_hf: Path, emit: Emit, *, threads: int = DEFAULT_THREADS,
            max_seconds: float = DEFAULT_MAX_SECONDS, chunk_bytes: int = CHUNK_BYTES,
            free_bytes: Callable[[Path], int] | None = None,
            progress_seconds: float = PROGRESS_SECONDS) -> bool:
    """Fill `local_hf` from `volume_hf`. True iff every planned file is now local."""
    sentinel = local_hf / SENTINEL
    if sentinel.is_file():
        emit("hydrate_done", files=0, bytes=0, seconds=0.0, mb_per_s=None, complete=True, reason="warm")
        return True
    started = time.monotonic()
    plan = build_tree(volume_hf, local_hf)
    total = sum(p[2] for p in plan)
    emit("hydrate_start", files=len(plan), bytes=total, threads=threads)
    deadline = started + max_seconds
    free = free_bytes or (lambda p: shutil.disk_usage(p).free)
    copied_bytes = 0
    lock = threading.Lock()
    last_report = [time.monotonic()]

    def on_bytes(n: int) -> None:
        nonlocal copied_bytes
        with lock:
            copied_bytes += n
            now = time.monotonic()
            if now - last_report[0] < progress_seconds:
                return
            last_report[0] = now
            so_far = copied_bytes
        emit("hydrate_progress", bytes=so_far, total=total, elapsed_s=round(now - started, 1))

    reason = None
    files = 0
    with ThreadPoolExecutor(max_workers=max(1, threads), thread_name_prefix="hydrate") as pool:
        for src, dest, size in plan:
            if free(local_hf) < size + FREE_MARGIN_BYTES:
                reason = "disk"
                break
            file_started = time.monotonic()
            try:
                _copy_file(src, dest, size, pool, deadline, on_bytes, chunk_bytes)
            except _Stop:
                reason = "time_cap"
                break
            except OSError as e:
                reason = f"read_error:{type(e).__name__}"
                break
            files += 1
            secs = time.monotonic() - file_started
            emit("hydrate_file", file=dest.relative_to(local_hf).as_posix(), bytes=size,
                 seconds=round(secs, 2), mb_per_s=round(size / 1e6 / secs, 1) if secs > 0 else None)
    seconds = time.monotonic() - started
    complete = reason is None
    if complete:
        sentinel.write_text(json.dumps({"bytes": total}))
    emit("hydrate_done", files=files, bytes=copied_bytes, seconds=round(seconds, 2),
         mb_per_s=round(copied_bytes / 1e6 / seconds, 1) if seconds > 0 else None,
         complete=complete, reason=reason)
    return complete


def maybe_hydrate(emit: Emit, env: dict | None = None) -> str:
    """The job's entry point. Repoints `HF_HOME` at the local tree when it is usable.

    Returns what happened, for the test and the event: `disabled`, `no_volume`, `too_late`,
    `failed`, or `local`. A partial hydrate is still `local`: the tree is complete."""
    e = env if env is not None else os.environ
    if not hydrate_enabled(e):
        return "disabled"
    vol = e.get("VJ_VOLUME_ROOT")
    hf_home = e.get("HF_HOME")
    local_root = Path(e.get("VJ_HYDRATE_ROOT") or "/tmp/vj-hydrate")
    local_hf = local_root / "hf-cache"
    volume_hf = Path(vol) / "hf-cache" if vol else None
    if volume_hf is None or hf_home not in (str(volume_hf), str(local_hf)) or not volume_hf.is_dir():
        return "no_volume"
    # huggingface_hub reads HF_HOME once, at import. Repointing after that would split the job
    # between two caches, so an imported hub must already agree, or the job stays on the volume.
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None and not str(getattr(constants, "HF_HUB_CACHE", "")).startswith(str(local_hf)):
        emit("hydrate_skipped", reason="hub_already_imported")
        return "too_late"
    try:
        threads = int(e.get("VJ_HYDRATE_THREADS") or DEFAULT_THREADS)
        max_seconds = float(e.get("VJ_HYDRATE_MAX_SECONDS") or DEFAULT_MAX_SECONDS)
        hydrate(volume_hf, local_hf, emit, threads=threads, max_seconds=max_seconds)
    except Exception as exc:  # noqa: BLE001 -- the volume still serves this job
        e["HF_HOME"] = str(volume_hf)
        emit("hydrate_failed", error=type(exc).__name__, message=str(exc)[:300])
        return "failed"
    e["HF_HOME"] = str(local_hf)
    return "local"
