"""Container disk guards (vivijure-backend#600).

A worker's model cache persists across jobs on purpose, so a warm worker reuses its weights. On a small disk one
large pull (the Wan i2v set is about 114 GiB, the stills image itself about 29 GiB) can fill it, and then every later
job on that worker fails within seconds on ENOSPC: one full worker eats the queue. Three guards stop that:

1. Before a model pull, its size is compared with the free space at its destination. A pull that cannot fit fails the
   job with the numbers instead of writing until the disk is full (`check_pull_fits`, called from models_mirror).
2. At job start, free space below a floor fails the job with the numbers (`check_job_floor`).
3. Either failure, or an ENOSPC raised anywhere in a job, makes the handler RETURN the error with RunPod's
   `refresh_worker` flag (see `is_disk_failure`), so RunPod replaces the worker after this one job.

Every function here is pure or best-effort apart from the two checks, which raise `DiskSpaceError` by design.
"""
from __future__ import annotations

import errno
import shutil
from pathlib import Path

GIB = 1024 ** 3

# Free space a job keeps for its own I/O (bundle, staged LoRAs, outputs) beyond any pull, and the floor below which a
# job does not start. A keyframe job's I/O is well under 1 GiB; 4 GiB leaves room for a film's clips and assembly.
DEFAULT_MIN_FREE_GB = 4.0


class DiskSpaceError(RuntimeError):
    """The container disk cannot hold what this job needs. The message names the numbers, in GiB."""


def free_bytes(path: Path | str) -> int:
    """Free bytes on the filesystem holding `path`. A path that does not exist yet is measured at its nearest
    existing parent, since that is where it will be created."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def free_gb(path: Path | str) -> float | None:
    """Free GiB at `path`, rounded to 0.1, for a progress event. None when it cannot be measured: an event must never
    fail a job, and a missing figure must not read as zero."""
    try:
        return round(free_bytes(path) / GIB, 1)
    except OSError:
        return None


def min_free_bytes(env: dict | None = None) -> int:
    """The floor, from VJ_MIN_FREE_GB when it is a non-negative number, else DEFAULT_MIN_FREE_GB."""
    raw = (env or {}).get("VJ_MIN_FREE_GB")
    try:
        gb = float(raw) if raw is not None else DEFAULT_MIN_FREE_GB
    except (TypeError, ValueError):
        gb = DEFAULT_MIN_FREE_GB
    return int(max(0.0, gb) * GIB)


def check_job_floor(path: Path | str, env: dict | None = None) -> None:
    """Raise DiskSpaceError when free space at `path` is below the floor."""
    free, floor = free_bytes(path), min_free_bytes(env)
    if free < floor:
        raise DiskSpaceError(f"worker disk nearly full: {free / GIB:.1f} GiB free at {path}, "
                             f"below the {floor / GIB:.1f} GiB a job needs to start")


def check_pull_fits(leg: str, needed: int, dest: Path | str, env: dict | None = None) -> None:
    """Raise DiskSpaceError when `needed` bytes plus the floor do not fit in the free space at `dest`. The floor stays
    free after the pull, for the job's own I/O."""
    free, floor = free_bytes(dest), min_free_bytes(env)
    if needed + floor > free:
        raise DiskSpaceError(f"model pull {leg!r} does not fit: needs {needed / GIB:.1f} GiB plus "
                             f"{floor / GIB:.1f} GiB for the job, {free / GIB:.1f} GiB free at {dest}")


def is_disk_failure(exc: BaseException | None) -> bool:
    """Whether `exc`, or anything in its cause or context chain, is a DiskSpaceError or an ENOSPC OSError. The chain
    is walked because boto3 and the mirror can wrap the OSError; a cycle in the chain is not followed twice."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, DiskSpaceError):
            return True
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            return True
        exc = exc.__cause__ or exc.__context__
    return False
