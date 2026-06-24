"""R2 object-key layout for a render.

Every key the worker reads or writes is defined here, in one place, so the scheme stays
consistent and stays aligned with the control plane's artifact routes (the planner UI fetches
these exact keys back). The inbound `bundle_key` is chosen by the control plane and passed in
the job; everything the worker produces is keyed off the project name by these helpers.

Pure string building, no I/O: trivially testable, and the single source of truth a deploy can
audit against the control plane.
"""
from __future__ import annotations

import posixpath


def _slug(project: str) -> str:
    """A project name reduced to an R2-safe path segment: keep it from smuggling a slash or
    whitespace into a key (which would scatter a render across phantom prefixes)."""
    return "_".join(str(project).strip().split()).replace("/", "_") or "untitled"


def output_key(project: str) -> str:
    """The final muxed MP4 the control plane polls for."""
    return f"renders/{_slug(project)}/full.mp4"


def state_key(project: str) -> str:
    """The project tree tarball for incremental re-render (contents-at-root; see r2.upload_state)."""
    return f"projects/{_slug(project)}/state.tar.gz"


def lora_key(project: str, slot: str) -> str:
    """A trained character adapter, by slot."""
    return f"loras/{_slug(project)}/{_slug(slot)}/pytorch_lora_weights.safetensors"


def keyframe_key(project: str, shot_id: str) -> str:
    """A rendered SDXL keyframe, by shot."""
    return f"renders/{_slug(project)}/keyframes/{_slug(shot_id)}.png"


def keyframe_hash_key(project: str, shot_id: str) -> str:
    """The render-param hash for a keyframe, as its OWN per-shot R2 object (sibling of the PNG). The
    incremental restore reads it to detect a param change without a shared state tarball -- so
    concurrent scatter shards never clobber a single mutable state object (#112)."""
    return f"renders/{_slug(project)}/keyframes/{_slug(shot_id)}.hash"


def clip_key(project: str, shot_id: str) -> str:
    """A per-shot i2v clip, by shot (the offloaded/per-shot finish emits these)."""
    return f"renders/{_slug(project)}/clips/{_slug(shot_id)}.mp4"


def progress_log_key(project: str, job_id: str) -> str:
    """The append-only NDJSON event stream for one render, keyed by project AND job id so
    concurrent or cancelled runs of the same project never clobber each other."""
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.ndjson"


def progress_snapshot_key(project: str, job_id: str) -> str:
    """The latest-state JSON snapshot for one render (the cheap thing a /status route or Uptime
    Kuma polls), keyed the same way."""
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.json"


def join(*parts: str) -> str:
    """POSIX-join key parts (R2 keys are always forward-slash, regardless of worker OS)."""
    return posixpath.join(*parts)
