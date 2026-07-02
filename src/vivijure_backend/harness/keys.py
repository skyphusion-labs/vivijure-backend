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


def clip_key(project: str, shot_id: str) -> str:
    """A per-shot i2v clip, by shot (the offloaded/per-shot finish emits these)."""
    return f"renders/{_slug(project)}/clips/{_slug(shot_id)}.mp4"


def i2v_clip_key(project: str, shot_id: str) -> str:
    """The standalone i2v_clip job's output clip (run_i2v_clip_job). Same _slug as every other
    key so one project never scatters across two slug spellings of its own name."""
    return f"renders/{_slug(project)}/clips/{_slug(shot_id)}_i2v.mp4"


def finished_clip_key(project: str, shot_id: str) -> str:
    """The standalone finish_clip job's output clip (run_finish_job). Same _slug rationale."""
    return f"renders/{_slug(project)}/clips/{_slug(shot_id)}_finished.mp4"


def check_job_key(key: str, *, prefixes: tuple[str, ...], what: str) -> str:
    """Validate a JOB-SUPPLIED R2 key before any store I/O.

    The job names WHERE the worker reads (bundle_key, audio_key, pretrained_loras, the standalone
    jobs' clip_key/keyframe_key); this pins that choice to the render key map, so a malformed or
    mis-scoped key fails loud BEFORE any transfer instead of pointing store I/O at an arbitrary
    bucket path. Pure string checks, raises ValueError naming the purpose (the caller wraps it in
    its own error type). Requirements: non-empty, no surrounding whitespace, relative (no leading
    /), forward slashes only, no `..` segment, and under one of the allowed prefixes."""
    k = str(key or "")
    ok = (
        bool(k)
        and k == k.strip()
        and not k.startswith("/")
        and "\\" not in k
        and ".." not in k.split("/")
        and k.startswith(prefixes)
    )
    if not ok:
        raise ValueError(
            f"{what}: R2 key {k!r} must be a plain relative key under "
            f"{' or '.join(prefixes)} (see the render key map)")
    return k


def progress_log_key(project: str, job_id: str) -> str:
    """The append-only NDJSON event stream for one render, keyed by project AND job id so
    concurrent or cancelled runs of the same project never clobber each other."""
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.ndjson"


def progress_snapshot_key(project: str, job_id: str) -> str:
    """The latest-state JSON snapshot for one render (the cheap thing a /status route or Uptime
    Kuma polls), keyed the same way."""
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.json"


def _verify_run(run_id: str) -> str:
    """A verify run id reduced to an R2-safe path segment (same discipline as _slug): a verify
    run must never smuggle a slash or whitespace into its key and scatter its channel."""
    return "_".join(str(run_id).strip().split()).replace("/", "_") or "unkeyed"


def verify_events_key(run_id: str, *, prefix: str = "verify") -> str:
    """The run-scoped NDJSON event stream a pod-side verify run writes (one JSON record per line).
    Run-scoped, not project-scoped: a verify run is a build-gate probe, not a project render, so it
    keys off its own run id under a dedicated `verify/` prefix and never collides with renders/."""
    return f"{_verify_run(prefix)}/{_verify_run(run_id)}/events.ndjson"


def verify_summary_key(run_id: str, *, prefix: str = "verify") -> str:
    """The run-scoped latest-state JSON snapshot for a verify run -- the cheap object the release
    gate polls until status is terminal, then reads the events array from. Same key rationale."""
    return f"{_verify_run(prefix)}/{_verify_run(run_id)}/summary.json"


def join(*parts: str) -> str:
    """POSIX-join key parts (R2 keys are always forward-slash, regardless of worker OS)."""
    return posixpath.join(*parts)
