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
import re


def _slug(project: str) -> str:
    """A project name reduced to an R2-safe path segment: keep it from smuggling a slash or
    whitespace into a key (which would scatter a render across phantom prefixes)."""
    return "_".join(str(project).strip().split()).replace("/", "_") or "untitled"


def output_key(project: str) -> str:
    """The final muxed MP4 the control plane polls for."""
    return f"renders/{_slug(project)}/full.mp4"


def lora_key(project: str, slot: str) -> str:
    """A trained character adapter, by slot."""
    return f"loras/{_slug(project)}/{_slug(slot)}/pytorch_lora_weights.safetensors"


def wan_lora_key(project: str, slot: str, expert: str) -> str:
    """A trained Wan 2.2 A14B character adapter, by slot and MoE expert (`high` / `low`).

    Wan A14B is a two-expert mixture, so a Wan character LoRA is TWO files (a high-noise and a
    low-noise adapter) that the render module feeds to `high_noise_loras` / `low_noise_loras`.
    Each expert lands at its own key beside the single-file SDXL `lora_key`. `expert` is pinned to
    {high, low} so a typo can never scatter a third phantom object into the project prefix."""
    e = str(expert).strip().lower()
    if e not in ("high", "low"):
        raise ValueError(f"wan_lora_key: expert must be 'high' or 'low', got {expert!r}")
    return f"loras/{_slug(project)}/{_slug(slot)}/wan_{e}_noise.safetensors"


def keyframe_key(project: str, shot_id: str) -> str:
    """A rendered SDXL keyframe, by shot."""
    return f"renders/{_slug(project)}/keyframes/{_slug(shot_id)}.png"


def keyframe_hash_key(project: str, shot_id: str) -> str:
    """The param-hash sidecar for a keyframe (#112): written next to the PNG so the next
    incremental render can decide reuse-vs-regenerate per shot straight from R2. Per-artifact
    keys are what replaced the shared projects/<slug>/state.tar.gz -- concurrent shards write
    disjoint objects, so there is no shared mutable state left to race on."""
    return f"renders/{_slug(project)}/keyframes/{_slug(shot_id)}.hash"


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


def bundle_key_matches_project(bundle_key: str, project: str) -> bool:
    """True when a job-supplied bundle_key belongs to the named project (defense-in-depth on the
    shared bucket: prefix-only checks are not enough). Accepts the flat, content-addressed, and
    nested bundle layouts the control plane emits."""
    slug = _slug(project)
    if not bundle_key.startswith("bundles/"):
        return False
    rest = bundle_key[len("bundles/"):]
    if rest.startswith(f"{slug}/"):
        return True
    if rest == f"{slug}.tar.gz":
        return True
    return bool(re.fullmatch(re.escape(slug) + r"-[0-9a-f]{16}\.tar\.gz", rest))


def check_bundle_key_for_project(bundle_key: str, project: str, *, what: str) -> str:
    """Validate bundle_key shape AND project tenancy before any store I/O."""
    k = check_job_key(bundle_key, prefixes=("bundles/",), what=what)
    if not bundle_key_matches_project(k, project):
        slug = _slug(project)
        raise ValueError(
            f"{what}: bundle_key {k!r} must belong to project {project!r} "
            f"(expected bundles/{slug}/..., bundles/{slug}.tar.gz, or "
            f"bundles/{slug}-<contenthash>.tar.gz)")
    return k


def is_cast_registry_lora_key(key: str) -> bool:
    """True when ``key`` is a cast-banked LoRA under the global cast registry layout.

    Cast adapters intentionally live outside the render project slug: the control plane resolves
    opaque cast ids to these keys (``resolveCastLoras``) and passes them as ``pretrained_loras``.
    Accept only the registry shapes the studio writes; arbitrary ``loras/<other-project>/`` paths
    remain blocked by ``check_scoped_job_key``.
    """
    if not key.startswith("loras/"):
        return False
    rest = key[len("loras/"):]
    # SDXL banked adapter: loras/cast-{id}/{timestamp}.safetensors (deriveLoraDestKey).
    if re.fullmatch(r"cast-\d+/[^/]+\.safetensors", rest):
        return True
    # Render/train output keyed by cast slug: loras/lora-{slug}-{timestamp}/A/...
    if rest.startswith("lora-") and "/" in rest:
        head = rest.split("/", 1)[0]
        if re.fullmatch(r"lora-[a-zA-Z0-9_-]+", head):
            return True
    return False


def check_scoped_job_key(key: str, *, project: str, prefixes: tuple[str, ...], what: str) -> str:
    """Validate a job-supplied read key is under the project slug within its prefix."""
    k = check_job_key(key, prefixes=prefixes, what=what)
    slug = _slug(project)
    if k.startswith("audio/"):
        # Studio beds are flat audio/<uuid>.<ext> (vivijure-cf upload route). No project prefix,
        # but reject nested paths so audio_key cannot smuggle renders/ or bundles/ reads.
        rest = k[len("audio/"):]
        if "/" in rest or not rest:
            raise ValueError(
                f"{what}: flat audio bed key {k!r} must be audio/<filename> with no extra slashes")
        return k
    if k.startswith("renders/") and not k.startswith(f"renders/{slug}/"):
        raise ValueError(
            f"{what}: R2 key {k!r} must be under renders/{slug}/ for project {project!r}")
    if k.startswith("loras/") and not k.startswith(f"loras/{slug}/"):
        if is_cast_registry_lora_key(k):
            return k
        raise ValueError(
            f"{what}: R2 key {k!r} must be under loras/{slug}/ for project {project!r}")
    return k


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


def job_done_error_key(project: str, job_id: str) -> str:
    """The run-scoped NDJSON log of RunPod /job-done POST rejections for one render (#90).

    Colocated with the render's other progress objects and keyed by project AND job id the same
    way, but a DISTINCT object: it must NEVER clobber the live progress snapshot the control plane
    polls. The SDK's job-done post (a late status mirror OR the terminal result) can be rejected
    (observed: a 400 on a successful job) with the reason printed only to worker stdout, which is
    not retrievable via GraphQL / runpodctl / MCP; mirroring each rejection here makes it
    inspectable in-band. One record appended per rejected post."""
    return f"renders/{_slug(project)}/progress/{_slug(job_id)}.job-done-errors.ndjson"


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
