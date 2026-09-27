"""Cold-start model mirror: populate the worker's local HF cache from R2 at startup.

Why pull at cold start instead of baking weights into the image: the full set is ~hundreds of
GB; a layer that size is rejected by GHCR and ingests slowly, while R2 has no layer limit and
is fast and already seeded. So the image stays tiny and a cold worker mirrors what it needs
from `r2:<bucket>/models` with `rclone --links` (faithfully reconstructing the HF cache); a
warm worker sees the completion sentinel and skips. The same R2 token does double duty here
and for job I/O; the worker holds no other credential.

The sentinel is written only after the pull fully succeeds, so a worker killed mid-pull leaves
no marker and re-runs (rclone copy is idempotent) rather than rendering against a half-mirror.
The rclone command is built by a pure helper so it tests without spawning anything.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Callable

# Module-level handle for the eager i2v prefetch thread (see start_i2v_prefetch).
# ensure_i2v_models joins it before checking the sentinel so warm behaviour applies whether
# the pull finished in the background or still in progress when i2v_pipeline is called.
_i2v_prefetch_thread: threading.Thread | None = None

# The heavy i2v repos (Wan I2V + the Lightning distill, ~120GB), pulled LAZILY by
# ensure_i2v_models() on the first i2v_pipeline() use. A keyframe/preview worker -- the common cheap
# op -- never calls it, so it never pulls them. These are folded into DEFAULT_SKIP_REPOS below so the
# cold-start pull always excludes exactly what the lazy path owns (no double-pull, no miss).
I2V_LAZY_REPOS = (
    "models--Wan-AI--Wan2.2-I2V-A14B-Diffusers",
    "models--lightx2v--Wan2.2-Lightning",
)

# Repos NOT pulled at cold start: the lazy i2v repos above, plus dead weight nothing in the model
# spec loads (T2V is never used; the stray SDXL repos are not in the spec). Storage in R2 is kept
# (cheap, safer); these are pull-time excludes only. Tune per the live model set, the deploy's call.
# This list MUST agree with the bake keep/drop set (deploy/bake-manifest.json): runtime and bake
# derive the same model set from the same source, so neither pulls/bakes a repo the other ignores.
DEFAULT_SKIP_REPOS = I2V_LAZY_REPOS + (
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers",          # text-to-video: never loaded
    "models--stabilityai--stable-diffusion-xl-base-1.0",  # not in the model spec (dead weight)
    "models--stabilityai--sdxl-turbo",                    # not in the model spec (dead weight)
    "models--cagliostrolab--animagine-xl-4.0",            # anime alt in a note only; not loaded (#13)
    "spaces--InstantX--InstantID",                        # the HF Space, not the model repo
)

# Separate completion sentinel for the lazy i2v pull (the cold-start pull has its own, SENTINEL).
I2V_SENTINEL = ".vj-i2v-mirror-complete"

# HF's abandoned download temp files; model-presence checks treat any *.incomplete as a broken
# repo, so never mirror them.
_INCOMPLETE_GLOB = "**/*.incomplete"

# Written under the models root only after the pull fully succeeds (see module docstring).
SENTINEL = ".vj-mirror-complete"

# Baked-image marker: present at VJ_MODELS_ROOT/.vj-baked when the model weights are BAKED INTO THE
# IMAGE. The marker says BAKED and nothing about which precision was baked; that lives in the stamp's
# own `precision=` line, which is the only thing entitled to answer it. It is written at build time,
# NOT by any pull. When present, every mirror entry point early-returns BEFORE touching the volume
# resolve or the R2 pull: a baked worker carries its weights, so it is datacenter-agnostic (no network
# volume pinning it to a provisioned DC) and never pays the R2 cold-pull tax. The R2 mirror stays the
# fallback ONLY when this marker is absent (a non-baked / legacy image), so this is purely additive.
# B (bf16-lazy-on-final) is the one exception that still pulls: the FINAL-tier i2v path deliberately
# fetches bf16 shards from R2 even on a baked worker (see models.ModelServer.i2v_pipeline); that pull
# is gated on the tier, not on this marker, so the baked default stays pull-free.
BAKED_SENTINEL = ".vj-baked"


# --------------------------------------------------------------------------- mirror credential
#
# The mirror's credential is NOT the tenant's. The backend uses R2 for two unrelated purposes with
# OPPOSITE sharing requirements: this mirror pulls identical shared weights from OUR bucket, while
# tenant job I/O (harness/r2.py) reads and writes THE TENANT's bucket.
#
# They used to be the same four environment names, which meant one credential and one bucket doing
# both jobs. That is not merely untidy. The control plane's `templateEnv` (vivijure-control-plane
# src/runpod.ts) sets `R2_BUCKET` to the TENANT's bucket, so on a provisioned tenant endpoint the
# weights mirror is pointed at a bucket with no `models/` prefix. It is masked today by the baked
# image and volume sentinels short-circuiting before any pull, not by being correct.
#
# So the mirror reads its OWN namespaced names first, and falls back to the legacy shared names for
# endpoints provisioned before the split. The legacy fallback is TRANSITIONAL and its removal is
# tracked: while it exists, an endpoint that carries the legacy names still has a usable tenant
# job-I/O credential in its environment, which is the fallback path a pooled endpoint must not have.
MIRROR_ENV = ("MODELS_R2_ENDPOINT", "MODELS_R2_ACCESS_KEY_ID", "MODELS_R2_SECRET_ACCESS_KEY",
              "MODELS_R2_BUCKET")
LEGACY_ENV = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

# The one name whose PRESENCE marks an endpoint as provisioned after the credential split. See
# `harness.r2.R2Config.from_payload_or_env`: on such an endpoint the tenant job-I/O fallback to the
# environment is refused, because the endpoint may serve more than one tenant.
MIRROR_MARKER_ENV = "MODELS_R2_ACCESS_KEY_ID"


def uses_namespaced_mirror_creds(env: dict | None = None) -> bool:
    """True iff this endpoint carries the mirror's OWN credential, i.e. it was provisioned after the
    credential split. Read by the tenant job-I/O config to decide whether an environment fallback is
    still allowed; see `harness.r2.R2Config.from_payload_or_env`."""
    e = env if env is not None else os.environ
    return bool(e.get(MIRROR_MARKER_ENV))


def mirror_env(env: dict | None = None) -> dict:
    """The environment the mirror should read, with its own credential resolved under the LEGACY
    names so every downstream reader (`rclone_env`, the bucket reads, the presence gates) is
    unchanged.

    `MODELS_R2_*` wins field by field; each field falls back to its legacy `R2_*` counterpart. The
    fallback deliberately preserves today's behaviour on an endpoint that has not been repinned,
    INCLUDING the tenant-bucket defect described above: silently repointing a live endpoint's weights
    pull at a different bucket would be an unrequested behaviour change on production. The defect is
    fixed by setting `MODELS_R2_BUCKET`, not by this function guessing."""
    e = env if env is not None else os.environ
    out = dict(e)
    for own, legacy in zip(MIRROR_ENV, LEGACY_ENV):
        value = e.get(own) or e.get(legacy)
        if value:
            out[legacy] = value
    return out


def is_baked(env: dict | None = None) -> bool:
    """True iff the model weights are baked into the image (the BAKED_SENTINEL marker is present at
    VJ_MODELS_ROOT). Pure: a filesystem check, no I/O beyond stat. Callers use it to skip every
    volume/R2 path. A baked worker is datacenter-agnostic and never pulls (except B's final-tier
    bf16 lazy-pull, which is gated on the render tier, not on this)."""
    e = env if env is not None else os.environ
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    return (models_root / BAKED_SENTINEL).exists()


def repo_in_hf_cache(repo_id: str, env: dict | None = None) -> bool:
    """True iff `repo_id` is offline-loadable from the local HF hub cache: the `models--<org>--<name>`
    dir exists AND holds a non-empty snapshot. A bare cache dir with no populated snapshot still raises
    LocalEntryNotFoundError under `local_files_only=True`, so a from_pretrained caller must gate on THIS,
    not on the dir alone. Pure: stat only, no I/O. (The sm_120 de-risk hit exactly this -- the runtime
    asked for a repo that was never baked, and a glob-only baked_probe masked it.)"""
    e = env if env is not None else os.environ
    hf_home = Path(e.get("HF_HOME") or Path(e.get("VJ_MODELS_ROOT", "/opt/models")) / "hf-cache")
    snaps = hf_home / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return False
    return any(p.is_dir() and any(p.iterdir()) for p in snaps.iterdir())


# Bump this constant (or set VJ_MODEL_VERSION in the worker env) whenever the model set
# in R2 changes so warm workers re-pull instead of silently using a stale cache.
_DEFAULT_MODEL_VERSION = "1"


def rclone_env(env: dict) -> dict:
    """Build the child-process environment that configures rclone's `r2:` remote ENTIRELY via
    RCLONE_CONFIG_* env vars, so the R2 secret is passed to the rclone subprocess in its environment
    and NEVER written to a config file on disk (py/clear-text-storage-sensitive-data). Raises if creds
    are incomplete so the worker fails here, loudly, not later at the model-presence gate.

    Returns a COPY of the current process env with the RCLONE_CONFIG_R2_* keys overlaid, suitable to
    pass as subprocess(..., env=...). PATH etc. are preserved (so rclone resolves) and the secret lives
    only in process/child memory. The remote name is "r2", matching the r2: prefix every leg uses."""
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT") if not env.get(k)]
    if missing:
        raise RuntimeError("models_mirror: incomplete R2 creds; missing: " + ", ".join(missing))
    child = dict(os.environ)
    child.update({
        "RCLONE_CONFIG_R2_TYPE": "s3",
        "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_R2_ACCESS_KEY_ID": env["R2_ACCESS_KEY_ID"],
        "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": env["R2_SECRET_ACCESS_KEY"],
        "RCLONE_CONFIG_R2_ENDPOINT": env["R2_ENDPOINT"],
        "RCLONE_CONFIG_R2_ACL": "private",
        "RCLONE_CONFIG_R2_NO_CHECK_BUCKET": "true",
    })
    return child


def mirror_cmd(src: str, dst: Path, *, skip_repos: tuple[str, ...] = ()) -> list[str]:
    """argv for one `rclone copy --links` mirror leg. The r2: remote resolves from the RCLONE_CONFIG_*
    child env (see rclone_env); there is no rclone config file, so the R2 secret never touches disk.
    Pure: built and asserted without rclone."""
    cmd = ["rclone", "copy", "--links",
           "--transfers", "16", "--checkers", "16",
           "--multi-thread-streams", "8", "--multi-thread-cutoff", "100M",
           "--stats", "60s", "--stats-one-line", "-v",
           "--exclude", _INCOMPLETE_GLOB]
    for repo in skip_repos:
        cmd += ["--exclude", f"hub/{repo}/**"]
    cmd += [src, str(dst)]
    return cmd


# ----------------------------------------------------------- cold-start telemetry (issue #55)
# Cold-start staging cost was only ever measured by SSHing into live pods (the 2026-06-13 load
# test). These helpers emit it as structured @event lines so the staging-time distribution
# across workers/deploys is readable from logs, which is the foundation for choosing a
# structural fix (bake vs pre-warm vs stage). Pure/best-effort: never abort a pull.

def _timed(fn: Callable, *args, **kwargs) -> float:
    """Run fn(*args, **kwargs) and return its wall-clock seconds (monotonic)."""
    t0 = time.monotonic()
    fn(*args, **kwargs)
    return time.monotonic() - t0


def _dir_bytes(path: Path) -> int:
    """Sum sizes of real files (not symlinks) under path. After symlink reconstruction the HF
    cache holds each weight once as a blob (snapshots are symlinks), so this counts true bytes
    pulled without double-counting. Best-effort: a stat error on one entry never raises."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _mirror_event(legs: list[tuple[str, float]], total_bytes: int, *, cold: bool,
                  model_version: str, event: str = "mirror_complete") -> str:
    """Build an @event telemetry line for a completed mirror run (pure; no disk/rclone).

    `legs` is a list of (leg_name, seconds). Emits per-leg + total timing, byte count, and
    derived throughput so a slow node (the 2026-06-13 Blackwell that took ~35 min) stands out
    against the H200 pack in the logs."""
    total_s = sum(s for _, s in legs)
    mbps = (total_bytes / 1e6 / total_s) if total_s > 0 else 0.0
    payload = {
        "cold": cold,
        "model_version": model_version,
        "total_seconds": round(total_s, 1),
        "total_bytes": total_bytes,
        "throughput_mbps": round(mbps, 1),
        "legs": {name: round(s, 1) for name, s in legs},
    }
    return "@event " + event + " " + json.dumps(payload, sort_keys=True)


def _skip_event(reason: str, *, event: str = "mirror_skipped") -> str:
    """Build an @event line for a skipped mirror (warm sentinel / no R2 creds). Counting these
    against mirror_complete across a deploy shows how often the per-deploy cold-start tax hits."""
    return "@event " + event + " " + json.dumps({"reason": reason}, sort_keys=True)


def _jitter_seconds(e: dict) -> float:
    """A bounded random pre-mirror delay (seconds) to de-stagger R2 egress when many cold workers
    fall back to the mirror at once (issue #55: the 8-way fan-out contention killer). Tunable via
    `VJ_MIRROR_JITTER_SEC` (the ceiling; default 0 = off). Factored out from the sleep so the draw
    is testable; returns 0 when off or misconfigured. Cheap insurance for the fallback path -- the
    preloaded volume is the real fix, so this defaults off."""
    try:
        ceiling = float(e.get("VJ_MIRROR_JITTER_SEC") or 0)
    except (TypeError, ValueError):
        return 0.0
    return random.uniform(0, ceiling) if ceiling > 0 else 0.0


# Job actions that can NEVER reach the Wan i2v stage. The volume readiness gate asks exactly one
# question about i2v -- "can an i2v stage run here at all" -- and the answer is already declared in
# code, so this introduces no new endpoint knob (#453):
#   - orchestrator.Action.PREVIEW     "keyframes-only preview: train -> keyframes, NO i2v, no MP4"
#   - orchestrator.Action.REGEN_SHOT  "regenerate named keyframes only, no i2v"
#   - orchestrator.Action.TRAIN_LORA  "train LoRAs only"
#   - the harness-only "finish_clip"  RIFE interpolation / face restore; never loads Wan
# A DENYLIST on purpose: an action this module has not heard of counts as i2v-capable, which is the
# conservative direction (the gate then demands the i2v corpus exactly as it always did), and it
# matches `orchestrator.Action.parse`, which also resolves an unknown value to RENDER.
#
# NOT the same question as `harness.handler._wants_i2v_prefetch`, which asks whether eagerly
# STARTING the ~120GB pull is worth it and deliberately keeps the prefetch for preview / regen_shot
# to warm a worker that may get a render next. That is a cost call; this is a correctness gate. They
# are kept apart on purpose -- merging them would make one of the two wrong.
I2V_FREE_ACTIONS = frozenset({"preview", "regen_shot", "train_lora", "finish_clip"})


def i2v_stage_possible(action: str | None) -> bool:
    """True iff a job with this `action` can reach the Wan i2v stage, and so needs the i2v corpus.
    Pure. `None` (an unnamed action) and any unrecognised action answer True: see I2V_FREE_ACTIONS
    for why the unknown case must be the demanding one."""
    return str(action if action is not None else "render").strip() not in I2V_FREE_ACTIONS


class VolumeState(Enum):
    """What the mounted network volume IS for the job in hand. A NAMED state, not a bare bool.

    `_resolve_volume` used to return True/False, which collapsed two very different situations into
    one reading: a volume preloaded with the honest post-2026-09-26 corpus (keyframes, no Wan) and a
    volume whose preload genuinely did not finish. Both surfaced as False plus a log line reading
    "not fully preloaded", i.e. "somebody botched the upload", and an operator re-ran a preload that
    could not fix it (#453). A monitor must be able to tell them apart WITHOUT parsing prose, so the
    distinction lives here, in the return value, and in the structured `@event` reason.

    `__bool__` is defined deliberately: any caller not yet updated to read `.usable` still gets the
    correct, fail-safe answer rather than the always-truthy behaviour of a plain Enum member. New
    code reads `.usable` (or the member) explicitly."""

    ABSENT = "absent"              # no VJ_VOLUME_ROOT, or it is not mounted on this worker
    READY = "ready"                # every sentinel present, i2v corpus included
    READY_NO_I2V = "ready_no_i2v"  # base corpus current; i2v corpus absent AND not needed here
    INCOMPLETE = "incomplete"      # a sentinel this job NEEDS is missing -> degrade wholesale

    @property
    def usable(self) -> bool:
        """True iff the worker should read its weights off the volume."""
        return self in (VolumeState.READY, VolumeState.READY_NO_I2V)

    def __bool__(self) -> bool:
        return self.usable


# The two env keys a volume repoint overwrites, and the values the ENDPOINT configured before any
# repoint happened. Captured once per process (see _remember_pristine_roots) so a later job that
# must NOT use the volume can be handed the endpoint own roots back instead of inheriting a stale
# repoint from an earlier job on the same warm worker.
_VOLUME_REPOINT_KEYS = ("VJ_MODELS_ROOT", "HF_HOME")
_PRISTINE_ROOTS: dict[str, str | None] | None = None


def _remember_pristine_roots(e: dict) -> None:
    """Record the endpoint own model roots, once per process, BEFORE any repoint can have landed.
    Only `_resolve_volume` and `_self_preload_volume` repoint, and the former calls this before it
    can, so the first call always observes the pristine values."""
    global _PRISTINE_ROOTS
    if _PRISTINE_ROOTS is None:
        _PRISTINE_ROOTS = {k: e.get(k) for k in _VOLUME_REPOINT_KEYS}


def _decline(e: dict, state: "VolumeState") -> "VolumeState":
    """Return `state` after taking BACK any volume repoint an earlier job on this warm worker made.

    The repoint is process-global (it lands in os.environ so the deferred torch/diffusers loads see
    it), but since #453 whether the volume is usable is a PER-JOB question: a `preview` job may
    legitimately accept a keyframe-only volume, and the very next `render` job on the same warm
    worker must not inherit that. Without this undo, that render would resolve its HF cache to the
    volume, find the base sentinel current, skip the R2 pull, and then try to mirror Wan ONTO the
    volume at i2v time -- precisely the failure the wholesale-degrade guard exists to prevent."""
    if _PRISTINE_ROOTS is None:
        return state
    for key, value in _PRISTINE_ROOTS.items():
        if value is None:
            e.pop(key, None)
        else:
            e[key] = value
    return state


def _volume_sentinel_ok(path: Path, model_version: str) -> bool:
    """True iff the sentinel file at `path` exists and matches `model_version`. Raises only OSError,
    which the caller treats as a miss."""
    return path.exists() and path.read_text().strip() == model_version


def _classify_volume(vol: Path, model_version: str, *,
                     i2v_required: bool) -> tuple["VolumeState", bool, bool]:
    """Read the volume sentinels and name what the volume is for a job with these requirements.

    Returns (state, base_ok, i2v_ok). Pure apart from two stats; raises only OSError, which the
    caller treats as a probe failure. The base sentinel is ALWAYS required: a volume without it has
    no corpus this worker can use. The i2v sentinel is required only when an i2v stage can run, and
    when it is required and missing the state is INCOMPLETE, which degrades wholesale exactly as
    before -- the i2v-absent case is a distinct state, never a relaxation of the guard."""
    base_ok = _volume_sentinel_ok(vol / SENTINEL, model_version)
    i2v_ok = _volume_sentinel_ok(vol / I2V_SENTINEL, model_version)
    if base_ok and i2v_ok:
        return VolumeState.READY, base_ok, i2v_ok
    if base_ok and not i2v_required:
        return VolumeState.READY_NO_I2V, base_ok, i2v_ok
    return VolumeState.INCOMPLETE, base_ok, i2v_ok


def _truthy(v) -> bool:
    """Env-flag truthiness: '1'/'true'/'yes'/'on' (any case) are on; everything else is off."""
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# Single-writer preload lock + how long before a held lock is presumed abandoned. A full fill takes
# ~6 min; 60 min means a lock older than that = a dead writer we can take over (see #55 Phase D).
_PRELOAD_LOCK = ".vj-preload.lock"
_LOCK_TTL_S = 3600


def _acquire_volume_lock(lockpath: Path, log: Callable[[str], None], ttl_s: int = _LOCK_TTL_S) -> bool:
    """Atomically claim the single-writer self-preload lock on the volume. `O_CREAT|O_EXCL` makes the
    create race-free across workers, so exactly one wins; a simultaneous cold fan-out cannot all
    start mirroring at once (RunPod warns concurrent writes corrupt a volume). If the lock already
    exists but is older than `ttl_s`, the prior writer is presumed dead and we take it over (unlink +
    retry). Returns True iff we now own it."""
    for _ in range(2):  # normal attempt + one retry after a stale take-over
        try:
            fd = os.open(str(lockpath), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(time.time()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - lockpath.stat().st_mtime
            except OSError:
                return False
            if age <= ttl_s:
                return False  # fresh lock held by a live writer
            log(f"models_mirror: stale preload lock at {lockpath} ({age:.0f}s > {ttl_s}s); taking over.")
            try:
                lockpath.unlink()
            except OSError:
                return False
        except OSError as exc:  # e.g. volume not mounted (ENOENT) / not writable -- never crash
            log(f"models_mirror: cannot claim preload lock at {lockpath} ({exc}); using R2 mirror.")
            return False
    return False


def _self_preload_volume(e: dict, model_version: str, log: Callable[[str], None],
                         *, i2v_required: bool = True) -> "VolumeState":
    """First-worker self-preload (opt-in via `VJ_VOLUME_SELF_PRELOAD`): when the mounted volume is
    empty/partial, win the single-writer lock and mirror R2 -> the volume (base + i2v) so every
    later worker in this datacenter is hot -- no manual preload pod, and version bumps self-heal.
    The easiest scale-out: attach an empty volume to a DC and the first worker primes it.

    Returns a `VolumeState`: a usable one (and a repoint of `HF_HOME`/`VJ_MODELS_ROOT` at the
    now-filled volume) iff WE filled what this job needs, INCOMPLETE to fall back to the local-disk
    R2 mirror for this job. `i2v_required` is the caller's readiness requirement and is applied to
    the filled volume by the same `_classify_volume` the read path uses, so a self-preload and a
    plain read can never disagree about what "preloaded" means. Only the lock winner writes; the
    losers fall back (correct, just not hot), which is what prevents concurrent-write corruption.
    The fill runs the standard mirror against a sub-env with the volume-read path disabled, so it
    does not recurse back into `_resolve_volume`."""
    vol = Path(e["VJ_VOLUME_ROOT"])
    lock = vol / _PRELOAD_LOCK
    if not _acquire_volume_lock(lock, log):
        log(f"models_mirror: another worker is preloading {vol}; this job uses the R2 mirror.")
        return VolumeState.INCOMPLETE
    log(f"models_mirror: self-preloading volume {vol} from R2 (sole writer)...")
    try:
        sub = dict(e)
        sub["VJ_MODELS_ROOT"] = str(vol)
        sub["HF_HOME"] = str(vol / "hf-cache")
        sub.pop("VJ_VOLUME_ROOT", None)          # disable the volume-read path in the nested calls
        sub.pop("VJ_VOLUME_SELF_PRELOAD", None)
        ensure_models(env=sub, log=log)          # base set + base sentinel -> volume
        ensure_i2v_models(env=sub, log=log)      # Wan set + i2v sentinel  -> volume
    except Exception as exc:  # noqa: BLE001 -- a failed fill must never abort the job; fall back
        log(f"models_mirror: self-preload of {vol} failed ({exc}); this job uses the R2 mirror.")
        return VolumeState.INCOMPLETE
    finally:
        try:
            lock.unlink()
        except OSError:
            pass
    try:
        state, base_ok, i2v_ok = _classify_volume(vol, model_version, i2v_required=i2v_required)
    except OSError:
        state, base_ok, i2v_ok = VolumeState.INCOMPLETE, False, False
    if state.usable:
        e["VJ_MODELS_ROOT"] = str(vol)
        e["HF_HOME"] = str(vol / "hf-cache")
        log(f"models_mirror: self-preload complete; reading weights from {vol}.")
        log(_skip_event("volume_self_preload"))
        return state
    log(f"models_mirror: self-preload of {vol} did not complete the sentinels this job needs "
        f"(base={base_ok}, i2v={i2v_ok}, i2v_required={i2v_required}); using R2 mirror.")
    return VolumeState.INCOMPLETE


def _resolve_volume(e: dict, model_version: str, log: Callable[[str], None],
                    *, action: str | None = None) -> "VolumeState":
    """Name what the mounted RunPod network volume (`VJ_VOLUME_ROOT`) is for THIS job, and when it is
    usable repoint `HF_HOME` + `VJ_MODELS_ROOT` at it so the worker reads its weights straight off
    the local-datacenter volume -- no 223 GB R2 copy (issue #55 Phase C: per-datacenter preloaded
    volumes, R2 mirror as the universal fallback). Returns a `VolumeState`, never a bare bool; an
    unusable state means the caller falls back to the R2 mirror on writable local disk.

    WHICH SENTINELS ARE REQUIRED IS THE JOB'S QUESTION, NOT THE VOLUME'S. The base sentinel is
    always required. The i2v sentinel is required only when this job's `action` can actually reach
    the Wan i2v stage (`i2v_stage_possible`). Conrad retired the local i2v path on 2026-09-26, so
    the honest post-ruling corpus is keyframes and no Wan: 60% of the old 107.8 GB weight set is
    simply not there any more. Requiring the i2v sentinel unconditionally made that correctly
    preloaded volume fail EVERY resolve, forever, with no error and nothing to show for it but a
    cold-start bill that reads as a volume nobody configured (#453).

    THE ORIGINAL GUARD IS KEPT, NOT WEAKENED. When an i2v stage CAN run and the volume lacks the
    i2v corpus, the volume is still refused WHOLESALE (INCOMPLETE) rather than repointed: after a
    repoint a standalone `i2v_clip` calls `ensure_i2v_models` against the volume root and would try
    to mirror Wan ONTO the volume, failing instead of falling back. That reasoning was right; only
    its premise moved. So the guard now fires on "the corpus this job needs is incomplete" instead
    of on "the corpus a 2026-06 worker needed is incomplete", and the deliberately-absent i2v corpus
    gets its own state (READY_NO_I2V) instead of being reported as a botched upload.
    (One volume per DC is still the design.)

    READ-ONLY by default: RunPod warns that concurrent writes from multiple workers corrupt a
    volume, so a worker normally never writes it. The exception is opt-in self-preload
    (`VJ_VOLUME_SELF_PRELOAD`, #55 Phase D): on an empty/partial volume, the FIRST worker wins a
    single-writer lock and fills it for the rest, so scale-out is just "attach an empty volume."
    Mutating `e` (os.environ in production) is how the repoint reaches the deferred torch/diffusers
    loads, which read `HF_HOME` at call time."""
    vol = e.get("VJ_VOLUME_ROOT")
    if not vol:
        return VolumeState.ABSENT
    # Before anything can repoint, record the roots the endpoint configured (see _decline).
    _remember_pristine_roots(e)
    # The volume must actually be mounted. If VJ_VOLUME_ROOT is set but the path isn't there (the
    # endpoint is configured for volumes but this worker landed in a DC without one, or the env was
    # set before the volume was attached), treat it as a clean miss -> R2 mirror. Without this, the
    # self-preload lock attempt would os.open() a non-existent dir and crash the worker.
    if not Path(vol).is_dir():
        log(f"models_mirror: VJ_VOLUME_ROOT={vol} is not mounted here; falling back to R2 mirror.")
        return _decline(e, VolumeState.ABSENT)
    i2v_required = i2v_stage_possible(action)
    try:
        state, base_ok, i2v_ok = _classify_volume(Path(vol), model_version,
                                                  i2v_required=i2v_required)
    except OSError as exc:  # noqa: BLE001 -- a volume probe failure must never abort the job
        log(f"models_mirror: volume probe at {vol} failed ({exc}); falling back to R2 mirror.")
        return _decline(e, VolumeState.INCOMPLETE)
    if state.usable:
        e["VJ_MODELS_ROOT"] = str(vol)
        e["HF_HOME"] = str(Path(vol) / "hf-cache")
        if state is VolumeState.READY:
            log(f"models_mirror: fully preloaded network volume at {vol} (v{model_version}); "
                "reading weights from it, skipping the R2 mirror.")
            log(_skip_event("volume"))
        else:
            # NOT a partial preload. Said in full so nobody re-runs a preload that cannot help.
            log(f"models_mirror: network volume at {vol} carries the corpus this job needs "
                f"(v{model_version}, base sentinel current). The i2v corpus is absent and "
                f"action={action!r} runs no i2v stage, so this is a DELIBERATELY i2v-free volume, "
                "not an unfinished preload. Reading weights from it, skipping the R2 mirror.")
            log(_skip_event("volume_no_i2v"))
        return state
    log(f"models_mirror: volume at {vol} is missing a sentinel THIS JOB NEEDS for v{model_version} "
        f"(base={base_ok}, i2v={i2v_ok}, i2v_required={i2v_required}); degrading wholesale.")
    if _truthy(e.get("VJ_VOLUME_SELF_PRELOAD")):
        filled = _self_preload_volume(e, model_version, log, i2v_required=i2v_required)
        return filled if filled.usable else _decline(e, filled)
    log("models_mirror: falling back to R2 mirror.")
    return _decline(e, state)


def ensure_models(*, env: dict | None = None, log: Callable[[str], None] = print,
                  skip_repos: tuple[str, ...] = DEFAULT_SKIP_REPOS,
                  action: str | None = None) -> bool:
    """Mirror the kept model set from R2 into the local HF cache + antelopev2 dir.

    Returns True if a pull ran, False if it was skipped (warm worker, or no R2 creds so weights
    are assumed pre-provisioned). Raises on a hard failure (missing rclone, failed pull).

    `action` is the job's action, and it is the ONLY thing that decides whether the network-volume
    readiness gate requires the i2v corpus (see `_resolve_volume`). Omitting it keeps the
    conservative pre-#453 behaviour: the i2v corpus is required.
    """
    e = mirror_env(env)
    model_version = e.get("VJ_MODEL_VERSION") or _DEFAULT_MODEL_VERSION

    # Baked image: weights are in the image at VJ_MODELS_ROOT; never touch a volume or R2 (a baked
    # image is datacenter-agnostic: no volume pinning, no cold-pull tax). Which precision was baked
    # is the `.vj-baked` stamp's business, not this guard's. Checked
    # FIRST so a baked worker short-circuits before the volume-resolve / R2 path entirely.
    if is_baked(e):
        log("models_mirror: baked image (.vj-baked present); skipping volume + R2 (weights baked in).")
        log(_skip_event("baked"))
        return False

    # Preloaded per-datacenter network volume (issue #55 Phase C): if one is mounted and current,
    # read weights straight off it and skip the R2 copy entirely. Falls through to the mirror below
    # on any miss, so the R2 path stays the universal fallback. Repoints HF_HOME/VJ_MODELS_ROOT.
    volume = _resolve_volume(e, model_version, log, action=action)
    # `e` is mirror_env's COPY, so the repoint -- or its UNDO -- lands there only. Carry both root
    # keys to the environment we were asked to use (os.environ in production), or the deferred
    # torch/diffusers loads never read the volume. Carried on EVERY outcome, not just a usable one:
    # readiness is per-job since #453, so a decline on a warm worker whose PREVIOUS job repointed at
    # the volume has to take that repoint back (see _decline).
    target = env if env is not None else os.environ
    for key in _VOLUME_REPOINT_KEYS:
        if key in e:
            target[key] = e[key]
        else:
            target.pop(key, None)
    if volume.usable:
        return False

    hf_home = Path(e.get("HF_HOME", "/opt/models/hf-cache"))
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    bucket = e.get("R2_BUCKET", "vivijure")
    sentinel = models_root / SENTINEL

    if sentinel.exists():
        if sentinel.read_text().strip() == model_version:
            log("models_mirror: warm worker (sentinel present); skipping R2 pull.")
            log(_skip_event("warm"))
            return False
        log(f"models_mirror: sentinel version mismatch (want {model_version!r}); re-mirroring.")
    if not e.get("R2_ACCESS_KEY_ID"):
        hub = hf_home / "hub"
        if not hub.is_dir() or not any(hub.iterdir()):
            log("models_mirror: WARNING: no R2 creds and HF cache appears empty -- "
                "model loads will fail; set R2_ACCESS_KEY_ID or pre-provision weights")
            log(_skip_event("no_creds_empty_cache"))
        else:
            log("models_mirror: no R2 creds; assuming weights are pre-provisioned.")
            log(_skip_event("no_creds"))
        return False
    if shutil.which("rclone") is None:
        raise RuntimeError("models_mirror: rclone is not installed in the image")

    renv = rclone_env(e)
    hf_home.mkdir(parents=True, exist_ok=True)
    log(f"models_mirror: cold worker -> mirroring r2:{bucket}/models to {hf_home} "
        f"(skipping {len(skip_repos)} lazy repos)...")
    jitter = _jitter_seconds(e)
    if jitter:
        log(f"models_mirror: jitter {jitter:.1f}s before R2 pull (de-stagger concurrent egress).")
        time.sleep(jitter)
    legs: list[tuple[str, float]] = []
    legs.append(("hf-cache", _timed(
        subprocess.run,
        mirror_cmd(f"r2:{bucket}/models/hf-cache", hf_home, skip_repos=skip_repos),
        check=True, env=renv)))

    antelope = models_root / "antelopev2"
    antelope.mkdir(parents=True, exist_ok=True)
    legs.append(("antelopev2", _timed(
        subprocess.run, mirror_cmd(f"r2:{bucket}/models/antelopev2", antelope), check=True, env=renv)))

    # Finishing-stage weights: stored at fixed paths under models_root (NOT in the HF cache).
    # ModelServer.frame_interpolator loads $VJ_MODELS_ROOT/rife/flownet.pkl and
    # ModelServer.face_restorer loads $VJ_MODELS_ROOT/GFPGANv1.4/GFPGANv1.4.pth directly,
    # so they need their own R2 mirror legs separate from the HF-cache pull above.
    for subdir in ("rife", "GFPGANv1.4", "facexlib"):
        dst = models_root / subdir
        dst.mkdir(parents=True, exist_ok=True)
        legs.append((subdir, _timed(
            subprocess.run, mirror_cmd(f"r2:{bucket}/models/{subdir}", dst), check=True, env=renv)))

    # rclone --links stores HF-cache symlinks as `<name>.rclonelink` text files (the link target),
    # and rclone >= 1.7x does NOT translate them back to real symlinks on download -- it leaves the
    # marker files in place, so the snapshot dirs end up with `config.json.rclonelink` instead of
    # `config.json -> ../../blobs/<hash>`, and an HF_HUB_OFFLINE load can't find the file. Rebuild
    # the symlinks from the markers ourselves so the cache is valid regardless of rclone's version.
    _reconstruct_symlinks(hf_home, log)
    # Write .no_exist stubs at the R2 revision (read from refs/main after the mirror). Build-time
    # stubs (bake_hf_configs.py) use the HF revision at build time which may differ; this call
    # ensures stubs are always at the revision R2 seeded, which is what offline probes check.
    write_no_exist_stubs(hf_home / "hub", HF_OFFLINE_STUBS, log)

    models_root.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(model_version + "\n")
    # Telemetry is best-effort: a traversal error in _dir_bytes (e.g. rglob hitting a
    # PermissionError mid-walk, which the per-entry guard does not catch) must never fail an
    # otherwise-successful cold mirror -- the exact path we are trying to stabilize.
    try:
        total_bytes = (_dir_bytes(hf_home) + _dir_bytes(antelope)
                       + _dir_bytes(models_root / "rife") + _dir_bytes(models_root / "GFPGANv1.4")
                       + _dir_bytes(models_root / "facexlib"))
        log(_mirror_event(legs, total_bytes, cold=True, model_version=model_version))
    except Exception as exc:
        log(f"models_mirror: mirror telemetry skipped ({exc})")
    log("models_mirror: model mirror from R2 complete.")
    return True


def ensure_i2v_models(*, env: dict | None = None, log: Callable[[str], None] = print,
                      repos: tuple[str, ...] = I2V_LAZY_REPOS, force: bool = False) -> bool:
    """Lazily mirror the heavy i2v models (Wan I2V + the Lightning distill) from R2 on first i2v use.

    Called from models.ModelServer.i2v_pipeline before the Wan weights load. A keyframe/preview
    worker never calls it, so it skips ~120GB at cold start (those repos are in DEFAULT_SKIP_REPOS).
    Idempotent via its own sentinel; returns True if a pull ran, False if skipped (warm, or no R2
    creds so weights are assumed pre-provisioned). Raises on a hard failure, same as ensure_models.

    `force` overrides the BAKED early-return ONLY. It exists for the one caller that has already
    decided a pull is required despite the image being baked: the FINAL-tier B seam in
    `models._select_i2v_weights`, which selects the bf16 repo on an fp8-baked image and needs those
    shards fetched. Without it the guard silently swallowed the pull its own caller had asked for and
    the load failed deep against a cache holding the other repo (#339). It does NOT override the warm
    sentinel or the missing-creds path: those are states, not decisions.
    """
    global _i2v_prefetch_thread
    e = mirror_env(env)
    hf_home = Path(e.get("HF_HOME", "/opt/models/hf-cache"))
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    bucket = e.get("R2_BUCKET", "vivijure")
    sentinel = models_root / I2V_SENTINEL

    # Baked image: the i2v weights this image was baked with are already on disk, so the DEFAULT is
    # never to pull. The only exception is an explicit `force` from a caller that has decided
    # otherwise (the B final-tier seam, #339).
    #
    # The old line here claimed "fp8 weights baked in". Nothing checks that, and the shipping bake is
    # bf16, so every cold start on every production worker printed an fp8 claim about an image that
    # contains no fp8 weight file (#364). The sentinel proves the image is BAKED and nothing more, so
    # that is all this line now says; the precision an operator wants is in the `.vj-baked` stamp and
    # in the `model_precision` event the i2v load emits.
    if is_baked(e) and not force:
        log("models_mirror: baked image (.vj-baked present); skipping i2v R2 pull "
            "(the weights this image was baked with are already on disk).")
        log(_skip_event("baked", event="i2v_mirror_skipped"))
        return False
    if is_baked(e) and force:
        log("models_mirror: baked image, but the caller forced an i2v R2 pull (final-tier bf16 seam); "
            "not skipping.")

    # Join the background prefetch thread (if started by start_i2v_prefetch) before checking
    # the sentinel so its result is visible. If the thread failed, fall through to pull normally.
    # Guard against calling join() on the current thread (which happens when ensure_i2v_models
    # is called from _within_ the prefetch thread via start_i2v_prefetch._pull).
    if (_i2v_prefetch_thread is not None
            and _i2v_prefetch_thread.is_alive()
            and threading.current_thread() is not _i2v_prefetch_thread):
        log("models_mirror: i2v prefetch in progress; waiting...")
        _i2v_prefetch_thread.join()

    model_version = e.get("VJ_MODEL_VERSION") or _DEFAULT_MODEL_VERSION
    if sentinel.exists():
        if sentinel.read_text().strip() == model_version:
            log("models_mirror: i2v models already mirrored (sentinel present); skipping.")
            log(_skip_event("warm", event="i2v_mirror_skipped"))
            return False
        log(f"models_mirror: i2v sentinel version mismatch (want {model_version!r}); re-mirroring.")
    if not e.get("R2_ACCESS_KEY_ID"):
        log("models_mirror: no R2 creds; i2v weights assumed pre-provisioned.")
        log(_skip_event("no_creds", event="i2v_mirror_skipped"))
        return False
    if shutil.which("rclone") is None:
        raise RuntimeError("models_mirror: rclone is not installed in the image")

    renv = rclone_env(e)
    hub = hf_home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    legs: list[tuple[str, float]] = []
    for repo in repos:
        log(f"models_mirror: lazy i2v pull -> mirroring {repo} from R2...")
        legs.append((repo, _timed(
            subprocess.run,
            mirror_cmd(f"r2:{bucket}/models/hf-cache/hub/{repo}", hub / repo), check=True, env=renv)))
    _reconstruct_symlinks(hf_home, log)
    sentinel.write_text(model_version + "\n")
    # Best-effort telemetry (see ensure_models): never fail a good lazy pull on a _dir_bytes walk.
    try:
        total_bytes = sum(_dir_bytes(hub / repo) for repo in repos)
        log(_mirror_event(legs, total_bytes, cold=True, model_version=model_version,
                          event="i2v_mirror_complete"))
    except Exception as exc:
        log(f"models_mirror: i2v mirror telemetry skipped ({exc})")
    log("models_mirror: i2v model mirror from R2 complete.")
    return True


def start_i2v_prefetch(*, env: dict | None = None, log: Callable[[str], None] = print) -> "threading.Thread | None":
    """Eager-start the Wan I2V pull in a background thread so it overlaps LoRA training.

    Call this right after ensure_models() returns (cold-start pull done, network free). The
    background thread runs ensure_i2v_models(); ensure_i2v_models() joins it before the sentinel
    check so i2v_pipeline() sees the weights already present rather than waiting serially.

    Idempotent: a second call while a thread is running returns the existing thread. Returns None
    on a warm worker (sentinel present) or when R2 creds are absent -- both are instant no-ops
    that don't need a thread.
    """
    global _i2v_prefetch_thread
    if _i2v_prefetch_thread is not None:
        return _i2v_prefetch_thread

    e = mirror_env(env)
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    if (models_root / I2V_SENTINEL).exists() or not e.get("R2_ACCESS_KEY_ID"):
        return None

    def _pull() -> None:
        try:
            ensure_i2v_models(env=env, log=log)
        except Exception as exc:
            log(f"models_mirror: i2v prefetch error: {exc}")

    t = threading.Thread(target=_pull, daemon=True, name="vj-i2v-prefetch")
    _i2v_prefetch_thread = t
    t.start()
    log("models_mirror: eager i2v prefetch started (overlaps LoRA training).")
    return t


# Known-absent files that diffusers probes for under HF_HUB_OFFLINE=1.
# Each tuple is (HF cache dir name, file path relative to the snapshot dir). These are files
# that don't exist in the repos; online, diffusers gets a graceful 404 and falls back; offline,
# the missing cache entry raises LocalEntryNotFoundError. An empty .no_exist stub at the right
# path (written once at image build time by deploy/bake_hf_configs.py) replicates the 404
# negative-cache entry. See write_no_exist_stubs below.
HF_OFFLINE_STUBS: tuple[tuple[str, str], ...] = (
    # probe 1: shard-index check for the VAE; single-file VAE has no index.json
    ("models--SG161222--RealVisXL_V5.0", "vae/diffusion_pytorch_model.safetensors.index.json"),
    # probe 2: same shard-index check for the xinsir ControlNet weights
    ("models--xinsir--controlnet-openpose-sdxl-1.0", "diffusion_pytorch_model.safetensors.index.json"),
    ("models--xinsir--controlnet-canny-sdxl-1.0", "diffusion_pytorch_model.safetensors.index.json"),
    # probe 3: PEFT adapter_config probe for the IP-Adapter image_encoder (not a PEFT model)
    ("models--h94--IP-Adapter", "sdxl_models/image_encoder/adapter_config.json"),
)


def write_no_exist_stubs(hub: Path, stubs: tuple[tuple[str, str], ...],
                         log: Callable[[str], None] = print) -> list[Path]:
    """Create empty HF-cache .no_exist stubs for known-absent repo files (build-time helper).

    diffusers probes for certain files (shard-index .index.json, PEFT adapter_config.json) that
    don't exist in the repos. Online these are graceful 404s. Under HF_HUB_OFFLINE=1 the missing
    cache entry raises LocalEntryNotFoundError. An empty stub at
    `hub/<cache-dir>/.no_exist/<revision>/<file>` replicates the negative-cache entry so the
    graceful fallback runs instead.

    The revision is read from refs/main written by snapshot_download. Returns the list of stub
    paths created; skips entries whose refs/main doesn't exist yet (warns instead)."""
    written = []
    for cache_dir, absent_path in stubs:
        ref_file = hub / cache_dir / "refs" / "main"
        if not ref_file.exists():
            log(f"models_mirror: no refs/main for {cache_dir}; skipping .no_exist stub")
            continue
        rev = ref_file.read_text().strip()
        stub = hub / cache_dir / ".no_exist" / rev / absent_path
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("")
        log(f"models_mirror: .no_exist stub: {cache_dir}/.no_exist/{rev[:12]}/{absent_path}")
        written.append(stub)
    return written


def _reconstruct_symlinks(root: Path, log: Callable[[str], None]) -> int:
    """Turn every `*.rclonelink` marker under `root` into the real symlink it describes (its file
    content is the link target). Idempotent; tolerant of an already-correct cache."""
    n = 0
    for marker in root.rglob("*.rclonelink"):
        target = marker.read_text().strip()
        link = marker.with_suffix("")  # drop the .rclonelink extension
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
            marker.unlink()
            n += 1
        except OSError as exc:  # noqa: PERF203
            log(f"models_mirror: could not rebuild symlink {link} -> {target} ({exc})")
    if n:
        log(f"models_mirror: rebuilt {n} HF-cache symlinks from .rclonelink markers")
    return n


def _preload_main() -> None:
    """Preload a RunPod network volume from R2 (issue #55 Phase C). Run this INSIDE a one-shot pod
    that has the volume mounted and the R2_* creds set, with VJ_MODELS_ROOT pointed at the mount:

        VJ_MODELS_ROOT=/runpod-volume python -m vivijure_backend.harness.models_mirror --preload

    It runs the SAME ensure_models + ensure_i2v_models the worker uses, so the on-volume layout
    (hf-cache/, antelopev2/, rife/, GFPGANv1.4/, both sentinels) is byte-for-byte what a worker
    reads via VJ_VOLUME_ROOT -- no separate copy logic to drift. VJ_VOLUME_ROOT must NOT be set
    here (that is the read-only worker path); the preloader is the volume's sole writer. The full
    set is pulled (base + the lazy i2v repos) by calling ensure_i2v_models explicitly, since one
    full volume per datacenter is the design (no base/i2v split)."""
    import sys
    print("models_mirror: preloading volume (base + i2v) from R2...", flush=True)
    ensure_models()                       # base set -> writes .vj-mirror-complete
    ensure_i2v_models()                   # Wan I2V + Lightning -> writes .vj-i2v-mirror-complete
    root = Path(os.environ.get("VJ_MODELS_ROOT", "/opt/models"))
    ok = (root / SENTINEL).exists() and (root / I2V_SENTINEL).exists()
    print(f"models_mirror: preload {'complete' if ok else 'INCOMPLETE'} at {root} "
          f"(base={ (root / SENTINEL).exists() }, i2v={ (root / I2V_SENTINEL).exists() })", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    import sys
    if "--preload" in sys.argv:
        _preload_main()
    else:
        print("usage: python -m vivijure_backend.harness.models_mirror --preload", flush=True)
