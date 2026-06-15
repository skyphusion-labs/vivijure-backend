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
import shutil
import subprocess
import tempfile
import threading
import time
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
# spec loads (T2V is never used; the two stray SDXL repos are not in the spec). Storage in R2 is kept
# (cheap, safer); these are pull-time excludes only. Tune per the live model set, the deploy's call.
DEFAULT_SKIP_REPOS = I2V_LAZY_REPOS + (
    "models--Wan-AI--Wan2.2-T2V-A14B-Diffusers",          # text-to-video: never loaded
    "models--stabilityai--stable-diffusion-xl-base-1.0",  # not in the model spec (dead weight)
    "models--stabilityai--sdxl-turbo",                    # not in the model spec (dead weight)
    "spaces--InstantX--InstantID",                        # the HF Space, not the model repo
)

# Separate completion sentinel for the lazy i2v pull (the cold-start pull has its own, SENTINEL).
I2V_SENTINEL = ".vj-i2v-mirror-complete"

# HF's abandoned download temp files; model-presence checks treat any *.incomplete as a broken
# repo, so never mirror them.
_INCOMPLETE_GLOB = "**/*.incomplete"

# Written under the models root only after the pull fully succeeds (see module docstring).
SENTINEL = ".vj-mirror-complete"

# Bump this constant (or set VJ_MODEL_VERSION in the worker env) whenever the model set
# in R2 changes so warm workers re-pull instead of silently using a stale cache.
_DEFAULT_MODEL_VERSION = "1"


def rclone_conf(env: dict, conf_dir: Path) -> Path:
    """Write an rclone.conf for the R2 store from the worker's R2_* env. Raises if creds are
    incomplete so the worker fails here, loudly, not later at the model-presence gate."""
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT") if not env.get(k)]
    if missing:
        raise RuntimeError("models_mirror: incomplete R2 creds; missing: " + ", ".join(missing))
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf = conf_dir / "rclone.conf"
    conf.write_text(
        "[r2]\ntype = s3\nprovider = Cloudflare\n"
        f"access_key_id = {env['R2_ACCESS_KEY_ID']}\n"
        f"secret_access_key = {env['R2_SECRET_ACCESS_KEY']}\n"
        f"endpoint = {env['R2_ENDPOINT']}\n"
        "acl = private\nno_check_bucket = true\n"
    )
    conf.chmod(0o600)
    return conf


def mirror_cmd(conf: Path, src: str, dst: Path, *, skip_repos: tuple[str, ...] = ()) -> list[str]:
    """argv for one `rclone copy --links` mirror leg. Pure: built and asserted without rclone."""
    cmd = ["rclone", "--config", str(conf), "copy", "--links",
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


def ensure_models(*, env: dict | None = None, log: Callable[[str], None] = print,
                  skip_repos: tuple[str, ...] = DEFAULT_SKIP_REPOS) -> bool:
    """Mirror the kept model set from R2 into the local HF cache + antelopev2 dir.

    Returns True if a pull ran, False if it was skipped (warm worker, or no R2 creds so weights
    are assumed pre-provisioned). Raises on a hard failure (missing rclone, failed pull).
    """
    e = env if env is not None else os.environ
    hf_home = Path(e.get("HF_HOME", "/opt/models/hf-cache"))
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    bucket = e.get("R2_BUCKET", "vivijure")
    sentinel = models_root / SENTINEL

    model_version = e.get("VJ_MODEL_VERSION") or _DEFAULT_MODEL_VERSION
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

    conf = rclone_conf(e, Path(tempfile.gettempdir()) / "vj-rclone")
    hf_home.mkdir(parents=True, exist_ok=True)
    log(f"models_mirror: cold worker -> mirroring r2:{bucket}/models to {hf_home} "
        f"(skipping {len(skip_repos)} lazy repos)...")
    legs: list[tuple[str, float]] = []
    legs.append(("hf-cache", _timed(
        subprocess.run,
        mirror_cmd(conf, f"r2:{bucket}/models/hf-cache", hf_home, skip_repos=skip_repos),
        check=True)))

    antelope = models_root / "antelopev2"
    antelope.mkdir(parents=True, exist_ok=True)
    legs.append(("antelopev2", _timed(
        subprocess.run, mirror_cmd(conf, f"r2:{bucket}/models/antelopev2", antelope), check=True)))

    # Finishing-stage weights: stored at fixed paths under models_root (NOT in the HF cache).
    # ModelServer.frame_interpolator loads $VJ_MODELS_ROOT/rife/flownet.pkl and
    # ModelServer.face_restorer loads $VJ_MODELS_ROOT/GFPGANv1.4/GFPGANv1.4.pth directly,
    # so they need their own R2 mirror legs separate from the HF-cache pull above.
    for subdir in ("rife", "GFPGANv1.4"):
        dst = models_root / subdir
        dst.mkdir(parents=True, exist_ok=True)
        legs.append((subdir, _timed(
            subprocess.run, mirror_cmd(conf, f"r2:{bucket}/models/{subdir}", dst), check=True)))

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
    total_bytes = (_dir_bytes(hf_home) + _dir_bytes(antelope)
                   + _dir_bytes(models_root / "rife") + _dir_bytes(models_root / "GFPGANv1.4"))
    log(_mirror_event(legs, total_bytes, cold=True, model_version=model_version))
    log("models_mirror: model mirror from R2 complete.")
    return True


def ensure_i2v_models(*, env: dict | None = None, log: Callable[[str], None] = print,
                      repos: tuple[str, ...] = I2V_LAZY_REPOS) -> bool:
    """Lazily mirror the heavy i2v models (Wan I2V + the Lightning distill) from R2 on first i2v use.

    Called from models.ModelServer.i2v_pipeline before the Wan weights load. A keyframe/preview
    worker never calls it, so it skips ~120GB at cold start (those repos are in DEFAULT_SKIP_REPOS).
    Idempotent via its own sentinel; returns True if a pull ran, False if skipped (warm, or no R2
    creds so weights are assumed pre-provisioned). Raises on a hard failure, same as ensure_models.
    """
    global _i2v_prefetch_thread
    e = env if env is not None else os.environ
    hf_home = Path(e.get("HF_HOME", "/opt/models/hf-cache"))
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    bucket = e.get("R2_BUCKET", "vivijure")
    sentinel = models_root / I2V_SENTINEL

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

    conf = rclone_conf(e, Path(tempfile.gettempdir()) / "vj-rclone")
    hub = hf_home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    legs: list[tuple[str, float]] = []
    for repo in repos:
        log(f"models_mirror: lazy i2v pull -> mirroring {repo} from R2...")
        legs.append((repo, _timed(
            subprocess.run,
            mirror_cmd(conf, f"r2:{bucket}/models/hf-cache/hub/{repo}", hub / repo), check=True)))
    _reconstruct_symlinks(hf_home, log)
    sentinel.write_text(model_version + "\n")
    total_bytes = sum(_dir_bytes(hub / repo) for repo in repos)
    log(_mirror_event(legs, total_bytes, cold=True, model_version=model_version,
                      event="i2v_mirror_complete"))
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

    e = env if env is not None else os.environ
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
