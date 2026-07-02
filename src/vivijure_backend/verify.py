"""Pod-side verify @event EMITTER -- the release-gate probe channel.

`python -m vivijure_backend.verify` runs ON A GPU POD during the staging gate (docs/release-gate.md)
and emits a small, machine-readable stream of structured events that the release harness
(`deploy/runpod_verify.py`) asserts on to decide whether a freshly built image may be promoted onto
the production serverless endpoint. It is the WRITER; `runpod_verify.evaluate` is the READER. This
module is the single source of the wire contract they share.

ACTIVE ONLY UNDER `VJ_VERIFY`. `main()` is a hard no-op (prints one skip line, exits 0) unless
`VJ_VERIFY` is set truthy, so the module has ZERO effect on a normal render: importing it starts
nothing, and an ordinary job never sets the flag.

THE CHANNEL (decided here; documented in docs/release-gate.md). Option B, an R2 object at a
run-scoped key, exactly like the render ProgressEmitter -- durable, queryable after the pod is gone,
no new infra and no new secret (the worker already holds the R2 token). Two objects per verify run,
keyed by run id under a dedicated `verify/` prefix so a probe never collides with a project render:

  - `verify/<run_id>/summary.json`   (application/json)      -- THE POLL TARGET. Latest state:
        {"schema": "vivijure-verify/1", "run_id": str,
         "status": "running" | "complete" | "error",
         "started_ts": float|null, "updated_ts": float|null, "last_event": str|null,
         "error": null | {"stage": str, "message": str},
         "events": [{"seq": int, "ts": float, "event": str, "payload": {...}}, ...]}
  - `verify/<run_id>/events.ndjson`  (application/x-ndjson)  -- the same records, one JSON object
        per line (a stream a human or a tail can follow; the summary is the machine poll target).

Both objects are rewritten in full on every emit: a verify run is a handful of events, one writer,
one process, so the emitter accumulates them in memory and re-PUTs the tiny object each time (no S3
append, no second writer to race). Last-writer-wins per event name matches `runpod_verify.find_event`.

REDUNDANT TRANSPORT. Every event is ALSO printed to stdout as `@event <name> {json-payload}`,
byte-identical to the wire `runpod_verify.parse_events` already reads, so the pod logs remain a
fallback channel if R2 is briefly unreachable. R2 writes are best-effort (a channel hiccup must not
abort the render before it can emit its `error`); the stdout mirror is unconditional.

THE EVENTS (payloads are EXACTLY the fields `runpod_verify.evaluate` reads -- one contract, no
translation layer):

  gpu_probe   {torch_cuda, kernel_ok, vj_baked, weights_on_disk, vram_free_gb, vram_total_gb,
               device_name}   -- pod-only insight (torch sees the GPU, the cu128 kernel LOADS on
               this card, the baked sentinel + weights are on disk, VRAM headroom).
  first_frame {seconds}       -- time-to-first-frame (measured from the render's progress channel).
  sharpness   {value, baseline}   -- the #118 method-ii quality gate: variance-of-Laplacian of the
               rendered first frame vs the runtime-quant baseline.
  complete    {output_key, clip_bytes}   -- render done, output object written.
  error       {stage, message}   -- a probe/render failure; also flips summary.status to "error".

HARNESS INTEGRATION (no prose parsing): poll `summary.json` until `status` is terminal, then feed
`events_from_summary(raw)` straight into the existing `runpod_verify.evaluate(events, cfg)`.

The emitter, the pure probe/metric helpers, and the run_verify orchestration are unit-tested on CPU
with a fake store and a fake render. The GPU draft render (`_pod_draft_render`) is a deferred,
pod-only seam -- its live proof is the authorized S1 pod run, exactly like the harness's own
RunpodMcpPodClient seam.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

from .harness import keys

SCHEMA = "vivijure-verify/1"

# The base events a verify run emits, in causal order. The harness polls for `complete` (or `error`).
BASE_EVENTS: tuple[str, ...] = ("gpu_probe", "first_frame", "sharpness", "complete")

# The R2 store env names the emitter needs (mirrors harness.r2.R2Config.from_env). Named here so
# a fatal-precondition `verify_fatal` can report the MISSING ones as a structured list, not a
# prose message the harness would have to parse.
R2_ENV_NAMES: tuple[str, ...] = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

# The gpu_probe payload keys, in one place so the emitter and the harness never drift.
GPU_PROBE_KEYS: tuple[str, ...] = (
    "torch_cuda", "kernel_ok", "vj_baked", "weights_on_disk",
    "vram_free_gb", "vram_total_gb", "device_name",
)

# Default runtime-quant sharpness baseline (the #118 method-ii reference). Overridable per run via
# VJ_SHARPNESS_BASELINE so the gate can be re-based without a code change; the harness asserts the
# RATIO (value/baseline), so the absolute number only sets the reference point.
DEFAULT_SHARPNESS_BASELINE = 100.0


# --------------------------------------------------------------------------- env gate

def is_enabled(env: dict | None = None) -> bool:
    """True iff the verify channel is armed. Truthy = 1/true/yes/on (case-insensitive); anything
    else (incl. unset/empty) is off, so a normal render can never accidentally arm it."""
    e = env if env is not None else os.environ
    return str(e.get("VJ_VERIFY", "")).strip().lower() in ("1", "true", "yes", "on")


def run_id_from(env: dict | None = None) -> str | None:
    """The run id the harness assigned (VJ_VERIFY_RUN_ID). None if absent -- the harness MUST set it
    so it knows the exact key to poll; we never invent one (and the runtime forbids clock/random)."""
    e = env if env is not None else os.environ
    rid = str(e.get("VJ_VERIFY_RUN_ID", "")).strip()
    return rid or None


def key_prefix_from(env: dict | None = None) -> str:
    e = env if env is not None else os.environ
    return str(e.get("VJ_VERIFY_KEY_PREFIX", "verify")).strip() or "verify"


def baseline_from(env: dict | None = None) -> float:
    e = env if env is not None else os.environ
    try:
        return float(e.get("VJ_SHARPNESS_BASELINE", DEFAULT_SHARPNESS_BASELINE))
    except (TypeError, ValueError):
        return DEFAULT_SHARPNESS_BASELINE


# --------------------------------------------------------------------------- pure probe + metrics

def gpu_probe_payload(facts: dict) -> dict:
    """Shape a raw facts dict into the exact gpu_probe payload (the 7 contract keys, missing ones
    defaulted falsey). Pure: the contract is asserted in tests without a GPU."""
    defaults: dict[str, Any] = {
        "torch_cuda": False, "kernel_ok": False, "vj_baked": False, "weights_on_disk": False,
        "vram_free_gb": 0.0, "vram_total_gb": 0.0, "device_name": "",
    }
    return {k: facts.get(k, defaults[k]) for k in GPU_PROBE_KEYS}


def collect_gpu_facts(env: dict | None = None) -> dict:
    """RUN ON THE POD (deferred torch import). Probes the insight a serverless worker hides: torch
    sees the GPU, the cu128 kernel actually loads on THIS card (a tiny cuda matmul -- the
    sm_120/'no kernel image' class fails HERE, loudly), the `.vj-baked` sentinel + weights are on
    disk, VRAM headroom. Best-effort: any probe error becomes a falsey fact so the harness FAILS the
    check rather than the probe crashing the pod. Pure enough to import on a CPU box (torch deferred);
    on a box with no CUDA it simply reports torch_cuda=False."""
    from pathlib import Path
    e = env if env is not None else os.environ
    facts = dict(gpu_probe_payload({}))  # start from the falsey defaults
    models_root = Path(e.get("VJ_MODELS_ROOT", "/opt/models"))
    facts["vj_baked"] = (models_root / ".vj-baked").exists()
    facts["weights_on_disk"] = (models_root / "hf-cache" / "hub").is_dir()
    try:
        import torch
        facts["torch_cuda"] = bool(torch.cuda.is_available())
        if facts["torch_cuda"]:
            facts["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            facts["vram_free_gb"] = round(free / 1024**3, 2)
            facts["vram_total_gb"] = round(total / 1024**3, 2)
            x = torch.randn(64, 64, device="cuda")  # the load-bearing kernel launch
            _ = (x @ x).sum().item()
            facts["kernel_ok"] = True
    except Exception as exc:  # noqa: BLE001 -- a probe failure is a CHECK failure, never a crash
        facts["device_name"] = facts["device_name"] or f"probe_error:{type(exc).__name__}"
    return facts


def frame_sharpness(frame) -> float:
    """Variance of the discrete Laplacian of a frame -- the focus/sharpness metric behind the #118
    method-ii quality gate. Accepts an HxW (grayscale) or HxWxC array-like; a color frame is reduced
    to luma-ish grayscale by channel mean. Pure numpy, no cv2, so it is testable on CPU with a tiny
    synthetic array. Returns 0.0 for a degenerate (too-small) frame rather than raising."""
    import numpy as np
    a = np.asarray(frame, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    # 4-neighbour discrete Laplacian on the interior (no wrap): -4*center + up+down+left+right.
    lap = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1] + a[2:, 1:-1]
        + a[1:-1, :-2] + a[1:-1, 2:]
    )
    return float(lap.var())


# --------------------------------------------------------------------------- channel readers (harness side)

def _records_to_events(records: list[dict]) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for r in records:
        if isinstance(r, dict) and isinstance(r.get("event"), str):
            payload = r.get("payload")
            out.append((r["event"], payload if isinstance(payload, dict) else {}))
    return out


def events_from_summary(raw: bytes | str) -> list[tuple[str, dict]]:
    """Parse a `summary.json` object into the `[(name, payload), ...]` list the harness feeds to
    `runpod_verify.evaluate` -- so the harness never touches JSON shape, never parses prose. Tolerant:
    a malformed/empty object yields an empty list rather than raising (the harness then fails the
    check on missing events, which is the correct outcome)."""
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(doc, dict):
        return []
    return _records_to_events(doc.get("events") or [])


def events_from_ndjson(raw: bytes | str) -> list[tuple[str, dict]]:
    """Parse an `events.ndjson` object (one record per line) into `[(name, payload), ...]`. The
    line-oriented fallback if a caller reads the stream object instead of the summary."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    records: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return _records_to_events(records)


# --------------------------------------------------------------------------- the emitter

class ChannelUnwritable(RuntimeError):
    """The verify R2 channel could not be written on the FIRST (authoritative) attempt. Unlike a
    mid-run best-effort hiccup, this means nothing the harness reads will EVER appear -- so the run
    is pointless and must fail LOUD, not silently-empty. Carries `stage` (r2_auth | r2_write) so the
    caller emits a precise verify_fatal."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


# boto3 surfaces a bad/quote-wrapped R2 credential as one of these on the write call (the values are
# present, so R2Config.from_env's presence check passes -- the failure only shows up at auth time).
_R2_AUTH_MARKERS = ("SignatureDoesNotMatch", "InvalidAccessKeyId", "AccessDenied", "Forbidden",
                    "403", "InvalidAccessKey", "CredentialsError", "NoCredentials", "Unauthorized")


def _classify_write_error(exc: Exception) -> str:
    """r2_auth for a credential/permission failure (the bad/quote-wrapped-R2-cred class that silently
    empties the channel), else r2_write for any other first-write failure. String inspection only, so
    no boto3 import is needed here; the marker set never contains a secret VALUE, only error codes."""
    blob = f"{type(exc).__name__}: {exc}"
    return "r2_auth" if any(m in blob for m in _R2_AUTH_MARKERS) else "r2_write"


class VerifyEmitter:
    """Writes a verify run's events to R2 (summary.json + events.ndjson) and mirrors each to stdout.

    Single run, single process: it owns the in-memory record list and the summary it rewrites on
    each emit. R2 writes are best-effort (a channel failure must never abort the render before it can
    emit `error`); the stdout mirror is unconditional so the log transport is always populated."""

    def __init__(self, store, run_id: str, *, prefix: str = "verify",
                 log: Callable[[str], Any] = print, clock: Callable[[], float] = time.time):
        self.store = store
        self.run_id = str(run_id)
        self.prefix = prefix
        self._log = log
        self._clock = clock
        self._seq = 0
        self._records: list[dict] = []
        self._summary: dict[str, Any] = {
            "schema": SCHEMA, "run_id": self.run_id, "status": "running",
            "started_ts": None, "updated_ts": None, "last_event": None,
            "error": None, "events": self._records,
        }

    def preflight(self) -> None:
        """AUTHORITATIVE first write: prove the channel is writable BEFORE any probe/render. The
        per-event `_write` is best-effort by design (a mid-run R2 hiccup must never abort the render),
        but that same swallow turns a bad credential into a SILENT empty channel: a quote-wrapped /
        malformed R2 secret passes `R2Config.from_env` (presence-only) yet fails auth on the first
        put_bytes, so every event vanishes and the harness sees nothing (the #249/#77 silent-degrade
        class -- it cost a multi-pod hunt). So this ONE write is authoritative: it seeds the running
        summary and, on failure, raises `ChannelUnwritable` so the caller emits a LOUD verify_fatal
        instead of a 30-minute empty prefix. A None store (the CPU tests / no-sink path) is a no-op."""
        if self.store is None:
            return
        try:
            self.store.put_bytes(
                json.dumps(self._summary, separators=(",", ":")).encode("utf-8"),
                keys.verify_summary_key(self.run_id, prefix=self.prefix),
                content_type="application/json")
        except Exception as exc:  # noqa: BLE001 -- classify + re-raise as a loud fatal, never swallow
            raise ChannelUnwritable(
                _classify_write_error(exc),
                f"verify channel first write failed ({type(exc).__name__}: {str(exc)[:200]}); "
                "R2 creds present but unusable (bad/quote-wrapped value?)") from exc

    # --- public emit API ---

    def emit(self, event: str, **payload) -> dict:
        """Record one event and fan it out (R2 + stdout), all best-effort. Returns the record."""
        ts = round(self._clock(), 3)
        rec = {"seq": self._seq, "ts": ts, "event": str(event), "payload": dict(payload)}
        self._seq += 1
        self._records.append(rec)
        s = self._summary
        if s["started_ts"] is None:
            s["started_ts"] = ts
        s["updated_ts"] = ts
        s["last_event"] = rec["event"]
        self._human(rec)
        self._write()
        return rec

    def gpu_probe(self, facts: dict) -> dict:
        return self.emit("gpu_probe", **gpu_probe_payload(facts))

    def first_frame(self, seconds: float) -> dict:
        return self.emit("first_frame", seconds=round(float(seconds), 3))

    def sharpness(self, value: float, baseline: float) -> dict:
        return self.emit("sharpness", value=round(float(value), 4), baseline=round(float(baseline), 4))

    def complete(self, output_key: str, clip_bytes: int) -> dict:
        self._summary["status"] = "complete"
        return self.emit("complete", output_key=str(output_key), clip_bytes=int(clip_bytes))

    def error(self, stage: str, message: object) -> dict:
        self._summary["status"] = "error"
        self._summary["error"] = {"stage": str(stage), "message": str(message)[:500]}
        return self.emit("error", stage=str(stage), message=str(message)[:500])

    def fatal(self, stage: str, missing: list[str], message: object) -> dict:
        """A FATAL PRECONDITION -- the verify entrypoint could not even start (missing run id, an
        incomplete/misnamed R2 config). Distinct from `error` (a stage that started and failed): it
        carries `missing`, the structured list of absent env names, so the harness/log diagnoses the
        cause without prose-parsing. Sets terminal status like `error`; on a store-less emitter it
        still mirrors to stdout, which is the whole point (the failure can never be silent)."""
        self._summary["status"] = "error"
        self._summary["error"] = {"stage": str(stage), "message": str(message)[:500]}
        return self.emit("verify_fatal", stage=str(stage), missing=list(missing),
                         message=str(message)[:500])

    # --- transports (best-effort) ---

    def _human(self, rec: dict) -> None:
        try:
            self._log("@event " + rec["event"] + " " + json.dumps(rec["payload"], sort_keys=True))
        except Exception:
            pass

    def _write(self) -> None:
        if self.store is None:
            return
        try:
            ndjson = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in self._records)
            self.store.put_bytes(
                ndjson.encode("utf-8"),
                keys.verify_events_key(self.run_id, prefix=self.prefix),
                content_type="application/x-ndjson")
            summary = json.dumps(self._summary, separators=(",", ":"))
            self.store.put_bytes(
                summary.encode("utf-8"),
                keys.verify_summary_key(self.run_id, prefix=self.prefix),
                content_type="application/json")
        except Exception:
            pass  # a channel write must NEVER abort the verify render


# --------------------------------------------------------------------------- orchestration

@dataclass
class RenderOutcome:
    """What a verify draft render produced. The render seam emits `first_frame` itself (measured from
    the render's progress channel); this carries the terminal facts run_verify needs to emit
    `sharpness` + `complete`. The seam hands back the rendered first frame (array-like) so sharpness is
    computed in one place."""
    output_key: str
    clip_bytes: int
    first_frame_array: Any = None            # the decoded first frame; sharpness computed from it
    sharpness_value: float | None = None     # or a precomputed value, if the seam measured it itself


RenderFn = Callable[["VerifyEmitter"], RenderOutcome]


def run_verify(store, *, run_id: str, render: RenderFn,
               probe: Callable[[], dict] = collect_gpu_facts,
               baseline: float = DEFAULT_SHARPNESS_BASELINE,
               prefix: str = "verify",
               log: Callable[[str], Any] = print,
               clock: Callable[[], float] = time.time) -> VerifyEmitter:
    """Drive one verify run and return the emitter (its `_summary` is the final state).

    Order: emit `gpu_probe` from the probe, then run the injected `render` (which emits `first_frame`
    and returns the outcome), then emit `sharpness` + `complete`. Any failure is caught and
    turned into an `error` event (status -> error), so a crash still leaves a terminal, machine-
    readable channel for the harness to read rather than a hung `running` state."""
    emitter = VerifyEmitter(store, run_id, prefix=prefix, log=log, clock=clock)
    try:
        emitter.preflight()   # authoritative: a bad-cred channel must fail LOUD, not silently-empty
    except ChannelUnwritable as e:
        emitter.fatal(e.stage, [], e.message)   # verify_fatal to stdout; status -> error
        return emitter
    try:
        emitter.gpu_probe(probe())
    except Exception as e:  # noqa: BLE001 -- a probe crash is a check failure, recorded not raised
        emitter.error("gpu_probe", e)
        return emitter
    try:
        outcome = render(emitter)
        value = outcome.sharpness_value
        if value is None:
            value = frame_sharpness(outcome.first_frame_array) if outcome.first_frame_array is not None else 0.0
        emitter.sharpness(value, baseline)
        emitter.complete(outcome.output_key, outcome.clip_bytes)
    except Exception as e:  # noqa: BLE001 -- surface the failure on the channel, then propagate up
        emitter.error("render", e)
    return emitter


# --------------------------------------------------------------------------- pod-only GPU render seam

def _pod_draft_render(emitter: "VerifyEmitter", env: dict) -> RenderOutcome:
    """RUN ON THE POD ONLY. Drive one draft-tier render through the REAL serverless entrypoint
    (`worker.handler`) and hand back the terminal facts. Deferred imports (the worker pulls in
    torch/boto3/the GPU pipeline) so this module stays CPU-importable and the unit tests never reach
    this path. Live proof is the authorized S1 pod run (docs/release-gate.md), exactly like the
    harness's own RunpodMcpPodClient seam.

    We delegate to `worker.handler`, NOT the inner `harness.handler.handler`: the worker entrypoint
    REGISTERS the per-job GPU pipeline (register_pipeline(build_pipeline(req))) before handing off,
    which the harness handler assumes was already done. `python -m vivijure_backend.verify` is a
    different entrypoint than the worker, so calling the harness handler directly left the pipeline
    registry empty and the render failed with "no GPU Pipeline registered" (found on the S1 watched
    pod). Routing through worker.handler means the verify render is byte-identical to a real
    serverless render, registration included, and can never drift from it. `first_frame`
    is then MEASURED from the render's own structured progress channel (the ProgressEmitter ndjson the
    render already wrote to R2): the first `i2v_step`/`keyframe_done` timestamp minus the `started`
    timestamp is the true time-to-first-frame -- no progress hook to race, no second source of truth.

    Inputs come from the environment the harness sets on the pod:
      VJ_VERIFY_BUNDLE_KEY  -- the draft project bundle the harness staged in R2 (bundles/...).
      VJ_VERIFY_PROJECT     -- the project name for the job/keys (default 'verify')."""
    from pathlib import Path
    import tempfile

    from . import worker as _worker
    from .harness.r2 import R2, R2Config

    bundle_key = str(env.get("VJ_VERIFY_BUNDLE_KEY", "")).strip()
    if not bundle_key:
        raise RuntimeError("VJ_VERIFY_BUNDLE_KEY is required on the pod: the harness stages a draft "
                           "bundle in R2 and names its key here")
    project = str(env.get("VJ_VERIFY_PROJECT", "verify")).strip() or "verify"
    job_id = "verify-" + emitter.run_id

    job = {"id": job_id, "input": {
        "action": "render", "project": project,
        "bundle_key": bundle_key, "quality_tier": "draft",
    }}
    result = _worker.handler(job)

    output_key = result.get("output_key") or ""
    if not output_key:
        raise RuntimeError("render returned no output_key")

    store = R2(R2Config.from_env(env))
    emitter.first_frame(_first_frame_seconds(store, project, job_id))

    # Pull the finished clip back and measure the first frame for the sharpness gate.
    workdir = Path(tempfile.mkdtemp(prefix="vj-verify-"))
    clip_path = workdir / "verify-out.mp4"
    store.get_file(output_key, clip_path)
    clip_bytes = clip_path.stat().st_size
    frame = _first_frame_array(clip_path)
    return RenderOutcome(output_key=output_key, clip_bytes=clip_bytes, first_frame_array=frame)


def _first_frame_seconds(store, project: str, job_id: str) -> float:
    """RUN ON THE POD ONLY. Measure time-to-first-frame from the render's own progress ndjson in R2:
    the first generated-frame event (`i2v_step`, else `keyframe_done`, else the first non-`started`
    event) timestamp minus the `started` timestamp. Best-effort: an unreadable channel yields 0.0
    (the harness then fails the first_frame bound honestly rather than the probe crashing)."""
    from .harness import keys
    try:
        raw = store.get_bytes(keys.progress_log_key(project, job_id))
        recs = [json.loads(line) for line in raw.decode("utf-8", "replace").splitlines() if line.strip()]
    except Exception:  # noqa: BLE001
        return 0.0
    started = next((r.get("ts") for r in recs if r.get("event") == "started"), None)
    if started is None:
        return 0.0
    for want in ("i2v_step", "keyframe_done"):
        ts = next((r.get("ts") for r in recs if r.get("event") == want), None)
        if ts is not None:
            return round(float(ts) - float(started), 3)
    first = next((r.get("ts") for r in recs if r.get("event") != "started"), None)
    return round(float(first) - float(started), 3) if first is not None else 0.0


def _first_frame_array(clip_path):
    """RUN ON THE POD ONLY. Decode the first frame of an mp4 to an array for the sharpness metric.
    Deferred cv2 import; returns None on any decode failure so sharpness degrades to 0.0 (a check
    failure) rather than crashing the run."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(clip_path))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        return frame if ok else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- entrypoint

def main(argv: list[str] | None = None, env: dict | None = None,
         store: Any = None, render: RenderFn | None = None) -> int:
    """`python -m vivijure_backend.verify`. Hard no-op unless VJ_VERIFY is armed (so a normal render
    is untouched). Armed: build the R2 store, run the draft verify, and return 0 iff the channel
    reached a `complete` status (not `error`). Tests inject `env`/`store`/`render`; the pod path uses
    the real R2 store and `_pod_draft_render`.

    NO SILENT FAILURE: every fatal precondition (missing VJ_VERIFY_RUN_ID, an incomplete/misnamed R2
    config) emits a structured terminal `verify_fatal {stage, missing}` event to stdout BEFORE returning non-zero, so a
    launch-side env mistake can never present as a 30-minute empty prefix with nothing to read."""
    e = env if env is not None else os.environ
    if not is_enabled(e):
        print("@event verify_skipped " + json.dumps({"reason": "VJ_VERIFY not set"}, sort_keys=True))
        return 0

    prefix = key_prefix_from(e)

    run_id = run_id_from(e)
    if not run_id:
        # No run id => no R2 key is even possible, but the failure must still be LOUD, never a silent
        # empty prefix (the exact 30-min-of-nothing failure a stubbed read_logs turns invisible). Emit
        # a structured terminal `error` through a store-less emitter so stdout -- the redundant
        # transport -- always carries the reason, then exit.
        VerifyEmitter(None, "unkeyed", prefix=prefix).fatal(
            "config", ["VJ_VERIFY_RUN_ID"], "VJ_VERIFY_RUN_ID is required when VJ_VERIFY is set")
        return 2

    baseline = baseline_from(e)

    if store is None:
        from .harness.r2 import R2, R2Config
        try:
            store = R2(R2Config.from_env(e))
        except Exception as exc:  # noqa: BLE001 -- a bad/misnamed R2 config must fail LOUD, not silent
            # R2 itself is the failure, so we cannot record it TO R2 (the same bind the handler has);
            # emit `verify_fatal` with the MISSING env names to stdout through a store-less emitter so a
            # cred-NAME mismatch (the F17 class: creds passed as R2_S3_*/AWS_* instead of R2_*) is never
            # an invisible hang -- the harness reads the missing list, no prose-parsing.
            missing = [k for k in R2_ENV_NAMES if not e.get(k)]
            VerifyEmitter(None, run_id, prefix=prefix).fatal("r2_config", missing, exc)
            return 1
    if render is None:
        render = lambda em: _pod_draft_render(em, e)  # noqa: E731 -- bind the env into the seam

    emitter = run_verify(store, run_id=run_id, render=render, baseline=baseline, prefix=prefix)
    return 0 if emitter._summary["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover -- the pod entrypoint
    raise SystemExit(main())
