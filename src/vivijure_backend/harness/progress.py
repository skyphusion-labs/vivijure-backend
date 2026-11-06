"""Structured progress channel for a render: discrete stage events, best-effort, never fatal.

The design fork, decided: **option B (an R2 structured channel) is primary.** The worker already
holds the R2 store and its token, so this adds no new infra and no new secret; it is durable and
queryable after the fact, reusable beyond RunPod, and it matches the @event structured-state
philosophy (machine state alongside the human stdout). **Option A (RunPod's progress_update) is
supported as an optional injected hook**, so a single `emit()` yields human stdout, R2 machine
state, AND the RunPod status field at once.

Per render the worker writes two objects, keyed by project AND job id so concurrent or cancelled
runs never clobber each other (we have had three in flight at once):
  - `renders/<project>/progress/<job_id>.ndjson` : the event stream, one JSON object per line.
    The worker owns the job, so it accumulates events in memory and rewrites the whole (tiny) log
    each emit, no S3 append needed and no second writer to race.
  - `renders/<project>/progress/<job_id>.json`   : the latest snapshot (status, counts, last
    event, error), the cheap thing a `/status/<project>/<job_id>` route or Uptime Kuma polls.

Events are discrete stages, not stdout scraping: `started`, `mirror_done`, `train_done{slot}`,
`keyframe_done{shot}`, `i2v_done{shot}`, `assemble_done`, `upload_done{key}`, `complete`, and
`error{stage}`; plus a throttled `train_step{slot,step,total,loss}` (training is the long pole,
so it is the one place we report sub-stage progress, and only every N steps).

EVERYTHING is best-effort: every R2 write, every stdout line, every hook call is wrapped so a
logging failure is swallowed and can never propagate into the render. We just spent four builds
clearing real crashes; a logging-induced one is not allowed.
"""
from __future__ import annotations

import json
import tempfile
import time
from typing import Any, Callable

from . import keys

# Events whose occurrences we count in the snapshot (the others are lifecycle: started/complete).
_COUNTED = ("train_done", "keyframe_done", "i2v_done", "assemble_done", "upload_done",
            "train_step", "i2v_step")

# Events that carry the container's free disk, so every job records it at its start and at its end
# (vivijure-backend#600: a worker whose model cache filled the disk failed every later job).
_DISK_EVENTS = ("started", "complete", "error")


class ProgressEmitter:
    """Emits a render's stage events to R2 (+ stdout + an optional RunPod hook). Single-job, single
    process: it owns the in-memory event list and the snapshot it rewrites on each emit."""

    def __init__(self, store, project: str, job_id: str, *,
                 on_progress: Callable[[dict], Any] | None = None,
                 log: Callable[[str], Any] = print, clock: Callable[[], float] = time.time,
                 disk_path: str | None = None):
        self.store = store
        # Where the job's own files live: the temp root the handler makes each job's workdir under.
        self._disk_path = disk_path if disk_path is not None else tempfile.gettempdir()
        self.project = project
        self.job_id = str(job_id)
        self.on_progress = on_progress          # option A hook: callable(snapshot) best-effort
        self._log = log
        self._clock = clock
        self._events: list[dict] = []
        self._snapshot: dict[str, Any] = {
            "project": project, "job_id": self.job_id, "status": "running",
            "started_ts": None, "updated_ts": None,
            "counts": {}, "last_event": None, "error": None,
        }

    # --- public emit API ---

    def emit(self, event: str, **fields) -> None:
        """Record one stage event and fan it out (R2 + stdout + hook), all best-effort."""
        if event in _DISK_EVENTS and "disk_free_gb" not in fields:
            # Best-effort: None when it cannot be measured, never a raise.
            from .disk import free_gb
            fields["disk_free_gb"] = free_gb(self._disk_path)
        rec = {"ts": round(self._clock(), 3), "event": event, **fields}
        self._events.append(rec)
        self._update_snapshot(rec)
        self._human(rec)
        self._write()
        self._hook()

    def complete(self, **fields) -> None:
        self._snapshot["status"] = "complete"
        self.emit("complete", **fields)

    def error(self, stage: str, message: object) -> None:
        self.emit("error", stage=stage, message=str(message)[:500])

    def train_step_cb(self, slot: str) -> Callable[[int, int, float], None]:
        """A throttled per-step training callback for `lora_train.train_slot` to call (it calls
        only every N steps, so no extra throttle here)."""
        def cb(step: int, total: int, loss: float) -> None:
            self.emit("train_step", slot=slot, step=int(step), total=int(total),
                      loss=round(float(loss), 4))
        return cb

    def i2v_step_cb(self, shot: str) -> Callable[[int, int], None]:
        """A per-step i2v callback for `i2v.animate` to call inside the Wan denoise loop. Unlike
        training (1000 steps, throttled every 50), i2v is 4 (draft) to 40 (final) steps at ~30s/step
        on the final tier, so it emits EVERY step: ~40 events naturally paced ~30s apart, the live
        signal that distinguishes a slow i2v from a hung one without SSHing the worker."""
        def cb(step: int, total: int) -> None:
            self.emit("i2v_step", shot=shot, step=int(step), total=int(total))
        return cb

    def finish_cb(self, shot: str) -> Callable[[str, int, int], None]:
        """A per-clip finishing-stage callback for `finish.finish_clip` to call. `stage` is the
        sub-pass ('face_restore' / 'interpolate'); `done`/`total` tick within that pass, so the
        snapshot shows which finish pass a clip is in (the interpolation pass especially can be the
        long pole when an i2v clip has many frames)."""
        def cb(stage: str, done: int, total: int) -> None:
            self.emit("finish_step", shot=shot, stage=str(stage), done=int(done), total=int(total))
        return cb

    # --- internals (every one best-effort) ---

    def _update_snapshot(self, rec: dict) -> None:
        s = self._snapshot
        if s["started_ts"] is None:
            s["started_ts"] = rec["ts"]
        s["updated_ts"] = rec["ts"]
        s["last_event"] = rec
        ev = rec["event"]
        if ev in _COUNTED:
            s["counts"][ev] = s["counts"].get(ev, 0) + 1
        if ev == "error":
            s["status"] = "error"
            s["error"] = {"stage": rec.get("stage"), "message": rec.get("message")}
        elif ev == "complete":
            s["status"] = "complete"

    def _human(self, rec: dict) -> None:
        try:
            body = json.dumps({k: v for k, v in rec.items() if k != "event"}, separators=(",", ":"))
            self._log(f"@event {rec['event']} {body}")
        except Exception:
            pass

    def _write(self) -> None:
        if self.store is None:
            return
        try:
            ndjson = ("\n".join(json.dumps(e, separators=(",", ":")) for e in self._events) + "\n")
            self.store.put_bytes(ndjson.encode("utf-8"),
                                 keys.progress_log_key(self.project, self.job_id),
                                 content_type="application/x-ndjson")
            snap = json.dumps(self._snapshot, separators=(",", ":"))
            self.store.put_bytes(snap.encode("utf-8"),
                                 keys.progress_snapshot_key(self.project, self.job_id),
                                 content_type="application/json")
        except Exception:
            pass  # a progress write must NEVER fail a render

    def _hook(self) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(dict(self._snapshot))
        except Exception:
            pass


class NullEmitter:
    """A no-op emitter so a pipeline runs without a progress channel wired (the default, and what
    the CPU tests use). Same surface as ProgressEmitter, all no-ops."""

    def emit(self, *a, **k) -> None: pass
    def complete(self, *a, **k) -> None: pass
    def error(self, *a, **k) -> None: pass
    def train_step_cb(self, slot: str): return None
    def i2v_step_cb(self, shot: str): return None
    def finish_cb(self, shot: str): return None


def read_snapshot(store, project: str, job_id: str) -> dict | None:
    """Read a job's latest snapshot back (for a /status route or a poll script). Returns None if
    it is not there yet or unreadable; never raises."""
    try:
        return json.loads(store.get_bytes(keys.progress_snapshot_key(project, job_id)))
    except Exception:
        return None
