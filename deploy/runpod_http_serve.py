"""RunPod-compatible job API over stdlib HTTP for homelab LOCAL_FINISH_* services.

Sidecars POST /run {"input": {...}} -> {"id"} and poll GET /status/<id> with the same envelope
RunPod serverless uses (IN_QUEUE / IN_PROGRESS / COMPLETED / FAILED). Stdlib only.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import signal
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


class JobStatus(str, Enum):
    IN_QUEUE = "IN_QUEUE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    payload: dict
    status: JobStatus = JobStatus.IN_QUEUE
    output: dict | None = None
    error: str | None = None
    _cancel: bool = field(default=False, repr=False)

    def status_dict(self) -> dict:
        d: dict = {"id": self.id, "status": self.status.value}
        if self.status is JobStatus.COMPLETED and self.output is not None:
            d["output"] = self.output
        if self.status is JobStatus.FAILED and self.error is not None:
            d["error"] = self.error
        return d


RunFn = Callable[[dict, Callable[[], bool]], dict]


class JobRegistry:
    def __init__(self, run_fn: RunFn, *, max_completed: int = 256) -> None:
        self._run_fn = run_fn
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._completed_order: deque[str] = deque()
        self._max_completed = max_completed
        self._worker: threading.Thread | None = None
        self._wake = threading.Condition(self._lock)
        self._stop = False

    def submit(self, payload: dict) -> str:
        job = Job(id=uuid.uuid4().hex, payload=payload)
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._ensure_worker_locked()
            self._wake.notify()
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return True
            if job.status is JobStatus.IN_QUEUE:
                try:
                    self._queue.remove(job_id)
                except ValueError:
                    pass
                job.status = JobStatus.FAILED
                job.error = "canceled before start"
                self._retain_locked(job_id)
                return True
            if job.status is JobStatus.IN_PROGRESS:
                job._cancel = True
            return True

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run_loop, name="finish-serve-jobs", daemon=True)
            self._worker.start()

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                while not self._queue and not self._stop:
                    self._wake.wait()
                if self._stop and not self._queue:
                    return
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None or job.status is not JobStatus.IN_QUEUE:
                    continue
                if job._cancel:
                    job.status = JobStatus.FAILED
                    job.error = "canceled before start"
                    self._retain_locked(job_id)
                    continue
                job.status = JobStatus.IN_PROGRESS
            try:
                output = self._run_fn(job.payload, lambda: self._is_cancelled(job_id))
                with self._lock:
                    job.output = output
                    job.status = JobStatus.COMPLETED
                    self._retain_locked(job_id)
            except Cancelled:
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = "canceled"
                    self._retain_locked(job_id)
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    job.status = JobStatus.FAILED
                    job.error = str(e)[:500]
                    self._retain_locked(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job._cancel)

    def _retain_locked(self, job_id: str) -> None:
        self._completed_order.append(job_id)
        while len(self._completed_order) > self._max_completed:
            old = self._completed_order.popleft()
            self._jobs.pop(old, None)

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
            self._wake.notify_all()


_STATUS_RE = re.compile(r"^/status/([A-Za-z0-9]+)$")
_CANCEL_RE = re.compile(r"^/cancel/([A-Za-z0-9]+)$")


def token_error(headers_token: str | None, expected: str) -> tuple[int, dict] | None:
    if not expected:
        return 503, {"ok": False, "error": "LOCAL_FINISH_TOKEN not configured: refusing open GPU endpoint"}
    if not headers_token or not hmac.compare_digest(headers_token, expected):
        return 401, {"ok": False, "error": "unauthorized"}
    return None


def route(
    method: str,
    path: str,
    body: dict | None,
    *,
    registry: JobRegistry,
    token: str | None,
    expected_token: str,
    service: str,
    version: str = "serve-1",
) -> tuple[int, dict]:
    if method == "GET" and path == "/health":
        return 200, {"ok": True, "service": service, "version": version, "mode": "local-finish-http"}

    if method == "POST" and path == "/run":
        payload = (body or {}).get("input", body or {})
        if (body or {}).get("selftest") or payload.get("selftest"):
            return 200, {"ok": True, "selftest": True, "service": service}
        err = token_error(token, expected_token)
        if err:
            return err
        job_id = registry.submit(payload)
        return 200, {"id": job_id}

    m = _STATUS_RE.match(path)
    if method == "GET" and m:
        err = token_error(token, expected_token)
        if err:
            return err
        job = registry.get(m.group(1))
        if job is None:
            return 404, {"status": 404, "title": "Not Found", "detail": "job not found"}
        return 200, job.status_dict()

    m = _CANCEL_RE.match(path)
    if method == "POST" and m:
        err = token_error(token, expected_token)
        if err:
            return err
        registry.cancel(m.group(1))
        return 200, {"ok": True}

    return 404, {"status": 404, "title": "Not Found", "detail": "no such route"}


def wrap_runpod_handler(handler_fn: Callable[[dict], dict]) -> RunFn:
    """Adapt a RunPod handler(job) to the registry run_fn(payload, should_cancel)."""

    def run(payload: dict, should_cancel: Callable[[], bool]) -> dict:
        if should_cancel():
            raise Cancelled()
        job = {"input": payload}
        result = handler_fn(job)
        if not isinstance(result, dict):
            raise RuntimeError(f"handler returned non-dict: {type(result).__name__}")
        # RunPod marks top-level `error` as FAILED; soft-degrade uses `detail` only.
        if result.get("error"):
            raise RuntimeError(str(result["error"])[:500])
        return result

    return run


def run_serve(
    handler_fn: Callable[[dict], dict],
    *,
    service: str,
    host: str | None = None,
    port: int | None = None,
    token_env: str = "LOCAL_FINISH_TOKEN",
    version: str = "serve-1",
) -> None:
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = int(port or os.environ.get("PORT", "8010") or "8010")
    expected_token = os.environ.get(token_env, "") or ""
    registry = JobRegistry(wrap_runpod_handler(handler_fn))

    class Handler(BaseHTTPRequestHandler):
        def _bearer(self) -> str | None:
            h = self.headers.get("authorization") or ""
            return h[7:] if h.lower().startswith("bearer ") else None

        def _body(self) -> dict | None:
            length = int(self.headers.get("content-length") or 0)
            if not length:
                return None
            try:
                return json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return None

        def _dispatch(self, method: str) -> None:
            status, payload = route(
                method,
                self.path,
                self._body() if method == "POST" else None,
                registry=registry,
                token=self._bearer(),
                expected_token=expected_token,
                service=service,
                version=version,
            )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("POST")

        def log_message(self, *args) -> None:
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)

    def _graceful(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful)

    print(f"{service} LOCAL_FINISH HTTP on {host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        httpd.server_close()
        registry.shutdown()
