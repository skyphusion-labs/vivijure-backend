"""Best-effort diagnostic sink for RunPod /job-done POST rejections (#90).

Why this exists: when RunPod rejects a job-done callback (observed: a 400 on a successful job),
the failure prints ONLY to serverless-worker stdout, which is not retrievable via GraphQL /
runpodctl / MCP. This mirrors that same record into the run-scoped R2 channel so the rejection is
inspectable in-band, next to the render's other progress objects, before we root-cause it.

Purely ADDITIVE and purely BEST-EFFORT: it never retries the post, never changes the job outcome,
and every write is wrapped so a diagnostic failure can never propagate into the worker. The
job-done rejection is already non-fatal (the terminal status comes from the handler's own
return/raise, and /status shows the real verdict); this only makes the rejection observable.

The SDK's _transmit patch (worker._patch_runpod_content_type) that catches the rejection runs
with NO run context -- only the URL and the posted body, neither of which reliably carries the
project or job id (the pointer-only result return omits both). So the handler REGISTERS the run
context (the R2 store it already built, plus project + job id) at job start, and the patch calls
record() on a rejection. RunPod serverless processes one job at a time per worker, so the
last-registered context is the job whose result is being posted -- including the terminal post
that fires just after the handler returns.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from . import keys

# One worker processes one job at a time, but the terminal result post and any late status-mirror
# posts run on different threads (the SDK spawns daemon threads for mirrors), so the shared context
# and the record accumulator are guarded by a lock.
_lock = threading.Lock()
_ctx: dict[str, Any] = {"store": None, "project": None, "job_id": None}
_records: list[dict] = []


def register(store: Any, project: str, job_id: str) -> None:
    """Record the current job's R2 store + identity so a later job-done rejection can be mirrored
    run-scoped, and reset the per-job record accumulator. Called by the handler at job start.
    Best-effort; never raises."""
    try:
        with _lock:
            _ctx["store"] = store
            _ctx["project"] = str(project)
            _ctx["job_id"] = str(job_id)
            _records.clear()
    except Exception:
        pass


def redact_url(url: Any) -> str:
    """Strip the query string from a job-done URL before it lands in the channel. RunPod's
    job-done URL carries only gpu / stream flags in the query today (no secret), but the path is
    what identifies the post, and dropping the query keeps any future token-bearing query out of
    the durable channel. Idempotent."""
    return str(url).split("?", 1)[0]


def build_record(*, url: Any, posted_status: Any, http_status: Any, body: Any,
                 content_type: Any) -> dict:
    """The one canonical shape of a job-done rejection record (stdout + R2 share it). `body` is
    truncated to 500 chars to keep the object small; `url` is redacted (query stripped)."""
    return {
        "status": int(http_status),
        "body": str(body)[:500],
        "content_type": content_type,
        "url": redact_url(url),
        "posted_status": posted_status,
    }


def record(*, url: Any, posted_status: Any, http_status: Any, body: Any,
           content_type: Any) -> dict:
    """Mirror one job-done rejection into the run-scoped R2 channel, best-effort, and return the
    record (so the caller can also log the same shape to stdout).

    Appends to an in-memory list and rewrites the whole (tiny) NDJSON object, mirroring the
    progress channel's rewrite-on-each-emit design (no S3 append, no second writer to race). A
    missing store or context (e.g. an R2-config failure, where R2 itself is the failure) or an R2
    error is swallowed -- stdout still carries the record. NEVER raises: a diagnostic write must
    not add a failure mode to the worker."""
    rec = build_record(url=url, posted_status=posted_status, http_status=http_status,
                        body=body, content_type=content_type)
    try:
        with _lock:
            store = _ctx["store"]
            project = _ctx["project"]
            job_id = _ctx["job_id"]
            _records.append(rec)
            if store is None or project is None or job_id is None:
                return rec
            ndjson = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in _records)
            store.put_bytes(
                ndjson.encode("utf-8"),
                keys.job_done_error_key(project, job_id),
                content_type="application/x-ndjson")
    except Exception:
        pass  # a diagnostic write must NEVER fail (or slow a retry of) the worker
    return rec
