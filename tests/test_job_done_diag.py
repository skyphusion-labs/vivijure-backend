"""CPU tests for the job-done rejection diagnostic sink (#90): the run-scoped R2 key, the
best-effort R2 mirror, the record shape (status / body / content_type / url / posted_status),
multi-post append, per-job reset, and the never-raise guarantee. Also proves the handler
registers the live store so the SDK _transmit patch can mirror run-scoped. No R2, no GPU."""
import json

import pytest

from vivijure_backend.harness import job_done_diag, keys


class RecordingStore:
    """Captures put_bytes; can be told to fail every write to exercise best-effort."""
    def __init__(self, fail=False):
        self.objects: dict[str, bytes] = {}
        self.fail = fail
        self.writes = 0

    def put_bytes(self, data, key, *, content_type=None, metadata=None):
        self.writes += 1
        if self.fail:
            raise RuntimeError("R2 is down")
        self.objects[key] = data
        return key


@pytest.fixture(autouse=True)
def _reset_diag_state():
    """The sink holds process-global run context + a record accumulator (one worker, one job at a
    time); reset it around every test so state never leaks between them."""
    job_done_diag._ctx.update(store=None, project=None, job_id=None)
    job_done_diag._records.clear()
    yield
    job_done_diag._ctx.update(store=None, project=None, job_id=None)
    job_done_diag._records.clear()


DONE_URL = "https://api.runpod.ai/v2/t9wcvlxh8rc5la/job-done/wk-1/job-1?gpu=NVIDIA&isStream=false"


# --------------------------------------------------------------------------- key layout

def test_key_is_run_scoped_and_distinct_from_progress_snapshot():
    k = keys.job_done_error_key("neon rain", "job-1")
    assert k == "renders/neon_rain/progress/job-1.job-done-errors.ndjson"
    # MUST NOT be the object the control plane polls -- a clobber there is the F17-class bug.
    assert k != keys.progress_snapshot_key("neon rain", "job-1")
    assert k != keys.progress_log_key("neon rain", "job-1")


# ----------------------------------------------------------------- record: R2 mirror + shape

def test_record_writes_run_scoped_ndjson_after_register():
    store = RecordingStore()
    job_done_diag.register(store, "neon rain", "job-1")
    job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                         http_status=400, body="bad request", content_type="text/plain")
    key = keys.job_done_error_key("neon rain", "job-1")
    assert key in store.objects
    rec = json.loads(store.objects[key].decode().strip())
    assert rec["status"] == 400
    assert rec["body"] == "bad request"
    assert rec["content_type"] == "text/plain"
    assert rec["posted_status"] == "COMPLETED"


def test_url_query_is_stripped_in_the_record():
    store = RecordingStore()
    job_done_diag.register(store, "neon", "j")
    rec = job_done_diag.record(url=DONE_URL, posted_status="IN_PROGRESS",
                               http_status=400, body="x", content_type="application/json")
    assert rec["url"] == "https://api.runpod.ai/v2/t9wcvlxh8rc5la/job-done/wk-1/job-1"
    assert "?" not in rec["url"] and "gpu" not in rec["url"]


def test_body_is_truncated_to_500_chars():
    store = RecordingStore()
    job_done_diag.register(store, "neon", "j")
    rec = job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                               http_status=400, body="z" * 5000, content_type="text/html")
    assert len(rec["body"]) == 500


def test_multiple_rejections_in_one_job_append_not_clobber():
    store = RecordingStore()
    job_done_diag.register(store, "neon", "job-A")
    # a late status mirror rejection, then the terminal result rejection
    job_done_diag.record(url=DONE_URL, posted_status="IN_PROGRESS",
                         http_status=400, body="mirror rejected", content_type="text/plain")
    job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                         http_status=400, body="result rejected", content_type="text/plain")
    lines = store.objects[keys.job_done_error_key("neon", "job-A")].decode().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["posted_status"] == "IN_PROGRESS"
    assert json.loads(lines[1])["posted_status"] == "COMPLETED"


def test_register_resets_the_accumulator_between_jobs():
    store = RecordingStore()
    job_done_diag.register(store, "neon", "job-A")
    job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                         http_status=400, body="one", content_type="text/plain")
    # a new job registers -> the next job's file starts fresh, no bleed from job-A
    store2 = RecordingStore()
    job_done_diag.register(store2, "neon", "job-B")
    job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                         http_status=400, body="two", content_type="text/plain")
    lines = store2.objects[keys.job_done_error_key("neon", "job-B")].decode().strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["body"] == "two"


# ---------------------------------------------------------------- best-effort / never fatal

def test_no_context_returns_record_but_writes_nothing():
    # never registered (store None): record() still returns the shape for the stdout line, and
    # raises nothing -- the R2 mirror is simply skipped (stdout stays the source of truth).
    rec = job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                               http_status=400, body="x", content_type="text/plain")
    assert rec["status"] == 400 and rec["posted_status"] == "COMPLETED"


def test_a_failing_store_is_swallowed_never_raises():
    job_done_diag.register(RecordingStore(fail=True), "neon", "job-1")
    # must not raise: a diagnostic write cannot become a worker failure mode
    rec = job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                               http_status=400, body="x", content_type="text/plain")
    assert rec["status"] == 400


def test_register_and_record_never_raise_on_bad_input():
    job_done_diag.register(RecordingStore(), "neon", "job-1")
    # None url / non-int-ish status still must not raise (record swallows and returns a shape)
    rec = job_done_diag.record(url=None, posted_status=None,
                               http_status=400, body=None, content_type=None)
    assert rec["status"] == 400


# ----------------------------------------------------------- handler wiring (registers store)

def test_handler_registers_the_live_store_so_a_rejection_mirrors_run_scoped(monkeypatch):
    from vivijure_backend.harness import handler as H, models_mirror, r2, pipeline_registry

    store = RecordingStore()
    monkeypatch.setattr(r2.R2Config, "from_env", lambda *a, **k: object())
    monkeypatch.setattr(r2, "R2", lambda cfg=None: store)
    # stop the job right after the store is built + registered (mirror gate raises)
    monkeypatch.setattr(models_mirror, "ensure_models",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop after register")))
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda: None)

    with pytest.raises(RuntimeError, match="stop after register"):
        H.handler({"input": {"project": "neon rain"}, "id": "job-reg"})

    # the handler registered the LIVE store; a later job-done rejection now lands run-scoped
    job_done_diag.record(url=DONE_URL, posted_status="COMPLETED",
                         http_status=400, body="rejected", content_type="text/plain")
    assert keys.job_done_error_key("neon rain", "job-reg") in store.objects
