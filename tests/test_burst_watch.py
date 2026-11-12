"""scripts/burst_watch.py (#545, option D): the read-only burst instrument. Offline: a fake clock and
a fake /health that plays back a burst."""
import importlib.util
import io
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts" / "burst_watch.py"
spec = importlib.util.spec_from_file_location("burst_watch", PATH)
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)


def health(in_queue=0, running=0, throttled=0, initializing=0, ready=0):
    return {"jobs": {"inQueue": in_queue, "inProgress": running}, "workers": {
        "running": running, "throttled": throttled, "initializing": initializing, "ready": ready}}


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def play(script, interval=30, minutes=2):
    clock = Clock()
    it = iter(script)
    calls = []

    def fetch(ep, key):
        calls.append((ep, key))
        h = next(it)
        if isinstance(h, Exception):
            raise h
        return h

    out = io.StringIO()
    summary = bw.watch("ep1", "k", interval, minutes, fetch=fetch, now=clock.now, sleep=clock.sleep, out=out)
    return summary, out.getvalue().splitlines(), calls


def test_a_cold_burst_reads_per_minute_and_flags_queued_not_scaling():
    # minute 0: 6 queued, nothing coming up (the #545 shape); minute 1: two initializing; minute 2: running
    script = [health(6), health(6, initializing=2), health(4, initializing=2), health(2, running=2, throttled=3),
              health(0, running=2)]
    summary, lines, calls = play(script, interval=30, minutes=2)
    assert len(calls) == 5 and all(c == ("ep1", "k") for c in calls)
    assert [r["minute"] for r in summary["minutes"]] == [0, 1, 2]
    assert summary["minutesQueuedNotScaling"] == [0, 1]
    assert summary["peakRunning"] == 2 and summary["peakThrottled"] == 3 and summary["peakInQueue"] == 6
    assert summary["minutes"][0]["max_initializing"] == 2
    assert summary["minutesQueuedWithReady"] == []

    assert lines[-1].startswith("@event burst.summary ")
    assert all(l.startswith("@event burst.sample ") for l in lines[:-1])


def test_a_failed_read_is_an_error_sample_not_a_gap():
    summary, lines, _ = play([health(1), RuntimeError("HTTP 502"), health(0)], interval=30, minutes=1)
    assert summary["samples"] == 3 and summary["errors"] == 1
    assert json.loads(lines[1].split(" ", 2)[2])["error"].startswith("RuntimeError: HTTP 502")


def test_the_key_is_never_printed():
    clock = Clock()
    out = io.StringIO()
    bw.watch("ep1", "secret-key-123", 30, 1, fetch=lambda e, k: health(), now=clock.now, sleep=clock.sleep, out=out)
    assert "secret-key-123" not in out.getvalue()


def test_only_the_health_route_is_ever_read(monkeypatch):
    seen = []

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"jobs": {}, "workers": {}}'

    def fake_urlopen(req, timeout):
        seen.append((req.get_method(), req.full_url))
        return R()

    monkeypatch.setattr(bw.urllib.request, "urlopen", fake_urlopen)
    bw.fetch_health("xe5d1ivs40ohjo", "k")
    assert seen == [("GET", "https://api.runpod.ai/v2/xe5d1ivs40ohjo/health")]


def test_jobs_queued_beside_a_ready_worker_are_flagged_apart():
    summary, _, _ = play([health(3, ready=2, initializing=1), health(0, running=2)], interval=30, minutes=1)
    assert summary["minutesQueuedWithReady"] == [0]
    assert summary["minutesQueuedNotScaling"] == []
