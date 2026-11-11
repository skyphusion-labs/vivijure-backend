"""Watch a RunPod endpoint's queue and worker states through a burst, read only (#545, option D).

    RUNPOD_API_KEY=... python3 scripts/burst_watch.py <endpoint_id> [--interval 15] [--minutes 20]

The #545 ruling (lead, 2026-11-09): keep QUEUE_DELAY 4 (option A), and measure the first burst
after a code-only release, with RUNNING and THROTTLED worker counts per minute (option D). Change
the scaler (option B) only if jobs still queue while the scaler is not bringing a worker up.

This is the instrument for D. It only ever calls GET https://api.runpod.ai/v2/<id>/health: it
submits nothing, changes no setting and costs nothing. Start it, then let the burst happen (a
release's first real traffic, or a burst someone else submits). It prints one JSON line per sample
(`@event burst.sample {...}`) and, at the end, one `@event burst.summary {...}` with, per minute:
the most workers seen in each state, the most jobs in queue, and whether that minute had jobs
queued while no worker was initializing (`queuedNotScaling`, the option B question), and whether
it had jobs queued while a worker stood ready (`queuedWithReady`, a dispatch delay, not a scaler one).

/health is RunPod's own rollup and can lag the per-worker view (list-endpoint-workers); read a
surprising minute against that before acting on it.

The key comes from RUNPOD_API_KEY and is never printed. A sample that fails is recorded as an
error sample, not dropped, so a gap in the data is visible.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "https://api.runpod.ai/v2"
STATES = ("idle", "initializing", "ready", "running", "throttled", "unhealthy")
JOB_FIELDS = ("inQueue", "inProgress", "completed", "failed", "retried")


def fetch_health(endpoint_id, key, timeout=20):
    req = urllib.request.Request(f"{BASE}/{endpoint_id}/health", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def sample(health, t):
    workers = health.get("workers") or {}
    jobs = health.get("jobs") or {}
    return {
        "t": t,
        "workers": {s: int(workers.get(s) or 0) for s in STATES},
        "jobs": {f: int(jobs.get(f) or 0) for f in JOB_FIELDS},
    }


def summarize(samples):
    """Per minute since the first sample: max workers per state, max inQueue, and queuedNotScaling."""
    good = [s for s in samples if "error" not in s]
    if not good:
        return {"samples": len(samples), "errors": len(samples), "minutes": []}
    t0 = good[0]["t"]
    minutes = {}
    for s in good:
        m = int((s["t"] - t0) // 60)
        row = minutes.setdefault(m, {"minute": m, "samples": 0, "maxInQueue": 0, "queuedNotScaling": False,
                                     "queuedWithReady": False,
                                     **{f"max_{st}": 0 for st in STATES}})
        row["samples"] += 1
        row["maxInQueue"] = max(row["maxInQueue"], s["jobs"]["inQueue"])
        for st in STATES:
            row[f"max_{st}"] = max(row[f"max_{st}"], s["workers"][st])
        if s["jobs"]["inQueue"] > 0 and s["workers"]["initializing"] == 0:
            row["queuedNotScaling"] = True
        if s["jobs"]["inQueue"] > 0 and s["workers"]["ready"] > 0:
            row["queuedWithReady"] = True
    rows = [minutes[k] for k in sorted(minutes)]
    return {
        "samples": len(samples),
        "errors": len(samples) - len(good),
        "minutes": rows,
        "minutesQueuedNotScaling": [r["minute"] for r in rows if r["queuedNotScaling"]],
        "minutesQueuedWithReady": [r["minute"] for r in rows if r["queuedWithReady"]],
        "peakRunning": max(r["max_running"] for r in rows),
        "peakThrottled": max(r["max_throttled"] for r in rows),
        "peakInQueue": max(r["maxInQueue"] for r in rows),
    }


def watch(endpoint_id, key, interval, minutes, fetch=fetch_health, now=time.time, sleep=time.sleep, out=sys.stdout):
    samples = []
    end = now() + minutes * 60
    while True:
        t = now()
        try:
            s = sample(fetch(endpoint_id, key), t)
        except Exception as e:  # noqa: BLE001 -- a failed read is data, not a crash
            s = {"t": t, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        samples.append(s)
        out.write("@event burst.sample " + json.dumps(s, sort_keys=True) + "\n")
        out.flush()
        if t + interval > end:
            break
        sleep(interval)
    summary = {"endpoint": endpoint_id, "interval": interval, **summarize(samples)}
    out.write("@event burst.summary " + json.dumps(summary, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("endpoint_id")
    ap.add_argument("--interval", type=int, default=15, help="seconds between samples (default 15)")
    ap.add_argument("--minutes", type=int, default=20, help="how long to watch (default 20)")
    a = ap.parse_args(argv)
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        sys.exit("RUNPOD_API_KEY is not set")
    if a.interval < 5 or a.minutes < 1:
        sys.exit("--interval must be at least 5 and --minutes at least 1")
    watch(a.endpoint_id, key, a.interval, a.minutes)


if __name__ == "__main__":
    main()
