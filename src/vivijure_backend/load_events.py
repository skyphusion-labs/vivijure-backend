"""Structured progress around each slow model load.

A weight load prints nothing useful while it runs. tqdm writes `\\r`, so a log line appears only
when the bar finishes, and a 7 minute UNet read over a volume looks exactly like a hang. These
events say what is loading, and how long it took, on the job's own progress channel.

Best-effort like every other progress path: an emitter failure never reaches the load.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

HEARTBEAT_SECONDS = 30.0

_emitter: Any = None


def set_emitter(emitter: Any) -> None:
    """Point the module at this job's emitter. The ModelServer is process-global, so the emitter
    is set per job rather than passed through every loader."""
    global _emitter
    _emitter = emitter


def _emit(event: str, **fields: Any) -> None:
    emitter = _emitter
    if emitter is None:
        return
    try:
        emitter.emit(event, **fields)
    except Exception:  # noqa: BLE001 -- progress must never fail a render
        pass


@contextmanager
def timed_load(name: str, *, heartbeat_seconds: float = HEARTBEAT_SECONDS) -> Iterator[None]:
    """Emit `load_start`, a `load_heartbeat` while it runs, then `load_done` or `load_failed`."""
    started = time.monotonic()
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(heartbeat_seconds):
            _emit("load_heartbeat", name=name, elapsed_s=round(time.monotonic() - started, 1))

    ticker = threading.Thread(target=beat, name=f"load-heartbeat-{name}", daemon=True)
    _emit("load_start", name=name)
    ticker.start()
    try:
        yield
    except BaseException as e:
        _emit("load_failed", name=name, elapsed_s=round(time.monotonic() - started, 1),
              error=type(e).__name__)
        raise
    else:
        _emit("load_done", name=name, elapsed_s=round(time.monotonic() - started, 1))
    finally:
        stop.set()
        ticker.join(timeout=1.0)
