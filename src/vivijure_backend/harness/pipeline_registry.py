"""The seam between the harness and the model layer.

The harness must not import torch, but at deploy time it needs a concrete GPU `Pipeline` to
run. The model layer registers one here (typically from the worker image's entry module,
before `handler` is first called); the harness fetches it by `get_pipeline()`. Keeping the
binding here, not an import in `handler`, is what lets the handler stay CPU-importable and the
GPU stages stay independently developed.
"""
from __future__ import annotations

from .handler import Pipeline

_PIPELINE: Pipeline | None = None


def register_pipeline(pipeline: Pipeline) -> None:
    """Bind the concrete GPU `Pipeline` the harness will run. The worker image's entry module
    calls this once, before the first job is handled."""
    global _PIPELINE
    _PIPELINE = pipeline


def get_pipeline() -> Pipeline:
    """Return the registered GPU `Pipeline`; raise if none was registered (a deploy error)."""
    if _PIPELINE is None:
        raise RuntimeError(
            "no GPU Pipeline registered; the worker image's entry module must call "
            "harness.pipeline_registry.register_pipeline(...) before handling a job"
        )
    return _PIPELINE
