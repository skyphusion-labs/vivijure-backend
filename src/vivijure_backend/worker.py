"""RunPod worker entry point: build the GPU pipeline for a job and run it.

This is the module the worker image runs. For each job it builds a `GpuPipeline` from the job's
typed `RenderConfig`, registers it on the harness seam, and delegates to the harness's `handler`
(which mirrors models on a cold worker, opens R2, plans on the CPU, and drives the pipeline). The
`ModelServer` is a process global so a warm worker reuses loaded models across jobs; only the
lightweight per-job pipeline wrapper is rebuilt, carrying that job's config.

CPU-importable: the torch-touching `ModelServer` is created lazily on first real GPU use, so this
module (and `build_pipeline`) import and test without a GPU.
"""
from __future__ import annotations

import dataclasses

from .contract import RenderRequest
from .harness.pipeline_registry import register_pipeline
from .pipeline import GpuPipeline

_SERVER = None  # process-global ModelServer; warm workers reuse loaded models across jobs


def _server(req: RenderRequest | None = None):
    """Return the process-global ModelServer, building it on first call.

    Model loading is expensive and models are shared across jobs on a warm worker, so the server
    is a singleton. On first call, specs from `req.config` (keyframe base/distill, i2v base/distill)
    are wired in so a non-default model repo configured in the job DOES take effect -- provided this
    is the cold-start job. On subsequent calls (warm worker, server already built) the existing
    model set is reused regardless of `req.config.*.model` fields; per-job hot-swapping requires a
    pod restart. If `req` is None the server uses DEFAULT_SPECS (e.g. standalone test usage)."""
    global _SERVER
    if _SERVER is None:
        from .models import ModelServer, ModelRole, DEFAULT_SPECS  # deferred: torch
        job_specs: dict = {}
        if req is not None:
            kc, ic = req.config.keyframe, req.config.i2v
            # Use dataclasses.replace to override ONLY the repo_id, preserving weight_name and
            # all other spec fields (KEYFRAME_FEWSTEP carries weight_name=<specific .safetensors>;
            # rebuilding ModelSpec positionally drops it and breaks the distill LoRA load).
            job_specs = {
                ModelRole.KEYFRAME_BASE: dataclasses.replace(
                    DEFAULT_SPECS[ModelRole.KEYFRAME_BASE], repo_id=kc.base_model),
                ModelRole.KEYFRAME_FEWSTEP: dataclasses.replace(
                    DEFAULT_SPECS[ModelRole.KEYFRAME_FEWSTEP], repo_id=kc.distill_model),
                ModelRole.I2V: dataclasses.replace(
                    DEFAULT_SPECS[ModelRole.I2V], repo_id=ic.model),
                ModelRole.I2V_DISTILL: dataclasses.replace(
                    DEFAULT_SPECS[ModelRole.I2V_DISTILL], repo_id=ic.distill_model),
            }
        _SERVER = ModelServer(specs=job_specs or None)
    return _SERVER


def build_pipeline(req: RenderRequest) -> GpuPipeline:
    """The GPU pipeline for one job: the job's typed config + its pretrained-LoRA references,
    over the shared model server. The server is initialized from `req.config` model fields on
    the first (cold-start) job; subsequent jobs reuse the already-loaded model set."""
    server = _server(req)
    _warn_model_divergence(req, server)
    return GpuPipeline(config=req.config, pretrained_loras=req.pretrained_loras, server=server)


def _warn_model_divergence(req: RenderRequest, server) -> None:
    """Emit a structured warning when a warm-worker job requests different models than those loaded.

    Specs freeze from the cold-start job. A warm worker cannot swap models without a restart; if it
    silently renders on the wrong set the result looks valid but is produced by the wrong model.
    The warning is best-effort (no import of torch/GPU deps) and printed to stdout so it appears in
    RunPod logs and can be scraped; it does not fail the job (that requires a pod-restart policy)."""
    try:
        from .models import ModelRole, DEFAULT_SPECS
        kc, ic = req.config.keyframe, req.config.i2v
        checks = {
            ModelRole.KEYFRAME_BASE: kc.base_model,
            ModelRole.KEYFRAME_FEWSTEP: kc.distill_model,
            ModelRole.I2V: ic.model,
            ModelRole.I2V_DISTILL: ic.distill_model,
        }
        mismatches = {
            role: (loaded.repo_id, requested)
            for role, requested in checks.items()
            if (loaded := server.specs.get(role)) and loaded.repo_id != requested
        }
        if mismatches:
            import json, time
            payload = {
                "ts": time.time(),
                "mismatches": {r.name: {"loaded": l, "requested": req}
                               for r, (l, req) in mismatches.items()},
            }
            print("@event model_spec_divergence " + json.dumps(payload), flush=True)
    except Exception:
        pass  # never let a warning abort a job


def handler(job: dict) -> dict:
    """RunPod serverless entry. Build the per-job pipeline from the request's RenderConfig,
    register it, and hand off to the harness (which owns model mirror, R2, plan, finish).

    finish_clip actions skip pipeline registration: they only need the ModelServer for RIFE
    and face-restore, which run_finish_job creates directly, so there is no render pipeline
    to register."""
    from .harness.handler import handler as harness_handler

    payload = job.get("input", job)
    if str(payload.get("action", "render")) != "finish_clip":
        register_pipeline(build_pipeline(RenderRequest.from_dict(payload)))
    return harness_handler(job)


def main() -> None:
    """The container's main process: start the RunPod serverless loop with our handler. The
    `runpod` SDK import is deferred so this module stays CPU/dep-light for tests; the worker
    image installs it. `python -m vivijure_backend.worker` runs this."""
    import runpod  # the worker image's serverless SDK
    _patch_runpod_content_type()
    runpod.serverless.start({"handler": handler})


def _patch_runpod_content_type() -> None:
    """Patch the RunPod SDK to send job-done callbacks as application/json and capture the
    response body on failure so we can diagnose 4xx rejections.

    The SDK's _transmit() hardcodes Content-Type: application/x-www-form-urlencoded while
    sending a JSON body, and RunPod's job-done endpoint rejects this with 400. Using aiohttp's
    json= kwarg lets the session encode properly and sets Content-Type automatically. No-ops
    and logs if SDK internals change -- never silently breaks the worker startup."""
    try:
        import json as _json
        import runpod.serverless.modules.rp_http as _rp_http

        async def _patched_transmit(client_session, url, job_data):
            import aiohttp as _aiohttp
            async with client_session.post(
                url,
                json=_json.loads(job_data),
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    import json as _j
                    print("@event job_done_error " + _j.dumps({
                        "status": resp.status,
                        "body": body[:500],
                        "content_type": resp.content_type,
                    }), flush=True)
                    raise _aiohttp.ClientResponseError(
                        resp.request_info, resp.history,
                        status=resp.status, message=body[:200],
                    )

        _rp_http._transmit = _patched_transmit
        print("@event patch_applied {\"patch\": \"job_done_content_type\"}", flush=True)
    except Exception as exc:
        print(f"@event patch_failed {{\"patch\": \"job_done_content_type\", \"error\": \"{exc}\"}}", flush=True)


if __name__ == "__main__":
    main()
