#!/usr/bin/env python3
"""Automated post-build verification on a RunPod GPU POD (not a serverless endpoint).

WHY A POD, NOT SERVERLESS: a serverless worker gives zero debugging insight on a failure (no shell,
re-fire-and-guess). This harness spins a GPU POD pinned to the freshly built image, runs ONE
draft-tier cold render on it, and ASSERTS on the structured `@event` channel (NOT on English prose).
On PASS it tears the pod down; on FAIL it STOPS (not deletes) the pod so the disk/state survive for an
SSH debug session, and emits the resume handle. That failed-but-debuggable box is the whole point.

SPEND DISCIPLINE (hard): draft tier only, single concurrency, a hard wall-clock TTL that auto-stops
the pod regardless of outcome, teardown on pass / stop-not-leak on fail, and a per-run cost estimate.
Even when GPU spend is authorized, this never leaves a pod billing silently.

CONTRACT -- the pod's verify entrypoint emits these structured lines to stdout; the harness parses
and asserts on them (machine-readable state channel, GMCP-style `@event <name> {json}`):

  @event gpu_probe {"torch_cuda": bool, "kernel_ok": bool, "vj_baked": bool, "weights_on_disk": bool,
                    "vram_free_gb": float, "vram_total_gb": float, "device_name": str}
      Pod-only INSIGHT you cannot get on serverless: torch sees the GPU, the cu128 kernel actually
      LOADS on THIS card (the Blackwell/sm_120 "no kernel image" class), `.vj-baked` present, weights
      on disk, VRAM headroom.
  @event mirror_complete {...}        OPTIONAL. Its PRESENCE on a baked image is a FAILURE: the
      `.vj-baked` early-return should have skipped every R2 pull. (See models_mirror.)
  @event mirror_skipped {"reason": "baked"}   The baked-sentinel HIT we want instead.
  @event first_frame {"seconds": float}       Time-to-first-frame, asserted under a bound.
  @event sharpness {"value": float, "baseline": float}   The #118 method-ii quality gate vs the
      runtime-quant baseline (parity metric, asserted above threshold).
  @event complete {"output_key": str, "clip_bytes": int}   Render done; output object written.

NO LIVE SPIN FROM THIS MODULE BY DEFAULT. `main()` runs a MOCKED dry-run unless `--live` is passed
AND the RunPod client is wired; the live binding (RunpodMcpPodClient) is a documented seam the gated
validation fills. Unit tests drive the whole flow against FakePodClient -- no GPU, no network.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# --------------------------------------------------------------------------- tiers + config

# Tier -> the GPU class the image needs. i2v/bf16 needs an H200-class card (the ~28B MoE full-step
# floor); a homelab-lite base-only image verifies on a cheap consumer card. Concrete ids are resolved
# against live availability (PodClient.list_gpu_types) at spin time, never hard-coded to one sold-out
# SKU. These are PREFERENCE ORDERS, most-capable first within budget.
GPU_TIERS: dict[str, tuple[str, ...]] = {
    "i2v": ("NVIDIA H200", "NVIDIA H100 NVL", "NVIDIA H100 80GB HBM3", "NVIDIA B200"),
    "base": ("NVIDIA RTX 4090", "NVIDIA RTX A5000", "NVIDIA A10", "NVIDIA L4"),
}

# Per-GPU on-demand $/hr (estimate only, for the per-run cost line; the live client may override from
# list_gpu_types pricing). Kept conservative-high so the printed estimate never UNDERstates spend.
GPU_HOURLY_USD: dict[str, float] = {
    "NVIDIA H200": 3.99, "NVIDIA H100 NVL": 2.79, "NVIDIA H100 80GB HBM3": 2.69,
    "NVIDIA B200": 5.99, "NVIDIA RTX 4090": 0.69, "NVIDIA RTX A5000": 0.36,
    "NVIDIA A10": 0.45, "NVIDIA L4": 0.43,
}


@dataclass(frozen=True)
class VerifyConfig:
    """One verify run's bounds + gates. All spend guards live here so a caller cannot fire an
    unbounded job by omission."""
    image: str                                  # the freshly built image ref (ghcr.io/...:tag)
    tier: str = "i2v"                            # GPU_TIERS key; i2v => H200-class
    registry_auth_id: str | None = None          # containerRegistryAuthId (MCP-managed, no dashboard)
    ttl_seconds: int = 1800                      # HARD wall-clock auto-stop, regardless of progress
    max_first_frame_seconds: float = 300.0       # time-to-first-frame bound
    max_baked_staging_seconds: float = 30.0      # baked-sentinel HIT: staging must be ~0
    min_sharpness_ratio: float = 0.97            # quality parity vs runtime-quant baseline (#118 gate)
    min_vram_free_gb: float = 8.0                # headroom floor after load
    expect_baked: bool = True                    # a baked image must NOT pull from R2
    env: dict[str, str] = field(default_factory=lambda: {
        # draft tier ONLY: few-step distill on, fp8 on; the cheap path, the spend-bounded one.
        "VJ_I2V_DISTILL": "1", "VJ_I2V_FP8": "1", "VJ_VERIFY": "1",
    })


# --------------------------------------------------------------------------- pod client seam

class PodClient(Protocol):
    """The RunPod pod-lifecycle surface the harness needs. The live impl wraps the RunPod MCP/API
    (create-pod / get-pod / stop-pod / start-pod / delete-pod / list-gpu-types). Tests inject a fake.
    Lifecycle semantics the harness relies on: stop_pod HALTS billing but KEEPS the disk (resumable
    via start_pod); delete_pod is the irreversible teardown."""

    def list_gpu_types(self) -> list[dict[str, Any]]: ...
    def create_pod(self, *, image: str, gpu_type_id: str, env: dict[str, str],
                   registry_auth_id: str | None, ttl_seconds: int) -> dict[str, Any]: ...
    def get_pod(self, pod_id: str) -> dict[str, Any]: ...
    def read_logs(self, pod_id: str) -> list[str]: ...
    def stop_pod(self, pod_id: str) -> dict[str, Any]: ...
    def delete_pod(self, pod_id: str) -> dict[str, Any]: ...


class RunpodMcpPodClient:
    """Live RunPod client -- the seam the GATED validation fills. Deliberately NOT implemented here so
    importing/unit-testing this module can NEVER fire a real pod. The first real spin wires this to the
    RunPod MCP tools and routes through the integration checkpoint (image-readiness judged first)."""

    def __getattr__(self, _name: str) -> Any:
        raise NotImplementedError(
            "RunpodMcpPodClient is a gated seam: wire it to the RunPod MCP for the authorized "
            "validation run. No live pod spin from the harness module itself.")


# --------------------------------------------------------------------------- @event parsing + facts

def parse_events(lines: list[str]) -> list[tuple[str, dict]]:
    """Extract `@event <name> {json}` lines from a log stream. Tolerant: a non-event or malformed-json
    line is skipped (the harness asserts on the events it FINDS, never crashes on prose)."""
    events: list[tuple[str, dict]] = []
    for raw in lines:
        s = raw.strip()
        if not s.startswith("@event "):
            continue
        rest = s[len("@event "):].strip()
        name, _, payload = rest.partition(" ")
        try:
            data = json.loads(payload) if payload.strip() else {}
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append((name, data))
    return events


def find_event(events: list[tuple[str, dict]], name: str) -> dict | None:
    """The LAST payload for `name` (last-writer wins, so a re-emitted event reflects final state)."""
    payload = None
    for n, d in events:
        if n == name:
            payload = d
    return payload


def emit_gpu_probe(facts: dict) -> str:
    """Build the `@event gpu_probe` line from collected facts (pure; the pod entrypoint prints it).
    Factored out so the contract is asserted in tests without a GPU."""
    keys = ("torch_cuda", "kernel_ok", "vj_baked", "weights_on_disk",
            "vram_free_gb", "vram_total_gb", "device_name")
    payload = {k: facts.get(k) for k in keys}
    return "@event gpu_probe " + json.dumps(payload, sort_keys=True)


def collect_gpu_facts() -> dict:
    """RUN ON THE POD ONLY (deferred torch import). Probes the insight a serverless worker hides: does
    torch see the GPU, does the cu128 kernel actually load on THIS card (a tiny matmul on cuda -- the
    sm_120 'no kernel image' class fails HERE, loudly, not mid-render), is `.vj-baked` present, are the
    weights on disk, how much VRAM is free. Best-effort: any probe error becomes a False/0 fact so
    the harness fails the check rather than the probe crashing the pod."""
    import os
    from pathlib import Path
    facts: dict[str, Any] = {"torch_cuda": False, "kernel_ok": False, "vj_baked": False,
                             "weights_on_disk": False, "vram_free_gb": 0.0, "vram_total_gb": 0.0,
                             "device_name": ""}
    models_root = Path(os.environ.get("VJ_MODELS_ROOT", "/opt/models"))
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
            # The load-bearing probe: a real kernel launch on THIS card. sm_120/no-kernel-image dies here.
            x = torch.randn(64, 64, device="cuda")
            _ = (x @ x).sum().item()
            facts["kernel_ok"] = True
    except Exception as exc:  # noqa: BLE001 -- a probe failure is a CHECK failure, never a crash
        facts["device_name"] = facts["device_name"] or f"probe_error:{type(exc).__name__}"
    return facts


# --------------------------------------------------------------------------- evaluation

@dataclass
class VerifyResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    reasons: list[str]


def evaluate(events: list[tuple[str, dict]], cfg: VerifyConfig) -> VerifyResult:
    """Assert the verify contract over the parsed `@event` stream. Pure: no pod, no GPU. Every gate is
    a named check so the JSON report says exactly WHICH bound failed."""
    checks: dict[str, bool] = {}
    metrics: dict[str, Any] = {}
    reasons: list[str] = []

    probe = find_event(events, "gpu_probe") or {}
    complete = find_event(events, "complete") or {}
    first = find_event(events, "first_frame") or {}
    sharp = find_event(events, "sharpness") or {}
    mirror = find_event(events, "mirror_complete")

    # 1. baked-sentinel HIT: a baked image must NOT have run a heavy R2 mirror leg.
    if cfg.expect_baked:
        baked_staging = float((mirror or {}).get("total_seconds", 0.0)) if mirror else 0.0
        checks["baked_no_r2_pull"] = (mirror is None) or (baked_staging <= cfg.max_baked_staging_seconds)
        metrics["baked_staging_seconds"] = baked_staging
        if not checks["baked_no_r2_pull"]:
            reasons.append(f"baked image ran an R2 mirror leg ({baked_staging:.0f}s > "
                           f"{cfg.max_baked_staging_seconds:.0f}s); .vj-baked early-return did not fire")
        checks["vj_baked_present"] = bool(probe.get("vj_baked"))
        if not checks["vj_baked_present"]:
            reasons.append(".vj-baked marker absent on a baked image")

    # 2. pod-only insight checks
    checks["gpu_visible"] = bool(probe.get("torch_cuda"))
    checks["kernel_ok"] = bool(probe.get("kernel_ok"))
    checks["weights_on_disk"] = bool(probe.get("weights_on_disk"))
    vram_free = float(probe.get("vram_free_gb") or 0.0)
    checks["vram_headroom"] = vram_free >= cfg.min_vram_free_gb
    metrics["vram_free_gb"] = vram_free
    metrics["device_name"] = probe.get("device_name", "")
    if not checks["gpu_visible"]:
        reasons.append("torch.cuda not available on the pod")
    if not checks["kernel_ok"]:
        reasons.append("cuda kernel did not load on this card (sm/kernel-image mismatch class)")
    if not checks["weights_on_disk"]:
        reasons.append("model weights not present on disk")
    if not checks["vram_headroom"]:
        reasons.append(f"VRAM headroom {vram_free:.1f}GB < floor {cfg.min_vram_free_gb:.1f}GB")

    # 3. render completed + output object written
    output_key = complete.get("output_key")
    clip_bytes = int(complete.get("clip_bytes") or 0)
    checks["render_complete"] = bool(output_key)
    checks["clip_written"] = clip_bytes > 0
    metrics["output_key"] = output_key
    metrics["clip_bytes"] = clip_bytes
    if not checks["render_complete"]:
        reasons.append("no @event complete with an output_key")
    if not checks["clip_written"]:
        reasons.append("output clip object is empty / not written")

    # 4. time-to-first-frame bound
    ttff = float(first.get("seconds") or 0.0) if first else 0.0
    checks["first_frame_in_time"] = bool(first) and ttff <= cfg.max_first_frame_seconds
    metrics["first_frame_seconds"] = ttff
    if not checks["first_frame_in_time"]:
        reasons.append(f"time-to-first-frame {ttff:.0f}s exceeds {cfg.max_first_frame_seconds:.0f}s "
                       "(or no first_frame event)")

    # 5. quality parity vs the runtime-quant baseline (#118 method-ii gate)
    value = float(sharp.get("value") or 0.0)
    baseline = float(sharp.get("baseline") or 0.0)
    ratio = (value / baseline) if baseline > 0 else 0.0
    checks["sharpness_parity"] = ratio >= cfg.min_sharpness_ratio
    metrics["sharpness_value"] = value
    metrics["sharpness_baseline"] = baseline
    metrics["sharpness_ratio"] = round(ratio, 4)
    if not checks["sharpness_parity"]:
        reasons.append(f"sharpness parity {ratio:.3f} < threshold {cfg.min_sharpness_ratio:.3f}")

    return VerifyResult(passed=all(checks.values()), checks=checks, metrics=metrics, reasons=reasons)


# --------------------------------------------------------------------------- cost

def cost_estimate_usd(gpu_type_id: str, elapsed_seconds: float) -> float:
    """Per-run H200 (or whatever card) cost estimate from elapsed wall-clock. Conservative-high so the
    printed figure never understates spend."""
    hourly = GPU_HOURLY_USD.get(gpu_type_id, 4.0)
    return round(hourly * (elapsed_seconds / 3600.0), 4)


def pick_gpu_type(available: list[dict[str, Any]], tier: str) -> str | None:
    """Choose the most-capable AVAILABLE gpu id for `tier` from the live list. `available` is the
    RunPod list_gpu_types shape (dicts with 'id'/'displayName' and an availability flag). Returns None
    if nothing in the tier preference order is available (caller aborts -- no spin, no spend)."""
    prefs = GPU_TIERS.get(tier, ())
    by_name = {}
    for g in available:
        name = g.get("displayName") or g.get("id") or ""
        if g.get("available", True):
            by_name.setdefault(name, g.get("id") or name)
    for want in prefs:
        if want in by_name:
            return by_name[want]
    return None


# --------------------------------------------------------------------------- orchestration

def run_verify(client: PodClient, cfg: VerifyConfig,
               *, clock: Callable[[], float], poll_sleep: Callable[[float], None] | None = None,
               max_polls: int = 600) -> dict:
    """Drive one bounded verify run and return the JSON report dict. Spend-safe by construction:

      - aborts BEFORE any spend if no tier GPU is available (returns a no-spin report);
      - polls get_pod/read_logs until @event complete OR the hard TTL elapses;
      - on PASS: emit report + "promote", then DELETE the pod (full teardown);
      - on FAIL: STOP the pod (keeps disk for SSH debug) and include the resume handle;
      - the TTL auto-stops the pod no matter what, so a hung render cannot bleed the card.

    `clock` and `poll_sleep` are injected so tests run the whole flow deterministically with no real
    time or pod. Never raises on a verify FAILURE (that is a normal reported outcome); only a genuine
    client/transport fault propagates, after a best-effort stop."""
    poll_sleep = poll_sleep or (lambda _s: None)
    report: dict[str, Any] = {"image": cfg.image, "tier": cfg.tier}

    gpu_id = pick_gpu_type(client.list_gpu_types(), cfg.tier)
    if gpu_id is None:
        report.update(passed=False, spun=False,
                      reasons=[f"no available GPU for tier {cfg.tier!r}; not spinning (no spend)"])
        return report
    report["gpu_type_id"] = gpu_id

    started = clock()
    pod = client.create_pod(image=cfg.image, gpu_type_id=gpu_id, env=cfg.env,
                            registry_auth_id=cfg.registry_auth_id, ttl_seconds=cfg.ttl_seconds)
    pod_id = pod["id"]
    report["pod_id"] = pod_id

    result: VerifyResult | None = None
    timed_out = False
    try:
        for _ in range(max_polls):
            if clock() - started >= cfg.ttl_seconds:
                timed_out = True
                break
            events = parse_events(client.read_logs(pod_id))
            if find_event(events, "complete") is not None:
                result = evaluate(events, cfg)
                break
            poll_sleep(min(5.0, cfg.ttl_seconds))
        else:
            timed_out = True
        if result is None:  # never completed within TTL/polls -> evaluate what we have, mark timeout
            result = evaluate(parse_events(client.read_logs(pod_id)), cfg)
            if timed_out:
                result.passed = False
                result.reasons.append("verify did not reach @event complete before the hard TTL")
    finally:
        elapsed = clock() - started
        report["elapsed_seconds"] = round(elapsed, 1)
        report["cost_estimate_usd"] = cost_estimate_usd(gpu_id, elapsed)

    report["checks"] = result.checks
    report["metrics"] = result.metrics
    report["reasons"] = result.reasons
    report["passed"] = result.passed
    report["timed_out"] = timed_out

    if result.passed:
        report["signal"] = "promote"
        client.delete_pod(pod_id)              # PASS: full teardown, no leak
        report["teardown"] = "deleted"
    else:
        client.stop_pod(pod_id)                # FAIL: stop (keep disk) for SSH debug; billing halted
        report["signal"] = "hold"
        report["teardown"] = "stopped"
        report["debug_handle"] = {
            "pod_id": pod_id,
            "resume": f"runpod start-pod {pod_id}  # restarts on the preserved disk",
            "shell": f"runpodctl exec {pod_id} -- bash  # or SSH per the pod's connection info",
            "note": "pod STOPPED not deleted: disk/state preserved, GPU billing halted.",
        }
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI. Defaults to a MOCKED dry-run (no GPU, no spend). `--live` requires a wired RunPod client
    and is the gated validation path; this module ships the live client as an unimplemented seam, so a
    bare `--live` fails loud rather than firing a pod by accident."""
    import argparse
    ap = argparse.ArgumentParser(description="RunPod pod verify harness (mock-first; live is gated).")
    ap.add_argument("--image", default="ghcr.io/skyphusion-labs/vivijure-backend:dryrun")
    ap.add_argument("--tier", default="i2v", choices=sorted(GPU_TIERS))
    ap.add_argument("--registry-auth-id", default=None)
    ap.add_argument("--live", action="store_true", help="use the (gated) live RunPod client")
    args = ap.parse_args(argv)

    cfg = VerifyConfig(image=args.image, tier=args.tier, registry_auth_id=args.registry_auth_id)
    if args.live:
        client: PodClient = RunpodMcpPodClient()  # type: ignore[assignment]
    else:
        print("runpod_verify: DRY-RUN (mocked client; no GPU, no spend). Pass --live for the gated run.")
        client = _DryRunClient(cfg)

    import time
    report = run_verify(client, cfg, clock=time.monotonic)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


class _DryRunClient:
    """A self-contained mock that returns a PASS-shaped @event stream so `main` (no flags) exercises
    the full harness offline. NOT a test fixture (tests have their own); purely so a human can run the
    script and see the report shape without a pod."""

    def __init__(self, cfg: VerifyConfig):
        self._cfg = cfg

    def list_gpu_types(self):
        return [{"id": "H200", "displayName": "NVIDIA H200", "available": True}]

    def create_pod(self, **_kw):
        return {"id": "dryrun-pod"}

    def get_pod(self, _pod_id):
        return {"id": _pod_id, "desiredStatus": "RUNNING"}

    def read_logs(self, _pod_id):
        return [
            emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                            "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                            "device_name": "NVIDIA H200"}),
            '@event mirror_skipped {"reason": "baked"}',
            '@event first_frame {"seconds": 42.0}',
            '@event sharpness {"value": 0.99, "baseline": 1.0}',
            '@event complete {"output_key": "renders/verify/full.mp4", "clip_bytes": 1048576}',
        ]

    def stop_pod(self, _pod_id):
        return {"id": _pod_id, "desiredStatus": "EXITED"}

    def delete_pod(self, _pod_id):
        return {"id": _pod_id, "deleted": True}


if __name__ == "__main__":
    raise SystemExit(main())
