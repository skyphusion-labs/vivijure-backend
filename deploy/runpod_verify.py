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

NO LIVE SPIN FROM THIS MODULE BY DEFAULT. `main()` runs a MOCKED dry-run unless `--live` is passed;
`--live` binds RunpodSdkPodClient, the SECURE-cloud-only live RunPod client (up/down/list via the
`runpod` SDK; RUNPOD_API_KEY from env). The pod-side verify EMITTER that produces the @event contract,
and the channel read_logs reads it back over, is the remaining seam (RunpodSdkPodClient.read_logs);
until it lands, `--live` proves pod lifecycle, not a full verify PASS. Unit tests drive the whole flow
against FakePodClient -- no GPU, no network.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol


# --------------------------------------------------------------------------- tiers + config

# Tier -> the GPU classes the image must run on. The i2v/bf16 serverless pool spans THREE DC arches,
# one card each: H200 (Hopper sm_90, 141GB), B200 (Blackwell DC sm_100, 180GB), RTX PRO 6000 Blackwell
# (sm_120, 96GB). 3-arch coverage rides on the prebuilt cu128 wheels (nothing compiles CUDA from source,
# so TORCH_CUDA_ARCH_LIST is a no-op and absent by design). H100 (ALSO Hopper sm_90) is EXCLUDED not for
# its kernel target -- H200 shares it -- but for its memory ENVELOPE: 80GB OOMs Wan2.2-A14B where the
# 141GB H200 fits (Conrad, 2026-06-29; H200 is Hopper, the earlier 'H200 = Blackwell floor' note was
# wrong). A homelab-lite base-only image verifies on a cheap consumer card. Concrete ids resolve against
# live availability (PodClient.list_gpu_types) at spin time, never a sold-out SKU. PREFERENCE ORDERS.
GPU_TIERS: dict[str, tuple[str, ...]] = {
    "i2v": ("NVIDIA H200", "NVIDIA B200", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    "base": ("NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA A10", "NVIDIA L4"),  # ids are the canonical RunPod get_gpus() ids
}

# Per-GPU on-demand $/hr (estimate only, for the per-run cost line; the live client may override from
# list_gpu_types pricing). Kept conservative-high so the printed estimate never UNDERstates spend.
GPU_HOURLY_USD: dict[str, float] = {
    "NVIDIA H200": 4.39, "NVIDIA H100 NVL": 2.79, "NVIDIA H100 80GB HBM3": 2.69,
    "NVIDIA B200": 5.99, "NVIDIA GeForce RTX 4090": 0.69, "NVIDIA RTX A5000": 0.36,
    "NVIDIA A10": 0.45, "NVIDIA L4": 0.43,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.49,
}


@dataclass(frozen=True)
class VerifyConfig:
    """One verify run's bounds + gates. All spend guards live here so a caller cannot fire an
    unbounded job by omission."""
    image: str                                  # the freshly built image ref (ghcr.io/...:tag)
    tier: str = "i2v"                            # GPU_TIERS key; i2v => the 3-arch pool (sm_90/100/120)
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


@dataclass(frozen=True)
class RegressionConfig(VerifyConfig):
    """A FULL capability regression run's bounds + gates (extends the #131 smoke `VerifyConfig`). One
    file, one contract -- no fork. Adds the CAP-1..6 + BAK-3/4 bounds and turns on the pod-side
    regression emitters via `VJ_REGRESSION`. Every bound is a hard wall sized ~3x the H200 median to
    absorb cold-start variance without being trivially loose (see docs/regression-plan.md)."""
    # CAP wall-clock bounds (seconds) -- exceed and the check fails; the pod TTL still auto-stops.
    max_keyframe_seconds: float = 120.0     # CAP-1 SDXL keyframe
    max_clip_seconds: float = 300.0         # CAP-2 Wan2.2 i2v draft clip
    max_rife_seconds: float = 60.0          # CAP-3 RIFE load + interpolate
    max_finish_seconds: float = 180.0       # CAP-4 finish (interp + GFPGAN + encode)
    max_e2e_seconds: float = 600.0          # CAP-6 2-shot end-to-end
    # Artifact-size backstops against blank/degenerate output the pipeline does not error on.
    min_keyframe_bytes: int = 50_000        # CAP-1 PNG not blank
    min_clip_bytes: int = 100_000           # CAP-4 finished clip not empty
    min_e2e_bytes: int = 200_000            # CAP-6 assembled film not empty
    # CAP-3 structural RIFE architecture the flownet.pkl weights expect (IFNet block0/1/2, IFBlock c=90).
    expected_rife_block_count: int = 3
    expected_rife_c: int = 90
    # BAK-4 precision: prod is now bf16-only (fp8 dropped off the datacenter path). The probe EXPECTS
    # bf16; a valid-but-different precision (fp8) WARNS, it does not fail. fp32 is never a valid bake.
    expected_precision: str = "bfloat16"
    valid_precisions: frozenset = frozenset({"float8_e4m3fn", "bfloat16"})
    env: dict[str, str] = field(default_factory=lambda: {
        # Draft tier (distill on) so spend stays bounded. Precision comes from the BAKE
        # (VJ_BAKE_PRECISION), NOT a runtime quant flag, so we do NOT force VJ_I2V_FP8 here (prod is
        # bf16-only). VJ_REGRESSION activates the pod-side CAP-1..6 + BAK-3/4 emitters; VJ_VERIFY keeps
        # the base gpu_probe / first_frame / sharpness path on.
        "VJ_I2V_DISTILL": "1", "VJ_VERIFY": "1", "VJ_REGRESSION": "1",
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


def _import_runpod(api_key):
    """Import the `runpod` SDK lazily and set the API key. Lazy so importing this module (and every unit
    test) never needs the SDK installed or a key present -- only a real `--live` run does. The key is
    read from RUNPOD_API_KEY (never hardcoded, never logged)."""
    import os
    import runpod  # type: ignore  # noqa: I001 -- optional dep, only for the live path
    key = api_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set; the live RunPod client needs it. Pass it via env "
                           "or a CI secret -- never hardcode a key.")
    runpod.api_key = key
    return runpod


def _null_log_fetcher(_pod_id):
    """Default @event log source for the live client: NONE wired. Returns [] and NEVER raises, so a
    `--live` run degrades HONESTLY -- the verify poll loop simply never sees `@event complete`, fails on
    the hard TTL with a clear reason, and the always-run teardown still fires (no leaked pod). The
    pod-side verify entrypoint that EMITS the @event contract, plus the channel it writes to (the
    read_logs pairing), is the seam this hook fills; until it lands, `--live` proves pod lifecycle
    (up/down/list), not a full verify PASS."""
    return []


class RunpodSdkPodClient:
    """Live RunPod pod-lifecycle client, SECURE cloud ONLY, backed by the `runpod` SDK (>=1.7).

    Implements the up/down/list verbs run_verify drives: list_gpu_types (SECURE GPUs only), create_pod
    (forces cloud_type=SECURE -- NEVER COMMUNITY: community caps pod disk ~20GB, the RunPod-secure-only
    rule), get_pod, stop_pod (halts billing, keeps disk for SSH debug), delete_pod (terminate,
    irreversible teardown). The hard TTL is enforced HARNESS-SIDE by run_verify (the SDK create_pod has
    no native auto-stop), so a hung render is stopped by the poll loop plus the always-run teardown.

    read_logs is the ONE seam that pairs with the pod-side verify EMITTER (backend lane): the pod writes
    the @event contract, and the channel it is read back over is injected as `log_fetcher`. The default
    (`_null_log_fetcher`) returns [] so a `--live` run can NEVER silently pass without a real @event
    stream and never leaks a pod on a missing/broken channel.

    The `runpod` SDK is imported lazily; unit tests inject a fake `sdk` and never import it."""

    def __init__(self, *, api_key=None, container_disk_gb=500, data_center_id=None,
                 log_fetcher=None, name_prefix="vj-verify", sdk=None):
        self._sdk = sdk if sdk is not None else _import_runpod(api_key)
        self._container_disk_gb = container_disk_gb
        self._data_center_id = data_center_id
        self._log_fetcher = log_fetcher or _null_log_fetcher
        self._name_prefix = name_prefix

    def list_gpu_types(self):
        """SECURE-cloud GPU types in the harness shape ({id, displayName, available}). The RunPod SDK
        `get_gpus()` list carries only id/displayName/memoryInGb -- NOT cloud availability -- and its
        `displayName` is a SHORT label ("L4"), while GPU_TIERS matches the canonical id ("NVIDIA L4").
        So `displayName` is set to the id (what pick_gpu_type matches on), and SECURE availability is
        resolved from the per-GPU `get_gpu(id)` detail (`secureCloud`). Only the ids some tier actually
        prefers are detail-fetched (a handful, not all ~50), so `available` is True exactly for a
        tier-relevant card RunPod offers on SECURE -- pick_gpu_type can never pick a community-only or
        off-tier SKU."""
        universe = [g.get("id") for g in (self._sdk.get_gpus() or []) if g.get("id")]
        wanted = {gid for prefs in GPU_TIERS.values() for gid in prefs}
        out = []
        for gid in universe:
            secure = False
            if gid in wanted:
                try:
                    secure = bool((self._sdk.get_gpu(gid) or {}).get("secureCloud"))
                except Exception:  # noqa: BLE001 -- a detail miss is "not available", never a crash
                    secure = False
            out.append({"id": gid, "displayName": gid, "available": secure})
        return out

    def create_pod(self, *, image, gpu_type_id, env, registry_auth_id, ttl_seconds):
        """Spin ONE SECURE-cloud GPU pod on `image`. `ttl_seconds` is enforced by the harness poll loop,
        not the SDK. `registry_auth_id` is rejected loudly (not silently ignored): our images are public
        GHCR (no auth needed); a private image must be pulled via a pre-created RunPod template carrying
        containerRegistryAuthId, spun from template_id -- a separate seam, not this call."""
        if registry_auth_id:
            raise ValueError("RunpodSdkPodClient pulls public GHCR images (no registry auth). For a "
                             "private image, provision a RunPod template with containerRegistryAuthId "
                             "and spin from its template_id -- do not pass registry_auth_id here.")
        pod = self._sdk.create_pod(
            name=("%s-%s" % (self._name_prefix, gpu_type_id)).replace(" ", "-")[:60],
            image_name=image, gpu_type_id=gpu_type_id, cloud_type="SECURE",
            container_disk_in_gb=self._container_disk_gb, env=dict(env or {}),
            data_center_id=self._data_center_id, support_public_ip=True, start_ssh=True)
        pod_id = pod.get("id") if isinstance(pod, dict) else None
        if not pod_id:
            raise RuntimeError("RunPod create_pod returned no id: %r" % (pod,))
        return {"id": pod_id}

    def get_pod(self, pod_id):
        return self._sdk.get_pod(pod_id) or {"id": pod_id}

    def read_logs(self, pod_id):
        try:
            lines = self._log_fetcher(pod_id)
            return list(lines) if lines else []
        except Exception:  # noqa: BLE001 -- a log-fetch fault must never leak the pod; degrade to []
            return []

    def stop_pod(self, pod_id):
        self._sdk.stop_pod(pod_id)
        return {"id": pod_id, "desiredStatus": "EXITED"}

    def delete_pod(self, pod_id):
        self._sdk.terminate_pod(pod_id)
        return {"id": pod_id, "deleted": True}

    def list_live_pod_ids(self):
        """Every pod id currently on the account -- for the post-run teardown-confirm (assert our pod id
        is gone => list-confirmed zero). Shape-tolerant across SDK versions."""
        pods = self._sdk.get_pods() or []
        if isinstance(pods, dict):
            pods = pods.get("pods") or (pods.get("myself") or {}).get("pods") or []
        return [p.get("id") for p in pods if isinstance(p, dict) and p.get("id")]


# Back-compat alias: this client used to be a NotImplementedError seam named RunpodMcpPodClient.
RunpodMcpPodClient = RunpodSdkPodClient


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


def format_event(name: str, payload: dict) -> str:
    """Build one `@event <name> {json}` line (pure; the pod entrypoint prints it). The single source
    of the wire format so every pod-side emitter and every test agree on the exact contract."""
    return "@event " + name + " " + json.dumps(payload, sort_keys=True)


def emit_gpu_probe(facts: dict) -> str:
    """Build the `@event gpu_probe` line from collected facts (pure; the pod entrypoint prints it).
    Factored out so the contract is asserted in tests without a GPU."""
    keys = ("torch_cuda", "kernel_ok", "vj_baked", "weights_on_disk",
            "vram_free_gb", "vram_total_gb", "device_name")
    return format_event("gpu_probe", {k: facts.get(k) for k in keys})


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


def _ifblock_channels(block: Any) -> int | None:
    """Derive a RIFE `IFBlock`'s hidden width `c` STRUCTURALLY (it is not stored as an attribute).
    `IFBlock.conv0` is `Sequential(conv(in_planes, c//2), conv(c//2, c))` and `conv(...)` is itself
    `Sequential(Conv2d, PReLU)`, so the LAST conv block's `Conv2d.out_channels` is exactly `c`. Returns
    None if the module is not shaped like an IFBlock (defensive: the caller maps None -> 'not found')."""
    try:
        conv0 = block.conv0
        last_conv = conv0[len(conv0) - 1]   # the second conv() in conv0
        return int(last_conv[0].out_channels)
    except Exception:  # noqa: BLE001 -- shape drift is a probe miss, never a crash
        return None


def rife_architecture_facts(flownet: Any) -> dict:
    """RUN ON THE POD ONLY (CAP-3 structural probe). Introspects the vendored RIFE `IFNet` instance
    and proves the architecture the freshly re-hosted (C2) package actually built matches the
    `flownet.pkl` weights it must run. The inference path uses block0/block1/block2 (block_tea is the
    training-only teacher and is EXCLUDED), all `IFBlock(c=90)`. This is a STRUCTURAL assertion on the
    vendored code, not a string match on a file name. Best-effort: any introspection error yields
    loaded=False so the harness FAILS the CAP-3 check rather than crashing the pod.

    Returns the body of `@event rife_model_probe` minus `flownet_pkl_bytes`, which the pod entrypoint
    fills from the on-disk weight size."""
    facts = {"block_count": 0, "c_per_block": 0, "loaded": False}
    try:
        widths: list[int] = []
        for name in ("block0", "block1", "block2"):
            block = getattr(flownet, name, None)
            if block is None:
                continue
            c = _ifblock_channels(block)
            if c is not None:
                widths.append(c)
        facts["block_count"] = len(widths)
        if widths and all(w == widths[0] for w in widths):
            facts["c_per_block"] = widths[0]
        facts["loaded"] = facts["block_count"] == 3 and facts["c_per_block"] > 0
    except Exception:  # noqa: BLE001
        return {"block_count": 0, "c_per_block": 0, "loaded": False}
    return facts


# --------------------------------------------------------------------------- evaluation

@dataclass
class VerifyResult:
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, Any]
    reasons: list[str]
    # WARN-only, non-fatal observations (e.g. a valid-but-unexpected precision); never flip `passed`.
    warnings: list[str] = field(default_factory=list)
    # Honestly recorded untested capabilities (e.g. CAP-5 LoRA); recorded, NEVER silently passed, and
    # do NOT flip `passed` (the #249/#77 degrade discipline: a gap is named, not hidden).
    coverage_gaps: dict[str, str] = field(default_factory=dict)


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


# The complete ordered list of `@event` names a FULL regression run must emit (BAK probes first, then
# the CAP renders, then the base quality gates + completion sentinel). Tests build a minimal mock
# stream from this; the pod entrypoint emits them in this order under VJ_REGRESSION=1.
REGRESSION_EVENTS: tuple[str, ...] = (
    "model_inventory",          # BAK-3 -- emitted BEFORE any render (fail-fast on a missing weight)
    "model_precision",          # BAK-4 -- baked dtype
    "gpu_probe",                # BAK-1/5 (base) -- sentinel + kernel + VRAM
    "mirror_skipped",           # BAK-2 (base) -- baked early-return hit, no R2 leg
    "keyframe_done",            # CAP-1 -- SDXL keyframe
    "clip_done",                # CAP-2 -- Wan2.2 i2v draft clip
    "rife_model_probe",         # CAP-3 -- structural architecture probe (load time)
    "rife_done",                # CAP-3 -- interpolation result
    "finish_done",              # CAP-4 -- finish path (interp + GFPGAN + encode)
    "first_frame",              # base -- time-to-first-frame gate
    "sharpness",                # base -- #118 method-ii quality parity gate
    "e2e_done",                 # CAP-6 -- 2-shot end-to-end
    "complete",                 # base -- render-done sentinel the harness polls for
)


def evaluate_regression(events: list[tuple[str, dict]], cfg: VerifyConfig) -> VerifyResult:
    """The FULL capability regression: run the base smoke gates (`evaluate`), then add the CAP-1..6 +
    BAK-3/4 checks onto the same named-check dict. Pure: no pod, no GPU. Every assertion targets a
    named field in a parsed `@event` payload, never English prose. `passed` is the AND of every gate;
    coverage gaps (CAP-5 LoRA) and warnings (a valid-but-unexpected precision) are recorded but never
    flip `passed`. Accepts a plain `VerifyConfig` (falls back to RegressionConfig defaults for the new
    bounds) so a caller cannot under-gate by passing the wrong config type."""
    rc = cfg if isinstance(cfg, RegressionConfig) else RegressionConfig(
        image=cfg.image, tier=cfg.tier, registry_auth_id=cfg.registry_auth_id,
        ttl_seconds=cfg.ttl_seconds, expect_baked=cfg.expect_baked)
    result = evaluate(events, cfg)
    checks, metrics, reasons = result.checks, result.metrics, result.reasons
    warnings = result.warnings
    coverage_gaps = result.coverage_gaps

    def gate(name: str, ok: bool, why: str) -> None:
        checks[name] = bool(ok)
        if not ok:
            reasons.append(why)

    # ---- BAK-3: every baked model file present (fail-fast: emitted before any render) ----
    inv = find_event(events, "model_inventory") or {}
    inv_keys = ("sdxl", "wan22", "rife_flownet", "gfpgan")
    metrics["model_inventory"] = {k: inv.get(k) for k in (*inv_keys, "all_present")}
    missing = [k for k in inv_keys if not inv.get(k)]
    gate("bak3_all_models_present", bool(inv.get("all_present")) and not missing,
         "BAK-3 baked model inventory incomplete: missing " + (", ".join(missing) or "model_inventory event"))

    # ---- BAK-4: precision is a VALID baked dtype (fp8 or bf16); fp32 fails, mismatch only WARNS ----
    prec = find_event(events, "model_precision") or {}
    dtype = prec.get("i2v_dtype")
    metrics["i2v_dtype"] = dtype
    gate("bak4_precision_valid", dtype in rc.valid_precisions,
         f"BAK-4 i2v precision {dtype!r} is not a valid baked precision "
         f"{sorted(rc.valid_precisions)} (fp32 is never a valid bake)")
    if dtype in rc.valid_precisions and dtype != rc.expected_precision:
        warnings.append(f"BAK-4 i2v precision {dtype!r} differs from the expected baked precision "
                        f"{rc.expected_precision!r} (valid, non-fatal)")

    # ---- CAP-1: SDXL keyframe ----
    kf = find_event(events, "keyframe_done") or {}
    kf_w, kf_h = int(kf.get("width") or 0), int(kf.get("height") or 0)
    kf_bytes, kf_t = int(kf.get("bytes") or 0), float(kf.get("elapsed_s") or 0.0)
    metrics["keyframe"] = {"width": kf_w, "height": kf_h, "bytes": kf_bytes, "elapsed_s": kf_t}
    gate("cap1_keyframe_format", kf.get("format") == "PNG", "CAP-1 keyframe format is not PNG")
    gate("cap1_keyframe_dims", kf_w > 0 and kf_h > 0,
         f"CAP-1 keyframe has a zero dimension ({kf_w}x{kf_h})")
    gate("cap1_keyframe_bytes", kf_bytes >= rc.min_keyframe_bytes,
         f"CAP-1 keyframe {kf_bytes}B < floor {rc.min_keyframe_bytes}B (blank/degenerate frame)")
    gate("cap1_keyframe_time", bool(kf) and kf_t <= rc.max_keyframe_seconds,
         f"CAP-1 keyframe {kf_t:.0f}s exceeds {rc.max_keyframe_seconds:.0f}s (or no keyframe_done event)")

    # ---- CAP-2: Wan2.2 i2v clip (the 6-field pointer contract) ----
    clip = find_event(events, "clip_done") or {}
    clip_t = float(clip.get("elapsed_s") or 0.0)
    metrics["clip"] = {"num_frames": clip.get("num_frames"), "fps": clip.get("fps"),
                       "seconds": clip.get("seconds"), "distilled": clip.get("distilled"),
                       "elapsed_s": clip_t}
    gate("cap2_clip_pointer", bool(clip.get("clip_key")), "CAP-2 clip_done has no clip_key")
    gate("cap2_clip_frames", int(clip.get("num_frames") or 0) >= 17,
         f"CAP-2 clip num_frames {clip.get('num_frames')} < 17 (likely an encode error)")
    gate("cap2_clip_fps", int(clip.get("fps") or 0) == 16,
         f"CAP-2 clip fps {clip.get('fps')} != 16 (Wan2.2 documented default)")
    gate("cap2_clip_seconds", float(clip.get("seconds") or 0.0) >= 1.0,
         f"CAP-2 clip seconds {clip.get('seconds')} < 1.0")
    gate("cap2_clip_distilled", clip.get("distilled") is True,
         "CAP-2 clip not distilled (draft-tier path did not run; spend risk)")
    gate("cap2_clip_time", bool(clip) and clip_t <= rc.max_clip_seconds,
         f"CAP-2 clip {clip_t:.0f}s exceeds {rc.max_clip_seconds:.0f}s (or no clip_done event)")

    # ---- CAP-3: RIFE interpolation -- the highest-priority check (structural + functional) ----
    probe = find_event(events, "rife_model_probe") or {}
    rdone = find_event(events, "rife_done") or {}
    rife_bc, rife_c = int(probe.get("block_count") or 0), int(probe.get("c_per_block") or 0)
    rife_t = float(rdone.get("elapsed_s") or 0.0)
    r_h, r_w = int(rdone.get("h") or 0), int(rdone.get("w") or 0)
    metrics["rife"] = {"block_count": rife_bc, "c_per_block": rife_c,
                       "flownet_pkl_bytes": probe.get("flownet_pkl_bytes"),
                       "output_frames": rdone.get("output_frames"), "elapsed_s": rife_t}
    gate("cap3_rife_loaded", bool(probe.get("loaded")), "CAP-3 RIFE model did not load")
    gate("cap3_rife_architecture",
         rife_bc == rc.expected_rife_block_count and rife_c == rc.expected_rife_c,
         f"CAP-3 RIFE architecture {rife_bc} blocks @ c={rife_c} != expected "
         f"{rc.expected_rife_block_count} @ c={rc.expected_rife_c} (vendored code drifted from flownet.pkl)")
    gate("cap3_rife_flownet_present", int(probe.get("flownet_pkl_bytes") or 0) > 0,
         "CAP-3 flownet.pkl bytes is 0 (weights not found)")
    gate("cap3_rife_output_frames", int(rdone.get("output_frames") or 0) == 3,
         f"CAP-3 RIFE output_frames {rdone.get('output_frames')} != 3 (factor=2: A, midpoint, B)")
    gate("cap3_rife_dims", r_h > 0 and r_w > 0,
         f"CAP-3 RIFE output has a zero dimension ({r_w}x{r_h})")
    gate("cap3_rife_time", bool(rdone) and rife_t <= rc.max_rife_seconds,
         f"CAP-3 RIFE {rife_t:.0f}s exceeds {rc.max_rife_seconds:.0f}s (or no rife_done event)")

    # ---- CAP-4: finish path (interp + GFPGAN face restore + encode) ----
    fin = find_event(events, "finish_done") or {}
    fin_bytes, fin_t = int(fin.get("bytes") or 0), float(fin.get("elapsed_s") or 0.0)
    fin_key = str(fin.get("clip_key") or "")
    metrics["finish"] = {"interpolated": fin.get("interpolated"), "face_restored": fin.get("face_restored"),
                         "out_fps": fin.get("out_fps"), "out_frames": fin.get("out_frames"),
                         "bytes": fin_bytes, "elapsed_s": fin_t}
    gate("cap4_finish_interpolated", fin.get("interpolated") is True, "CAP-4 finish did not interpolate")
    # CAP-4 uses a portrait prompt ("close-up portrait of a person, cinematic") so GFPGAN has a face.
    gate("cap4_finish_face_restored", fin.get("face_restored") is True,
         "CAP-4 finish did not restore a face (portrait prompt should give GFPGAN a face)")
    gate("cap4_finish_fps", int(fin.get("out_fps") or 0) == 32,
         f"CAP-4 finish out_fps {fin.get('out_fps')} != 32 (16fps * factor 2)")
    gate("cap4_finish_frames", int(fin.get("out_frames") or 0) >= 33,
         f"CAP-4 finish out_frames {fin.get('out_frames')} < 33 (17 in * 2 - 1 minimum)")
    gate("cap4_finish_bytes", fin_bytes >= rc.min_clip_bytes,
         f"CAP-4 finished clip {fin_bytes}B < floor {rc.min_clip_bytes}B")
    gate("cap4_finish_key", fin_key.endswith("_finished.mp4"),
         f"CAP-4 finished clip_key {fin_key!r} does not end in _finished.mp4")
    gate("cap4_finish_time", bool(fin) and fin_t <= rc.max_finish_seconds,
         f"CAP-4 finish {fin_t:.0f}s exceeds {rc.max_finish_seconds:.0f}s (or no finish_done event)")
    # 16:9 aspect probe is DEFERRED for Phase C (Conrad answer #1); not gated here.

    # ---- CAP-5: LoRA apply -- a recorded coverage gap, NEVER a silent skip, never flips passed ----
    coverage_gaps["lora_apply"] = "skipped -- #280 open, no baked LoRA adapter (blocks the LoRA claim, not Phase C)"

    # ---- CAP-6: end-to-end 2-shot (shot ordering + ffmpeg concat + audio mux) ----
    e2e = find_event(events, "e2e_done") or {}
    e2e_bytes, e2e_t = int(e2e.get("bytes") or 0), float(e2e.get("elapsed_s") or 0.0)
    metrics["e2e"] = {"shots": e2e.get("shots"), "has_audio": e2e.get("has_audio"),
                      "duration_s": e2e.get("duration_s"), "bytes": e2e_bytes, "elapsed_s": e2e_t}
    gate("cap6_e2e_shots", int(e2e.get("shots") or 0) == 2, f"CAP-6 shots {e2e.get('shots')} != 2")
    gate("cap6_e2e_bytes", e2e_bytes >= rc.min_e2e_bytes,
         f"CAP-6 assembled film {e2e_bytes}B < floor {rc.min_e2e_bytes}B")
    gate("cap6_e2e_audio", e2e.get("has_audio") is True,
         "CAP-6 assembled film has no audio track (mux path unproven; 440Hz sine is sufficient)")
    gate("cap6_e2e_duration", float(e2e.get("duration_s") or 0.0) >= 2.0,
         f"CAP-6 duration {e2e.get('duration_s')}s < 2.0")
    gate("cap6_e2e_time", bool(e2e) and e2e_t <= rc.max_e2e_seconds,
         f"CAP-6 e2e {e2e_t:.0f}s exceeds {rc.max_e2e_seconds:.0f}s (or no e2e_done event)")

    result.passed = all(checks.values())
    return result


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
               max_polls: int = 600,
               evaluator: Callable[[list[tuple[str, dict]], VerifyConfig], VerifyResult] = evaluate,
               event_reader: Callable[[], tuple[list[tuple[str, dict]], str | None]] | None = None,
               on_pod_created: Callable[[str], None] | None = None) -> dict:
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
    if on_pod_created is not None:
        try:
            on_pod_created(pod_id)
        except Exception:  # noqa: BLE001 -- bookkeeping only; must never fail the run
            pass

    def _log_reader() -> tuple[list[tuple[str, dict]], str | None]:
        evs = parse_events(client.read_logs(pod_id))
        return evs, ("complete" if find_event(evs, "complete") is not None else None)

    reader = event_reader or _log_reader

    result: VerifyResult | None = None
    timed_out = False
    try:
        for _ in range(max_polls):
            if clock() - started >= cfg.ttl_seconds:
                timed_out = True
                break
            events, terminal = reader()
            if terminal is not None:                 # "complete" OR "error" -- a terminal channel state
                result = evaluator(events, cfg)
                if terminal == "error":
                    result.passed = False
                    result.reasons.append("verify channel reported status=error before all gates passed")
                break
            poll_sleep(min(5.0, cfg.ttl_seconds))
        else:
            timed_out = True
        if result is None:  # never reached a terminal status within TTL/polls -> evaluate what we have
            events, _ = reader()
            result = evaluator(events, cfg)
            if timed_out:
                result.passed = False
                result.reasons.append("verify did not reach a terminal status (complete/error) before the hard TTL")
    finally:
        elapsed = clock() - started
        report["elapsed_seconds"] = round(elapsed, 1)
        report["cost_estimate_usd"] = cost_estimate_usd(gpu_id, elapsed)

    report["checks"] = result.checks
    report["metrics"] = result.metrics
    report["reasons"] = result.reasons
    if result.warnings:
        report["warnings"] = result.warnings
    if result.coverage_gaps:
        report["coverage_gaps"] = result.coverage_gaps
    report["passed"] = result.passed
    report["timed_out"] = timed_out

    # Teardown doctrine (docs/release-gate.md): capture evidence, then DELETE + list-confirm ZERO on
    # EVERY path. A stopped pod still bills disk and leaves a pad standing, so it is NEVER a CI exit
    # state (stop is only a mid-debug state while a human is actively attached in-session). The failure
    # evidence is durable regardless: the R2 channel (summary.json + events.ndjson) outlives the pod;
    # the pod-log tail is captured into the report BEFORE delete for the workflow evidence artifact.
    report["signal"] = "promote" if result.passed else "hold"
    if not result.passed:
        try:
            tail = client.read_logs(pod_id)
            if tail:
                report["pod_logs_tail"] = list(tail)[-200:]
        except Exception:  # noqa: BLE001 -- evidence capture is best-effort; it never blocks teardown
            pass
        run_id = (cfg.env or {}).get("VJ_VERIFY_RUN_ID")
        if run_id:
            prefix = (cfg.env or {}).get("VJ_VERIFY_KEY_PREFIX", "verify")
            report["evidence"] = {
                "summary_key": "%s/%s/summary.json" % (prefix, run_id),
                "events_key": "%s/%s/events.ndjson" % (prefix, run_id),
                "note": "durable R2 verify channel; outlives the deleted pod",
            }
    client.delete_pod(pod_id)                  # DELETE on EVERY path -- no stopped pad left billing
    report["teardown"] = "deleted"

    lister = getattr(client, "list_live_pod_ids", None)
    if callable(lister):
        try:
            live_ids = list(lister())
            report["live_pods_after"] = len(live_ids)
            report["teardown_confirmed"] = pod_id not in live_ids   # ZERO on every path, PASS or FAIL
        except Exception:  # noqa: BLE001 -- a confirm-list fault must not mask the verify result
            report["teardown_confirmed"] = None
    return report


# --------------------------------------------------------------------------- live channel + promote

PROD_ENDPOINT_ID = "t9wcvlxh8rc5la"  # the production serverless endpoint (docs/release-gate.md)


def r2_summary_event_reader(env: dict, run_id: str, *, prefix: str = "verify"):
    """LIVE event source: a zero-arg callable that polls the run-scoped R2 ``summary.json`` the pod-side
    emitter (``vivijure_backend.verify``) writes, and returns ``(events, terminal_status)`` for
    ``run_verify``. The FROZEN contract, no prose parsing: ``verify.events_from_summary`` does the JSON
    shape and the summary's own ``status`` is the terminal signal. A not-yet-written or transiently
    unreadable object yields ``([], None)`` so the poll loop keeps waiting until the pod writes it (or
    the hard TTL fires) -- an honest timeout, never a crash."""
    from vivijure_backend import verify as _verify
    from vivijure_backend.harness import keys as _keys
    from vivijure_backend.harness.r2 import R2, R2Config

    store = R2(R2Config.from_env(env))
    summary_key = _keys.verify_summary_key(run_id, prefix=prefix)

    def read() -> tuple[list[tuple[str, dict]], str | None]:
        try:
            raw = store.get_bytes(summary_key)
        except Exception:  # noqa: BLE001 -- not written yet / transient: keep polling, never leak or crash
            return [], None
        try:
            status = json.loads(raw).get("status")
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            status = None
        terminal = status if status in ("complete", "error") else None
        return _verify.events_from_summary(raw), terminal

    return read


def promote_image(image: str, *, endpoint_id: str = PROD_ENDPOINT_ID, api_key: str | None = None,
                  transport: "Callable[..., Any] | None" = None) -> dict:
    """Promote a verified image onto the PRODUCTION serverless endpoint by repinning its ``imageName``.
    This is the ONLY path an image reaches prod (docs/release-gate.md doctrine); the caller gates it
    behind an explicit ``--promote`` go, so a bare verify run never touches prod. ``transport`` is
    injected in tests (no network). The RunPod API key is read from RUNPOD_API_KEY and NEVER logged."""
    key = api_key or os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError("promote needs RUNPOD_API_KEY (never hardcode a key)")
    if transport is None:
        import requests

        def transport(url, *, headers, payload):  # noqa: ANN001 -- thin default HTTP leg
            resp = requests.patch(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json() if resp.content else {}

    url = "https://rest.runpod.io/v1/endpoints/%s" % endpoint_id
    body = transport(url,
                     headers={"Authorization": "Bearer %s" % key, "Content-Type": "application/json"},
                     payload={"imageName": image})
    return {"endpoint_id": endpoint_id, "image": image, "response": body}


def _github_env_writer(name: str) -> "Callable[[str], None]":
    """Return a callback that records ``name=<value>`` to ``$GITHUB_ENV`` (so an always-run teardown
    backstop can STOP the pod even if this process is later killed/cancelled), falling back to a stderr
    note off CI. Never raises."""
    def write(value: str) -> None:
        path = os.environ.get("GITHUB_ENV")
        try:
            if path:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("%s=%s\n" % (name, value))
            else:
                print("%s=%s" % (name, value), file=sys.stderr)
        except Exception:  # noqa: BLE001 -- a bookkeeping write must never fail the run
            pass
    return write


def reap_pod(pod_id: str, *, client=None, summary_sink: "Callable[[str], None] | None" = None):
    """Teardown backstop: STOP (fast billing halt) then DELETE (terminate = zero) a pod, then
    list-confirm it is gone. The workflow always-run step calls this so a killed/cancelled run ends at
    ZERO pods -- a stopped pad still bills disk and is never an allowed exit state. Best-effort per verb
    (an already-gone pod is fine; delete is the one that matters). Returns True if confirmed gone, False
    if confirmed STILL LIVE (loud -- the caller fails the step), None if the live set was unreadable."""
    client = client or RunpodSdkPodClient()
    for verb in ("stop_pod", "delete_pod"):
        try:
            getattr(client, verb)(pod_id)
        except Exception:  # noqa: BLE001 -- already stopped/deleted is fine
            pass
    gone = None
    lister = getattr(client, "list_live_pod_ids", None)
    if callable(lister):
        try:
            gone = pod_id not in list(lister())
        except Exception:  # noqa: BLE001
            gone = None
    if gone is False:
        msg = "teardown backstop ALERT: pod %s is STILL LIVE after delete -- check RunPod NOW" % pod_id
    elif gone is True:
        msg = "teardown backstop: pod %s deleted, list-confirmed zero" % pod_id
    else:
        msg = "teardown backstop: pod %s delete issued; live set unreadable, could not confirm zero" % pod_id
    print(msg)
    if summary_sink is not None:
        summary_sink(msg)
    return gone


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
    ap.add_argument("--regression", action="store_true",
                    help="run the FULL capability regression (CAP-1..6 + BAK-3/4), not just the smoke")
    ap.add_argument("--data-center-id", default=None,
                    help="pin the SECURE pod to a specific RunPod data center (default: RunPod picks)")
    ap.add_argument("--container-disk-gb", type=int, default=500,
                    help="pod container disk in GB (default 500; a ~87GB baked image fits with room)")
    ap.add_argument("--run-id", default=None,
                    help="verify run id (live); the harness assigns one if omitted -- it names the R2 "
                         "summary key the pod writes and the harness polls")
    ap.add_argument("--bundle-key", default="bundles/Packet_Chase.tar.gz",
                    help="R2 key of the draft project bundle the pod renders (VJ_VERIFY_BUNDLE_KEY)")
    ap.add_argument("--project", default="verify", help="verify project slug (VJ_VERIFY_PROJECT)")
    ap.add_argument("--key-prefix", default="verify",
                    help="R2 verify key prefix (VJ_VERIFY_KEY_PREFIX)")
    ap.add_argument("--sharpness-baseline", type=float, default=100.0,
                    help="sharpness reference the ratio gate compares against (VJ_SHARPNESS_BASELINE)")
    ap.add_argument("--promote", action="store_true",
                    help="on a PASS, repin the PROD serverless endpoint to the verified image. OFF by "
                         "default: a proof run validates up|verify|down without ever touching prod")
    ap.add_argument("--promote-endpoint", default=PROD_ENDPOINT_ID,
                    help="the production serverless endpoint id to promote onto")
    ap.add_argument("--reap-pod", default=None,
                    help="teardown backstop: STOP then DELETE this pod id and list-confirm it is gone, "
                         "then exit. The workflow always-run step calls it so a killed/cancelled run "
                         "ends at ZERO pods, never a stopped pad still billing disk")
    ap.add_argument("--report-file", default=None,
                    help="also write the JSON run report to this path (for a workflow evidence artifact)")
    args = ap.parse_args(argv)

    if args.reap_pod:
        def _summary(msg: str) -> None:
            path = os.environ.get("GITHUB_STEP_SUMMARY")
            try:
                if path:
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
            except Exception:  # noqa: BLE001 -- a summary write must never fail the backstop
                pass
        gone = reap_pod(args.reap_pod, summary_sink=_summary)
        return 0 if gone is not False else 1  # confirmed-still-live => fail the step, loud

    if args.regression:
        cfg: VerifyConfig = RegressionConfig(image=args.image, tier=args.tier,
                                             registry_auth_id=args.registry_auth_id)
        evaluator = evaluate_regression
    else:
        cfg = VerifyConfig(image=args.image, tier=args.tier, registry_auth_id=args.registry_auth_id)
        evaluator = evaluate
    event_reader = None
    on_pod_created = None
    if args.live:
        import uuid
        run_id = args.run_id or ("s1-" + uuid.uuid4().hex[:12])
        pod_env = dict(cfg.env)  # VJ_I2V_* + VJ_VERIFY (+ VJ_REGRESSION)
        for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
            v = os.environ.get(k)
            if v:
                pod_env[k] = v
        pod_env.update({
            "VJ_VERIFY_RUN_ID": run_id,
            "VJ_VERIFY_KEY_PREFIX": args.key_prefix,
            "VJ_VERIFY_BUNDLE_KEY": args.bundle_key,
            "VJ_VERIFY_PROJECT": args.project,
            "VJ_SHARPNESS_BASELINE": repr(args.sharpness_baseline),
        })
        cfg = replace(cfg, env=pod_env)
        client: PodClient = RunpodSdkPodClient(  # type: ignore[assignment]
            data_center_id=args.data_center_id, container_disk_gb=args.container_disk_gb)
        event_reader = r2_summary_event_reader(pod_env, run_id, prefix=args.key_prefix)
        on_pod_created = _github_env_writer("VJ_VERIFY_POD_ID")
        print("runpod_verify: LIVE run_id=%s image=%s bundle=%s promote=%s"
              % (run_id, cfg.image, args.bundle_key, "on" if args.promote else "OFF"))
    else:
        mode = "REGRESSION " if args.regression else ""
        print(f"runpod_verify: {mode}DRY-RUN (mocked client; no GPU, no spend). Pass --live for the gated run.")
        client = _DryRunClient(cfg)

    import time
    report = run_verify(client, cfg, clock=time.monotonic, evaluator=evaluator,
                        event_reader=event_reader, on_pod_created=on_pod_created)

    if report.get("passed") and report.get("signal") == "promote":
        if args.promote:
            try:
                report["promote"] = promote_image(cfg.image, endpoint_id=args.promote_endpoint)
                report["promoted"] = True
            except Exception as e:  # noqa: BLE001 -- a promote fault is loud; teardown already ran
                report["promoted"] = False
                report["promote_error"] = "%s: %s" % (type(e).__name__, e)
        else:
            report["promoted"] = False
            report["promote_note"] = ("PASS is promote-eligible; promote skipped "
                                      "(no --promote, prod untouched)")

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report_file:
        try:
            with open(args.report_file, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
        except Exception:  # noqa: BLE001 -- the evidence artifact is a convenience; never fail on it
            pass
    ok = bool(report.get("passed"))
    if args.promote and ok and not report.get("promoted"):
        ok = False  # promote requested on a passing image but did not land -> fail the gate
    return 0 if ok else 1


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
        lines = [
            emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                            "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                            "device_name": "NVIDIA H200"}),
            '@event mirror_skipped {"reason": "baked"}',
            '@event first_frame {"seconds": 42.0}',
            '@event sharpness {"value": 0.99, "baseline": 1.0}',
            '@event complete {"output_key": "renders/verify/full.mp4", "clip_bytes": 1048576}',
        ]
        if isinstance(self._cfg, RegressionConfig):
            # A full PASS-shaped regression stream (BAK probes first, then the CAP renders) so a human
            # can `--regression` the script and see the report shape with no pod.
            lines[1:1] = [
                format_event("model_inventory", {"sdxl": True, "wan22": True, "rife_flownet": True,
                                                 "gfpgan": True, "all_present": True}),
                format_event("model_precision", {"i2v_dtype": "bfloat16"}),
                format_event("keyframe_done", {"shot_id": "s0", "key": "kf/s0.png", "width": 1280,
                                               "height": 720, "format": "PNG", "bytes": 820_000,
                                               "elapsed_s": 31.0}),
                format_event("clip_done", {"shot_id": "s0", "clip_key": "clips/s0.mp4", "num_frames": 49,
                                           "fps": 16, "seconds": 3.0, "distilled": True, "elapsed_s": 78.0}),
                format_event("rife_model_probe", {"block_count": 3, "c_per_block": 90,
                                                  "flownet_pkl_bytes": 23_400_000, "loaded": True}),
                format_event("rife_done", {"shot_id": "s0", "input_frames": 2, "output_frames": 3,
                                           "factor": 2, "h": 720, "w": 1280, "elapsed_s": 9.0}),
                format_event("finish_done", {"shot_id": "s0", "clip_key": "clips/s0_finished.mp4",
                                             "interpolated": True, "face_restored": True, "out_frames": 97,
                                             "out_fps": 32, "bytes": 1_400_000, "elapsed_s": 52.0}),
                format_event("e2e_done", {"shots": 2, "output_key": "renders/e2e/film.mp4",
                                          "has_audio": True, "duration_s": 6.0, "bytes": 3_200_000,
                                          "elapsed_s": 420.0}),
            ]
        return lines

    def stop_pod(self, _pod_id):
        return {"id": _pod_id, "desiredStatus": "EXITED"}

    def delete_pod(self, _pod_id):
        return {"id": _pod_id, "deleted": True}


if __name__ == "__main__":
    raise SystemExit(main())
