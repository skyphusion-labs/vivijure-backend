#!/usr/bin/env python3
"""3-arch de-risk driver for the baked bf16 image (vivijure-backend).

Runs ON the pod, inside the baked image's conda env. Drives the REAL pipeline
(keyframe -> i2v -> RIFE -> finish) directly over GpuPipeline.execute, with NO R2:
the bundle is built in-memory and outputs are written to local disk. Emits a
machine-readable @event channel (GMCP-style) we assert on, never English prose.

Headline it proves: a baked worker NEVER pulls from R2 at boot or at render. The
rclone tripwire (a fake rclone earlier in PATH that touches VJ_RCLONE_TRIPWIRE on
spawn) makes that definitive: if the baked short-circuit is honest, rclone is never
spawned and the tripwire file never appears. Every subcommand re-checks it.

THREE-ARCH DE-RISK (#15): the serverless pool bounces across three DC cards --
H200 (Hopper sm_90), B200 (Blackwell DC sm_100), RTX PRO 6000 Blackwell
(sm_120). 3-arch coverage rides on the prebuilt cu128 wheels (nothing compiles
CUDA from source, so TORCH_CUDA_ARCH_LIST is a no-op and is absent by design).
The `arch-gate` subcommand is the build-time STOP-gate: it asserts the three BASE
targets {sm_90, sm_100, sm_120} are present in torch.cuda.get_arch_list(). It is
CPU-only (get_arch_list reads the wheel's build config, no GPU needed), so it runs
inside the image in CI. `probe` + `render` then prove the kernels actually LOAD and
RUN on each real card, and MEASURE the per-arch first-call (Triton/inductor) JIT
cost that CPU CI cannot pre-warm.

Usage (on the pod, per arch):
    python vj_derisk.py arch-gate                                   # build-time STOP-gate, CPU-only
    python vj_derisk.py probe                                        # gpu_probe + baked-sentinel + kernel
    python vj_derisk.py render --aspect portrait  --tier final --out /workspace/out
    python vj_derisk.py render --aspect landscape --tier final --out /workspace/out
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# src/ layout -> import path. The image puts our package at /opt/vivijure; fall back to a repo checkout.
for cand in ("/opt/vivijure", str(Path(__file__).resolve().parents[0] / "vivijure-backend" / "src")):
    if cand not in sys.path and Path(cand).exists():
        sys.path.insert(0, cand)


def ev(event: str, **fields) -> None:
    print("@event " + event + " " + json.dumps(fields, sort_keys=True), flush=True)


# --------------------------------------------------------------------------- arch gate

# The three BASE arch targets the serverless pool requires, one per pooled DC card:
#   sm_90  -> H200 (Hopper)        sm_100 -> B200 (Blackwell DC)        sm_120 -> RTX PRO 6000 (Blackwell)
# Base targets only. The sm_*a accelerated variants and the compute_* PTX entry torch may also list are
# BONUS, never a substitute: a kernel compiled ONLY against an 'a' variant would still pass this gate but
# could trip a real forward -- that residual is the runtime backstop the per-arch render (below) catches.
ARCH_GATE: tuple[str, ...] = ("sm_90", "sm_100", "sm_120")


def missing_arches(arch_list) -> list[str]:
    """Pure: the required base arches absent from torch's compiled arch list (literal membership,
    'a' variants do NOT satisfy a base target). Testable without torch or a GPU."""
    present = set(arch_list)
    return [a for a in ARCH_GATE if a not in present]


def arch_gate() -> int:
    """Build-time STOP-gate: assert the cu128 wheel carries SASS/PTX for all three base arches.
    CPU-only (get_arch_list reads the build config, not the device), so it runs inside the image in CI."""
    import torch
    arch_list = list(torch.cuda.get_arch_list())
    miss = missing_arches(arch_list)
    ev("arch_gate",
       torch_version=torch.__version__,
       arch_list=arch_list,                       # RAW, verbatim -- not a yes/no
       required=list(ARCH_GATE),
       present=[a for a in ARCH_GATE if a in arch_list],
       missing=miss,
       passed=(not miss))
    if miss:
        print("ARCH GATE FAIL: %s missing from torch arch list %s" % (miss, arch_list), file=sys.stderr)
        return 1
    return 0


def _check_tripwire(where: str) -> None:
    """Emit the rclone no-pull proof: the fake rclone touches VJ_RCLONE_TRIPWIRE on spawn, so on a
    truthfully-baked image the file must NEVER exist. Absent env -> tripwire not armed (reported)."""
    trip = os.environ.get("VJ_RCLONE_TRIPWIRE")
    if not trip:
        ev("rclone_tripwire", where=where, armed=False)
        return
    ev("rclone_tripwire", where=where, armed=True, path=trip, fired=Path(trip).exists())


# --------------------------------------------------------------------------- probe

def probe() -> int:
    """gpu_probe + baked-sentinel inspection. No render, no weight load past a tiny CUDA kernel."""
    models_root = os.environ.get("VJ_MODELS_ROOT", "/opt/models")
    sentinel = Path(models_root) / ".vj-baked"
    baked_meta = {}
    if sentinel.is_file():
        for line in sentinel.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                baked_meta[k.strip()] = v.strip()

    # Which i2v repos are actually baked into the HF cache?
    hub = Path(models_root) / "hf-cache" / "hub"
    i2v_repos = sorted(p.name for p in hub.glob("models--*Wan*")) if hub.is_dir() else []
    finish_dirs = sorted(p.name for p in Path(models_root).iterdir()
                         if p.is_dir() and p.name in ("antelopev2", "rife", "GFPGANv1.4"))

    from vivijure_backend.harness.models_mirror import is_baked, repo_in_hf_cache
    from vivijure_backend.models import DEFAULT_SPECS, ModelRole, _select_i2v_weights
    baked = is_baked()

    # $0 PRE-GPU gate: assert the EXACT repos the render will from_pretrained ARE cached, BEFORE the
    # CUDA kernel below. A models--*Wan* glob is not enough -- it matched the bf16 repo while the
    # runtime asked for the un-baked -fp8 repo, and that LocalEntryNotFoundError reached render. Reuse
    # the runtime's own i2v decision (_select_i2v_weights) so the probe can never assert a different
    # repo than the load resolves; check the keyframe base too (same un-baked-repo-offline class).
    # Finish models load direct file paths (finish_dirs below), not HF snapshots, so they are separate.
    i2v_spec = DEFAULT_SPECS[ModelRole.I2V]
    i2v_final = os.environ.get("VJ_I2V_DISTILL", "1") == "0"
    i2v_fp8_present = bool(i2v_spec.fp8_repo_id) and repo_in_hf_cache(i2v_spec.fp8_repo_id) if baked else False
    i2v_repo_active, _, _ = _select_i2v_weights(
        i2v_spec, baked=baked, final_tier=i2v_final, fp8_present=i2v_fp8_present)
    render_repos = {"i2v": i2v_repo_active,
                    "keyframe_base": DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id}
    missing_repos = {role: r for role, r in render_repos.items() if not repo_in_hf_cache(r)}
    if missing_repos:
        ev("derisk_fail", stage="render_repo_not_cached", missing_repos=missing_repos,
           render_repos=render_repos, i2v_final_tier=i2v_final,
           hint="runtime would from_pretrained these offline and raise LocalEntryNotFoundError")
        return 1

    import torch
    cuda_ok = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else None
    cap = torch.cuda.get_device_capability(0) if cuda_ok else None
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if cuda_ok else 0.0
    free_b, total_b = torch.cuda.mem_get_info() if cuda_ok else (0, 0)
    arch_list = list(torch.cuda.get_arch_list())

    # The real test: does a cu128 kernel actually LOAD + RUN on THIS arch (the Blackwell sm_120
    # "no kernel image is available" class of failure)? A tiny bf16 matmul on-device proves it. Time
    # it: the FIRST device kernel launch bears the lazy-load/JIT cost, the headline per-arch metric.
    kernel_ok = False
    kernel_first_call_seconds = None
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        c = (a @ b).float().sum().item()
        torch.cuda.synchronize()
        kernel_first_call_seconds = round(time.time() - t0, 4)
        kernel_ok = c == c  # not NaN
    except Exception as exc:  # noqa: BLE001
        ev("kernel_error", error=str(exc))

    # Base-arch coverage of THIS card vs the wheel: a card whose capability is missing from the wheel's
    # arch list is the exact silent-drop the gate exists for; surface it here on real silicon too.
    cap_sm = ("sm_%d%d" % cap) if cap else None
    ev("gpu_probe",
       torch_cuda=cuda_ok, kernel_ok=kernel_ok, device_name=dev_name,
       capability=("%d.%d" % cap) if cap else None,
       capability_sm=cap_sm,
       capability_in_arch_list=(cap_sm in arch_list) if cap_sm else None,
       kernel_first_call_seconds=kernel_first_call_seconds,
       arch_list=arch_list,
       vram_total_gb=vram_total,
       vram_free_gb=round(free_b / 1e9, 2), vram_total_meminfo_gb=round(total_b / 1e9, 2))
    ev("baked_probe",
       vj_baked=baked, sentinel_present=sentinel.is_file(), sentinel_meta=baked_meta,
       i2v_repos_baked=i2v_repos, finish_dirs=finish_dirs,
       render_repos_active=render_repos, i2v_repo_active=i2v_repo_active)
    _check_tripwire("probe")
    return 0


# --------------------------------------------------------------------------- render

class _Probe:
    """Minimal progress emitter: timestamps stage events + samples peak VRAM, resetting per stage.
    Also records every i2v step timestamp so render_done can isolate the per-arch first-call JIT cost
    (first step bears the Triton/inductor compile; steady steps do not)."""
    def __init__(self):
        import torch
        self._torch = torch
        self.t0 = time.time()
        self.last = self.t0
        self.stages = []
        self.i2v_start = None                 # set when the keyframe stage finishes (i2v begins)
        self.i2v_step_times: dict[str, list] = {}
        torch.cuda.reset_peak_memory_stats()

    def _mark(self, name):
        now = time.time()
        peak = round(self._torch.cuda.max_memory_allocated() / 1e9, 2)
        rec = {"stage": name, "seconds": round(now - self.last, 1), "peak_vram_gb": peak}
        self.stages.append(rec)
        ev("stage_done", **rec)
        if name == "keyframe_done":
            self.i2v_start = now              # i2v compute starts the instant keyframe finishes
        self.last = now
        self._torch.cuda.reset_peak_memory_stats()

    def emit(self, event, **fields):
        if event in ("keyframe_done", "i2v_done", "finish_done"):
            self._mark(event)
        else:
            ev("pipeline_" + event, **fields)

    def complete(self, **fields):
        ev("pipeline_complete", **fields)

    def train_step_cb(self, slot):
        return None

    def i2v_step_cb(self, shot):
        times = self.i2v_step_times.setdefault(shot, [])
        def cb(step, total):
            times.append((step, time.time()))
            if step == 1 or step == total:
                ev("i2v_step", shot=shot, step=step, total=total,
                   peak_vram_gb=round(self._torch.cuda.max_memory_allocated() / 1e9, 2))
        return cb

    def finish_cb(self, shot):
        return None

    def i2v_jit_summary(self) -> list[dict]:
        """Per-shot first-call JIT cost: first-step wall-clock (compile + compute) minus the steady
        per-step median (compute only). Defensive -- returns partial/null fields if data is thin."""
        out = []
        for shot, times in self.i2v_step_times.items():
            times = sorted(times)
            rec = {"shot": shot, "steps": len(times),
                   "first_step_seconds": None, "steady_step_seconds": None,
                   "first_call_jit_seconds": None}
            if len(times) >= 2 and self.i2v_start is not None:
                first_dur = times[0][1] - self.i2v_start
                deltas = [times[i][1] - times[i - 1][1] for i in range(1, len(times))]
                steady = statistics.median(deltas)
                rec["first_step_seconds"] = round(first_dur, 3)
                rec["steady_step_seconds"] = round(steady, 3)
                rec["first_call_jit_seconds"] = round(max(first_dur - steady, 0.0), 3)
            out.append(rec)
        return out


ASPECTS = {
    "portrait":  (576, 1024),
    "landscape": (1024, 576),
}


def _ffprobe_dims(path: Path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames,duration",
             "-of", "json", str(path)], text=True)
        s = json.loads(out)["streams"][0]
        return {k: s.get(k) for k in ("width", "height", "nb_frames", "duration")}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def _thumb_b64_from_image(path: Path, px=256) -> str:
    from PIL import Image
    import io
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _thumb_b64_from_clip(path: Path, px=256) -> str:
    import tempfile
    tmp = Path(tempfile.mkstemp(suffix=".jpg")[1])
    subprocess.check_call(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path),
         "-vf", "thumbnail,scale=%d:-1" % px, "-frames:v", "1", str(tmp)])
    b = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink(missing_ok=True)
    return b


def build_render_inputs(aspect: str, tier: str, out_dir: Path, frames: int, i2v_steps: int, kf_steps: int):
    """Build the GPU-free render inputs (contract objects + config) for a de-risk render. Pure of any
    CUDA/GPU work, so CI can exercise the whole construction path against the REAL contract -- the exact
    coverage gap that let a bad Scene kwarg (duration_seconds, not target_seconds) reach the pod and
    derisk_fail at stage=render_portrait. Returns (req, sb, cast, bundle, cfg, work, w, h)."""
    from vivijure_backend.contract import Scene, Storyboard, Cast, Bundle, RenderRequest
    from vivijure_backend.config import RenderConfig

    w, h = ASPECTS[aspect]
    work = out_dir / aspect
    work.mkdir(parents=True, exist_ok=True)

    scene = Scene(
        id="shot_01",
        prompt=("a lone lighthouse on a rocky coast at golden hour, sweeping clouds, "
                "cinematic wide shot, gentle camera push-in"),
        character_slots=[],
        target_seconds=2.0,
    )
    sb = Storyboard(title="derisk-%s" % aspect, scenes=[scene],
                    duration_seconds=2.0, use_characters=[])
    cast = Cast(characters={})
    bundle = Bundle(root=work / "project", storyboard=sb, cast=cast)
    bundle.root.mkdir(parents=True, exist_ok=True)

    overrides = {
        "keyframe": {"width": w, "height": h, "steps": kf_steps},
        "i2v": {"num_frames": frames, "steps": i2v_steps},
        "finish": {"interpolate": True, "interpolation_factor": 2, "face_restore": "gfpgan"},
    }
    cfg = RenderConfig.from_request(tier, overrides)
    req = RenderRequest(action="render", project="derisk", bundle_key="local",
                        quality_tier=tier, config=cfg)
    return req, sb, cast, bundle, cfg, work, w, h


def render(aspect: str, tier: str, out_dir: Path, frames: int, i2v_steps: int, kf_steps: int) -> int:
    from vivijure_backend.orchestrator import plan as make_plan
    from vivijure_backend.pipeline import GpuPipeline
    from vivijure_backend.assemble import ClipInput, assemble, order_for_storyboard
    import torch

    req, sb, _cast, bundle, cfg, work, w, h = build_render_inputs(
        aspect, tier, out_dir, frames, i2v_steps, kf_steps)
    ev("render_config", aspect=aspect, tier=tier, width=w, height=h,
       i2v_distill_env=os.environ.get("VJ_I2V_DISTILL"),
       finish_enabled=cfg.finish.enabled,
       i2v_steps=cfg.i2v.steps, i2v_frames=cfg.i2v.num_frames, kf_steps=cfg.keyframe.steps)

    rplan = make_plan(req, sb)
    ev("plan", train=list(rplan.lora.train), keyframes=rplan.keyframes_to_generate,
       shots=rplan.shots_to_animate, summary=rplan.summary())

    pipe = GpuPipeline(config=cfg)
    probe = _Probe()
    pipe.set_progress(probe)

    t0 = time.time()
    outputs = pipe.execute(rplan, bundle, work)
    total = round(time.time() - t0, 1)

    # Assemble the finished clips into one film, off-GPU (the harness's job, done here locally).
    ordered = order_for_storyboard(
        [ClipInput(shot_id=s, path=p) for s, p in outputs.clips], sb)
    film = work / "film.mp4"
    assemble(ordered, film)

    # Capture artifacts.
    kf = outputs.keyframes.get("shot_01")
    kf_bytes = kf.stat().st_size if kf and kf.is_file() else 0
    clip_path = outputs.clips[0][1] if outputs.clips else None
    clip_bytes = clip_path.stat().st_size if clip_path and clip_path.is_file() else 0
    film_bytes = film.stat().st_size if film.is_file() else 0

    ev("render_done", aspect=aspect, total_seconds=total,
       keyframe_bytes=kf_bytes, clip_bytes=clip_bytes, film_bytes=film_bytes,
       clip_dims=_ffprobe_dims(clip_path) if clip_path else None,
       film_dims=_ffprobe_dims(film),
       peak_vram_gb_overall=round(torch.cuda.max_memory_allocated() / 1e9, 2),
       i2v_jit=probe.i2v_jit_summary(),
       stages=probe.stages)
    if kf and kf.is_file():
        ev("thumb_keyframe", aspect=aspect, b64=_thumb_b64_from_image(kf))
    if film.is_file():
        ev("thumb_film_frame", aspect=aspect, b64=_thumb_b64_from_clip(film))
    _check_tripwire("render")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arch-gate")
    sub.add_parser("probe")
    r = sub.add_parser("render")
    r.add_argument("--aspect", choices=list(ASPECTS), required=True)
    r.add_argument("--tier", default="final")
    r.add_argument("--out", type=Path, default=Path("/workspace/out"))
    r.add_argument("--frames", type=int, default=25)
    r.add_argument("--i2v-steps", type=int, default=8)
    r.add_argument("--kf-steps", type=int, default=20)
    args = ap.parse_args()

    if args.cmd == "arch-gate":
        return arch_gate()
    if args.cmd == "probe":
        return probe()
    return render(args.aspect, args.tier, args.out, args.frames, args.i2v_steps, args.kf_steps)


if __name__ == "__main__":
    raise SystemExit(main())
