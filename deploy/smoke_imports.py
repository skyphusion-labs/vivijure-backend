"""Import smoke for finishing-stage runtime deps, plus the render-path GPU provider check.

Run inside the built Docker image to verify every required package is present BEFORE the image is
pushed to GHCR:

    docker run --rm <image> conda run --no-capture-output -n vivijure python /opt/vivijure/smoke_imports.py

Exit 0 = all checks OK. Non-zero = at least one problem; FAIL lines on stderr identify what.
A dep missing here fails CI in seconds rather than crashing the worker after a 33-min GPU render.

Closes the loop with issue #9 (CI coverage of the runtime dep surface).

It also checks that onnxruntime still offers a CUDA execution provider (backend#346). That is a
CAPABILITY check, not an import check: a CPU-only onnxruntime build imports perfectly and then runs
the InstantID face embedding on CPU forever without ever erroring, which is exactly what shipped in
backend-v1.0.11.

Read the coverage limits honestly, because they bit us once already:
  * The provider-NAME check below needs no GPU and catches the two cheap failures: the CPU
    onnxruntime wheel shadowing onnxruntime-gpu, and a CUDA-13 wheel that cannot be imported at all.
  * A provider NAME is not a working provider. onnxruntime-gpu 1.28.0 imports cleanly, advertises
    CUDAExecutionProvider, and then silently hands the session back on CPUExecutionProvider against
    a CUDA 12.8 base. Only creating a real session catches that, and that needs a real GPU.
So when a GPU is visible this module creates a real session and asserts the EP actually attaches;
when one is not, it says so LOUDLY rather than reporting a pass it did not earn.
"""
import importlib
import os
import sys

CHECKS = [
    ("av",                                       "PyAV: imageio pyav plugin for finish_clip"),
    ("gfpgan",                                   "GFPGAN blind face restorer"),
    ("basicsr.utils.registry",                   "basicsr ARCH_REGISTRY (codeformer path)"),
    ("facexlib.utils.face_restoration_helper",   "facexlib face detection helper"),
    ("rife.RIFE_HDv3",                           "vendored RIFE HDv3 frame interpolator (Model loader)"),
    ("vivijure_backend.finish",                  "finishing stage (must stay CPU-importable)"),
    ("diffusers.pipelines.wan.pipeline_wan_i2v", "Wan i2v pipeline (catches torchao/torch mismatch)"),
]

REQUIRED_PROVIDER = "CUDAExecutionProvider"
# Baked with the antelopev2 pack the InstantID face analyzer uses; the detection model is the one
# models.py loads first, so a session here exercises the same wheel on the same file.
PROBE_MODEL = os.path.join(os.environ.get("VJ_MODELS_ROOT", "/opt/models"),
                           "antelopev2", "scrfd_10g_bnkps.onnx")

failed = []
for mod, label in CHECKS:
    try:
        importlib.import_module(mod)
        print(f"OK    {mod}  ({label})")
    except Exception as exc:
        print(f"FAIL  {mod}  ({label}): {exc}", file=sys.stderr)
        failed.append(mod)

if failed:
    print(f"\n{len(failed)} import(s) failed; see FAIL lines above.", file=sys.stderr)
    sys.exit(len(failed))

print(f"\nAll {len(CHECKS)} finish-stage imports OK.")

# ---------------------------------------------------------------- render-path GPU provider check
try:
    import onnxruntime
except Exception as exc:
    print(f"FAIL  onnxruntime failed to import: {exc}", file=sys.stderr)
    print("      An ImportError naming libcudart.so.13 means a CUDA 13 onnxruntime-gpu wheel got "
          "past the cap in deploy/requirements.txt (backend#346).", file=sys.stderr)
    sys.exit(1)

providers = onnxruntime.get_available_providers()
if REQUIRED_PROVIDER not in providers:
    print(f"FAIL  onnxruntime {onnxruntime.__version__} does not expose {REQUIRED_PROVIDER}; "
          f"available: {providers}. The InstantID face embedding would run on CPU.", file=sys.stderr)
    print("      A CPU-only onnxruntime build is shadowing onnxruntime-gpu; "
          "deploy/ensure_onnxruntime_gpu.py is what prevents this.", file=sys.stderr)
    sys.exit(1)

print(f"OK    onnxruntime {onnxruntime.__version__} exposes {REQUIRED_PROVIDER}.")

# The name is necessary but NOT sufficient. Only a real session proves the EP attaches.
try:
    import torch
    gpu_visible = torch.cuda.is_available()
except Exception:
    gpu_visible = False

if not gpu_visible:
    print("SKIP  session-level CUDA EP check: no GPU visible to this container.")
    print("      THIS RUN DID NOT PROVE THE CUDA EP INITIALISES. A wheel can advertise "
          f"{REQUIRED_PROVIDER} and still fall back to CPU at session creation "
          "(onnxruntime-gpu 1.28.0 does exactly that on a CUDA 12.8 base).")
    print("      Re-run with `docker run --gpus all` to turn this into real evidence.")
    sys.exit(0)

if not os.path.exists(PROBE_MODEL):
    print(f"FAIL  GPU is visible but the probe model is missing: {PROBE_MODEL}", file=sys.stderr)
    sys.exit(1)

session = onnxruntime.InferenceSession(
    PROBE_MODEL, providers=[REQUIRED_PROVIDER, "CPUExecutionProvider"])
applied = session.get_providers()
if REQUIRED_PROVIDER not in applied:
    print(f"FAIL  onnxruntime {onnxruntime.__version__} advertised {REQUIRED_PROVIDER} but the "
          f"session fell back to {applied}. The InstantID face embedding would run on CPU.",
          file=sys.stderr)
    sys.exit(1)

print(f"OK    session-level check: {REQUIRED_PROVIDER} attached (session providers: {applied}).")
