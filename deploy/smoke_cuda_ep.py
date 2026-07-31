"""Session-level CUDA execution-provider check for the RUNTIME BASE image.

Run inside the runtime base, with a GPU actually attached:

    docker run --rm --gpus all -v "$PWD/deploy/smoke_cuda_ep.py:/tmp/smoke_cuda_ep.py:ro" \
      <runtime-base-image> conda run --no-capture-output -n vivijure python /tmp/smoke_cuda_ep.py

Exit 0 = onnxruntime creates a real session on the CUDA execution provider. Non-zero = it does not,
and the InstantID face path would run on CPU in production.

WHY THIS EXISTS SEPARATELY FROM deploy/smoke_imports.py
smoke_imports.py is the RELEASE image check: it is COPY-ed into the worker image and also imports
vivijure_backend.finish, which does not exist in the runtime base (the base has no app code). This
module is the strict subset that can run against the BASE, which is where the deps are actually
installed and therefore where a bad wheel is born. Keep the two in agreement; smoke_imports.py is
the fuller check and stays the gate on the release image.

WHY A GPU IS MANDATORY HERE, NOT OPTIONAL
A provider NAME is not a working provider. Measured on real hardware during backend#346:

    onnxruntime-gpu  import   advertises CUDA EP   actual session
    1.26.0           ok       yes                  CUDAExecutionProvider
    1.27.0           ImportError: libcudart.so.13  n/a
    1.28.0           ok       yes                  silently falls back to CPUExecutionProvider

The 1.28.0 row is the whole argument. It imports, it advertises, and it hands back a CPU session
against the CUDA 12.8 base. Every CPU-only check passes. So this module REFUSES to report a pass it
did not earn: no GPU visible is a FAILURE here, not a skip. The caller passed --gpus all; if that
did not produce a GPU, the check did not run, and a check that did not run must not be green.
"""
import os
import sys

REQUIRED_PROVIDER = "CUDAExecutionProvider"
# Baked with the antelopev2 pack the InstantID face analyzer uses. The detection model is the one
# models.py loads first, so a session here exercises the same wheel on the same file.
PROBE_MODEL = os.path.join(os.environ.get("VJ_MODELS_ROOT", "/opt/models"),
                           "antelopev2", "scrfd_10g_bnkps.onnx")

try:
    import torch
except Exception as exc:
    print(f"FAIL  torch failed to import: {exc}", file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print("FAIL  no GPU is visible to this container, so the session-level CUDA EP check cannot "
          "run.", file=sys.stderr)
    print("      This step is invoked with `docker run --gpus all`. A missing GPU here means the "
          "nvidia container runtime did not attach one, not that the check is inapplicable. "
          "Failing rather than skipping: a check that did not run must not report green.",
          file=sys.stderr)
    sys.exit(1)

print(f"OK    GPU visible: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")

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
