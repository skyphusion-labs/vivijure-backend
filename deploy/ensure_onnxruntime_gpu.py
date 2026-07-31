"""Guarantee the image ends up with a CUDA-capable onnxruntime, and PROVE it (backend#346).

Two independent defects put the InstantID face-embedding path on CPU in production without failing
a single build. This script fixes both and then asserts the invariant, so neither can come back
silently.

  (A) insightface declares a bare "onnxruntime" requirement (the CPU build). pip happily installs
      it ALONGSIDE our onnxruntime-gpu: the two wheels share one onnxruntime/ package directory,
      and whichever lands last owns the native module. When the CPU build wins, its binary has no
      CUDA execution provider compiled in at all, so CUDAExecutionProvider is not merely failing
      to start, it is absent from get_available_providers(). Measured in backend-v1.0.11: the
      providers were [AzureExecutionProvider, CPUExecutionProvider] on a healthy GPU, with
      torch.cuda.is_available() returning True. An AzureExecutionProvider entry is the fingerprint
      of the CPU wheel.

  (B) onnxruntime-gpu moved its default wheel from a CUDA 12.8 build to a CUDA 13.0 build at
      1.27.0. Against the CUDA 12.8 base of this image that wheel cannot even be imported
      (ImportError: libcudart.so.13). deploy/requirements.txt therefore caps the floor below 1.27.0.

The ORDER matters: (A) was masking (B). Removing the CPU wheel WITHOUT the cap in requirements.txt
turns a silent CPU fallback into a hard ImportError at worker start. Never do one without the other.

Why the reinstall is mandatory, not cosmetic: uninstalling onnxruntime deletes files inside the
shared onnxruntime/ tree that onnxruntime-gpu also owns, leaving the GPU package present in
metadata but broken on disk. The forced reinstall repairs it.

Usage (inside the conda env, from the Dockerfiles):
    python deploy/ensure_onnxruntime_gpu.py /tmp/requirements.txt

WHAT THIS GATE CANNOT DO, stated plainly so nobody trusts it further than it goes: a provider
NAME is not a working provider. onnxruntime-gpu 1.28.0 imports cleanly against this CUDA 12.8
base, advertises CUDAExecutionProvider, and then silently hands back a CPU session. It would
sail through the check below. Catching that needs a real GPU session, which a docker build step
does not have. The CAP in requirements.txt is therefore the actual guarantee; this gate is the
backstop that catches the two cheap failures (the CPU wheel shadowing the GPU one, and a CUDA-13
wheel that cannot import). deploy/smoke_imports.py does the real session check when a GPU is
visible to it.

Exit 0 only when a fresh interpreter reports CUDAExecutionProvider as available. Any other outcome
exits non-zero and fails the build LOUD. The check needs no GPU: both failure modes above are
visible on a CPU-only runner (one is an ImportError, the other a missing provider name). That was
verified by running the gate with the GPU hidden, it is not an assumption.
"""
import re
import subprocess
import sys

CPU_DIST = "onnxruntime"
GPU_DIST = "onnxruntime-gpu"
REQUIRED_PROVIDER = "CUDAExecutionProvider"


def read_gpu_spec(requirements_path):
    """Return the onnxruntime-gpu requirement EXACTLY as requirements.txt states it.

    requirements.txt stays the single source of the version set, so the cap lives in one place and
    this script can never drift from it.
    """
    with open(requirements_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name = re.split(r"[<>=!~\[ ]", line, 1)[0].strip().lower().replace("_", "-")
            if name == GPU_DIST:
                return line
    raise SystemExit("ensure_onnxruntime_gpu: no %s requirement found in %s"
                     % (GPU_DIST, requirements_path))


def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip"] + list(args),
                          capture_output=True, text=True)


def installed(dist):
    return pip("show", dist).returncode == 0


def main():
    requirements_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/requirements.txt"
    spec = read_gpu_spec(requirements_path)
    print("ensure_onnxruntime_gpu: requirements.txt asks for %s" % spec)

    # (A) Drop the CPU wheel that insightface drags in. Absent is fine; this is idempotent.
    if installed(CPU_DIST):
        print("ensure_onnxruntime_gpu: removing the CPU %s wheel (insightface transitive dep)"
              % CPU_DIST)
        result = pip("uninstall", "-y", CPU_DIST)
        if result.returncode != 0:
            sys.stderr.write(result.stdout + result.stderr)
            raise SystemExit("ensure_onnxruntime_gpu: failed to uninstall %s" % CPU_DIST)
    else:
        print("ensure_onnxruntime_gpu: CPU %s wheel not present" % CPU_DIST)

    # Repair onnxruntime-gpu: the uninstall above removes shared files it also owns.
    print("ensure_onnxruntime_gpu: reinstalling %s to repair the shared package tree" % spec)
    result = pip("install", "--force-reinstall", "--no-deps", spec)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit("ensure_onnxruntime_gpu: failed to reinstall %s" % spec)

    # THE GATE. A fresh interpreter, so nothing is served from an in-process import cache.
    probe = ("import onnxruntime;"
             " print(onnxruntime.__version__);"
             " print(\",\".join(onnxruntime.get_available_providers()))")
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout + result.stderr)
        raise SystemExit(
            "ensure_onnxruntime_gpu: onnxruntime failed to IMPORT. An ImportError naming "
            "libcudart.so.13 here means the resolved wheel is a CUDA 13 build and the cap in "
            "requirements.txt is missing or too loose (backend#346 defect B).")

    lines = result.stdout.strip().splitlines()
    version = lines[0].strip() if lines else "unknown"
    providers = lines[1].split(",") if len(lines) > 1 else []
    if REQUIRED_PROVIDER not in providers:
        sys.stderr.write("providers reported: %s\n" % providers)
        raise SystemExit(
            "ensure_onnxruntime_gpu: %s is ABSENT from onnxruntime %s. A CPU-only onnxruntime "
            "build is shadowing onnxruntime-gpu (backend#346 defect A); the face-embedding path "
            "would silently run on CPU." % (REQUIRED_PROVIDER, version))

    print("ensure_onnxruntime_gpu: OK -- onnxruntime %s exposes %s (providers: %s)"
          % (version, REQUIRED_PROVIDER, ", ".join(providers)))


if __name__ == "__main__":
    main()
