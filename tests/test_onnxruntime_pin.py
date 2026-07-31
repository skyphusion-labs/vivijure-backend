"""Regression guards for the onnxruntime pin (backend#346).

The InstantID face-embedding path shipped on CPU in backend-v1.0.11 and nothing failed. Two
independent defects caused it, and CI could not see either one:

  A. insightface hard-depends on the CPU `onnxruntime` wheel. It shares the `onnxruntime/` package
     directory with onnxruntime-gpu, and when it wins CUDAExecutionProvider disappears entirely.
     deploy/ensure_onnxruntime_gpu.py removes it and repairs the GPU wheel at build time.
  B. onnxruntime-gpu switched its default wheel to a CUDA 13 build at 1.27.0. Against this CUDA 12.8
     image, 1.27.0 raises `ImportError: libcudart.so.13`, and 1.28.0 is worse: it imports, advertises
     CUDAExecutionProvider, and then silently runs the session on CPU.

These tests are the CPU-only half of the defence: they hold the CAP in requirements.txt in place, so
a dependency bump that raises the floor past the CUDA-13 switch fails CI instead of shipping a
silent CPU fallback. The GPU half (a real session actually attaching the EP) lives in
deploy/smoke_imports.py, which cannot run here because CI has no GPU.

deploy/ scripts sit off the src/ pythonpath, so this imports by path the same way test_bake_layers
does."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deploy"))
import ensure_onnxruntime_gpu  # noqa: E402

REQUIREMENTS = REPO_ROOT / "deploy" / "requirements.txt"

# The first onnxruntime-gpu release built against CUDA 13. Everything from here up is incompatible
# with the deliberately-pinned CUDA 12.8 runtime base.
CUDA13_SWITCH = (1, 27, 0)


def _parse_version(text):
    return tuple(int(part) for part in text.strip().split(".")[:3])


def test_requirements_caps_onnxruntime_gpu_below_the_cuda13_switch():
    """The cap is the ACTUAL guarantee (the build-time gate cannot catch the 1.28.0 case), so it
    gets a test of its own. If this fails, read deploy/requirements.txt before touching it."""
    spec = ensure_onnxruntime_gpu.read_gpu_spec(str(REQUIREMENTS))
    uppers = [chunk.split("<", 1)[1] for chunk in spec.split(",") if chunk.strip().startswith("<")]
    assert uppers, (
        "deploy/requirements.txt must cap onnxruntime-gpu with an upper bound; got %r. "
        "Raising it past %s requires moving the runtime base to CUDA 13 and re-validating the "
        "pipeline on a GPU pod (backend#346)." % (spec, ".".join(str(p) for p in CUDA13_SWITCH))
    )
    for upper in uppers:
        assert _parse_version(upper) <= CUDA13_SWITCH, (
            "onnxruntime-gpu upper bound %s allows a CUDA-13 wheel on the CUDA 12.8 base "
            "(backend#346)." % upper
        )


def test_read_gpu_spec_returns_the_requirement_line():
    spec = ensure_onnxruntime_gpu.read_gpu_spec(str(REQUIREMENTS))
    assert spec.lower().startswith("onnxruntime-gpu")
    # Must be the GPU distribution, never the CPU one that insightface drags in.
    assert not spec.lower().startswith("onnxruntime==")


def test_read_gpu_spec_ignores_comments_and_the_cpu_distribution(tmp_path):
    """A bare `onnxruntime` line (defect A) must never be mistaken for the GPU requirement, and a
    commented-out pin must not satisfy the lookup."""
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# onnxruntime-gpu>=9.9.9 commented out, must be ignored\n"
        "onnxruntime\n"
        "onnxruntime-gpu>=1.21.0,<1.27.0  # the real one\n",
        encoding="utf-8",
    )
    assert ensure_onnxruntime_gpu.read_gpu_spec(str(path)) == "onnxruntime-gpu>=1.21.0,<1.27.0"


def test_read_gpu_spec_raises_when_the_requirement_is_missing(tmp_path):
    """Positive control for the parser: it must FAIL LOUD rather than silently return nothing, or
    the build step would happily reinstall an empty spec."""
    path = tmp_path / "requirements.txt"
    path.write_text("insightface>=1.0.1\nonnxruntime\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ensure_onnxruntime_gpu.read_gpu_spec(str(path))
