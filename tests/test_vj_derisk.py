"""Unit tests for the de-risk driver's pure logic (no torch, no GPU).

Only the arch-gate set logic is exercised here: probe/render need the baked image + a real CUDA
device, so they are proven on the pod, not in CI. Importing the module is import-light (the heavy
vivijure_backend imports are deferred inside probe/render)."""
import importlib.util
from pathlib import Path

import pytest

# Load deploy/vj_derisk.py directly (not a package import).
_SPEC = importlib.util.spec_from_file_location(
    "vj_derisk", Path(__file__).resolve().parents[1] / "deploy" / "vj_derisk.py")
vj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vj)


def test_arch_gate_requires_the_three_base_targets():
    assert vj.ARCH_GATE == ("sm_90", "sm_100", "sm_120")


def test_missing_arches_passes_a_full_cu128_list():
    # A representative torch 2.7 + cu128 arch list: the three base targets present (+ bonus arches/PTX).
    full = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
    assert vj.missing_arches(full) == []


def test_missing_arches_reports_each_absent_base_target():
    assert vj.missing_arches(["sm_80", "sm_90"]) == ["sm_100", "sm_120"]
    assert vj.missing_arches(["sm_90", "sm_100"]) == ["sm_120"]
    assert vj.missing_arches([]) == ["sm_90", "sm_100", "sm_120"]


def test_accelerated_variant_does_not_satisfy_a_base_target():
    # sm_120a (accelerated variant) is BONUS, never a substitute for the base sm_120 target: a kernel
    # built only against the 'a' variant can still trip a real forward (the #15 runtime backstop).
    assert "sm_120" in vj.missing_arches(["sm_90", "sm_100", "sm_120a"])


def test_missing_arches_is_order_independent():
    assert vj.missing_arches(["sm_120", "sm_100", "sm_90"]) == []
