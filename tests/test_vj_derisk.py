"""Unit tests for the de-risk driver's CPU-only logic (no torch, no GPU).

Three GPU-free surfaces are exercised here:
  - the arch-gate set logic (which base targets the cu128 wheel must carry),
  - build_render_inputs, the render() prologue that constructs the contract objects + config, and
  - the RenderPlan API render() consumes (make_plan signature + plan property/method shapes).
The actual probe/render (kernel load, weight load, real i2v) need the baked image + a CUDA device,
so they are proven on the pod, not in CI. Importing the module is import-light (the heavy
vivijure_backend imports are deferred inside build_render_inputs/probe/render)."""
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


def test_build_render_inputs_constructs_against_the_real_contract(tmp_path):
    # Regression guard: render() once passed an unsupported Scene kwarg (duration_seconds), which only
    # blew up ON THE POD at derisk_fail stage=render_portrait because nothing in CI exercised the
    # construction path. build_render_inputs is GPU-free, so the whole prologue is provable here against
    # the REAL vivijure_backend contract. Both aspects, so a bad ASPECTS entry is caught too.
    for aspect, (w, h) in vj.ASPECTS.items():
        req, sb, cast, bundle, cfg, work, gw, gh = vj.build_render_inputs(
            aspect, "final", tmp_path, frames=25, i2v_steps=8, kf_steps=20)
        assert (gw, gh) == (w, h)
        # The Scene duration hint the contract actually supports is target_seconds, NOT duration_seconds.
        assert sb.scenes[0].target_seconds == 2.0
        assert sb.scenes[0].id == "shot_01"
        # Storyboard has its OWN distinct duration_seconds field (this one IS valid on Storyboard).
        assert sb.duration_seconds == 2.0
        assert req.config is cfg
        assert work.is_dir() and bundle.root.is_dir()


def test_scene_contract_rejects_duration_seconds():
    # Pin the exact mismatch that caused the pod failure: Scene does not accept duration_seconds.
    from vivijure_backend.contract import Scene
    with pytest.raises(TypeError):
        Scene(prompt="x", duration_seconds=2.0)


def test_render_plan_api_matches_driver_usage(tmp_path):
    # render() walks the CPU path build_render_inputs -> make_plan(req, sb) -> reads the plan. Two more
    # driver/contract mismatches lurked PAST the prologue crash, each a separate GPU-spend re-fire crash:
    #   - make_plan was called with a cast= kwarg that plan() does not accept, and
    #   - keyframes_to_generate / shots_to_animate are @property (read without parens), not methods.
    # Pin the exact API shapes render() depends on so a contract drift fails in CI, not on a $/hr pod.
    from vivijure_backend.orchestrator import plan as make_plan
    req, sb, _cast, _bundle, _cfg, _work, _gw, _gh = vj.build_render_inputs(
        "portrait", "final", tmp_path, frames=25, i2v_steps=8, kf_steps=20)
    rplan = make_plan(req, sb)                              # signature: positional (request, storyboard)
    assert isinstance(rplan.keyframes_to_generate, int)    # @property, NOT callable
    assert isinstance(rplan.shots_to_animate, int)         # @property, NOT callable
    assert isinstance(rplan.summary(), str)                # summary() IS a method
    assert rplan.keyframes_to_generate == 1 and rplan.shots_to_animate == 1
