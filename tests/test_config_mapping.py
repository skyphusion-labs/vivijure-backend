"""Config->engine mapping completeness guard.

Two layers of protection:
1. Behavioral assertions: each test drives the real mapper and checks a concrete value reaches
   the engine param (catching dropped knobs like [B]'s missing height).
2. Meta-test: dataclasses.fields() cross-checked against the asserted+allowlisted set so a
   newly added config field FAILS until it is explicitly mapped or allowlisted.
   This closes the "silent unmapped field" gap that #20 exists to prevent.
"""
from __future__ import annotations

import dataclasses
import pytest

from vivijure_backend.config import (
    FeatureCache,
    FinishConfig,
    I2VConfig,
    KeyframeConfig,
    QualityTier,
    RenderConfig,
)
from vivijure_backend.pipeline import keyframe_params_from, i2v_params_from, finish_params_from

# ------------------------------------------------------------------ helpers

def _scene(seconds=5.0):
    class S:
        target_seconds = seconds
        id = "shot_01"
    return S()


def _rcfg(kf=None, i2v=None, fin=None) -> RenderConfig:
    return RenderConfig.from_request(QualityTier.FINAL, {
        **({"keyframe": kf} if kf else {}),
        **({"i2v": i2v} if i2v else {}),
        **({"finish": fin} if fin else {}),
    })


# --------------------------------------------------------- keyframe mapping

def test_keyframe_width_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"width": 1280}))
    assert p.width == 1280


def test_keyframe_height_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"height": 720}))
    assert p.height == 720


def test_keyframe_guidance_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"guidance_scale": 9.0}))
    assert p.guidance_scale == 9.0


def test_keyframe_seed_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"seed": 12345}))
    assert p.seed == 12345


def test_keyframe_distill_selects_distill_steps():
    p = keyframe_params_from(_rcfg(kf={"distill": True, "distill_steps": 6}))
    assert p.few_step is True
    assert p.steps == 6


def test_keyframe_full_steps_used_when_not_distill():
    p = keyframe_params_from(_rcfg(kf={"distill": False, "steps": 25}))
    assert p.few_step is False
    assert p.steps == 25


def test_keyframe_scheduler_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"scheduler": "dpmpp_2m_karras"}))
    assert p.scheduler == "dpmpp_2m_karras"


def test_keyframe_identity_method_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"identity_method": "instantid"}))
    assert p.identity_method == "instantid"


def test_keyframe_instantid_ip_adapter_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"instantid_ip_adapter_scale": 0.3}))
    assert p.instantid_ip_adapter_scale == pytest.approx(0.3)


def test_keyframe_multi_char_lora_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"lora_scale_per_slot": 0.5}}))
    assert p.lora_scale == pytest.approx(0.5)


def test_keyframe_ip_adapter_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"ip_adapter_scale": 0.4}))
    assert p.ip_adapter_scale == pytest.approx(0.4)


def test_keyframe_pose_conditioning_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"pose_conditioning": False}}))
    assert p.pose_conditioning is False


def test_keyframe_controlnet_pose_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"controlnet_pose_scale": 0.8}}))
    assert p.controlnet_pose_scale == pytest.approx(0.8)


def test_keyframe_region_gutter_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"region_gutter": 32}}))
    assert p.region_gutter == 32


def test_keyframe_max_slots_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"max_slots": 3}}))
    assert p.max_slots == 3


# ---------------------------------------------------------- i2v mapping

def test_i2v_fps_reaches_params():
    cfg = _rcfg(i2v={"fps": 24})
    p = i2v_params_from(cfg, _scene(5.0))
    assert p.fps == 24


def test_i2v_guidance_scale_reaches_params():
    cfg = _rcfg(i2v={"distill": False, "guidance_scale": 4.0})
    p = i2v_params_from(cfg, _scene())
    assert p.guidance_scale == pytest.approx(4.0)


def test_i2v_steps_selected_by_distill():
    cfg = _rcfg(i2v={"distill": False, "steps": 30})
    p = i2v_params_from(cfg, _scene())
    assert p.steps == 30


def test_i2v_distill_steps_selected_when_distill():
    cfg = _rcfg(i2v={"distill": True, "distill_steps": 4})
    p = i2v_params_from(cfg, _scene())
    assert p.distill is True
    assert p.steps == 4


def test_i2v_feature_cache_reaches_params():
    cfg = _rcfg(i2v={"distill": False, "feature_cache": "mixcache"})
    p = i2v_params_from(cfg, _scene())
    assert p.feature_cache == FeatureCache.MIXCACHE


def test_i2v_negative_prompt_reaches_params():
    cfg = _rcfg(i2v={"negative_prompt": "blurry, watermark"})
    p = i2v_params_from(cfg, _scene())
    # Custom prompt is prepended and the anti-static base guard is appended (additive, not replace).
    assert p.negative_prompt.startswith("blurry, watermark")
    assert "static, still, frozen" in p.negative_prompt


def test_i2v_num_frames_from_scene_duration():
    # 5s * 16fps = 80 frames -> snap up to 81 (4*20+1)
    cfg = _rcfg(i2v={"fps": 16})
    p = i2v_params_from(cfg, _scene(5.0))
    assert p.num_frames == 81
    assert (p.num_frames - 1) % 4 == 0


def test_i2v_seed_reaches_params():
    cfg = _rcfg(i2v={"seed": 77777})
    p = i2v_params_from(cfg, _scene())
    assert p.seed == 77777


def test_i2v_flow_shift_reaches_params():
    cfg = _rcfg(i2v={"distill": False, "flow_shift": 3.5})
    p = i2v_params_from(cfg, _scene())
    assert p.flow_shift == pytest.approx(3.5)


# -------------------------------------------------------- finish mapping

def test_finish_interpolate_reaches_params():
    cfg = _rcfg(fin={"interpolate": True, "interpolation_factor": 4})
    p = finish_params_from(cfg)
    assert p.interpolate is True
    assert p.factor == 4


def test_finish_target_fps_reaches_params():
    cfg = _rcfg(fin={"interpolate": True, "target_fps": 60})
    p = finish_params_from(cfg)
    assert p.target_fps == 60


def test_finish_face_restore_maps_enum_to_bool_and_backend():
    cfg = _rcfg(fin={"face_restore": "gfpgan"})
    p = finish_params_from(cfg)
    assert p.face_restore is True
    assert p.face_restore_backend == "gfpgan"


def test_finish_face_restore_none_maps_to_false():
    cfg = _rcfg(fin={"face_restore": "none"})
    p = finish_params_from(cfg)
    assert p.face_restore is False


def test_finish_face_fidelity_reaches_params():
    cfg = _rcfg(fin={"face_restore": "gfpgan", "face_fidelity": 0.3})
    p = finish_params_from(cfg)
    assert p.face_fidelity == pytest.approx(0.3)


def test_finish_only_faces_reaches_params():
    cfg = _rcfg(fin={"face_restore": "gfpgan", "only_faces": False})
    p = finish_params_from(cfg)
    assert p.only_faces is False


# ------------------------------------------------------ completeness guard
#
# The sets below declare which fields are covered by the behavioral tests above
# (ASSERTED) vs intentionally not forwarded to engine params (UNMAPPED).
# The meta-tests use dataclasses.fields() to ensure every field in the config
# dataclass appears in exactly one of these sets. A new config field therefore
# fails CI until it is explicitly mapped or allowlisted.

KEYFRAME_ASSERTED = {
    "steps", "distill_steps", "guidance_scale", "width", "height", "seed",
    "distill", "scheduler", "identity_method",
    "instantid_ip_adapter_scale",
    "ip_adapter_scale",
    "multi_char",
}

KEYFRAME_UNMAPPED = {
    "base_model",      # deploy-time model repo; wired to ModelServer.specs at pod startup
    "distill_model",   # deploy-time model repo
}

I2V_ASSERTED = {
    "num_frames", "fps", "steps", "distill_steps", "guidance_scale", "distill",
    "feature_cache", "negative_prompt", "seed", "flow_shift",
}

I2V_UNMAPPED = {
    "model",           # deploy-time model repo
    "distill_model",   # deploy-time model repo
    "seconds_per_shot",  # used by I2VConfig.frames_for() to derive num_frames; not a direct param
    "loader",          # model-loading strategy (diffusers vs LightX2V); not a generation param
}

FINISH_ASSERTED = {
    "interpolate", "interpolation_factor", "target_fps",
    "face_restore", "face_fidelity", "only_faces",
}

FINISH_UNMAPPED: set = set()


def _top_level_field_names(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)}


def test_keyframe_config_fields_all_accounted_for():
    """Every KeyframeConfig field must be in KEYFRAME_ASSERTED or KEYFRAME_UNMAPPED."""
    fields = _top_level_field_names(KeyframeConfig)
    unaccounted = fields - KEYFRAME_ASSERTED - KEYFRAME_UNMAPPED
    assert not unaccounted, (
        f"KeyframeConfig fields not covered: {sorted(unaccounted)}. "
        "Add a behavioral test and add the field name to KEYFRAME_ASSERTED, "
        "or add to KEYFRAME_UNMAPPED with rationale.")


def test_i2v_config_fields_all_accounted_for():
    """Every I2VConfig field must be in I2V_ASSERTED or I2V_UNMAPPED."""
    fields = _top_level_field_names(I2VConfig)
    unaccounted = fields - I2V_ASSERTED - I2V_UNMAPPED
    assert not unaccounted, (
        f"I2VConfig fields not covered: {sorted(unaccounted)}. "
        "Add a behavioral test and add the field name to I2V_ASSERTED, "
        "or add to I2V_UNMAPPED with rationale.")


def test_finish_config_fields_all_accounted_for():
    """Every FinishConfig field must be in FINISH_ASSERTED or FINISH_UNMAPPED."""
    fields = _top_level_field_names(FinishConfig)
    unaccounted = fields - FINISH_ASSERTED - FINISH_UNMAPPED
    assert not unaccounted, (
        f"FinishConfig fields not covered: {sorted(unaccounted)}. "
        "Add a behavioral test and add the field name to FINISH_ASSERTED, "
        "or add to FINISH_UNMAPPED with rationale.")
