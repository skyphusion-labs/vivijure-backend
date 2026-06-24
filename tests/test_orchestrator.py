"""The planner's whole job is to decide the render on the CPU and eliminate GPU work before
it fires. These tests pin every elimination path and the cost arithmetic."""
from pathlib import Path

from vivijure_backend.contract import Cast, Character, RenderRequest, Storyboard
from vivijure_backend.device import Tier
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.orchestrator import (
    MULTI_CHAR_DEFAULTS,
    Action,
    KeyframeMode,
    plan,
    validate,
)


def _sb(scenes, use_characters=("A", "B"), **extra):
    return Storyboard.from_dict({"use_characters": list(use_characters), "scenes": scenes, **extra})


def _req(action="render", quality="final", **extra):
    return RenderRequest.from_dict({"action": action, "project": "p", "bundle_key": "b",
                                    "quality_tier": quality, **extra})


TWO_SCENES = [
    {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
    {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
]


# ------------------------------------------------------------------------- full render

def test_render_trains_all_and_generates_all():
    p = plan(_req(), _sb(TWO_SCENES))
    assert sorted(p.lora.train) == ["A", "B"]
    assert p.lora.reuse == []
    assert p.keyframes_to_generate == 2
    assert p.shots_to_animate == 2
    assert p.assemble_off_gpu is True
    assert all(s.keyframe_mode is KeyframeMode.GENERATE for s in p.scenes)


def test_multi_char_scene_gets_anti_bleed_defaults():
    p = plan(_req(), _sb(TWO_SCENES))
    single, multi = p.scenes
    assert single.multi_char == {}
    assert multi.is_multi_character
    assert multi.multi_char == MULTI_CHAR_DEFAULTS
    assert multi.multi_char["pose_conditioning"] is True


def test_i2v_tier_follows_quality():
    assert plan(_req(quality="draft"), _sb(TWO_SCENES)).scenes[0].i2v_tier is Tier.RTX_PRO_6000
    assert plan(_req(quality="standard"), _sb(TWO_SCENES)).scenes[0].i2v_tier is Tier.H200
    assert plan(_req(quality="final"), _sb(TWO_SCENES)).scenes[0].i2v_tier is Tier.B200


# ----------------------------------------------------------------------- eliminations

def test_already_trained_lora_is_reused_not_retrained():
    p = plan(_req(), _sb(TWO_SCENES), trained_slots={"A"})
    assert p.lora.train == ["B"]
    assert p.lora.reuse == ["A"]
    assert any("already trained" in s for s in p.skips)


def test_pretrained_lora_passthrough():
    p = plan(_req(pretrained_loras={"A": "loras/A.safetensors"}), _sb(TWO_SCENES))
    assert p.lora.train == ["B"]
    assert "A" in p.lora.reuse
    assert any("pretrained" in s for s in p.skips)


def test_injected_start_image_skips_keyframe_gen():
    scenes = [dict(TWO_SCENES[0]), {**TWO_SCENES[1], "start_image": "clips/shot_02_keyframe.png"}]
    p = plan(_req(), _sb(scenes))
    modes = {s.shot_id: s.keyframe_mode for s in p.scenes}
    assert modes["shot_01"] is KeyframeMode.GENERATE
    assert modes["shot_02"] is KeyframeMode.INJECT
    assert p.keyframes_to_generate == 1
    assert any("inject" in s for s in p.skips)


def test_existing_keyframe_is_reused():
    p = plan(_req(), _sb(TWO_SCENES), existing_keyframes={"shot_01": None})
    modes = {s.shot_id: s.keyframe_mode for s in p.scenes}
    assert modes["shot_01"] is KeyframeMode.REUSE
    assert modes["shot_02"] is KeyframeMode.GENERATE
    assert p.keyframes_to_generate == 1


# --------------------------------------------------------------------------- actions

def test_hash_mismatch_forces_keyframe_regenerate():
    """A shot in existing_keyframes with a mismatched hash must be regenerated, not reused."""
    from vivijure_backend.contract import RenderRequest
    req = _req()
    cur = kf_hash(req.config.keyframe)
    stale_hash = "0000000000000000"  # a hash that cannot match the real config hash
    assert stale_hash != cur, "test expects stale_hash to differ from cur"
    p = plan(req, _sb(TWO_SCENES), existing_keyframes={"shot_01": stale_hash})
    modes = {s.shot_id: s.keyframe_mode for s in p.scenes}
    assert modes["shot_01"] is KeyframeMode.GENERATE  # mismatch -> regenerate
    assert modes["shot_02"] is KeyframeMode.GENERATE


def test_hash_match_allows_keyframe_reuse():
    """A shot in existing_keyframes with a matching hash is reused."""
    req = _req()
    cur = kf_hash(req.config.keyframe)
    p = plan(req, _sb(TWO_SCENES), existing_keyframes={"shot_01": cur})
    modes = {s.shot_id: s.keyframe_mode for s in p.scenes}
    assert modes["shot_01"] is KeyframeMode.REUSE
    assert modes["shot_02"] is KeyframeMode.GENERATE


def test_none_hash_allows_reuse_for_backward_compat():
    """A shot in existing_keyframes with None (old state, no hash stored) is reused conservatively."""
    p = plan(_req(), _sb(TWO_SCENES), existing_keyframes={"shot_01": None})
    assert p.scenes[0].keyframe_mode is KeyframeMode.REUSE


def test_finalize_is_i2v_only_over_reused_keyframes():
    # finalize must never train or generate keyframes, even with nothing trained yet.
    p = plan(_req(action="finalize"), _sb(TWO_SCENES))
    assert p.action is Action.FINALIZE
    assert p.lora.train == []
    assert p.keyframes_to_generate == 0
    assert all(s.keyframe_mode is KeyframeMode.REUSE for s in p.scenes)
    assert p.shots_to_animate == 2


def test_preview_is_keyframes_only_with_training():
    # The keyframes-only preview: train the LoRAs + draw every keyframe, but NO i2v and no MP4
    # (so the user can eyeball shots before committing GPU-seconds to Wan motion).
    p = plan(_req(action="preview"), _sb(TWO_SCENES))
    assert p.action is Action.PREVIEW
    assert sorted(p.lora.train) == ["A", "B"]          # untrained slots still train for a true preview
    assert p.keyframes_to_generate == 2                # all keyframes drawn
    assert p.shots_to_animate == 0                     # but nothing animated
    assert all(not s.needs_i2v for s in p.scenes)
    assert all(s.i2v_tier is None for s in p.scenes)   # no GPU tier reserved for motion


def test_preview_reuses_ready_lora_instead_of_retraining():
    # With the cast LoRA already trained/pretrained, a preview must reuse it (the bug that made a
    # cold preview retrain Marcus/Aria was a control-plane reuse miss, not a planner one).
    p = plan(_req(action="preview", pretrained_loras={"A": "loras/cast-3/x.safetensors"}),
             _sb(TWO_SCENES))
    assert "A" in p.lora.reuse
    assert p.lora.train == ["B"]
    assert p.shots_to_animate == 0


def test_keyframes_only_render_short_circuits_before_i2v():
    # Issue #119: a `render` carrying keyframes_only:true must draw its keyframes (training the
    # LoRAs they need) but STOP before any i2v/finish -- the cost door. The render never reaches
    # the GPU motion stage, so no shot animates and no i2v GPU tier is reserved.
    p = plan(_req(keyframes_only=True), _sb(TWO_SCENES))
    assert p.action is Action.RENDER                   # intent preserved (still a render, not a preview)
    assert sorted(p.lora.train) == ["A", "B"]          # keyframes still need their LoRAs
    assert p.keyframes_to_generate == 2                # all keyframes drawn
    assert p.shots_to_animate == 0                     # but NOTHING animated -- short-circuit
    assert all(not s.needs_i2v for s in p.scenes)
    assert all(s.i2v_tier is None for s in p.scenes)   # no GPU tier reserved for motion


def test_keyframes_only_costs_no_more_than_a_preview():
    # The whole point is the cost drop: a keyframes_only render must estimate the SAME GPU-seconds
    # as the equivalent preview (keyframes + training only), never the full render's i2v seconds.
    only = plan(_req(keyframes_only=True), _sb(TWO_SCENES))
    preview = plan(_req(action="preview"), _sb(TWO_SCENES))
    full = plan(_req(), _sb(TWO_SCENES))
    assert only.estimated_gpu_seconds == preview.estimated_gpu_seconds
    assert only.estimated_gpu_seconds < full.estimated_gpu_seconds


def test_keyframes_only_parsed_off_the_request():
    # The flag is part of the typed contract now, not silently dropped (the #119 root cause).
    assert _req(keyframes_only=True).keyframes_only is True
    assert _req().keyframes_only is False
    assert RenderRequest.from_dict({"action": "render", "keyframes_only": "true"}).keyframes_only is True


def test_regen_shot_is_keyframe_only_no_i2v():
    p = plan(_req(action="regen_shot"), _sb(TWO_SCENES))
    assert p.shots_to_animate == 0
    assert p.keyframes_to_generate == 2


def test_train_lora_costs_only_training_no_per_scene_work():
    # A LoRA-only job has no per-scene render; it must not estimate keyframe or i2v GPU time.
    p = plan(_req(action="train_lora"), _sb(TWO_SCENES))
    assert p.action is Action.TRAIN_LORA
    assert sorted(p.lora.train) == ["A", "B"]
    assert p.keyframes_to_generate == 0
    assert p.shots_to_animate == 0
    assert p.estimated_gpu_seconds == 2 * 180.0  # two LoRAs, nothing else


def test_process_shot_ids_scopes_the_render():
    p = plan(_req(action="finalize", process_shot_ids=["shot_02"]), _sb(TWO_SCENES))
    assert [s.shot_id for s in p.scenes] == ["shot_02"]
    assert any("out of process_shot_ids scope" in s for s in p.skips)


def test_unknown_action_falls_back_to_render():
    assert plan(_req(action="banana"), _sb(TWO_SCENES)).action is Action.RENDER


# -------------------------------------------------------------------------- validate

def test_validate_flags_slots_not_in_cast():
    sb = _sb([{"id": "s1", "prompt": "x", "character_slots": ["A", "B"]}], use_characters=["A"])
    errs = validate(_req(), sb)
    assert any("not in use_characters" in e for e in errs)


def test_validate_flags_unknown_process_shot_ids():
    errs = validate(_req(process_shot_ids=["shot_99"]), _sb(TWO_SCENES))
    assert any("process_shot_ids not in storyboard" in e for e in errs)


def test_validate_flags_empty_scene_prompt():
    sb = _sb([{"id": "s1", "prompt": "", "character_slots": ["A"]},
              {"id": "s2", "prompt": "fine", "character_slots": ["B"]}])
    errs = validate(_req(), sb)
    assert any("empty prompt" in e and "s1" in e for e in errs)
    assert not any("s2" in e for e in errs)


def test_validate_flags_whitespace_only_prompt():
    sb = _sb([{"id": "s1", "prompt": "   ", "character_slots": ["A"]}])
    errs = validate(_req(), sb)
    assert any("empty prompt" in e for e in errs)


def _cast(slots: list[str], with_refs: bool = True) -> Cast:
    chars = {}
    for slot in slots:
        chars[slot] = Character(slot=slot, name=slot, prompt="",
                                ref_paths=[Path(f"/fake/{slot}.png")] if with_refs else [])
    return Cast(characters=chars)


def test_validate_flags_missing_cast_slot():
    sb = _sb(TWO_SCENES, use_characters=["A", "B"])
    cast = _cast(["A"])  # B is missing
    errs = validate(_req(), sb, cast=cast)
    assert any("no entry in the cast registry" in e and "'B'" in e for e in errs)
    assert not any("'A'" in e for e in errs)


def test_validate_clean_storyboard_has_no_errors():
    assert validate(_req(), _sb(TWO_SCENES)) == []


def test_validate_clean_storyboard_with_cast_has_no_errors():
    sb = _sb(TWO_SCENES, use_characters=["A", "B"])
    assert validate(_req(), sb, cast=_cast(["A", "B"])) == []


# ------------------------------------------------------------------------------ cost

def test_cost_grows_with_quality_and_is_enumerated():
    draft = plan(_req(quality="draft"), _sb(TWO_SCENES))
    final = plan(_req(quality="final"), _sb(TWO_SCENES))
    assert final.estimated_gpu_seconds > draft.estimated_gpu_seconds
    # every reuse/skip is spelled out for the operator
    p = plan(_req(), _sb(TWO_SCENES), trained_slots={"A"}, existing_keyframes={"shot_01": None})
    assert p.skips  # non-empty: at least the reused LoRA and reused keyframe
    assert "saved:" in p.summary()
