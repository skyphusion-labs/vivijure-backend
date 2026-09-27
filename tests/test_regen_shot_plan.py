"""regen_shot redraws keyframes and trains no LoRAs (docs/contract.md, actions table)."""
from vivijure_backend.contract import RenderRequest, Storyboard
from vivijure_backend.orchestrator import Action, plan

SCENES = [
    {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
    {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
]


def _plan(action, **req_extra):
    req = RenderRequest.from_dict({"action": action, "project": "p", "bundle_key": "b",
                                   "quality_tier": "final", **req_extra})
    sb = Storyboard.from_dict({"use_characters": ["A", "B"], "scenes": SCENES})
    return plan(req, sb)


def test_regen_shot_trains_no_loras():
    p = _plan("regen_shot")
    assert p.action is Action.REGEN_SHOT
    assert p.lora.train == []
    assert p.keyframes_to_generate == 2          # still redraws keyframes
    assert p.shots_to_animate == 0               # still no motion


def test_regen_shot_estimate_is_keyframes_only():
    assert _plan("regen_shot").estimated_gpu_seconds == 2 * 6.0   # two keyframes, no LoRA training


def test_regen_shot_says_it_skipped_training():
    assert any("regen_shot" in s and "not trained" in s for s in _plan("regen_shot").skips)


def test_regen_shot_still_reuses_trained_and_pretrained_slots():
    p = _plan("regen_shot", pretrained_loras={"A": "loras/x/A/pytorch_lora_weights.safetensors"})
    assert p.lora.train == [] and p.lora.reuse == ["A"]


def test_preview_and_render_still_train_missing_slots():
    for action in ("preview", "render"):
        assert sorted(_plan(action).lora.train) == ["A", "B"]
