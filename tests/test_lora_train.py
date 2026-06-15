"""CPU-testable surface of the LoRA trainer: the parts that decide *what* to train before
any GPU is touched. The training loop itself needs CUDA and is validated on the pod."""
from pathlib import Path

import pytest

from vivijure_backend.contract import Character
from vivijure_backend.lora_train import (
    LoraTrainConfig,
    TrainedLora,
    _loss_target,
    _time_ids,
    caption_for,
    default_base_repo,
    train_slot,
)


def _char(slot="A", name="Vesper", prompt="teal-haired netrunner", refs=()):
    return Character(slot=slot, name=name, prompt=prompt, ref_paths=[Path(p) for p in refs])


def test_default_base_repo_is_the_keyframe_sdxl():
    # The LoRA must train against the same checkpoint the keyframe stage draws with.
    assert default_base_repo() == "SG161222/RealVisXL_V5.0"


def test_caption_uses_name_as_trigger_and_appends_prompt():
    assert caption_for(_char(), LoraTrainConfig().caption_template) == "Vesper, teal-haired netrunner"


def test_caption_falls_back_to_slot_when_unnamed():
    assert caption_for(_char(name="", prompt=""), LoraTrainConfig().caption_template) == "A"


def test_caption_drops_dangling_comma_when_prompt_is_empty():
    # An empty prompt must not leak a trailing ", " into the caption.
    assert caption_for(_char(prompt=""), LoraTrainConfig().caption_template) == "Vesper"


def test_train_slot_rejects_a_character_with_no_refs():
    # Refuse on the CPU before allocating a GPU for a slot that has nothing to learn from.
    with pytest.raises(ValueError, match="no reference images"):
        train_slot(_char(refs=()), Path("/tmp/never"))


def test_config_defaults_fit_a_few_reference_character():
    cfg = LoraTrainConfig()
    assert cfg.rank == 16
    assert cfg.resolution == 1024
    assert cfg.gradient_checkpointing is True
    assert cfg.max_steps == 1000


def test_trained_lora_carries_the_trigger_token():
    tl = TrainedLora(slot="A", path=Path("a.safetensors"), trigger="Vesper",
                     steps=1000, rank=16, ref_count=8, base_repo="x")
    assert tl.trigger == "Vesper"


def test_caption_default_template_works():
    assert caption_for(_char(), "{name}, {prompt}") == "Vesper, teal-haired netrunner"


def test_caption_rejects_unknown_placeholder():
    with pytest.raises(ValueError, match="unsupported placeholders"):
        caption_for(_char(), "{name}, {evil}")


def test_caption_rejects_nested_format_spec():
    # {name:{prompt.__class__.__mro__}} passes a str.Formatter inspection of field_name
    # but evaluates the format_spec as an attribute access; str.replace never reaches it.
    with pytest.raises(ValueError, match="unsupported placeholders"):
        caption_for(_char(), "{name:{prompt.__class__}}")


def test_caption_rejects_attribute_access():
    with pytest.raises(ValueError, match="unsupported placeholders"):
        caption_for(_char(), "{name.__class__}, {prompt}")


def test_caption_rejects_stray_braces():
    with pytest.raises(ValueError, match="unsupported placeholders"):
        caption_for(_char(), "{name}, {0[x]}")


def test_caption_allows_plain_text_no_braces():
    # A template with no placeholders at all is fine (literal caption)
    result = caption_for(_char(), "a test caption")
    assert result == "a test caption"


# ------------------------------------------------------------- new fields (issue #27)

def test_lora_alpha_defaults_to_none():
    assert LoraTrainConfig().lora_alpha is None


def test_trained_lora_checkpoint_dirs_defaults_to_empty():
    tl = TrainedLora(slot="A", path=Path("a.safetensors"), trigger="V",
                     steps=100, rank=16, ref_count=3, base_repo="x")
    assert tl.checkpoint_dirs == []


# ----------------------------------------- effective_steps scaling

def _effective_steps(max_steps: int, n: int) -> int:
    return max(50, max_steps * n // 5)


def test_effective_steps_at_five_refs_equals_max_steps():
    assert _effective_steps(1000, 5) == 1000


def test_effective_steps_scales_down_for_fewer_refs():
    assert _effective_steps(1000, 2) == 400
    assert _effective_steps(1000, 1) == 200


def test_effective_steps_floors_at_fifty():
    # A very small max_steps * few refs might compute below 50.
    assert _effective_steps(100, 1) == 50


def test_effective_steps_scales_up_for_more_refs():
    assert _effective_steps(1000, 10) == 2000


# ----------------------------------------- _time_ids shape

def test_time_ids_returns_a_1x6_tensor():
    torch = pytest.importorskip("torch")
    t = _time_ids((1024, 768), (0, 128), (1024, 1024), "cpu", torch.float32)
    assert t.shape == (1, 6)
    vals = t[0].tolist()
    # original_h, original_w, crop_top, crop_left, target_h, target_w
    assert vals == [1024.0, 768.0, 0.0, 128.0, 1024.0, 1024.0]


# ----------------------------------------- _loss_target prediction types

class _FakeScheduler:
    def __init__(self, prediction_type):
        self.config = type("C", (), {"prediction_type": prediction_type})()

    def get_velocity(self, latent, noise, timestep):
        return latent - noise  # deterministic fake velocity


def test_loss_target_returns_noise_for_epsilon_prediction():
    torch = pytest.importorskip("torch")
    latent = torch.ones(1, 4, 8, 8)
    noise = torch.zeros(1, 4, 8, 8)
    timestep = torch.tensor([0])
    sched = _FakeScheduler("epsilon")
    target = _loss_target(sched, latent, noise, timestep)
    assert (target == noise).all()


def test_loss_target_returns_velocity_for_v_prediction():
    torch = pytest.importorskip("torch")
    latent = torch.ones(1, 4, 8, 8)
    noise = torch.zeros(1, 4, 8, 8)
    timestep = torch.tensor([0])
    sched = _FakeScheduler("v_prediction")
    target = _loss_target(sched, latent, noise, timestep)
    # fake velocity = latent - noise = 1 - 0 = 1
    assert (target == latent - noise).all()
