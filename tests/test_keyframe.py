"""The keyframe engine's decisions are pure and tested on CPU: the prompt it builds, the region
geometry that confines each identity, and the single-vs-regional path choice. The generation
body needs torch/diffusers/PIL and is validated on a pod."""
from vivijure_backend.contract import Cast, Scene, Storyboard
from vivijure_backend.keyframe import (
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_LORA_SCALE,
    DISTILL_GUIDANCE,
    KeyframeParams,
    _bind_loras,
    _box_mask,
    _ensure_ip_adapter,
    _guidance,
    _normalize_lora_state_dict,
    _pose_skeleton,
    build_prompt,
    engine_for,
    region_boxes,
    slot_trigger,
)
import pytest

CAST = Cast.from_registry({"characters": {
    "A": {"name": "Vesper", "prompt": "teal-haired netrunner"},
    "B": {"name": "Rhode", "prompt": "bearded fixer"},
}})


def _sb(**extra):
    return Storyboard.from_dict({"style_prefix": "cyberpunk anime,", "use_characters": ["A", "B"],
                                 "scenes": [{"id": "s", "prompt": "x"}], **extra})


# --------------------------------------------------------------------------- triggers/prompt

def test_slot_trigger_is_the_character_name():
    assert slot_trigger(CAST, "A") == "Vesper"
    assert slot_trigger(CAST, "Z") == "Z"  # unknown slot falls back to the slot id


def test_prompt_combines_style_scene_and_triggers():
    sc = Scene.from_dict({"id": "shot_01", "prompt": "rooftop standoff", "character_slots": ["A", "B"]}, 0)
    p = build_prompt(sc, CAST, _sb())
    assert p == "cyberpunk anime, rooftop standoff, Vesper, Rhode"


def test_prompt_has_no_dangling_comma_without_style():
    sc = Scene.from_dict({"id": "s", "prompt": "wide skyline", "character_slots": []}, 0)
    p = build_prompt(sc, CAST, _sb(style_prefix=""))
    assert p == "wide skyline"  # empty style + no characters -> no leading/trailing comma


def test_prompt_appends_real_style_preset():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A"]}, 0)
    p = build_prompt(sc, CAST, _sb(style_preset="neon noir"))
    assert p.endswith("neon noir")
    # the literal "None" preset is never appended
    assert "None" not in build_prompt(sc, CAST, _sb())


# --------------------------------------------------------------------------- region geometry

def test_single_region_is_full_canvas():
    assert region_boxes(1024, 1024, 1) == [(0, 0, 1024, 1024)]


def test_two_regions_split_vertically():
    boxes = region_boxes(1024, 1024, 2)
    assert boxes == [(0, 0, 512, 1024), (512, 0, 1024, 1024)]


def test_last_region_absorbs_odd_remainder():
    # 1025 / 2 = 512 step; the last region must reach the full width, not stop at 1024
    boxes = region_boxes(1025, 1024, 2)
    assert boxes[-1][2] == 1025


def test_horizontal_orientation_stacks():
    boxes = region_boxes(512, 1024, 2, orientation="horizontal")
    assert boxes == [(0, 0, 512, 512), (0, 512, 512, 1024)]


def test_region_gutter_carves_a_dead_band_between_regions():
    (l0, t0, r0, b0), (l1, t1, r1, b1) = region_boxes(1024, 1024, 2, gutter=64)
    assert l0 == 0 and r1 == 1024          # outer canvas edges stay flush
    assert r0 == 512 - 32 and l1 == 512 + 32  # interior edges inset by gutter//2
    assert l1 - r0 == 64                    # a 64px dead band the masks cannot blend across
    # gutter=0 reproduces the old adjacent split exactly
    assert region_boxes(1024, 1024, 2, gutter=0) == [(0, 0, 512, 1024), (512, 0, 1024, 1024)]


def test_two_regions_split_a_landscape_canvas_at_its_real_width():
    # 16:9: the vertical split must run on the true width (1920), not a square dimension; each
    # region keeps the full canvas height so a portrait subject is not clipped.
    boxes = region_boxes(1920, 1080, 2)
    assert boxes == [(0, 0, 960, 1080), (960, 0, 1920, 1080)]
    assert all(t == 0 and b == 1080 for (l, t, r, b) in boxes)  # full height per region


def test_two_regions_split_a_vertical_canvas():
    # 9:16 portrait: still a side-by-side split, on the narrow width and tall height.
    boxes = region_boxes(720, 1280, 2)
    assert boxes == [(0, 0, 360, 1280), (360, 0, 720, 1280)]


def test_box_mask_matches_a_non_square_canvas():
    # The IP-Adapter region mask must be the canvas size (W x H), not a square, so it lines up
    # with a 16:9 generation.
    mask = _box_mask(1920, 1080, (0, 0, 960, 1080))
    assert mask.size == (1920, 1080)
    assert mask.getpixel((10, 10)) == 255       # inside the left region box
    assert mask.getpixel((1900, 10)) == 0       # outside it (right half is black)


def test_pose_skeleton_uses_the_non_square_canvas_and_both_halves():
    boxes = region_boxes(1920, 1080, 2, gutter=64)
    img = _pose_skeleton(1920, 1080, boxes)
    assert img.size == (1920, 1080)             # rendered on the real 16:9 canvas
    assert img.crop((0, 0, 960, 1080)).getbbox() is not None
    assert img.crop((960, 0, 1920, 1080)).getbbox() is not None


def test_keyframe_params_default_is_square():
    # Regression: the default keyframe is still 1024x1024 so existing square renders are unchanged.
    p = KeyframeParams()
    assert (p.width, p.height) == (1024, 1024)


def test_pose_skeleton_plants_one_figure_per_region():
    boxes = region_boxes(512, 512, 2, gutter=64)
    img = _pose_skeleton(512, 512, boxes)
    assert img.size == (512, 512)
    # a distinct figure drawn in EACH half (getbbox() is None only for an all-black crop)
    assert img.crop((0, 0, 256, 512)).getbbox() is not None
    assert img.crop((256, 0, 512, 512)).getbbox() is not None
    # the n=1 single-figure case stays centered and non-empty
    assert _pose_skeleton(512, 512, region_boxes(512, 512, 1)).getbbox() is not None


# ----------------------------------------------------------------------- engine path choice

def test_single_path_for_zero_or_one_character():
    assert engine_for(Scene.from_dict({"id": "s", "prompt": "x", "character_slots": []}, 0), KeyframeParams()) == "single"
    assert engine_for(Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A"]}, 0), KeyframeParams()) == "single"


def test_regional_path_for_two_characters():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A", "B"]}, 0)
    assert engine_for(sc, KeyframeParams()) == "regional"


def test_over_cap_falls_back_to_single():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A", "B", "C"]}, 0)
    assert engine_for(sc, KeyframeParams(max_slots=2)) == "single"


# --------------------------------------------------------------------------- params/defaults

def test_anti_bleed_defaults():
    p = KeyframeParams()
    assert p.lora_scale == DEFAULT_LORA_SCALE == 0.3
    assert p.ip_adapter_scale == DEFAULT_IP_ADAPTER_SCALE == 0.7
    assert p.pose_conditioning is True
    assert p.max_slots == 2


# ------------------------------------------------------------------- LoRA binding (shared pipe)

class _FakePipe:
    """Minimal stand-in for the SDXL pipeline's peft surface. `load_lora_weights` rejects a
    duplicate adapter name the way diffusers/peft does, so a test that re-binds a slot across
    scenes fails exactly as the real worker did until _bind_loras cleared the prior scene."""
    def __init__(self, preloaded=()):  # preloaded = persistent base adapters (e.g. a distill LoRA)
        self.adapters = list(preloaded)
        self.active = None

    def load_lora_weights(self, path, adapter_name):
        if adapter_name in self.adapters:
            raise ValueError(f"Adapter name {adapter_name} already in use in the model - "
                             "please select a new adapter name.")
        self.adapters.append(adapter_name)

    def get_list_adapters(self):
        return {"unet": list(self.adapters)}

    def delete_adapters(self, names):
        self.adapters = [a for a in self.adapters if a not in names]

    def set_adapters(self, names, adapter_weights):
        self.active = list(zip(names, adapter_weights))


def test_bind_loras_reuses_slot_across_scenes_on_shared_pipe():
    # The keyframe pipe is process-global; scene 2 must be able to bind "A" again without the
    # "Adapter name A already in use" crash that failed the first deployed multi-scene render.
    pipe = _FakePipe()
    _bind_loras(pipe, {"A": "a.safetensors", "B": "b.safetensors"}, 0.3)
    _bind_loras(pipe, {"A": "a.safetensors"}, 0.3)  # next scene, slot A again -> no raise
    assert pipe.adapters == ["A"]                   # B from the prior scene was cleared
    assert pipe.active == [("A", 0.3)]


class _NoRegisterPipe(_FakePipe):
    """`load_lora_weights` accepts the file but registers nothing -- exactly what diffusers does
    when the safetensors keys do not match its convention (the cast trainer's raw PEFT, no-`unet.`-
    prefix format that crashed the first real LoRA-reuse render)."""
    def load_lora_weights(self, path, adapter_name):
        pass  # silently loads zero modules; no adapter registered


def test_bind_loras_fails_loud_when_a_lora_registers_no_adapter():
    with pytest.raises(ValueError, match="registered no adapter"):
        _bind_loras(_NoRegisterPipe(), {"A": "a.safetensors"}, 0.3)
    # the base-only / no-character case must NOT raise (nothing to load, nothing to check)
    assert _bind_loras(_NoRegisterPipe(), {}, 0.3) == []


# -------------------------------------------------------- _normalize_lora_state_dict (pure CPU)

def test_normalize_lora_state_dict_passthrough_for_diffusers_format():
    # Already-diffusers keys (unet. prefix, lora.down/up naming) must come back unchanged.
    sd = {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight": 1,
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight": 2,
    }
    assert _normalize_lora_state_dict(sd) is sd


def test_normalize_lora_state_dict_converts_raw_peft_with_base_model_prefix():
    # Raw PEFT from get_peft_model_state_dict(unet): base_model.model. prefix, lora_A/lora_B keys.
    raw = {
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": 1,
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight": 2,
    }
    out = _normalize_lora_state_dict(raw)
    assert set(out.keys()) == {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight",
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight",
    }
    assert list(out.values()) == [1, 2]


def test_normalize_lora_state_dict_converts_raw_peft_without_base_model_prefix():
    # Variant: no base_model.model. prefix but still lora_A/lora_B (legacy save path).
    raw = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": 10,
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight": 20,
    }
    out = _normalize_lora_state_dict(raw)
    assert "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight" in out
    assert "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.up.weight" in out
    assert out["unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora.down.weight"] == 10


def test_normalize_lora_state_dict_preserves_non_lora_keys():
    # Any keys that don't contain .lora_A. / .lora_B. are still remapped to unet. scope but otherwise
    # left untouched (e.g. alpha scalars stored alongside the weights in some PEFT saves).
    raw = {
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight": 1,
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.weight": 2,
        "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_alpha": 16,
    }
    out = _normalize_lora_state_dict(raw)
    assert "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_alpha" in out
    assert out["unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_alpha"] == 16


def test_bind_loras_keeps_base_distill_adapter_across_scenes():
    pipe = _FakePipe(preloaded=["distill"])         # a persistent base adapter
    _bind_loras(pipe, {"A": "a.safetensors"}, 0.3)
    _bind_loras(pipe, {"B": "b.safetensors"}, 0.3)  # different slot, new scene
    assert pipe.adapters == ["distill", "B"]         # base stays, A (prior scene) is gone
    assert pipe.active == [("distill", 1.0), ("B", 0.3)]


def test_bind_loras_zeroes_distill_on_the_full_step_path():
    # Final tier (few_step=False) must render WITHOUT the distill LoRA: it stays loaded on the warm
    # pipe but at weight 0.0 (inert), so the same pipe serves draft and final without a reload.
    pipe = _FakePipe(preloaded=["distill"])
    _bind_loras(pipe, {"A": "a.safetensors"}, 0.3, few_step=False)
    assert pipe.active == [("distill", 0.0), ("A", 0.3)]


def test_bind_loras_tracks_distill_weight_for_a_no_character_scene():
    # The distill weight must follow the tier even when a scene has no character to bind, so a
    # no-character draft still rides the few-step LoRA and a no-character final does not.
    pipe = _FakePipe(preloaded=["distill"])
    _bind_loras(pipe, {}, 0.3, few_step=True)
    assert pipe.active == [("distill", 1.0)]
    _bind_loras(pipe, {}, 0.3, few_step=False)
    assert pipe.active == [("distill", 0.0)]


def test_bind_loras_clears_identity_for_a_no_character_scene():
    pipe = _FakePipe()
    _bind_loras(pipe, {"A": "a.safetensors"}, 0.3)
    _bind_loras(pipe, {}, 0.3)                        # a scene with no character
    assert pipe.adapters == []                        # prior identity does not leak in


# ------------------------------------------------------------- IP-Adapter count (shared pipe)

class _FakeIPPipe:
    """Tracks the IP-Adapter encoder count the way diffusers' set_ip_adapter_scale validates it:
    a scalar or a list whose length matches the loaded count is fine; a mismatch raises exactly as
    the worker did ("Cannot assign N scale_configs to M IP-Adapter")."""
    def __init__(self):
        self.ip_count = 0
        self.ip_scale = None

    def load_ip_adapter(self, repo, subfolder, weight_name):
        self.ip_count = len(weight_name) if isinstance(weight_name, list) else 1

    def unload_ip_adapter(self):
        self.ip_count = 0

    def set_ip_adapter_scale(self, scale):
        k = len(scale) if isinstance(scale, list) else 1
        if self.ip_count == 0 or k != self.ip_count:
            raise ValueError(f"Cannot assign {k} scale_configs to {self.ip_count} IP-Adapter.")
        self.ip_scale = scale


def test_ip_adapter_shrinks_from_regional_to_single():
    # The exact bug that failed the deployed render: a 2-char scene left 2 encoders, then a single
    # scene set a scalar scale -> "Cannot assign 1 scale_configs to 2 IP-Adapter".
    pipe = _FakeIPPipe()
    _ensure_ip_adapter(pipe, 2)        # regional scene
    assert pipe.ip_count == 2
    _ensure_ip_adapter(pipe, 1)        # next scene single -> must reduce to exactly 1
    assert pipe.ip_count == 1
    pipe.set_ip_adapter_scale(0.7)     # scalar now matches the single encoder (no raise)


def test_ip_adapter_grows_from_single_to_regional():
    pipe = _FakeIPPipe()
    _ensure_ip_adapter(pipe, 1)
    _ensure_ip_adapter(pipe, 2)        # single -> regional, must grow to 2
    assert pipe.ip_count == 2
    pipe.set_ip_adapter_scale([0.7, 0.7])   # list of 2 matches


def test_ip_adapter_clears_for_no_character_scene():
    pipe = _FakeIPPipe()
    _ensure_ip_adapter(pipe, 2)
    _ensure_ip_adapter(pipe, 0)        # a no-character scene drops all encoders
    assert pipe.ip_count == 0


def test_ip_adapter_noop_when_count_unchanged():
    pipe = _FakeIPPipe()
    _ensure_ip_adapter(pipe, 1)
    first = pipe.ip_count
    _ensure_ip_adapter(pipe, 1)        # same count -> no reload churn
    assert pipe.ip_count == first == 1


def test_few_step_distill_clamps_guidance():
    """Regression: the few-step distill was wired on (few_step=True) but guidance_scale stayed at the
    full-step default 5.0, so the Hyper-SD/DMD2 keyframe sampled at CFG 5.0 -- over-saturated, torn,
    chroma-split frames. A distilled keyframe must sample (near-)guidance-free; enforce it at the
    source so a caller that forgets to lower CFG can't break the render."""
    assert _guidance(KeyframeParams(few_step=True, guidance_scale=5.0)) == DISTILL_GUIDANCE
    assert DISTILL_GUIDANCE == 1.0  # matches the i2v distill (guidance-free)


def test_full_step_keeps_real_cfg():
    """Final tier (few_step=False, full steps) keeps its configured guidance -- the clamp is only for
    the distilled path."""
    assert _guidance(KeyframeParams(few_step=False, guidance_scale=5.0)) == 5.0
    assert _guidance(KeyframeParams(few_step=False, guidance_scale=7.5)) == 7.5


# ----------------------------------------------- which identity path a single render took (#350)

def test_single_identity_path_distinguishes_paths_that_all_produce_a_keyframe():
    from vivijure_backend.keyframe import IDENTITY_PATHS, single_identity_path

    assert single_identity_path("a", has_ref_image=True, has_lora=True) == "ip_adapter"
    assert single_identity_path("a", has_ref_image=True, has_lora=False) == "ip_adapter"
    # A slot with a LoRA and no usable reference renders fine and is NOT the IP-Adapter path.
    assert single_identity_path("a", has_ref_image=False, has_lora=True) == "lora_only"
    assert single_identity_path("a", has_ref_image=False, has_lora=False) == "none"
    # No slot at all is never an identity path, whatever else is lying around.
    assert single_identity_path(None, has_ref_image=True, has_lora=True) == "none"
    for slot in ("a", None):
        for ref in (True, False):
            for lora in (True, False):
                assert single_identity_path(slot, has_ref_image=ref, has_lora=lora) in IDENTITY_PATHS


def test_keyframe_result_defaults_do_not_claim_an_identity_path():
    # An unset field must not read as a path that ran.
    from pathlib import Path as _Path
    from vivijure_backend.keyframe import IDENTITY_PATHS, KeyframeResult
    res = KeyframeResult(shot_id="s", path=_Path("x"), slots=[], multi_char=False,
                         prompt="p", seed=0, engine="single")
    assert res.identity_path == ""
    assert res.identity_path not in IDENTITY_PATHS
