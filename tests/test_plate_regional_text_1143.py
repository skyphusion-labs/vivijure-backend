"""vivijure-cf#1143 option (a): per-region text in the two-person plate.

What is tested here, on CPU: the knob is off by default and in the hash; the prompt split gives each person THEIR
traits and never the other's, and keeps the scene, action, setting and adult clause in the shared text every
region reads; and the layer geometry that maps a region box onto each UNet level. The processor itself is checked
against diffusers' own Attention in scripts/check_regional_text.py (needs torch)."""
import ast
import itertools
from pathlib import Path

import pytest

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig
from vivijure_backend.contract import Cast, RenderRequest, Scene, Storyboard
from vivijure_backend.keyframe import (
    PROMPT_TOKEN_BUDGET,
    PROMPT_TOKEN_MARGIN,
    KeyframeParams,
    adult_clause,
    estimate_tokens,
    plate_region_prompts,
    plate_regional_text_on,
    region_boxes,
)
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.regional_text import box_to_grid, layer_hw

MARA = "Mara, an adult woman in her thirties, dark shoulder-length curls, sharp brown eyes, charcoal field jacket over a black top, calm presence."
DEV = "Dev, an adult man in his thirties, dark buzz cut, light stubble, grey zip hoodie, lean and alert."
REAL = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": MARA}, "B": {"name": "Dev", "prompt": DEV}}})
SHOT = "Mara and Dev stand side by side at a workbench in the dim workshop, looking at a laptop."


def _sb(**extra):
    return Storyboard.from_dict({"style_prefix": "photoreal, 35mm,", "scenes": [{"id": "s", "prompt": "x"}], **extra})


def _scene(prompt=SHOT, slots=("A", "B")):
    return Scene.from_dict({"id": "s", "prompt": prompt, "character_slots": list(slots)}, 0)


# --------------------------------------------------------------------------- the knob

def test_off_by_default_everywhere():
    assert KeyframeConfig().plate_regional_text is False
    assert KeyframeParams().plate_regional_text is False
    assert plate_regional_text_on(_scene(), KeyframeParams()) is False


@pytest.mark.parametrize("slots,want", [(("A", "B"), True), (("A",), False), ((), False), (("A", "B", "C"), False)])
def test_two_person_plates_only(slots, want):
    assert plate_regional_text_on(_scene(slots=slots), KeyframeParams(plate_regional_text=True)) is want


def test_it_is_in_the_reuse_hash():
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "x"}]})
    a = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz"})
    b = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                 "render_overrides": {"keyframe": {"plate_regional_text": True}}})
    assert b.config.keyframe.plate_regional_text is True
    assert kf_hash(a.config.keyframe, scene=sb.scenes[0], storyboard=sb) != kf_hash(b.config.keyframe, scene=sb.scenes[0], storyboard=sb)


# --------------------------------------------------------------------------- the right traits to the right region

def test_each_person_gets_their_own_traits_and_never_the_others():
    shared, (mara, dev) = plate_region_prompts(_scene(), REAL, _sb())
    for trait in ("dark shoulder-length curls", "charcoal jacket"):
        assert trait in mara and trait not in dev and trait not in shared
    for trait in ("dark buzz cut", "grey zip hoodie"):
        assert trait in dev and trait not in mara and trait not in shared


def test_the_shared_text_carries_the_scene_and_its_action_and_opens_every_region():
    shared, persons = plate_region_prompts(_scene(), REAL, _sb())
    assert "looking at a laptop" in shared
    for p in persons:
        assert p.startswith(shared)


def test_the_setting_is_shared():
    shared, persons = plate_region_prompts(_scene("Mara and Dev talk."), REAL, _sb(setting="a dim workshop at night"))
    assert "a dim workshop at night" in shared
    assert all("a dim workshop at night" in p for p in persons)


def test_off_the_plate_prompt_is_unchanged():
    """With the knob off, Pass A is fed plate_spec.prompt exactly as before (read from the syntax tree)."""
    fn = next(n for n in ast.walk(ast.parse(Path(kf.__file__).read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_scene_lock")
    src = ast.unparse(fn)
    assert "plate_prompt = plate_spec.prompt" in src
    assert "if plate_regional_text_on(scene, cfg):" in src
    # The plate call runs inside regional_text.applied, which always restores (tested below).
    withs = [w for w in ast.walk(fn) if isinstance(w, ast.With)]
    assert any("regional_text.applied" in ast.unparse(w.items[0].context_expr) and
               any(isinstance(c, ast.Call) and any(k.arg == "num_inference_steps" for k in c.keywords)
                   for c in ast.walk(w)) for w in withs)


class _Unet:
    def __init__(self):
        self.procs = {"down.attn1.processor": "self", "down.attn2.processor": "cross"}

    @property
    def attn_processors(self):
        return dict(self.procs)

    def set_attn_processor(self, d):
        self.procs = dict(d)


class _Pipe:
    vae_scale_factor = 8

    def __init__(self):
        self.unet = _Unet()


def test_applied_wraps_only_cross_attention_and_always_restores():
    from vivijure_backend import regional_text as rt
    pipe = _Pipe()
    with rt.applied(pipe, [("emb", (0, 0, 8, 8))], width=64, height=64, cfg_batch=True):
        procs = pipe.unet.attn_processors
        assert procs["down.attn1.processor"] == "self"
        assert isinstance(procs["down.attn2.processor"], rt.RegionalTextProcessor)
        assert procs["down.attn2.processor"].inner == "cross"
    assert pipe.unet.attn_processors == {"down.attn1.processor": "self", "down.attn2.processor": "cross"}
    with pytest.raises(RuntimeError):
        with rt.applied(pipe, [("emb", (0, 0, 8, 8))], width=64, height=64, cfg_batch=True):
            raise RuntimeError("the plate call failed")
    assert pipe.unet.attn_processors == {"down.attn1.processor": "self", "down.attn2.processor": "cross"}


def test_applied_with_no_regions_touches_nothing():
    from vivijure_backend import regional_text as rt
    pipe = _Pipe()
    before = pipe.unet.procs
    with rt.applied(pipe, [], width=64, height=64, cfg_batch=True):
        assert pipe.unet.procs is before


# --------------------------------------------------------------------------- the safety clause stays global

NO_AGE = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": "dark curls, charcoal jacket"},
                                            "B": {"name": "Dev", "prompt": "buzz cut, grey hoodie"}}})
STYLES = ["", "photoreal, 35mm,", "x " * 120]
SCENES = ["walks in", SHOT, "word " * 60, "word " * 200]


@pytest.mark.parametrize("style,scene,cast", list(itertools.product(STYLES, SCENES, [REAL, NO_AGE])))
def test_the_adult_clause_is_in_every_region_and_inside_the_budget(style, scene, cast):
    """Per-region text must never mask the adult statement away: it is in the shared text the background reads AND
    in every person's own prompt, and it ends inside the encoder's budget in each."""
    sb = Storyboard.from_dict({"style_prefix": style, "scenes": [{"id": "s", "prompt": "x"}]})
    shared, persons = plate_region_prompts(_scene(scene), cast, sb)
    clause = adult_clause(["A", "B"], named=False)
    assert clause == "two adults"
    for p in [shared, *persons]:
        assert clause in p
        assert estimate_tokens(p[: p.index(clause) + len(clause)]) <= PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN


# --------------------------------------------------------------------------- geometry

def test_layer_hw_walks_the_sdxl_levels():
    # 1344 x 768 -> a 168 x 96 latent; SDXL attends at 1/1, 1/2 and 1/4 of it.
    assert layer_hw(96 * 168, 96, 168) == (96, 168)
    assert layer_hw(48 * 84, 96, 168) == (48, 84)
    assert layer_hw(24 * 42, 96, 168) == (24, 42)
    with pytest.raises(ValueError):
        layer_hw(12345, 96, 168)


def test_a_vertical_region_maps_to_its_own_columns_and_the_gutter_to_neither():
    boxes = region_boxes(1344, 768, 2, orientation="vertical", gutter=64)
    left = box_to_grid(boxes[0], 1344, 768, 24, 42)
    right = box_to_grid(boxes[1], 1344, 768, 24, 42)
    row_l, row_r = left[10], right[10]
    assert row_l[0] and not row_l[-1] and row_r[-1] and not row_r[0]
    assert not any(a and b for a, b in zip(row_l, row_r))       # no cell is both people's
    assert any(not a and not b for a, b in zip(row_l, row_r))   # the gutter keeps the shared text
    assert all(r == row_l for r in left)                        # a vertical strip is the same on every row
