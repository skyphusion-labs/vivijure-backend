"""vivijure-cf#1143 option (c): with per-region text on, the band outside both people reads a people-free prompt.

Measured on 1.1.9: under option (a) the band read the shot text "Mara and Dev stand ...", and a third person
appeared there at 2 of 2 seeds. The processor side (the band's cells read the background prompt) is checked in
scripts/check_regional_text.py; here, the prompt, the gating, the safety clause and the plumbing."""
import ast
import itertools
from pathlib import Path

import pytest

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig
from vivijure_backend.contract import Cast, RenderRequest, Scene, Storyboard
from vivijure_backend.keyframe import (
    KeyframeParams,
    adult_clause,
    plate_background_prompt,
    plate_people_free_background_on,
    plate_region_prompts,
)
from vivijure_backend.orchestrator import kf_hash

MARA = "Mara, an adult woman in her thirties, dark shoulder-length curls, charcoal field jacket over a black top."
DEV = "Dev, an adult man in his thirties, dark buzz cut, grey zip hoodie."
CAST = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": MARA}, "B": {"name": "Dev", "prompt": DEV}}})


def _scene(prompt="Mara and Dev stand side by side at a workbench, looking at a laptop.", slots=("A", "B")):
    return Scene.from_dict({"id": "s", "prompt": prompt, "character_slots": list(slots)}, 0)


def test_off_by_default_and_only_with_regional_text():
    assert KeyframeConfig().plate_people_free_background is False
    assert KeyframeParams().plate_people_free_background is False
    assert plate_people_free_background_on(_scene(), KeyframeParams(plate_regional_text=False, plate_people_free_background=True)) is False
    # vivijure-cf#1281: per-region text is on by default for this shot, so the band follows it when asked for.
    assert plate_people_free_background_on(_scene(), KeyframeParams(plate_people_free_background=True)) is True
    assert plate_people_free_background_on(_scene(), KeyframeParams(plate_regional_text=True, plate_people_free_background=True)) is True
    assert plate_people_free_background_on(_scene(slots=("A",)), KeyframeParams(plate_regional_text=True, plate_people_free_background=True)) is False


def test_it_is_in_the_reuse_hash():
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "x"}]})
    a = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz"})
    b = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                 "render_overrides": {"keyframe": {"plate_people_free_background": True}}})
    assert b.config.keyframe.plate_people_free_background is True
    assert kf_hash(a.config.keyframe, scene=sb.scenes[0], storyboard=sb) != kf_hash(b.config.keyframe, scene=sb.scenes[0], storyboard=sb)


def test_the_background_is_style_and_setting_only():
    sb = Storyboard.from_dict({"style_prefix": "photoreal, 35mm,", "setting": "a dim workshop at night", "scenes": [{"prompt": "x"}]})
    assert plate_background_prompt(sb) == "photoreal, 35mm, a dim workshop at night"
    assert plate_background_prompt(Storyboard.from_dict({"scenes": [{"prompt": "x"}]})) == ""


SHOTS = ["Mara and Dev stand side by side at a workbench, looking at a laptop.", "two of them talk", "Mara hands Dev a wrench."]
STYLES = ["", "photoreal, 35mm,", "photoreal, dim industrial workshop at night,"]
SETTINGS = ["", "a dim workshop at night"]


@pytest.mark.parametrize("shot,style,setting", list(itertools.product(SHOTS, STYLES, SETTINGS)))
def test_the_background_never_names_a_person_or_a_count_or_the_shot(shot, style, setting):
    sb = Storyboard.from_dict({"style_prefix": style, "setting": setting, "scenes": [{"prompt": shot}]})
    bg = plate_background_prompt(sb).lower()
    for word in ("mara", "dev", "adult", "two", "people", "person", "man", "woman"):
        assert word not in bg.split() and word not in bg.replace(",", " ").split(), (word, bg)
    assert shot.lower() not in bg


@pytest.mark.parametrize("style,setting", list(itertools.product(STYLES, SETTINGS)))
def test_every_region_that_draws_a_person_keeps_the_adult_clause(style, setting):
    """(c) changes only the band. Each person's region still reads the shared text, adult clause included."""
    sb = Storyboard.from_dict({"style_prefix": style, "setting": setting, "scenes": [{"prompt": "x"}]})
    _, persons = plate_region_prompts(_scene(), CAST, sb)
    clause = adult_clause(["A", "B"], named=False)
    assert all(clause in p for p in persons)
    assert clause not in plate_background_prompt(sb)


def test_the_background_reaches_the_processors_and_only_when_on():
    fn = next(n for n in ast.walk(ast.parse(Path(kf.__file__).read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_scene_lock")
    src = ast.unparse(fn)
    assert "if regions and plate_people_free_background_on(scene, cfg):" in src
    assert "plate_background_prompt(storyboard)" in src
    withs = [w for w in ast.walk(fn) if isinstance(w, ast.With)]
    assert any(any(k.arg == "background" and ast.unparse(k.value) == "background" for k in w.items[0].context_expr.keywords)
               for w in withs)


def test_applied_hands_the_background_to_every_cross_attention_processor():
    from vivijure_backend import regional_text as rt

    class _Unet:
        def __init__(self):
            self.procs = {"a.attn1.processor": "self", "a.attn2.processor": "cross", "b.attn2.processor": "cross2"}

        @property
        def attn_processors(self):
            return dict(self.procs)

        def set_attn_processor(self, d):
            self.procs = dict(d)

    pipe = type("P", (), {"vae_scale_factor": 8})()
    pipe.unet = _Unet()
    with rt.applied(pipe, [("emb", (0, 0, 8, 8))], width=64, height=64, cfg_batch=True, background="bg"):
        wrapped = [v for v in pipe.unet.attn_processors.values() if isinstance(v, rt.RegionalTextProcessor)]
        assert len(wrapped) == 2 and all(v.background == "bg" for v in wrapped)


def test_the_plate_pose_scale_is_the_configured_one():
    """(d): the Pass A pose ControlNet uses cfg.controlnet_pose_scale, which the studio now forwards."""
    fn = next(n for n in ast.walk(ast.parse(Path(kf.__file__).read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == "_render_scene_lock")
    assert "cn_scale = cfg.controlnet_pose_scale" in ast.unparse(fn)
