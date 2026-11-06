"""vivijure-cf#1143 option (b): the plate can take each person's whole cast portrait, confined to their region.

Off by default, so the plate is unchanged until a live sweep picks a scale. The decision, the plumbing and the
argument building are tested here; the image it makes is validated on a pod."""
import ast
import sys
import types
from pathlib import Path

import pytest

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig
from vivijure_backend.contract import RenderRequest, Scene, Storyboard
from vivijure_backend.keyframe import KeyframeParams, plate_ip_adapter_scale_for
from vivijure_backend.orchestrator import kf_hash


def _scene(slots):
    return Scene.from_dict({"id": "s", "prompt": "x", "character_slots": slots}, 0)


def test_off_by_default_everywhere():
    assert KeyframeConfig().plate_ip_adapter_scale == 0.0
    assert KeyframeParams().plate_ip_adapter_scale == 0.0
    assert plate_ip_adapter_scale_for(_scene(["A", "B"]), KeyframeParams()) == 0.0


@pytest.mark.parametrize("slots,want", [(["A", "B"], 0.4), (["A"], 0.0), ([], 0.0)])
def test_only_a_two_person_plate_takes_it(slots, want):
    assert plate_ip_adapter_scale_for(_scene(slots), KeyframeParams(plate_ip_adapter_scale=0.4)) == want


def test_more_people_than_the_regional_path_handles_is_not_conditioned():
    assert plate_ip_adapter_scale_for(_scene(["A", "B", "C"]), KeyframeParams(plate_ip_adapter_scale=0.4)) == 0.0


def test_it_is_clamped():
    assert plate_ip_adapter_scale_for(_scene(["A", "B"]), KeyframeParams(plate_ip_adapter_scale=3.0)) == 1.0
    assert plate_ip_adapter_scale_for(_scene(["A", "B"]), KeyframeParams(plate_ip_adapter_scale=-1.0)) == 0.0


def test_it_is_in_the_reuse_hash():
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "x"}]})
    a = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz"})
    b = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                 "render_overrides": {"keyframe": {"plate_ip_adapter_scale": 0.4}}})
    assert b.config.keyframe.plate_ip_adapter_scale == 0.4
    assert kf_hash(a.config.keyframe, scene=sb.scenes[0], storyboard=sb) != kf_hash(b.config.keyframe, scene=sb.scenes[0], storyboard=sb)


def test_the_plate_call_spreads_the_portrait_arguments():
    """Read from the syntax tree: the Pass A pipe call (the one fed the plate prompt) spreads **plate_ip_kw."""
    src = Path(kf.__file__).read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_render_scene_lock")
    plate_calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and "plate_spec.prompt" in ast.unparse(c)
                   and any(k.arg == "num_inference_steps" for k in c.keywords)]
    assert len(plate_calls) == 1
    assert any(k.arg is None and ast.unparse(k.value) == "plate_ip_kw" for k in plate_calls[0].keywords)


class _Pipe:
    def __init__(self):
        self.loaded = None
        self.scales = None

    def set_ip_adapter_scale(self, s):
        self.scales = s


@pytest.fixture
def fake_masks(monkeypatch):
    mod = types.ModuleType("diffusers.image_processor")

    class IPAdapterMaskProcessor:
        def preprocess(self, masks, height, width):
            return ("masks", len(masks), height, width)

    mod.IPAdapterMaskProcessor = IPAdapterMaskProcessor
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.image_processor", mod)
    monkeypatch.setattr(kf, "_ensure_ip_adapter", lambda pipe, n=1: setattr(pipe, "loaded", n))
    monkeypatch.setattr(kf, "_box_mask", lambda w, h, b: b)


def test_each_person_gets_their_portrait_in_their_region(monkeypatch, fake_masks):
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count=1: [f"portrait_{slot}"])
    pipe = _Pipe()
    cfg = KeyframeParams(plate_ip_adapter_scale=0.4)
    kw = kf._plate_ip_kwargs(pipe, None, _scene(["A", "B"]), cfg, 0.4)
    assert kw["ip_adapter_image"] == ["portrait_A", "portrait_B"]
    assert kw["cross_attention_kwargs"]["ip_adapter_masks"] == ("masks", 2, cfg.height, cfg.width)
    assert pipe.loaded == 2 and pipe.scales == [0.4, 0.4]


def test_a_person_with_no_portrait_is_not_conditioned(monkeypatch, fake_masks):
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count=1: ["portrait_A"] if slot == "A" else [])
    pipe = _Pipe()
    kw = kf._plate_ip_kwargs(pipe, None, _scene(["A", "B"]), KeyframeParams(), 0.4)
    assert pipe.scales == [0.4, 0.0] and len(kw["ip_adapter_image"]) == 2


def test_no_portraits_at_all_draws_the_plate_as_before(monkeypatch, fake_masks):
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count=1: [])
    pipe = _Pipe()
    assert kf._plate_ip_kwargs(pipe, None, _scene(["A", "B"]), KeyframeParams(), 0.4) == {}
    assert pipe.loaded is None
