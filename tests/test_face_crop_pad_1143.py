"""vivijure-cf#1143 step 1: the Pass B identity crop padding is a knob, and its default is today's crop.

Measured on 1.1.6 (seed 110501): the two-shot loses Mara's hair on the scene-lock path, whose masked
IP-Adapter gets a face crop padded 0.35, and keeps it on the path that gets the full portrait. This knob lets
one bake be swept live. The GPU crop itself is validated on a pod; its geometry and its plumbing are here."""
import ast
from pathlib import Path

import pytest

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, QualityTier, RenderConfig
from vivijure_backend.contract import RenderRequest, Storyboard
from vivijure_backend.instantid import crop_face, crop_from_bbox
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.pipeline import keyframe_params_from


def _rcfg(kfd=None):
    return RenderConfig.from_request(QualityTier.FINAL, {"keyframe": kfd} if kfd else {})


def test_the_default_is_todays_crop():
    import inspect
    assert kf.DEFAULT_FACE_CROP_PAD == 0.35
    assert KeyframeConfig().face_crop_pad == 0.35
    assert kf.KeyframeParams().face_crop_pad == 0.35
    assert inspect.signature(crop_face).parameters["pad_ratio"].default == 0.35
    assert keyframe_params_from(_rcfg()).face_crop_pad == 0.35


@pytest.mark.parametrize("given,want", [(1.0, 1.0), (0.0, 0.0), (2.0, 2.0), (5.0, 2.0), (-1.0, 0.0), ("x", 0.35)])
def test_it_is_read_and_clamped_and_reaches_the_engine(given, want):
    assert keyframe_params_from(_rcfg({"face_crop_pad": given})).face_crop_pad == want


def test_it_is_in_the_reuse_hash():
    """A still drawn at one padding is never reused for another (vivijure-backend#573)."""
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "x"}]})
    a = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz"})
    b = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                 "render_overrides": {"keyframe": {"face_crop_pad": 1.0}}})
    assert b.config.keyframe.face_crop_pad == 1.0
    assert kf_hash(a.config.keyframe, scene=sb.scenes[0], storyboard=sb) != kf_hash(b.config.keyframe, scene=sb.scenes[0], storyboard=sb)


def test_a_larger_pad_keeps_more_around_the_face():
    from PIL import Image
    img = Image.new("RGB", (1000, 1000))
    box = (400, 300, 600, 550)  # a 200 x 250 face
    tight = crop_from_bbox(img, box, pad_ratio=0.35)
    wide = crop_from_bbox(img, box, pad_ratio=1.0)
    assert tight.size == (340, 425)
    assert wide.size == (600, 750)  # the face box plus one face width and height on each side


def test_both_pass_b_calls_hand_the_knob_to_the_crop():
    """Read from the syntax tree: each call to _face_crops_for_slots passes pad_ratio=cfg.face_crop_pad."""
    tree = ast.parse(Path(kf.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_face_crops_for_slots"]
    assert len(calls) == 2
    for c in calls:
        kw = {k.arg: ast.unparse(k.value) for k in c.keywords}
        assert kw.get("pad_ratio") == "cfg.face_crop_pad", ast.unparse(c)


def test_the_crop_helper_passes_the_pad_on(monkeypatch):
    seen = []
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count=1: ["ref.png"])
    import vivijure_backend.instantid as iid
    monkeypatch.setattr(iid, "crop_face", lambda analyzer, image, pad_ratio=0.35: seen.append(pad_ratio) or "crop")
    server = type("S", (), {"face_analyzer": lambda self: object()})()
    crops, any_face = kf._face_crops_for_slots(server, None, ["A", "B"], pad_ratio=1.25)
    assert seen == [1.25, 1.25] and crops == ["crop", "crop"] and any_face
