"""vivijure-cf#1143: `multi_char.pose_spread` closes the empty band between two pose figures.

The 1.1.11 diagnostic (debug_intermediates) showed the extra person in a two-shot is drawn by the plate, in the
band the skeleton leaves empty between two figures placed at the centres of the two halves. This knob moves the
figures toward the frame centre. Tested here: off is byte-identical to the skeleton drawn before the knob existed
(and keeps the reuse hash), on moves each figure by the computed shift and never out of its own region, the
value is parsed and clamped, it reaches both pose call sites, and it is in the hash when set."""
import pytest
from PIL import Image, ImageDraw
from test_debug_intermediates_1143 import gpu_fakes  # noqa: F401  the scene-lock fakes

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, MultiCharConfig, QualityTier
from vivijure_backend.contract import RenderRequest
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.pipeline import keyframe_params_from

W, H = 1344, 768
SW, SH = 448, 256  # the fake renders: wide enough that the default 64 px gutter leaves real regions


def _legacy_skeleton(width, height, boxes):
    """The skeleton exactly as drawn before pose_spread (backend 69ff143), kept as the reference."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (l, t, r, b) in boxes:
        bw, bh = r - l, b - t
        my = int(bh * 0.04)
        pts = [(l + x * bw, t + my + y * (bh - 2 * my)) for (x, y) in kf._POSE_KP]
        lw = max(2, bw // 70)
        for i, (a, c) in enumerate(kf._POSE_LIMBS):
            draw.line([pts[a], pts[c]], fill=kf._POSE_COLORS[i % len(kf._POSE_COLORS)], width=lw)
        rad = max(2, bw // 90)
        for i, (px, py) in enumerate(pts):
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=kf._POSE_COLORS[i % len(kf._POSE_COLORS)])
    return img


def _boxes(n=2, gutter=64):
    return kf.region_boxes(W, H, n, orientation="vertical", gutter=gutter)


def _x_extent(img, box):
    """Leftmost and rightmost lit column inside one region box."""
    l, t, r, b = box
    cols = [x for x in range(l, r) if any(img.getpixel((x, y)) != (0, 0, 0) for y in range(t, b, 4))]
    return min(cols), max(cols)


@pytest.mark.parametrize("n,gutter", [(1, 64), (2, 0), (2, 64), (3, 64)])
def test_off_is_byte_identical_to_the_old_skeleton(n, gutter):
    boxes = _boxes(n, gutter)
    assert kf._pose_skeleton(W, H, boxes).tobytes() == _legacy_skeleton(W, H, boxes).tobytes()
    assert kf._pose_skeleton(W, H, boxes, spread=0.0).tobytes() == _legacy_skeleton(W, H, boxes).tobytes()


def test_spread_moves_the_two_figures_together_and_keeps_each_in_its_region():
    boxes = _boxes()
    before = [_x_extent(_legacy_skeleton(W, H, boxes), b) for b in boxes]
    on = kf._pose_skeleton(W, H, boxes, spread=0.3)
    after = [_x_extent(on, b) for b in boxes]
    gap_before = before[1][0] - before[0][1]
    gap_after = after[1][0] - after[0][1]
    assert gap_after < gap_before - 200
    # Centres land where the spread puts them: 0.5 -/+ 0.15 of the width.
    for (lo, hi), want in zip(after, (W * 0.35, W * 0.65)):
        assert abs((lo + hi) / 2 - want) < 8
    # And nothing is drawn outside a region box (the gutter stays dark).
    r0 = boxes[0][2]
    l1 = boxes[1][0]
    assert all(on.getpixel((x, y)) == (0, 0, 0) for x in range(r0, l1) for y in range(0, H, 4))


def test_shifts_are_clamped_inside_the_region():
    boxes = _boxes()
    lo, hi = min(x for x, _ in kf._POSE_KP), max(x for x, _ in kf._POSE_KP)
    for spread in (0.05, 0.15, 0.3, 0.6, 0.9):
        for (l, _t, r, _b), dx in zip(boxes, kf.pose_figure_shifts(W, boxes, spread)):
            bw = r - l
            assert l <= l + lo * bw + dx and l + hi * bw + dx <= r
    assert kf.pose_figure_shifts(W, boxes, 0.0) == [0.0, 0.0]
    assert kf.pose_figure_shifts(W, _boxes(1), 0.5) == [0.0]


@pytest.mark.parametrize("value,want", [(None, 0.0), (0.3, 0.3), (-1, 0.0), (5, 0.9), ("x", 0.0)])
def test_config_parses_and_clamps(value, want):
    d = {} if value is None else {"pose_spread": value}
    assert MultiCharConfig.from_dict(d).pose_spread == want


def test_request_reaches_the_engine_params():
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                   "render_overrides": {"keyframe": {"multi_char": {"pose_spread": 0.25}}}})
    assert keyframe_params_from(req.config).pose_spread == 0.25
    assert kf.KeyframeParams().pose_spread == 0.0


# kf_hash of KeyframeConfig.for_tier(tier) on main at 69ff143, before pose_spread existed. A still drawn then must
# keep its hash while the knob is off, or every existing project redraws for a no-op.
HASH_BEFORE_POSE_SPREAD = {QualityTier.DRAFT: "0922f3dc0cb2ccca", QualityTier.STANDARD: "23b7e0c279e5ed2e",
                           QualityTier.FINAL: "23b7e0c279e5ed2e"}


@pytest.mark.parametrize("tier", list(HASH_BEFORE_POSE_SPREAD))
def test_hash_off_equals_the_hash_from_before_the_knob(tier):
    assert kf_hash(KeyframeConfig.for_tier(tier)) == HASH_BEFORE_POSE_SPREAD[tier]


def test_hash_unchanged_when_off_and_changes_when_set():
    kc = KeyframeConfig.for_tier(QualityTier.STANDARD)
    base = kf_hash(kc)
    kc.multi_char.pose_spread = 0.0
    assert kf_hash(kc) == base
    kc.multi_char.pose_spread = 0.3
    on = kf_hash(kc)
    assert on != base
    kc.multi_char.pose_spread = 0.25
    assert kf_hash(kc) not in (base, on)


def test_the_plate_pose_control_carries_the_spread(tmp_path, gpu_fakes):  # noqa: F811
    off, _ = _spread_render(tmp_path, 0.0)
    on, pipe = _spread_render(tmp_path, 0.3)
    assert Image.open(on.intermediates["pose"]).tobytes() == pipe.calls[0]["image"].tobytes()
    assert Image.open(on.intermediates["pose"]).tobytes() != Image.open(off.intermediates["pose"]).tobytes()
    boxes = kf.region_boxes(SW, SH, 2, orientation="vertical", gutter=kf.KeyframeParams().region_gutter)
    assert pipe.calls[0]["image"].tobytes() == kf._pose_skeleton(SW, SH, boxes, spread=0.3).tobytes()
    assert Image.open(off.intermediates["pose"]).tobytes() == _legacy_skeleton(SW, SH, boxes).tobytes()


def test_the_no_plate_regional_path_carries_it_too(tmp_path, gpu_fakes, monkeypatch):  # noqa: F811
    seen = []
    real = kf._pose_skeleton
    monkeypatch.setattr(kf, "_pose_skeleton", lambda w, h, boxes, *, spread=0.0: seen.append(spread) or real(w, h, boxes, spread=spread))
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count: [Image.new("RGB", (8, 8))])
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", scene_lock=False, pose_spread=0.4)
    from test_debug_intermediates_1143 import CAST, SB, _Pipe, _scene, _Server
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(_Pipe(SW, SH)), tmp_path / "x" / "shot_03.png",
                       params=params, lora_paths={})
    assert seen == [0.4]


def _spread_render(tmp_path, spread):
    from test_debug_intermediates_1143 import CAST, SB, _Pipe, _scene, _Server
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", pose_spread=spread)
    pipe = _Pipe(params.width, params.height)
    out = tmp_path / f"spread{spread}" / "shot_03.png"
    res = kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(pipe), out, params=params,
                             lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")},
                             save_intermediates=True)
    return res, pipe
