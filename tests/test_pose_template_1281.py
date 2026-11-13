"""vivijure-cf#1281: `multi_char.pose_template` picks the figure the two-person pose skeleton draws.

`stand` is the figure drawn since the skeleton existed: upright, full length, head-on. It is the default, and it
must stay byte-identical and keep every reuse hash. `reach` is for two people at a shared surface: waist-up (no
knees, no ankles), the arm nearer the partner forward, and the head turned toward the partner (the far eye and
far ear dropped, the COCO three-quarter convention). The figure on the right is the mirror of the one on the left,
with its own left and right swapped, so both face the frame centre. `auto` picks `reach` from the shot text.

Why: on the workbench board, per-region plate text draws faces head-on and Pass B opens the eyes to the lens
(cf#1281 comments 6687630211 and 6839323287). The standing figure is head-on too, so it cannot turn a head."""
import pytest
from PIL import Image, ImageDraw
from test_debug_intermediates_1143 import gpu_fakes  # noqa: F401  the scene-lock fakes

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, MultiCharConfig, QualityTier
from vivijure_backend.contract import RenderRequest, Scene
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.pipeline import keyframe_params_from

W, H = 1344, 768
SW, SH = 448, 256


def _legacy_skeleton(width, height, boxes, spread=0.0):
    """The skeleton exactly as drawn on main c7a6f02, before pose_template existed."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (l, t, r, b), dx in zip(boxes, kf.pose_figure_shifts(width, boxes, spread)):
        bw, bh = r - l, b - t
        my = int(bh * 0.04)
        pts = [(l + x * bw + dx, t + my + y * (bh - 2 * my)) for (x, y) in kf._POSE_KP]
        lw = max(2, bw // 70)
        for i, (a, c) in enumerate(kf._POSE_LIMBS):
            draw.line([pts[a], pts[c]], fill=kf._POSE_COLORS[i % len(kf._POSE_COLORS)], width=lw)
        rad = max(2, bw // 90)
        for i, (px, py) in enumerate(pts):
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=kf._POSE_COLORS[i % len(kf._POSE_COLORS)])
    return img


def _boxes(n=2, gutter=64):
    return kf.region_boxes(W, H, n, orientation="vertical", gutter=gutter)


def _lit(img, box, colour):
    l, t, r, b = box
    return [(x, y) for x in range(l, r, 2) for y in range(t, b, 2) if img.getpixel((x, y)) == colour]


# --- stand: the old figure, byte for byte --------------------------------------------------------------------

@pytest.mark.parametrize("n,gutter,spread", [(1, 64, 0.0), (2, 0, 0.0), (2, 64, 0.0), (2, 64, 0.3), (3, 64, 0.0)])
def test_stand_is_byte_identical_to_the_old_skeleton(n, gutter, spread):
    boxes = _boxes(n, gutter)
    old = _legacy_skeleton(W, H, boxes, spread).tobytes()
    assert kf._pose_skeleton(W, H, boxes, spread=spread).tobytes() == old
    assert kf._pose_skeleton(W, H, boxes, spread=spread, template="stand").tobytes() == old


# --- reach: the new figure -----------------------------------------------------------------------------------

def test_reach_draws_a_different_control():
    boxes = _boxes()
    assert kf._pose_skeleton(W, H, boxes, template="reach").tobytes() != _legacy_skeleton(W, H, boxes).tobytes()


def test_reach_figure_drops_legs_and_the_far_eye_and_ear():
    pts = kf.pose_keypoints("reach", mirrored=False)
    assert len(pts) == 18
    for i in (9, 10, 12, 13):        # knees and ankles: waist-up
        assert pts[i] is None
    assert pts[15] is None and pts[17] is None   # facing right (its own left): the left eye and ear are far
    assert pts[14] is not None and pts[16] is not None
    # Hips sit at the bottom of the box, so the figure is framed waist-up, not full length.
    assert min(pts[8][1], pts[11][1]) > 0.85


def test_reach_near_arm_goes_forward_toward_the_partner():
    left = kf.pose_keypoints("reach", mirrored=False)   # stands on the left, partner on the right
    # its left arm (5 shoulder, 6 elbow, 7 wrist) extends rightward, past its own body, at about bench height
    assert left[7][0] > left[6][0] > left[5][0]
    assert left[7][0] > 0.9
    assert 0.4 < left[7][1] < 0.8
    # and the nose is turned toward the partner: right of the neck
    assert left[0][0] > left[1][0]


def test_the_right_figure_is_the_mirror_with_sides_swapped():
    left = kf.pose_keypoints("reach", mirrored=False)
    right = kf.pose_keypoints("reach", mirrored=True)
    swap = {2: 5, 3: 6, 4: 7, 5: 2, 6: 3, 7: 4, 8: 11, 9: 12, 10: 13, 11: 8, 12: 9, 13: 10,
            14: 15, 15: 14, 16: 17, 17: 16}
    for i in range(18):
        j = swap.get(i, i)
        if left[i] is None:
            assert right[j] is None
        else:
            assert right[j] == pytest.approx((1 - left[i][0], left[i][1]))
    # the right figure reaches leftward, toward the partner
    assert right[4][0] < 0.1


def test_stand_keypoints_are_the_old_figure():
    assert kf.pose_keypoints("stand", mirrored=False) == list(kf._POSE_KP)


def test_reach_hands_meet_in_the_middle_and_nothing_leaves_its_region():
    boxes = _boxes()
    img = kf._pose_skeleton(W, H, boxes, template="reach")
    # the wrist joint colour (index 7 for the left figure's left wrist, index 4 for the right figure's right wrist)
    lw = _lit(img, boxes[0], kf._POSE_COLORS[7])
    rw = _lit(img, boxes[1], kf._POSE_COLORS[4])
    assert lw and rw
    assert max(x for x, _ in lw) > boxes[0][2] - 0.15 * (boxes[0][2] - boxes[0][0])
    assert min(x for x, _ in rw) < boxes[1][0] + 0.15 * (boxes[1][2] - boxes[1][0])
    # the gutter between the regions stays dark
    assert all(img.getpixel((x, y)) == (0, 0, 0) for x in range(boxes[0][2], boxes[1][0]) for y in range(0, H, 4))


def test_reach_shifts_stay_inside_the_region():
    boxes = _boxes()
    for spread in (0.05, 0.3, 0.6, 0.9):
        shifts = kf.pose_figure_shifts(W, boxes, spread, template="reach")
        for k, ((l, _t, r, _b), dx) in enumerate(zip(boxes, shifts)):
            xs = [p[0] for p in kf.pose_keypoints("reach", mirrored=k == 1) if p is not None]
            bw = r - l
            assert l <= l + min(xs) * bw + dx and l + max(xs) * bw + dx <= r


# --- auto: from the shot text --------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("Mara hands Dev a steaming coffee mug across the kitchen counter.", "reach"),
    ("Mara and Dev shake hands firmly across an office desk.", "reach"),
    ("Mara and Dev stand side by side at a workbench in the dim workshop, looking at a laptop.", "reach"),
    ("Mara and Dev sit across a small wooden table in a quiet cafe, leaning over a chess board.", "reach"),
    ("Mara and Dev walk down a rainy street at night.", "stand"),
    ("Two friends pose for a photo on the beach.", "stand"),
])
def test_auto_reads_the_shot_text(text, want):
    scene = Scene(prompt=text, character_slots=["A", "B"])
    assert kf.pose_template_for(scene, kf.KeyframeParams(pose_template="auto")) == want


def test_explicit_templates_ignore_the_text():
    walk = Scene(prompt="They walk down a street.", character_slots=["A", "B"])
    bench = Scene(prompt="They work at a bench.", character_slots=["A", "B"])
    assert kf.pose_template_for(bench, kf.KeyframeParams()) == "stand"
    assert kf.pose_template_for(bench, kf.KeyframeParams(pose_template="stand")) == "stand"
    assert kf.pose_template_for(walk, kf.KeyframeParams(pose_template="reach")) == "reach"


# --- config, request, hash -----------------------------------------------------------------------------------

@pytest.mark.parametrize("value,want", [(None, "stand"), ("stand", "stand"), ("reach", "reach"), ("auto", "auto"),
                                        ("REACH", "stand"), ("sit", "stand"), (1, "stand")])
def test_config_parses_and_refuses_unknown_values(value, want):
    d = {} if value is None else {"pose_template": value}
    assert MultiCharConfig.from_dict(d).pose_template == want


def test_request_reaches_the_engine_params():
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                   "render_overrides": {"keyframe": {"multi_char": {"pose_template": "reach"}}}})
    assert keyframe_params_from(req.config).pose_template == "reach"
    assert kf.KeyframeParams().pose_template == "stand"


# kf_hash of KeyframeConfig.for_tier(tier) on main c7a6f02, before pose_template existed. A still drawn then must
# keep its hash under the default, or every existing project redraws for a no-op.
HASH_BEFORE_POSE_TEMPLATE = {QualityTier.DRAFT: "bc7df897df3558e8", QualityTier.STANDARD: "ee77025e4bf000ed",
                             QualityTier.FINAL: "ee77025e4bf000ed"}


@pytest.mark.parametrize("tier", list(HASH_BEFORE_POSE_TEMPLATE))
def test_stand_keeps_the_hash_from_before_the_knob(tier):
    kc = KeyframeConfig.for_tier(tier)
    assert kf_hash(kc) == HASH_BEFORE_POSE_TEMPLATE[tier]
    kc.multi_char.pose_template = "stand"
    assert kf_hash(kc) == HASH_BEFORE_POSE_TEMPLATE[tier]


def test_reach_and_auto_each_change_the_hash():
    kc = KeyframeConfig.for_tier(QualityTier.STANDARD)
    base = kf_hash(kc)
    kc.multi_char.pose_template = "reach"
    reach = kf_hash(kc)
    kc.multi_char.pose_template = "auto"
    auto = kf_hash(kc)
    assert len({base, reach, auto}) == 3


# --- both pose call sites carry it ---------------------------------------------------------------------------

def test_the_plate_pose_control_carries_the_template(tmp_path, gpu_fakes):  # noqa: F811
    from test_debug_intermediates_1143 import CAST, SB, _Pipe, _scene, _Server
    boxes = kf.region_boxes(SW, SH, 2, orientation="vertical", gutter=kf.KeyframeParams().region_gutter)
    seen = {}
    for tpl in ("stand", "reach"):
        params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", pose_conditioning=True,
                                   pose_template=tpl, plate_count_retries=0)
        pipe = _Pipe(SW, SH)
        res = kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(pipe), tmp_path / tpl / "shot_03.png",
                                 params=params, lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")},
                                 save_intermediates=True)
        seen[tpl] = Image.open(res.intermediates["pose"]).tobytes()
        assert pipe.calls[0]["image"].tobytes() == kf._pose_skeleton(SW, SH, boxes, template=tpl).tobytes()
    assert seen["stand"] == _legacy_skeleton(SW, SH, boxes).tobytes()
    assert seen["reach"] != seen["stand"]


def test_the_no_plate_regional_path_carries_it_too(tmp_path, gpu_fakes, monkeypatch):  # noqa: F811
    seen = []
    real = kf._pose_skeleton
    monkeypatch.setattr(kf, "_pose_skeleton",
                        lambda w, h, boxes, **kw: seen.append(kw.get("template", "stand")) or real(w, h, boxes, **kw))
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count: [Image.new("RGB", (8, 8))])
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", scene_lock=False,
                               pose_template="reach")
    from test_debug_intermediates_1143 import CAST, SB, _Pipe, _scene, _Server
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(_Pipe(SW, SH)), tmp_path / "x" / "shot_03.png",
                       params=params, lora_paths={})
    assert seen == ["reach"]
