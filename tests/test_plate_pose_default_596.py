"""backend#596: a guarded two-plus-person scene-lock plate draws without the pose skeleton by default.

Measured on vivijure-cf#1281: under the fixed standing skeleton the pair posed for the camera instead of acting
(2 of 6 held, right people); with no skeleton 5 of 6, and the plate count guard caught every third person. So the
default is coupled to the guard, an explicit `pose_conditioning` still wins on both paths, the plain regional path
keeps its pose, and the reuse hash changes only for the stills whose drawing changes. The render tests run the real
scene-lock body over the fakes from the #1143 tests.
"""
import pytest
from test_debug_intermediates_1143 import (  # noqa: F401
    CAST,
    SB,
    _scene,
    gpu_fakes,  # the fixture, by name
)
from test_plate_count_guard_1143 import _run

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, MultiCharConfig
from vivijure_backend.contract import Scene
from vivijure_backend.orchestrator import kf_hash


def _shot(n: int) -> Scene:
    slots = ["A", "B", "C", "D"][:n]
    return Scene.from_dict({"id": "s", "prompt": "at a workbench, looking at a laptop", "character_slots": slots}, 0)


# ------------------------------------------------------------------ the rule

@pytest.mark.parametrize("people,want", [(0, False), (1, False), (2, True), (3, False)])
def test_the_default_takes_the_pose_off_only_a_two_plus_person_plate(people, want):
    # max_slots 2: three people are outside the regional path and never drew a pose.
    assert kf.plate_pose_default_off(people, max_slots=2, pose_conditioning=None, plate_count_retries=3) is want


def test_pose_off_with_the_guard_off_is_never_a_default():
    for retries in (0, -1, "0"):
        assert kf.plate_pose_default_off(2, max_slots=2, pose_conditioning=None, plate_count_retries=retries) is False
    # And the shipped defaults themselves: the plate draws without the pose only because the guard is on.
    params = kf.KeyframeParams()
    assert params.pose_conditioning is None and params.plate_count_retries > 0
    assert kf.plate_pose_on(_shot(2), params) is False
    assert kf.plate_pose_on(_shot(2), kf.KeyframeParams(plate_count_retries=0)) is True
    cfg = KeyframeConfig()
    assert cfg.multi_char.pose_conditioning is None and cfg.plate_count_retries > 0


@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_setting_wins_with_the_guard_on_or_off(explicit):
    for retries in (0, 3):
        params = kf.KeyframeParams(pose_conditioning=explicit, plate_count_retries=retries)
        assert kf.plate_pose_on(_shot(2), params) is explicit
        assert kf.plate_pose_default_off(2, max_slots=2, pose_conditioning=explicit, plate_count_retries=retries) is False


def test_a_one_person_or_empty_plate_never_draws_a_pose():
    for n in (0, 1):
        assert kf.plate_pose_on(_shot(n), kf.KeyframeParams(pose_conditioning=True)) is False


def test_the_plain_regional_path_keeps_its_pose():
    assert kf.regional_pose_on(kf.KeyframeParams()) is True
    assert kf.regional_pose_on(kf.KeyframeParams(pose_conditioning=True)) is True
    assert kf.regional_pose_on(kf.KeyframeParams(pose_conditioning=False)) is False


def test_the_planner_follows_the_rule():
    assert kf.plan_keyframe_passes(_shot(2), CAST, SB, kf.KeyframeParams())[0].controlnet == "none"
    assert kf.plan_keyframe_passes(_shot(2), CAST, SB, kf.KeyframeParams(pose_conditioning=True))[0].controlnet == "pose"
    assert kf.plan_keyframe_passes(_shot(2), CAST, SB, kf.KeyframeParams(plate_count_retries=0))[0].controlnet == "pose"


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("given,want", [({}, None), ({"pose_conditioning": None}, None),
                                        ({"pose_conditioning": True}, True), ({"pose_conditioning": False}, False),
                                        ({"controlnet_pose_scale": 0.55}, None)])
def test_a_job_that_does_not_name_the_pose_gets_the_default(given, want):
    # The last case is what the studio keyframe module sends today: the scale, never the switch.
    assert MultiCharConfig.from_dict(given).pose_conditioning is want


# ------------------------------------------------------------------ the real render body

def _plate_call(pipe):
    return pipe.calls[0]


@pytest.mark.usefixtures("gpu_fakes")
def test_the_default_plate_is_drawn_with_the_pose_controlnet_inert(tmp_path):
    res, pipe, _server = _run(tmp_path, "default", retries=3, counts=[2], save=True)
    assert _plate_call(pipe)["controlnet_conditioning_scale"] == 0.0
    assert "pose" not in res.intermediates


@pytest.mark.usefixtures("gpu_fakes")
def test_a_job_can_turn_the_plates_pose_back_on(tmp_path):
    res, pipe, _server = _run(tmp_path, "on", retries=3, counts=[2], save=True, pose=True)
    assert _plate_call(pipe)["controlnet_conditioning_scale"] == kf.KeyframeParams().controlnet_pose_scale
    assert "pose" in res.intermediates


@pytest.mark.usefixtures("gpu_fakes")
def test_turning_the_guard_off_restores_the_plates_pose(tmp_path):
    res, pipe, _server = _run(tmp_path, "guard-off", retries=0, save=True)
    assert _plate_call(pipe)["controlnet_conditioning_scale"] == kf.KeyframeParams().controlnet_pose_scale
    assert "pose" in res.intermediates


@pytest.mark.usefixtures("gpu_fakes")
@pytest.mark.parametrize("pose,want", [(None, True), (True, True), (False, False)])
def test_the_plain_regional_render_keeps_its_pose_by_default(tmp_path, monkeypatch, pose, want):
    # Scene lock off: the change does not reach this path, so unset still draws under the skeleton.
    from test_plate_count_guard_1143 import SH, SW, _CountingServer, _DistinctPipe
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", seed=110501,
                               scene_lock=False, pose_conditioning=pose)
    from PIL import Image
    # gpu_fakes stubs the reference images to none (that keeps its tests on scene lock); this path needs one each.
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count: [Image.new("RGB", (64, 64), (90, 90, 90))])
    pipe = _DistinctPipe(SW, SH)
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _CountingServer(pipe, []), tmp_path / "shot_03.png",
                       params=params, lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")})
    scale = pipe.calls[0]["controlnet_conditioning_scale"]
    assert (scale == params.controlnet_pose_scale) is want and (scale == 0.0) is (not want)


# ------------------------------------------------------------------ reuse hash, as in #589

# Computed with the code on main at 08cd5da, the commit before this change, by the same `_hash` call (KeyframeConfig
# defaults, the scene below, no storyboard). A stored still carries one of these today. A shot the change does not
# touch must keep its exact value, so pinned literals are the test, not a comparison of the new code with itself.
BEFORE_596 = {
    "n0": "23b7e0c279e5ed2e",
    "n1": "23b7e0c279e5ed2e",
    "n2": "bc5635c44b87c426",
    "n2_guard_off": "23b7e0c279e5ed2e",
    "n2_lock_off": "34f6f9b2d77ff564",
}


def _hash(n, **kw):
    kc = KeyframeConfig()
    for k, v in kw.items():
        if k == "pose":
            kc.multi_char.pose_conditioning = v
        else:
            setattr(kc, k, v)
    return kf_hash(kc, scene=_shot(n))


@pytest.mark.parametrize("n", [0, 1])
def test_a_shot_the_change_does_not_touch_keeps_its_stored_hash(n):
    assert _hash(n) == BEFORE_596[f"n{n}"]


def test_a_two_person_still_with_the_guard_off_keeps_its_stored_hash():
    assert _hash(2, plate_count_retries=0) == BEFORE_596["n2_guard_off"]


def test_a_two_person_still_with_scene_lock_off_keeps_its_stored_hash():
    assert _hash(2, scene_lock=False) == BEFORE_596["n2_lock_off"]


def test_asking_for_the_pose_draws_and_hashes_exactly_as_the_old_default():
    assert _hash(2, pose=True) == BEFORE_596["n2"]


def test_a_two_person_guarded_scene_lock_still_is_redrawn():
    assert _hash(2) != BEFORE_596["n2"]
    assert _hash(2) != _hash(2, pose=False)  # unset and explicitly off draw alike but are different requests
