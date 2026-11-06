"""vivijure-cf#1281: per-region plate text is ON by default for a guarded two-person scene-lock plate with no pose.

Measured on vivijure-cf#1281, board 2 (6 seeds, keyframes-only through the studio): the backend#596 default (no pose
skeleton, plate count guard on) held the action with the right people on 3 of 6, because the plate drew each person's
look from the whole shot text and Dev came out as a woman or long-haired. With per-region text on top, 6 of 6; with
the skeleton back on as well, 5 of 6. So the default is coupled to that measured cell: per-region text and the pose
both unset, and the pose default taking the skeleton off (which needs the guard). An explicit setting of either is
the job's own choice. The reuse hash changes only for the stills whose drawing changes. Render tests run the real
scene-lock body over the fakes from the #1143 tests.
"""
import pytest
from test_debug_intermediates_1143 import (  # noqa: F401
    CAST,
    SB,
    _scene,
    gpu_fakes,  # the fixture, by name
)
from test_plate_count_guard_1143 import SH, SW, _CountingServer, _DistinctPipe

import vivijure_backend.keyframe as kf
from vivijure_backend import regional_text as rt
from vivijure_backend.config import KeyframeConfig
from vivijure_backend.contract import RenderRequest, Scene
from vivijure_backend.orchestrator import kf_hash


def _shot(n: int) -> Scene:
    slots = ["A", "B", "C", "D"][:n]
    return Scene.from_dict({"id": "s", "prompt": "at a workbench, looking at a laptop", "character_slots": slots}, 0)


def _rule(people=2, *, text=None, pose=None, retries=3, max_slots=2):
    return kf.plate_regional_text_default_on(people, max_slots=max_slots, plate_regional_text=text,
                                             pose_conditioning=pose, plate_count_retries=retries)


# ------------------------------------------------------------------ the rule

@pytest.mark.parametrize("people,want", [(0, False), (1, False), (2, True), (3, False), (None, True)])
def test_the_default_reaches_only_a_two_person_plate(people, want):
    # max_slots 2: three people are outside the regional path. None is the config-only hash.
    assert _rule(people) is want


@pytest.mark.parametrize("pose", [True, False])
def test_an_explicit_pose_is_outside_the_measured_cell(pose):
    # The measured cell is the pose LEFT to its default. A job that names the pose draws as it did before.
    assert _rule(pose=pose) is False
    assert kf.plate_regional_text_on(_shot(2), kf.KeyframeParams(pose_conditioning=pose)) is False


def test_the_default_needs_the_guard():
    # With the guard off the plate keeps the skeleton (backend#596), so it is outside the cell too.
    for retries in (0, -1, "0"):
        assert _rule(retries=retries) is False
    assert kf.plate_regional_text_on(_shot(2), kf.KeyframeParams(plate_count_retries=0)) is False


@pytest.mark.parametrize("explicit", [True, False])
def test_an_explicit_setting_wins_everywhere(explicit):
    assert _rule(text=explicit) is False
    for pose in (None, True, False):
        for retries in (0, 3):
            params = kf.KeyframeParams(plate_regional_text=explicit, pose_conditioning=pose, plate_count_retries=retries)
            assert kf.plate_regional_text_on(_shot(2), params) is explicit


def test_the_shipped_defaults_turn_it_on_for_a_two_person_shot_only():
    params = kf.KeyframeParams()
    assert params.plate_regional_text is None and params.pose_conditioning is None and params.plate_count_retries > 0
    assert kf.plate_regional_text_on(_shot(2), params) is True
    for n in (0, 1, 3):
        assert kf.plate_regional_text_on(_shot(n), params) is False


def test_the_renderer_and_the_hash_share_one_rule():
    # Both call sites read plate_regional_text_default_on; neither re-derives it.
    import ast
    import inspect

    import vivijure_backend.orchestrator as orch
    for fn in (kf.plate_regional_text_on, orch.kf_hash):
        names = {n.func.id for n in ast.walk(ast.parse(inspect.getsource(fn)))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "plate_regional_text_default_on" in names, fn.__name__


# ------------------------------------------------------------------ config

@pytest.mark.parametrize("given,want", [({}, None), ({"plate_regional_text": None}, None),
                                        ({"plate_regional_text": True}, True), ({"plate_regional_text": False}, False),
                                        ({"plate_regional_text": "true"}, None), ({"plate_regional_text": 1}, None)])
def test_only_a_real_bool_is_a_choice(given, want):
    assert KeyframeConfig.from_dict(given).plate_regional_text is want


def test_the_wire_reaches_the_keyframe_config():
    for given, want in (({}, None), ({"plate_regional_text": True}, True), ({"plate_regional_text": False}, False)):
        r = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                     "render_overrides": {"keyframe": given}})
        assert r.config.keyframe.plate_regional_text is want


# ------------------------------------------------------------------ the real render body

def _render(tmp_path, name, **kw):
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", seed=110501, **kw)
    pipe = _DistinctPipe(SW, SH)
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _CountingServer(pipe, [2]), tmp_path / name / "shot_03.png",
                       params=params, lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")})
    return pipe


def _plate_regional(pipe) -> bool:
    return isinstance(pipe.cross_procs[0], rt.RegionalTextProcessor)


@pytest.mark.usefixtures("gpu_fakes")
def test_the_default_plate_draws_with_per_region_text_and_no_pose(tmp_path):
    pipe = _render(tmp_path, "default")
    assert _plate_regional(pipe) is True
    assert pipe.calls[0]["controlnet_conditioning_scale"] == 0.0
    # Pass B is not the plate: it never reads per-region text, and the processors are back after the plate.
    assert all(not isinstance(p, rt.RegionalTextProcessor) for p in pipe.cross_procs[1:])
    assert pipe.unet.procs["down.attn2.processor"] == "cross"


@pytest.mark.usefixtures("gpu_fakes")
@pytest.mark.parametrize("kw", [{"plate_regional_text": False}, {"pose_conditioning": True},
                                {"pose_conditioning": False}, {"plate_count_retries": 0}])
def test_outside_the_cell_the_plate_draws_one_prompt_as_before(tmp_path, kw):
    assert _plate_regional(_render(tmp_path, "off", **kw)) is False


@pytest.mark.usefixtures("gpu_fakes")
def test_a_job_can_still_ask_for_it_with_the_pose_on(tmp_path):
    assert _plate_regional(_render(tmp_path, "pr", plate_regional_text=True, pose_conditioning=True)) is True


# ------------------------------------------------------------------ reuse hash, as in #589 and #597

# Computed with the code on main at 435e562, the commit before this change, by the same `_hash` call (KeyframeConfig
# defaults, the scene below, no storyboard). A stored still carries one of these today. A shot the change does not
# touch must keep its exact value, so pinned literals are the test, not a comparison of the new code with itself.
BEFORE_1281 = {
    "n0": "23b7e0c279e5ed2e",
    "n1": "23b7e0c279e5ed2e",
    "n2": "ee77025e4bf000ed",
    "n2_guard_off": "23b7e0c279e5ed2e",
    "n2_lock_off": "34f6f9b2d77ff564",
    "n2_pose_on": "bc5635c44b87c426",
    "n2_pose_off": "bd6bbde31373e1d3",
    "n2_text_on": "d0b6ec77bed12e18",
    "n2_text_on_pose_on": "1e12c610016b0c9c",
    "n1_text_on": "1e091d7aec0259c2",
}


def _hash(n, **kw):
    kc = KeyframeConfig()
    for k, v in kw.items():
        if k == "pose":
            kc.multi_char.pose_conditioning = v
        else:
            setattr(kc, k, v)
    return kf_hash(kc, scene=_shot(n))


@pytest.mark.parametrize("case,kw", [
    ("n0", {}), ("n1", {}), ("n2_guard_off", {"plate_count_retries": 0}), ("n2_lock_off", {"scene_lock": False}),
    ("n2_pose_on", {"pose": True}), ("n2_pose_off", {"pose": False}), ("n2_text_on", {"plate_regional_text": True}),
    ("n2_text_on_pose_on", {"plate_regional_text": True, "pose": True}), ("n1_text_on", {"plate_regional_text": True}),
])
def test_a_still_the_change_does_not_touch_keeps_its_stored_hash(case, kw):
    n = int(case[1])
    assert _hash(n, **kw) == BEFORE_1281[case]


def test_explicit_off_draws_and_hashes_exactly_as_the_old_default():
    assert _hash(2, plate_regional_text=False) == BEFORE_1281["n2"]


def test_a_two_person_guarded_scene_lock_still_with_the_pose_unset_is_redrawn():
    assert _hash(2) != BEFORE_1281["n2"]
    # Unset and explicitly on draw alike but are different requests.
    assert _hash(2) != _hash(2, plate_regional_text=True)
