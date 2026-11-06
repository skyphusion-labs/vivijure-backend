"""vivijure-cf#1281: under per-region plate text, the plate count guard also redraws a plate short of a person.

Measured on vivijure-cf#1281 (hand board, 6 seeds, through the studio): with `plate_regional_text` on, 4 of 6
stills held ONE person, from plates the guard read as 1 face, and the guard passed them because it counts extras
only. Without per-region text the same board read 1 face on 1 of 6. So the floor rides with per-region text: on
every other path a one-face plate can be a person facing away, and the guard still accepts it there.

Render tests run the real scene-lock body over the fakes from the #1143 tests, with the cross-attention install
stubbed (the floor is about which plate is chosen, not how a region is masked).
"""
import contextlib

import pytest
from test_debug_intermediates_1143 import (  # noqa: F401
    CAST,
    SB,
    _scene,
    gpu_fakes,  # the fixture, by name
)
from test_plate_count_guard_1143 import SH, SW, _CountingServer, _DistinctPipe

import vivijure_backend.keyframe as kf
from vivijure_backend import regional_text
from vivijure_backend.config import KeyframeConfig, QualityTier
from vivijure_backend.orchestrator import kf_hash

# kf_hash of KeyframeConfig.for_tier(tier) on main 57a0f27, before the floor: the default, per-region text on,
# and per-region text on with the guard off.
HASH_MAIN = {
    QualityTier.DRAFT: {"default": "bc7df897df3558e8", "R": "6702f358e484f059", "R_noguard": "b7273c16479f9f91"},
    QualityTier.STANDARD: {"default": "ee77025e4bf000ed", "R": "d0b6ec77bed12e18", "R_noguard": "1e091d7aec0259c2"},
}


# ------------------------------------------------------------------ pure rules

@pytest.mark.parametrize("faces,expected,floor,want", [
    (3, 2, False, True), (2, 2, False, False), (1, 2, False, False), (0, 2, False, False),
    (3, 2, True, True), (2, 2, True, False), (1, 2, True, True), (0, 2, True, True),
])
def test_needs_redraw(faces, expected, floor, want):
    assert kf.plate_needs_redraw(faces, expected, floor=floor) is want


@pytest.mark.parametrize("counts,want", [
    ([2], 0), ([1, 2], 1), ([1, 1, 2], 2), ([3, 2], 1),
    ([1, 1, 1, 1], 0),   # never exact: the closest from below, earliest
    ([1, 3, 1, 1], 0),   # a missing person beats an extra one
    ([3, 1], 1),
    ([0, 1, 0], 1),      # the most faces not over expected
    ([3, 4, 3], 0),      # nothing at or under: the fewest, earliest
])
def test_choose_plate_under_the_floor(counts, want):
    assert kf.choose_plate(counts, 2, floor=True) == want


@pytest.mark.parametrize("counts,want", [([1], 0), ([1, 2], 0), ([3, 1], 1), ([3, 3], 0)])
def test_choose_plate_without_the_floor_is_unchanged(counts, want):
    assert kf.choose_plate(counts, 2) == want


def test_record_under_the_floor_is_honest_about_a_short_plate():
    assert kf.plate_count_record([1, 1], 2, 1, floor=True) == {
        "expected": 2, "faces": [1, 1], "chosen": 0, "clean": False, "retries": 1, "floor": True}
    assert kf.plate_count_record([1, 2], 2, 3, floor=True)["clean"] is True


def test_record_without_the_floor_keeps_its_shape():
    assert kf.plate_count_record([1], 2, 3) == {"expected": 2, "faces": [1], "chosen": 0, "clean": True,
                                                "retries": 3}


@pytest.mark.parametrize("slots,text,retries,want", [
    (("A", "B"), True, 3, True),
    (("A", "B"), False, 3, False),   # no per-region text: a one-face plate may be a person facing away
    (("A", "B"), True, 0, False),    # no guard, no floor
    (("A",), True, 3, False),        # one person: no regional plate
])
def test_floor_rides_with_per_region_text_and_the_guard(slots, text, retries, want):
    params = kf.KeyframeParams(plate_regional_text=text, plate_count_retries=retries)
    assert kf.plate_face_floor_on(_scene(slots), params) is want


# ------------------------------------------------------------------ render

def _run(tmp_path, monkeypatch, name, *, text, counts, retries=3, seed=130601):
    monkeypatch.setattr(regional_text, "applied", lambda *a, **k: contextlib.nullcontext())
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", seed=seed,
                               plate_count_retries=retries, plate_regional_text=text)
    pipe = _DistinctPipe(SW, SH)
    server = _CountingServer(pipe, counts)
    res = kf.render_keyframe(_scene(("A", "B")), CAST, SB, server, tmp_path / name / "shot_01.png", params=params,
                             lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")})
    return res, pipe


@pytest.mark.usefixtures("gpu_fakes")
def test_a_short_plate_is_redrawn_under_per_region_text(tmp_path, monkeypatch):
    res, pipe = _run(tmp_path, monkeypatch, "short", text=True, counts=[1, 1, 2])
    assert len(pipe.calls) == 4  # three plates, then Pass B once
    assert [c["generator"].seed for c in pipe.calls[1:3]] == [kf.plate_retry_seed(130601, 1),
                                                              kf.plate_retry_seed(130601, 2)]
    assert res.plate_count == {"expected": 2, "faces": [1, 1, 2], "chosen": 2, "clean": True, "retries": 3,
                               "floor": True}
    assert pipe.calls[3]["image"].tobytes() == kf._canny_preprocess(pipe.outputs[2]).tobytes()


@pytest.mark.usefixtures("gpu_fakes")
def test_a_plate_never_whole_ships_the_closest_and_says_so(tmp_path, monkeypatch):
    res, pipe = _run(tmp_path, monkeypatch, "never", text=True, counts=[1, 3, 1, 1])
    assert len(pipe.calls) == 5  # bounded at retries
    assert res.plate_count == {"expected": 2, "faces": [1, 3, 1, 1], "chosen": 0, "clean": False, "retries": 3,
                               "floor": True}


@pytest.mark.usefixtures("gpu_fakes")
def test_without_per_region_text_a_short_plate_is_still_accepted(tmp_path, monkeypatch):
    res, pipe = _run(tmp_path, monkeypatch, "plain", text=False, counts=[1])
    assert len(pipe.calls) == 2  # one plate, Pass B
    assert res.plate_count == {"expected": 2, "faces": [1], "chosen": 0, "clean": True, "retries": 3}


# ------------------------------------------------------------------ reuse hash

@pytest.mark.parametrize("tier", list(HASH_MAIN))
def test_only_per_region_text_stills_with_the_guard_redraw(tier):
    kc = KeyframeConfig.for_tier(tier)
    assert kf_hash(kc) == HASH_MAIN[tier]["default"]
    kc = KeyframeConfig.for_tier(tier)
    kc.plate_regional_text = True
    assert kf_hash(kc) != HASH_MAIN[tier]["R"]  # a per-region text still drawn before the floor redraws
    kc = KeyframeConfig.for_tier(tier)
    kc.plate_regional_text = True
    kc.plate_count_retries = 0
    assert kf_hash(kc) == HASH_MAIN[tier]["R_noguard"]  # no guard, no floor, same drawing
