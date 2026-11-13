"""vivijure-cf#1442: `keyframe.plate_gaze` asks the two-person plate to turn each head toward what the shot names.

Measured on the live endpoint first: under per-region plate text no Pass B knob moved the lens gaze on the workbench
board (Pass B text via pass_b_gaze, the cast LoRA at 0.2 and 0.1, canny 0.9, identity scale 0.5). The eyes follow the
plate's head-on outline. So `auto` adds a head-direction clause to each person's own region prompt in Pass A. The
shared prompt, the background band and Pass B are never changed. `off` is the default and must leave every prompt and
every reuse hash byte-identical."""
import hashlib

import pytest
from test_debug_intermediates_1143 import CAST, SB, _scene
from test_debug_intermediates_1143 import gpu_fakes  # noqa: F401  the scene-lock fakes

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, QualityTier
from vivijure_backend.contract import RenderRequest, Scene
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.pipeline import keyframe_params_from

# Measured on main cca5a58 (backend 1.1.22), before plate_gaze existed, for the fixture two-person shot.
SHARED_BEFORE = "1ed47acffb023cde"
PERSONS_BEFORE = ["5598660de42ea361", "ed1619d71254334f"]
PASS_B_BEFORE = "7e43ac76322b35ce"
# kf_hash with plate_regional_text on: (config only, two-person shot).
HASH_BEFORE = {QualityTier.STANDARD: ("af360e2ad2301bd9", "a14e5249f170dc1a"),
               QualityTier.DRAFT: ("3edebd64c948f63c", "a7cb8c68122871e7")}
CLAUSE = "head turned down toward the laptop, three-quarter view, eyes on the laptop"


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _shot(prompt, slots=("A", "B")):
    return Scene.from_dict({"id": "shot_03", "prompt": prompt, "character_slots": list(slots)}, 0)


# --- off: byte for byte what main drew ---------------------------------------------------------------------

def test_off_keeps_the_region_prompts_byte_identical():
    shared, persons = kf.plate_region_prompts(_scene(("A", "B")), CAST, SB)
    assert _sha(shared) == SHARED_BEFORE and [_sha(p) for p in persons] == PERSONS_BEFORE
    shared, persons = kf.plate_region_prompts(_scene(("A", "B")), CAST, SB, gaze="")
    assert _sha(shared) == SHARED_BEFORE and [_sha(p) for p in persons] == PERSONS_BEFORE
    assert kf.plate_gaze_clause(_scene(("A", "B")), "off", 2, CAST) == ""
    assert kf.KeyframeParams().plate_gaze == "off"


@pytest.mark.parametrize("tier", list(HASH_BEFORE))
def test_off_keeps_every_hash_from_before_the_knob(tier):
    kc = KeyframeConfig.for_tier(tier)
    kc.plate_regional_text = True
    assert kc.plate_gaze == "off"
    assert (kf_hash(kc), kf_hash(kc, _scene(("A", "B")), CAST, SB)) == HASH_BEFORE[tier]


# --- auto: each person's region, nothing else -----------------------------------------------------------------

def test_auto_adds_the_clause_to_each_person_region_only():
    clause = kf.plate_gaze_clause(_scene(("A", "B")), "auto", 2, CAST)
    assert clause == CLAUSE
    shared, persons = kf.plate_region_prompts(_scene(("A", "B")), CAST, SB, gaze=clause)
    assert _sha(shared) == SHARED_BEFORE  # the shared prompt (and so the background) never carries it
    _, before = kf.plate_region_prompts(_scene(("A", "B")), CAST, SB)
    for p, b in zip(persons, before):
        assert p.startswith(shared + ", " + CLAUSE)  # right after the shared text, ahead of the description
        assert p.replace(", " + CLAUSE, "") == b     # the description is unchanged when there is room


def test_pass_b_is_untouched_by_plate_gaze():
    for mode in ("off", "auto"):
        p = kf.KeyframeParams(identity_method="ip_adapter", plate_regional_text=True, plate_gaze=mode)
        assert _sha(kf.plan_keyframe_passes(_scene(("A", "B")), CAST, SB, p)[1].prompt) == PASS_B_BEFORE


def test_auto_changes_the_hash():
    kc = KeyframeConfig.for_tier(QualityTier.STANDARD)
    kc.plate_regional_text = True
    kc.plate_gaze = "auto"
    assert kf_hash(kc) != HASH_BEFORE[QualityTier.STANDARD][0]
    assert kf_hash(kc, _scene(("A", "B")), CAST, SB) != HASH_BEFORE[QualityTier.STANDARD][1]


def test_the_plate_reads_the_same_object_as_pass_b():
    for prompt in ("Mara and Dev lean over the table, looking down at the map and talking.",
                   "Mara and Dev looking at each other.", "Mara and Dev at a workbench."):
        sc = _shot(prompt)
        thing = kf.gaze_object(sc, 2, CAST)
        assert bool(kf.plate_gaze_clause(sc, "auto", 2, CAST)) == bool(thing) == bool(
            kf.pass_b_gaze_clause(sc, "auto", 2, CAST))
    assert kf.plate_gaze_clause(_shot("Mara at a bench, looking at a laptop.", ("A",)), "auto", 2, CAST) == ""


def test_the_description_is_fitted_after_the_clause_and_stays_in_budget():
    long = ("Mara and Dev stand side by side at a long cluttered workbench in the dim workshop at night, rain on the "
            "windows, tools everywhere, looking at a laptop.")
    sc = _shot(long)
    clause = kf.plate_gaze_clause(sc, "auto", 2, CAST)
    _, persons = kf.plate_region_prompts(sc, CAST, SB, gaze=clause)
    for p in persons:
        assert clause in p
        assert kf.estimate_tokens(p) <= kf.PROMPT_TOKEN_BUDGET


# --- the render path hands it to the plate -------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.mark.parametrize("mode,want", [("auto", CLAUSE), ("off", "")])
def test_the_scene_lock_plate_passes_the_clause(tmp_path, monkeypatch, gpu_fakes, mode, want):  # noqa: F811
    from test_debug_intermediates_1143 import _Pipe, _Server
    seen = {}

    def spy(scene, cast, storyboard, *, gaze=""):
        seen["gaze"] = gaze
        raise _Stop

    monkeypatch.setattr(kf, "plate_region_prompts", spy)
    params = kf.KeyframeParams(width=448, height=256, identity_method="ip_adapter", plate_regional_text=True,
                               plate_gaze=mode, plate_count_retries=0)
    with pytest.raises(_Stop):
        kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(_Pipe(448, 256)), tmp_path / "shot_03.png",
                           params=params, lora_paths={})
    assert seen["gaze"] == want


# --- the wire ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,want", [("auto", "auto"), ("off", "off"), ("on", "off"), (True, "off"),
                                        (None, "off"), ("AUTO", "off")])
def test_only_a_known_mode_is_read(value, want):
    assert KeyframeConfig.from_dict({"plate_gaze": value}).plate_gaze == want


def test_request_reaches_the_engine_params():
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                   "render_overrides": {"keyframe": {"plate_gaze": "auto"}}})
    assert keyframe_params_from(req.config).plate_gaze == "auto"
