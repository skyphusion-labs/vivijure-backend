"""vivijure-cf#1442: `keyframe.pass_b_gaze` lets Pass B say where the eyes go on a two-person scene-lock shot.

Under per-region plate text the plate holds the action (heads and eyes down at the work), and Pass B turns the faces
to the lens: it draws each face through an identity adapter and cast LoRA trained on head-on portraits, and its
prompt says nothing about the eyes (cf#1281 comments 6687630211 and 6840340795). `auto` reads the object of the shot
text's own "looking at <X>" and adds a gaze clause to Pass B only. `off` is the default and must leave the prompt and
every reuse hash byte-identical, so no stored still redraws."""
import hashlib

import pytest
from test_debug_intermediates_1143 import CAST, SB, _scene

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, QualityTier
from vivijure_backend.contract import RenderRequest, Scene
from vivijure_backend.orchestrator import kf_hash
from vivijure_backend.pipeline import keyframe_params_from

# Measured on main 852eac6, before pass_b_gaze existed: kf_hash(config only), kf_hash(two-person shot),
# kf_hash(one-person shot), for the fixtures below.
HASH_BEFORE = {
    QualityTier.STANDARD: ("ee77025e4bf000ed", "1ec3ce1c7e4e931e", "cdec19e71d2cf1ab"),
    QualityTier.DRAFT: ("bc7df897df3558e8", "fdf2bc5ed129fc39", "a8bb50490883fd3f"),
}
# sha256[:16] of the plate and Pass B prompts for the two-person shot on main 852eac6.
PLATE_BEFORE, PASS_B_BEFORE = "28bc33762121e6b8", "7e43ac76322b35ce"


def _solo():
    return Scene.from_dict({"id": "shot_03", "prompt": "Mara at a workbench, looking at a laptop.",
                            "character_slots": ["A"]}, 0)


def _shot(prompt, slots=("A", "B")):
    return Scene.from_dict({"id": "shot_03", "prompt": prompt, "character_slots": list(slots)}, 0)


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _passes(mode, scene=None):
    p = kf.KeyframeParams(identity_method="ip_adapter", plate_regional_text=True, pass_b_gaze=mode)
    return kf.plan_keyframe_passes(scene or _scene(("A", "B")), CAST, SB, p)


# --- off: byte for byte what main drew ---------------------------------------------------------------------

@pytest.mark.parametrize("tier", list(HASH_BEFORE))
def test_off_keeps_every_hash_from_before_the_knob(tier):
    kc = KeyframeConfig.for_tier(tier)
    assert kc.pass_b_gaze == "off"
    got = (kf_hash(kc), kf_hash(kc, _scene(("A", "B")), CAST, SB), kf_hash(kc, _solo(), CAST, SB))
    assert got == HASH_BEFORE[tier]
    kc.pass_b_gaze = "off"
    assert (kf_hash(kc), kf_hash(kc, _scene(("A", "B")), CAST, SB), kf_hash(kc, _solo(), CAST, SB)) == got


def test_off_keeps_both_prompts_byte_identical():
    plate, ident = _passes("off")
    assert (_sha(plate.prompt), _sha(ident.prompt)) == (PLATE_BEFORE, PASS_B_BEFORE)
    assert kf.KeyframeParams().pass_b_gaze == "off"


# --- auto: Pass B only ---------------------------------------------------------------------------------------

def test_auto_adds_the_clause_to_pass_b_only():
    plate, ident = _passes("auto")
    assert _sha(plate.prompt) == PLATE_BEFORE
    want = "looking down at the laptop, eyes on the laptop, not at the camera"
    assert want in ident.prompt
    assert ident.prompt.index(kf.PASS_B_LOCK_CLAUSE) < ident.prompt.index(want)
    off = _passes("off")[1].prompt
    assert ident.prompt.replace(want + ", ", "") == off


def test_auto_changes_the_two_person_hash_and_only_says_so_where_set():
    kc = KeyframeConfig.for_tier(QualityTier.STANDARD)
    kc.pass_b_gaze = "auto"
    assert kf_hash(kc, _scene(("A", "B")), CAST, SB) != HASH_BEFORE[QualityTier.STANDARD][1]
    assert kf_hash(kc) != HASH_BEFORE[QualityTier.STANDARD][0]
    kc.scene_lock = False  # no Pass B to carry it: the built prompt is the old one, only the mode key moves
    off = KeyframeConfig.for_tier(QualityTier.STANDARD)
    off.scene_lock = False
    assert kf_hash(kc, _scene(("A", "B")), CAST, SB) != kf_hash(off, _scene(("A", "B")), CAST, SB)


@pytest.mark.parametrize("prompt,thing", [
    ("Mara and Dev stand side by side at a workbench in the dim workshop, looking at a laptop.", "laptop"),
    ("Mara and Dev lean over the table, looking down at the map and talking.", "map"),
    ("Two engineers looking at their notes while the kettle boils", "notes"),
    ("Mara and Dev looking at a laptop on the bench", "laptop"),
    ("Mara and Dev, looking at the old circuit board.", "old circuit board"),
])
def test_auto_reads_the_object_from_the_shot_text(prompt, thing):
    got = kf.pass_b_gaze_clause(_shot(prompt), "auto", 2, CAST)
    assert got == f"looking down at the {thing}, eyes on the {thing}, not at the camera"


@pytest.mark.parametrize("prompt", [
    "Mara and Dev at a workbench.",                          # no gaze phrase
    "Mara and Dev looking at each other across the bench.",  # at a person
    "Mara looking at Dev across the bench.",                 # at a cast member, by name
    "Mara and Dev looking at the camera.",                   # at the lens
    "Mara and Dev looking at them.",                         # pronoun
])
def test_auto_says_nothing_without_a_thing_to_look_at(prompt):
    assert kf.pass_b_gaze_clause(_shot(prompt), "auto", 2, CAST) == ""


def test_auto_says_nothing_on_a_one_person_or_over_cap_shot_or_when_off():
    p = "Mara at a workbench, looking at a laptop."
    assert kf.pass_b_gaze_clause(_shot(p, ("A",)), "auto", 2, CAST) == ""
    assert kf.pass_b_gaze_clause(_shot(p, ("A", "B", "C")), "auto", 2, CAST) == ""
    assert kf.pass_b_gaze_clause(_shot(p), "off", 2, CAST) == ""


# --- the wire ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,want", [("auto", "auto"), ("off", "off"), ("on", "off"), (True, "off"),
                                        (None, "off"), ("AUTO", "off")])
def test_only_a_known_mode_is_read(value, want):
    assert KeyframeConfig.from_dict({"pass_b_gaze": value}).pass_b_gaze == want
    assert KeyframeConfig.from_dict({}).pass_b_gaze == "off"


def test_request_reaches_the_engine_params():
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                   "render_overrides": {"keyframe": {"pass_b_gaze": "auto"}}})
    assert keyframe_params_from(req.config).pass_b_gaze == "auto"
