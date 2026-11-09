"""vivijure-cf#1143 and #1145: the keyframe prompt's budget, its adult clause, and the film setting.

Pure, on CPU. The chunked encoder's torch half is validated on a pod; its split is tested here."""
import itertools

import pytest

from vivijure_backend.contract import Cast, Scene, Storyboard
from vivijure_backend.keyframe import (
    PROMPT_TOKEN_BUDGET,
    PROMPT_TOKEN_MARGIN,
    adult_clause,
    build_prompt,
    clause_rank,
    estimate_tokens,
    fit_description,
    plan_keyframe_passes,
    setting_for,
    slot_trigger,
    KeyframeParams,
)
from vivijure_backend.prompt_chunks import CLIP_CHUNK_TOKENS, PROMPT_CHUNKS, chunk_token_ids, frame_chunk

MARA = "Mara, an adult woman in her thirties, dark shoulder-length curls, sharp brown eyes, charcoal field jacket over a black top, calm presence."
DEV = "Dev, an adult man in his thirties, dark buzz cut, light stubble, grey zip hoodie, lean and alert."
REAL = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": MARA}, "B": {"name": "Dev", "prompt": DEV}}})
STYLE = "photoreal, cinematic film still, natural skin texture, 35mm,"


def _sb(**extra):
    return Storyboard.from_dict({"style_prefix": STYLE, "use_characters": ["A", "B"],
                                 "scenes": [{"id": "s", "prompt": "x"}], **extra})


def _scene(prompt, slots):
    return Scene.from_dict({"id": "s", "prompt": prompt, "character_slots": slots}, 0)


# --------------------------------------------------------------------------- the budget (#1143 option 1)

def test_the_budget_is_two_chunks():
    assert PROMPT_TOKEN_BUDGET == CLIP_CHUNK_TOKENS * PROMPT_CHUNKS == 150


def test_the_measured_two_shot_keeps_both_hair_and_both_garments_in_the_identity_pass():
    # film-076828c5 shot_03: the identity pass lost Mara's curls and Dev's hoodie at 75 tokens.
    sc = _scene("Mara and Dev stand side by side at a workbench in the dim workshop, looking at a laptop.", ["A", "B"])
    plate, ident = plan_keyframe_passes(sc, REAL, _sb(), KeyframeParams())
    for p in (plate.prompt, ident.prompt):
        assert "dark shoulder-length curls" in p
        assert "charcoal jacket" in p
        assert "dark buzz cut" in p
        assert "grey zip hoodie" in p
        assert estimate_tokens(p) <= PROMPT_TOKEN_BUDGET


def test_a_two_person_description_ranks_its_age_after_the_look():
    assert clause_rank("an adult woman in her thirties") == 0
    assert clause_rank("an adult woman in her thirties", adult_stated=True) == 2
    squeezed = fit_description(MARA, 10, name="Mara", adult_stated=True)
    assert "dark shoulder-length curls" in squeezed and "adult" not in squeezed


# --------------------------------------------------------------------------- the adult clause (#1143 option 3)

@pytest.mark.parametrize("names,named,want", [
    (["Mara"], True, "Mara, an adult"),
    (["Mara", "Dev"], True, "Mara and Dev, both adults"),
    (["Mara", "Dev", "Kai"], True, "Mara, Dev and Kai, all adults"),
    (["Mara"], False, "an adult"),
    (["Mara", "Dev"], False, "two adults"),
    (["A", "B", "C", "D"], False, "four adults"),
    ([], True, ""),
    ([], False, ""),
])
def test_adult_clause(names, named, want):
    assert adult_clause(names, named=named) == want


NO_AGE = Cast.from_registry({"characters": {
    "A": {"name": "Mara", "prompt": "dark curls, charcoal jacket"},
    "B": {"name": "Dev", "prompt": "buzz cut, grey hoodie"},
    "C": {"name": "Kai", "prompt": ""},
}})
STYLES = ["", STYLE, "x " * 120]                    # none, real, and past the 256-char cap
SCENES = ["walks in", "Mara and Dev talk at the bench in the workshop", "word " * 60, "word " * 200]
CASTS = [REAL, NO_AGE]
SLOTS = [["A"], ["B"], ["A", "B"], ["A", "B", "C"]]


@pytest.mark.parametrize("style,scene,cast,slots,ident", list(itertools.product(STYLES, SCENES, CASTS, SLOTS, [True, False])))
def test_no_person_can_lose_the_adult_statement(style, scene, cast, slots, ident):
    """The safety clause. For every shot shape, including descriptions with no age, a style and a scene past
    their caps, and three people: the prompt says everyone in it is an adult, and says it inside the budget,
    so the encoder can never be the thing that cuts it."""
    sb = Storyboard.from_dict({"style_prefix": style, "scenes": [{"id": "s", "prompt": "x"}]})
    sc = Scene.from_dict({"id": "s", "prompt": scene, "character_slots": slots}, 0)
    names = [slot_trigger(cast, s) for s in slots]  # a slot missing from the cast is named by its id
    p = build_prompt(sc, cast, sb, include_triggers=ident, lock_clause=ident)
    clause = adult_clause(names, named=ident)
    assert clause and clause in p
    assert estimate_tokens(p[: p.index(clause) + len(clause)]) <= PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN
    if ident:
        for n in names:
            assert n in clause


def test_an_empty_shot_has_no_adult_clause():
    p = build_prompt(_scene("an empty workshop", []), REAL, _sb())
    assert "adult" not in p


# --------------------------------------------------------------------------- the setting (#1145)

def test_a_shot_with_no_place_gets_the_setting():
    sb = _sb(setting="a dim workshop at night")
    sc = _scene("Mara wipes grease from her hands with a rag and frowns at a gauge.", ["A"])
    for p in plan_keyframe_passes(sc, REAL, sb, KeyframeParams()):
        assert "a dim workshop at night" in p.prompt, p.name


def test_the_setting_comes_right_after_the_scene():
    sb = _sb(setting="a dim workshop at night")
    sc = _scene("Mara frowns at a gauge.", ["A"])
    p = build_prompt(sc, REAL, sb, include_triggers=True, lock_clause=True)
    assert p.index("frowns at a gauge") < p.index("a dim workshop at night") < p.index("Mara, an adult")


def test_a_shot_that_says_the_setting_does_not_say_it_twice():
    sb = _sb(setting="a dim workshop at night")
    sc = _scene("Mara walks into A Dim Workshop At Night.", ["A"])
    assert build_prompt(sc, REAL, sb).lower().count("a dim workshop at night") == 1


def test_a_style_prefix_that_says_the_setting_is_enough():
    sb = _sb(style_prefix="photoreal, a dim workshop at night,", setting="a dim workshop at night")
    assert setting_for(_scene("Mara frowns.", ["A"]), sb) == ""


def test_no_setting_leaves_the_prompt_exactly_as_before():
    sc = _scene("Mara frowns at a gauge.", ["A"])
    for ident in (True, False):
        assert build_prompt(sc, REAL, _sb(), include_triggers=ident) == build_prompt(sc, REAL, _sb(setting=""), include_triggers=ident)
        assert build_prompt(sc, REAL, _sb(), include_triggers=ident) == build_prompt(sc, REAL, _sb(setting="   "), include_triggers=ident)


def test_the_setting_never_costs_a_person_their_look():
    # A shot that fills the budget: the descriptions stay, the setting goes.
    sb = _sb(setting="a dim workshop at night with generators")
    long_scene = "Mara " + "word " * 85
    p = build_prompt(_scene(long_scene, ["A"]), REAL, sb, include_triggers=True, lock_clause=True)
    assert "dark shoulder-length curls" in p
    assert "generators" not in p


def test_an_empty_shot_gets_the_setting_too():
    p = build_prompt(_scene("a wide shot of tools on a bench", []), REAL, _sb(setting="a dim workshop at night"))
    assert "a dim workshop at night" in p


def test_the_setting_is_read_from_the_board_and_trimmed():
    assert Storyboard.from_dict({"setting": "  a hilltop  ", "scenes": [{"prompt": "x"}]}).setting == "a hilltop"
    assert Storyboard.from_dict({"scenes": [{"prompt": "x"}]}).setting == ""


# --------------------------------------------------------------------------- the chunk split (#1143 option 1)

COMMA = 9


def test_a_short_prompt_is_one_chunk():
    chunks, dropped = chunk_token_ids(list(range(100, 140)), COMMA)
    assert chunks == [list(range(100, 140))] and dropped == 0


def test_an_empty_prompt_is_one_empty_chunk():
    assert chunk_token_ids([], COMMA) == ([[]], 0)


def test_chunks_never_exceed_75_and_lose_nothing_inside_the_limit():
    ids = list(range(100, 250))  # 150, no commas
    chunks, dropped = chunk_token_ids(ids, COMMA)
    assert [len(c) for c in chunks] == [75, 75] and dropped == 0
    assert sum(chunks, []) == ids


def test_a_split_prefers_the_last_comma_near_the_boundary():
    ids = list(range(100, 170)) + [COMMA] + list(range(200, 230))  # comma at index 70
    chunks, _ = chunk_token_ids(ids, COMMA)
    assert chunks[0][-1] == COMMA and len(chunks[0]) == 71
    assert sum(chunks, []) == ids


def test_a_comma_too_far_back_is_not_used():
    ids = [COMMA] + list(range(100, 200))  # comma at index 0, outside the 20-token window
    chunks, _ = chunk_token_ids(ids, COMMA)
    assert len(chunks[0]) == 75


def test_past_the_limit_the_rest_is_counted_as_dropped():
    chunks, dropped = chunk_token_ids(list(range(100, 300)), COMMA)  # 200 tokens
    assert len(chunks) == PROMPT_CHUNKS and dropped == 200 - 150


def test_a_framed_chunk_is_77_tokens():
    row = frame_chunk([5, 6, 7], bos=1, eos=2, pad=0)
    assert row[:5] == [1, 5, 6, 7, 2] and len(row) == 77 and set(row[5:]) == {0}
    with pytest.raises(ValueError):
        frame_chunk(list(range(76)), 1, 2, 0)


# --------------------------------------------------------------------------- every pipe call is chunked

def test_no_pipe_call_hands_sdxl_a_raw_prompt():
    """A pipe given `prompt=` or `negative_prompt=` truncates at 77 tokens again, silently. Every SDXL
    call in keyframe.py must take the chunked embeds instead. Read from the syntax tree, not by text, so a
    comment or a docstring cannot satisfy or break it."""
    import ast
    from pathlib import Path

    import vivijure_backend.keyframe as kf

    tree = ast.parse(Path(kf.__file__).read_text())
    raw, chunked = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kws = {k.arg for k in node.keywords}
        if kws & {"prompt", "negative_prompt"} and "num_inference_steps" in kws:
            raw.append(node.lineno)
        if "num_inference_steps" in kws and any(
                k.arg is None and isinstance(k.value, ast.Call) and getattr(k.value.func, "id", "") == "encode_long_prompt"
                for k in node.keywords):
            chunked += 1
    assert raw == []
    assert chunked == 6  # plate, Pass B single, Pass B regional, single, regional, InstantID
