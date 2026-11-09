"""vivijure-backend#573: a stored keyframe is reused only when nothing that draws it has changed.

On 1.1.6, film-0afa679b re-rendered a 1.1.5 board with the same params and got the 1.1.5 stills back as
`reused: true`, because the hash covered the params and nothing else."""
import vivijure_backend.orchestrator as orch
from vivijure_backend.contract import Cast, RenderRequest, Scene, Storyboard
from vivijure_backend.orchestrator import KeyframeMode, kf_hash, plan

CAST = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": "dark curls, charcoal jacket"}}})


def _req():
    return RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz"})


def _sb(prompt="Mara frowns at a gauge.", setting=""):
    return Storyboard.from_dict({"title": "p", "setting": setting,
                                 "scenes": [{"id": "shot_01", "prompt": prompt, "character_slots": ["A"]}]})


def _h(sb, cast=CAST):
    return kf_hash(_req().config.keyframe, scene=sb.scenes[0], cast=cast, storyboard=sb)


def test_the_same_shot_hashes_the_same():
    assert _h(_sb()) == _h(_sb())


def test_a_new_engine_version_changes_the_hash(monkeypatch):
    before = _h(_sb())
    monkeypatch.setattr(orch, "KEYFRAME_ENGINE_VERSION", "kf-next")
    assert _h(_sb()) != before


def test_new_prompt_building_changes_the_hash(monkeypatch):
    """The 1.1.5 -> 1.1.6 case: same params, same board, different code building the prompt."""
    import vivijure_backend.keyframe as kf
    before = _h(_sb())
    real = kf.adult_clause
    monkeypatch.setattr(kf, "adult_clause", lambda names, named: real(names, named=named) + ", grown")
    assert _h(_sb()) != before


def test_an_edited_shot_changes_the_hash():
    assert _h(_sb("Mara frowns at a gauge.")) != _h(_sb("Mara smiles at a gauge."))


def test_a_changed_description_changes_the_hash():
    other = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": "red hair, green coat"}}})
    assert _h(_sb(), CAST) != _h(_sb(), other)


def test_a_new_setting_changes_the_hash():
    assert _h(_sb()) != _h(_sb(setting="a dim workshop at night"))


def test_params_still_change_it():
    req = _req()
    sb = _sb()
    a = kf_hash(req.config.keyframe, scene=sb.scenes[0], cast=CAST, storyboard=sb)
    req.config.keyframe.seed = req.config.keyframe.seed + 1
    assert kf_hash(req.config.keyframe, scene=sb.scenes[0], cast=CAST, storyboard=sb) != a


def test_a_still_from_older_code_is_redrawn(monkeypatch):
    sb = _sb()
    monkeypatch.setattr(orch, "KEYFRAME_ENGINE_VERSION", "kf-old")
    stored = _h(sb)
    monkeypatch.undo()
    p = plan(_req(), sb, existing_keyframes={"shot_01": stored}, cast=CAST)
    assert p.scenes[0].keyframe_mode is KeyframeMode.GENERATE


def test_a_still_drawn_for_this_exact_shot_is_reused():
    sb = _sb()
    p = plan(_req(), sb, existing_keyframes={"shot_01": _h(sb)}, cast=CAST)
    assert p.scenes[0].keyframe_mode is KeyframeMode.REUSE


def test_an_edited_board_in_the_same_project_redraws():
    stored = _h(_sb("Mara frowns at a gauge."))
    p = plan(_req(), _sb("Mara smiles at a gauge."), existing_keyframes={"shot_01": stored}, cast=CAST)
    assert p.scenes[0].keyframe_mode is KeyframeMode.GENERATE


def test_a_reader_that_lost_the_cast_redraws_rather_than_reuses():
    """The safe direction: if the planner and the writer ever disagree on the inputs, the hashes differ."""
    sb = _sb()
    p = plan(_req(), sb, existing_keyframes={"shot_01": _h(sb)})  # no cast passed
    assert p.scenes[0].keyframe_mode is KeyframeMode.GENERATE


def test_a_plate_only_change_changes_the_hash(monkeypatch):
    """The plate is its own prompt: a change only it carries (the "solo" token) must still redraw."""
    import vivijure_backend.keyframe as kf
    before = _h(_sb())
    monkeypatch.setattr(kf, "PLATE_SOLO", "alone")
    assert _h(_sb()) != before
