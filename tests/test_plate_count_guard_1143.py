"""vivijure-cf#1143: the plate count guard (`keyframe.plate_count_retries`).

The 1.1.11 and 1.1.12 runs showed Pass A draws a third person at about half of seeds and Pass B never adds one. The
guard counts faces in the plate with the analyser the path already loads and redraws ONLY the plate, with a derived
seed, while it holds more faces than the shot has people. Tested here, with the real scene-lock body over a fake
pipe, fake torch and a scripted face analyser: off makes no count and draws exactly as before; a clean first plate
draws exactly the still the guard-off path draws; a dirty plate is redrawn with the derived seeds and Pass B gets the
chosen draw; no clean draw ships the fewest-faces draw with `clean: false`; one-person shots are never counted; and
the record reaches the result. What CPU cannot see is the real detector, which was measured offline on 19 plates."""
import pytest
from PIL import Image
from test_debug_intermediates_1143 import (  # noqa: F401  the scene-lock fakes
    CAST,
    SB,
    FakeStore,
    _bundle_tar,
    _call_fingerprint,
    _Gen,
    _Pipe,
    _req,
    _scene,
    _Server,
    gpu_fakes,
)

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, QualityTier
from vivijure_backend.contract import Keyframe, RenderResult
from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import Outputs, run_job
from vivijure_backend.orchestrator import kf_hash

SW, SH = 448, 256


class _DistinctPipe(_Pipe):
    """Each draw has its own shape (a square at a different place), so their canny maps differ too: a solid colour
    with one shared square gives the same edges whatever the colour, which would hide a wrong plate choice."""
    def __call__(self, **kw):
        out = super().__call__(**kw)
        i = len(self.outputs) - 1
        img = out.images[0]
        img.paste((255, 255, 255), (8 + 40 * i, 8, 30 + 40 * i, 30))
        return out


class _Analyzer:
    """Returns the scripted face count for each successive image it is asked about."""
    def __init__(self, counts):
        self.counts = list(counts)
        self.seen = 0

    def get(self, arr):
        n = self.counts[self.seen]
        self.seen += 1
        return [object()] * n


class _CountingServer(_Server):
    def __init__(self, pipe, counts):
        super().__init__(pipe)
        self.analyzer = _Analyzer(counts)
        self.analyzer_loads = 0

    def face_analyzer(self):
        self.analyzer_loads += 1
        return self.analyzer


def _run(tmp_path, name, *, retries, counts=(), slots=("A", "B"), save=False, seed=110501, pose=None):
    params = kf.KeyframeParams(width=SW, height=SH, identity_method="ip_adapter", seed=seed,
                               plate_count_retries=retries, pose_conditioning=pose)
    pipe = _DistinctPipe(SW, SH)
    server = _CountingServer(pipe, counts)
    res = kf.render_keyframe(_scene(slots), CAST, SB, server, tmp_path / name / "shot_03.png", params=params,
                             lora_paths={s: tmp_path / f"{s}.safetensors" for s in slots}, save_intermediates=save)
    return res, pipe, server


# ------------------------------------------------------------------ pure helpers

@pytest.mark.parametrize("counts,expected,want", [
    ([2], 2, 0), ([3, 2], 2, 1), ([3, 3, 2, 2], 2, 2), ([1], 2, 0),
    ([3, 3, 3, 3], 2, 0), ([4, 3, 3, 3], 2, 1), ([5, 4, 3, 4], 2, 2),
])
def test_choose_plate(counts, expected, want):
    assert kf.choose_plate(counts, expected) == want


def test_choose_plate_needs_a_draw():
    with pytest.raises(ValueError):
        kf.choose_plate([], 2)


def test_record_is_honest_about_a_dirty_plate():
    assert kf.plate_count_record([3, 3], 2, 1) == {"expected": 2, "faces": [3, 3], "chosen": 0, "clean": False,
                                                   "retries": 1}
    assert kf.plate_count_record([3, 2], 2, 3)["clean"] is True


def test_retry_seeds_are_derived_and_distinct():
    seeds = [kf.plate_retry_seed(110501, a) for a in (1, 2, 3)]
    assert len(set(seeds + [110501])) == 4
    assert seeds == [kf.plate_retry_seed(110501, a) for a in (1, 2, 3)]  # deterministic
    assert all(0 <= s < 2**31 for s in seeds + [kf.plate_retry_seed(2**31 - 1, 3)])


def test_count_faces_hands_bgr_to_the_analyser():
    seen = {}

    class A:
        def get(self, arr):
            seen["px"] = tuple(int(v) for v in arr[0, 0])
            return [1, 2, 3]

    assert kf.count_faces(A(), Image.new("RGB", (4, 4), (10, 20, 30))) == 3
    assert seen["px"] == (30, 20, 10)


@pytest.mark.parametrize("slots,set_to,want", [(("A", "B"), 2, 2), (("A", "B"), 9, 3), (("A", "B"), -1, 0),
                                               (("A",), 3, 0), ((), 3, 0)])
def test_two_person_plus_shots_only_and_clamped(slots, set_to, want):
    assert kf.plate_count_retries_for(_scene(slots), kf.KeyframeParams(plate_count_retries=set_to)) == want


# ------------------------------------------------------------------ the scene-lock body

@pytest.mark.usefixtures("gpu_fakes")
def test_off_counts_nothing(tmp_path):
    res, pipe, server = _run(tmp_path, "off", retries=0)
    assert server.analyzer_loads == 0 and res.plate_count is None and len(pipe.calls) == 2


@pytest.mark.usefixtures("gpu_fakes")
def test_one_person_is_never_counted(tmp_path):
    res, pipe, server = _run(tmp_path, "one", retries=3, slots=("A",))
    assert server.analyzer_loads == 0 and res.plate_count is None and len(pipe.calls) == 2


@pytest.mark.usefixtures("gpu_fakes")
def test_a_clean_first_plate_draws_exactly_the_guard_off_still(tmp_path):
    # The pose is pinned on both sides: unset, turning the guard off also restores the plate's pose (backend#596),
    # which is a different property, held by test_plate_pose_default_596.
    off, pipe_off, _ = _run(tmp_path, "off", retries=0, pose=True)
    on, pipe_on, server = _run(tmp_path, "on", retries=3, counts=[2], pose=True)
    assert [_call_fingerprint(c) for c in pipe_on.calls] == [_call_fingerprint(c) for c in pipe_off.calls]
    assert on.path.read_bytes() == off.path.read_bytes()
    assert on.plate_count == {"expected": 2, "faces": [2], "chosen": 0, "clean": True, "retries": 3}
    assert server.analyzer.seen == 1


@pytest.mark.usefixtures("gpu_fakes")
def test_a_dirty_plate_is_redrawn_and_pass_b_gets_the_chosen_draw(tmp_path):
    res, pipe, _server = _run(tmp_path, "dirty", retries=3, counts=[3, 3, 2], save=True)
    # Three plate draws, then Pass B once.
    assert len(pipe.calls) == 4
    assert [c["generator"].seed for c in pipe.calls[1:3]] == [kf.plate_retry_seed(110501, 1),
                                                              kf.plate_retry_seed(110501, 2)]
    assert pipe.calls[0]["generator"].seed == 110501
    assert pipe.calls[3]["generator"] is pipe.calls[0]["generator"]  # Pass B keeps the original generator
    assert res.plate_count == {"expected": 2, "faces": [3, 3, 2], "chosen": 2, "clean": True, "retries": 3}
    # Pass B's control is the canny of draw 2, and the saved plate is draw 2.
    assert pipe.calls[3]["image"].tobytes() == kf._canny_preprocess(pipe.outputs[2]).tobytes()
    assert Image.open(res.intermediates["plate"]).tobytes() == pipe.outputs[2].tobytes()
    for i in range(3):
        assert Image.open(res.intermediates[f"plate_try{i}"]).tobytes() == pipe.outputs[i].tobytes()
    assert "plate_try3" not in res.intermediates
    # Every plate draw got the same inputs apart from its generator.
    fps = [_call_fingerprint(c) for c in pipe.calls[:3]]
    for fp in fps:
        fp.pop("generator")
    assert fps[0] == fps[1] == fps[2]


@pytest.mark.usefixtures("gpu_fakes")
def test_no_clean_draw_ships_the_fewest_and_says_so(tmp_path):
    res, pipe, _ = _run(tmp_path, "never", retries=3, counts=[4, 3, 3, 3])
    assert len(pipe.calls) == 5  # four plates, Pass B once: bounded at retries
    assert res.plate_count == {"expected": 2, "faces": [4, 3, 3, 3], "chosen": 1, "clean": False, "retries": 3}
    assert pipe.calls[4]["image"].tobytes() == kf._canny_preprocess(pipe.outputs[1]).tobytes()
    assert len({kf._canny_preprocess(o).tobytes() for o in pipe.outputs}) == len(pipe.outputs)  # the fake can tell


@pytest.mark.usefixtures("gpu_fakes")
def test_the_bound_is_the_configured_one(tmp_path):
    res, pipe, _ = _run(tmp_path, "one-retry", retries=1, counts=[3, 3])
    assert len(pipe.calls) == 3 and res.plate_count["faces"] == [3, 3]


# ------------------------------------------------------------------ config, hash, result

# kf_hash of KeyframeConfig.for_tier(tier) on main before this knob (same values as before pose_spread).
HASH_BEFORE = {QualityTier.DRAFT: "0922f3dc0cb2ccca", QualityTier.STANDARD: "23b7e0c279e5ed2e"}


@pytest.mark.parametrize("tier", list(HASH_BEFORE))
def test_hash_unchanged_when_off(tier):
    kc = KeyframeConfig.for_tier(tier)
    kc.plate_count_retries = 0
    assert kf_hash(kc) == HASH_BEFORE[tier]


def test_default_is_on_at_three():
    # Ruled 2026-11-06 on the live 1.1.13 check (vivijure-cf#1143).
    assert KeyframeConfig().plate_count_retries == 3
    assert kf.KeyframeParams().plate_count_retries == 3
    assert _req().config.keyframe.plate_count_retries == 3
    assert _req(render_overrides={"keyframe": {"plate_count_retries": 0}}).config.keyframe.plate_count_retries == 0


def _shot(slots):
    from vivijure_backend.contract import Storyboard
    sb = Storyboard.from_dict({"scenes": [{"id": "shot_01", "prompt": "x", "character_slots": list(slots)}]})
    return sb.scenes[0], sb


@pytest.mark.parametrize("slots,changes", [((), False), (("A",), False), (("A", "B"), True),
                                           (("A", "B", "C"), False)])
def test_default_on_redraws_only_the_shots_it_guards(slots, changes):
    # The default flip redraws existing two-person stills once (accepted) and must NOT touch the others.
    sc, sb = _shot(slots)
    on = KeyframeConfig.for_tier(QualityTier.STANDARD)
    off = KeyframeConfig.for_tier(QualityTier.STANDARD)
    off.plate_count_retries = 0
    same = kf_hash(on, scene=sc, storyboard=sb) == kf_hash(off, scene=sc, storyboard=sb)
    assert same is (not changes)


def test_hash_changes_when_set():
    kc = KeyframeConfig.for_tier(QualityTier.STANDARD)
    kc.plate_count_retries = 0
    base = kf_hash(kc)
    kc.plate_count_retries = 2
    two = kf_hash(kc)
    kc.plate_count_retries = 3
    assert len({base, two, kf_hash(kc)}) == 3


def test_request_parses_and_clamps():
    assert _req(render_overrides={"keyframe": {"plate_count_retries": 2}}).config.keyframe.plate_count_retries == 2
    assert _req(render_overrides={"keyframe": {"plate_count_retries": 7}}).config.keyframe.plate_count_retries == 3
    assert _req().config.keyframe.plate_count_retries == 3


def test_result_carries_the_record_only_when_the_guard_ran():
    rec = {"expected": 2, "faces": [3, 2], "chosen": 1, "clean": True, "retries": 3}
    out = RenderResult(project="p", keyframes=[Keyframe("a", "k1", plate_count=rec), Keyframe("b", "k2")]).to_dict()
    assert out["keyframes"][0]["plate_count"] == rec
    assert "plate_count" not in out["keyframes"][1]


class _GuardPipeline:
    def set_pretrained_loras(self, mapping):
        self.pretrained_loras = mapping

    def execute(self, plan, bundle, workdir):
        out = Outputs()
        for slot in plan.lora.train:
            p = workdir / f"lora_{slot}.safetensors"; p.write_bytes(b"x"); out.loras[slot] = p
        for s in plan.scenes:
            if s.keyframe_mode.value == "generate":
                p = workdir / f"{s.shot_id}.png"; p.write_bytes(b"x"); out.keyframes[s.shot_id] = p
        out.plate_counts["shot_02"] = {"expected": 2, "faces": [3, 3, 3, 3], "chosen": 0, "clean": False,
                                       "retries": 3}
        return out


def test_harness_reports_the_record_on_the_still(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job({"action": "preview", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
                   "quality_tier": "standard"}, pipeline=_GuardPipeline(), store=store, workdir=tmp_path / "w",
                  job_id="j")
    by = {k["shot_id"]: k for k in res["keyframes"]}
    assert by["shot_02"]["plate_count"]["clean"] is False
    assert "plate_count" not in by["shot_01"]
    assert by["shot_02"]["key"] == keys.job_keyframe_key("neon", "j", "shot_02")


def test_gpu_pipeline_carries_the_record(tmp_path, monkeypatch):
    from vivijure_backend import pipeline as pl
    from vivijure_backend.contract import Bundle
    from vivijure_backend.orchestrator import plan as make_plan
    rec = {"expected": 2, "faces": [3, 2], "chosen": 1, "clean": True, "retries": 3}

    def fake_render(scene, cast, storyboard, server, out_path, **_):
        out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_bytes(b"x")
        return kf.KeyframeResult(shot_id=scene.id, path=out_path, slots=list(scene.character_slots),
                                 multi_char=True, prompt="p", seed=0, engine="regional",
                                 plate_count=rec if len(scene.character_slots) > 1 else None)

    events = []

    class Prog:
        def emit(self, name, **kw):
            events.append((name, kw))

    monkeypatch.setattr(pl._keyframe, "render_keyframe", fake_render)
    bundle = Bundle.extract(_bundle_tar(tmp_path / "b.tar.gz"), tmp_path / "project")
    req = _req()
    pipe = pl.GpuPipeline(config=req.config, server=object())
    pipe._progress = Prog()
    monkeypatch.setattr(pipe, "_onnx_provider_facts", lambda: None)
    out = pipe.execute(make_plan(req, bundle.storyboard, trained_slots={"A", "B"}), bundle, tmp_path / "w")
    assert out.plate_counts == {"shot_02": rec}
    done = {kw["shot"]: kw for name, kw in events if name == "keyframe_done"}
    assert done["shot_02"]["plate_count"] == rec and done["shot_01"]["plate_count"] is None
