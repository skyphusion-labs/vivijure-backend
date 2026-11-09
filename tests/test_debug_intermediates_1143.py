"""vivijure-cf#1143 diagnostic: a job-level flag that saves the scene-lock intermediates beside the keyframe.

The question the flag answers is which pass draws an extra person in a two-person shot: the Pass A plate, or
Pass B on top of it. So what must hold, and what is tested here on CPU:

- the flag is a TOP-LEVEL job field, only a real `true` turns it on, and it is not a generation knob (it is not
  in RenderConfig, so it cannot reach the keyframe reuse hash);
- with it on, the saved plate IS the image Pass A drew, the saved pose IS the control Pass A was given, and the
  saved canny IS the control Pass B was given. The scene-lock body runs for real here over a fake pipe, so a
  save that took the wrong object, or a copy taken at the wrong time, goes red;
- with it on or off, every input to every pass is the same, so the flag changes nothing that is drawn;
- the harness uploads them beside the still under the job's own segment and lists them in the result, and only
  when the JOB asked: a pipeline that kept files cannot publish them unasked.

What this cannot see: the real SDXL pipe. Whether the plate on a pod shows the duplicate is the live run."""
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image
from test_harness import FakeStore, _bundle_tar  # the shared fakes for the job flow

import vivijure_backend.keyframe as kf
from vivijure_backend.contract import (
    Cast,
    RenderRequest,
    RenderResult,
    Scene,
    Storyboard,
)
from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import Outputs, run_job

# ------------------------------------------------------------------ the request flag


def _req(**over):
    return RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz", **over})


def test_off_by_default():
    assert _req().debug_intermediates is False


@pytest.mark.parametrize("value,want", [(True, True), (False, False), ("true", False), (1, False), (None, False),
                                        ({"on": True}, False)])
def test_only_a_real_true_turns_it_on(value, want):
    assert _req(debug_intermediates=value).debug_intermediates is want


def test_it_is_not_a_generation_knob():
    # Under render_overrides it means nothing; at the top level it leaves the typed config alone, so the
    # keyframe reuse hash (built from config.keyframe) cannot see it.
    nested = _req(render_overrides={"keyframe": {"debug_intermediates": True}, "debug_intermediates": True})
    assert nested.debug_intermediates is False
    assert not hasattr(nested.config.keyframe, "debug_intermediates")
    assert _req(debug_intermediates=True).config == _req().config


def test_the_worker_hands_it_to_the_pipeline():
    from vivijure_backend import worker
    assert worker.build_pipeline(_req(debug_intermediates=True)).debug_intermediates is True
    assert worker.build_pipeline(_req()).debug_intermediates is False


def test_result_lists_them_only_when_there_are_any():
    assert "debug_intermediates" not in RenderResult(project="p").to_dict()
    item = {"shot_id": "s", "stage": "plate", "key": "k"}
    assert RenderResult(project="p", debug_intermediates=[item]).to_dict()["debug_intermediates"] == [item]


# ------------------------------------------------------------------ keys


def test_key_sits_beside_the_still_under_the_job():
    still = keys.job_keyframe_key("p", "job-1", "shot_03")
    for stage in kf.INTERMEDIATE_STAGES:
        k = keys.job_keyframe_intermediate_key("p", "job-1", "shot_03", stage)
        assert k == still[: -len(".png")] + f".{stage}.png"


def test_key_refuses_an_unknown_stage():
    with pytest.raises(ValueError):
        keys.job_keyframe_intermediate_key("p", "j", "s", "../../x")
    with pytest.raises(ValueError):
        kf.intermediate_path(Path("/w/s.png"), "final")


def test_the_studio_never_reads_one_as_a_still():
    # The studio's keyframe adopter takes the file name minus its extension as the shot id and keeps only ids
    # in the storyboard. An intermediate's name minus its extension is `<shot>.<stage>`, never a shot id.
    for stage in kf.INTERMEDIATE_STAGES:
        name = keys.job_keyframe_intermediate_key("p", "j", "shot_03", stage).rsplit("/", 1)[1]
        assert name.rsplit(".", 1)[0] != "shot_03"


# ------------------------------------------------------------------ the scene-lock body, run for real over fakes

class _Gen:
    def manual_seed(self, seed):
        self.seed = seed
        return self


class _Pipe:
    """Records every sampling call and returns a distinct solid image for each."""
    COLORS = ((200, 10, 10), (10, 200, 10), (10, 10, 200))

    def __init__(self, w, h):
        self.calls: list[dict] = []
        self.outputs: list[Image.Image] = []
        self.w, self.h = w, h

    def __call__(self, **kw):
        self.calls.append(kw)
        img = Image.new("RGB", (self.w, self.h), self.COLORS[len(self.outputs) % 3])
        # A plate with structure, so its canny map is not empty.
        img.paste((255, 255, 255), (self.w // 4, self.h // 4, self.w // 2, self.h // 2))
        self.outputs.append(img)
        return types.SimpleNamespace(images=[img])

    def set_ip_adapter_scale(self, scale):
        self.ip_scale = scale


class _Server:
    def __init__(self, pipe):
        self.pipe = pipe
        self.controlnets: list = []

    def keyframe_pipeline(self):
        return self.pipe

    def set_keyframe_controlnet(self, role):
        self.controlnets.append(role)


MARA = "Mara, an adult woman in her thirties, dark shoulder-length curls, charcoal field jacket over a black top."
DEV = "Dev, an adult man in his thirties, dark buzz cut, grey zip hoodie."
CAST = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": MARA}, "B": {"name": "Dev", "prompt": DEV}}})
SB = Storyboard.from_dict({"style_prefix": "photoreal,", "scenes": [{"id": "shot_03", "prompt": "x"}]})


def _scene(slots):
    return Scene.from_dict({"id": "shot_03", "prompt": "Mara and Dev at a workbench, looking at a laptop.",
                            "character_slots": list(slots)}, 0)


@pytest.fixture
def gpu_fakes(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(Generator=lambda device=None: _Gen()))
    diffusers = types.ModuleType("diffusers")
    image_processor = types.ModuleType("diffusers.image_processor")

    class _MaskProc:
        def preprocess(self, masks, height, width):
            return ("masks", len(masks))

    image_processor.IPAdapterMaskProcessor = _MaskProc
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.image_processor", image_processor)
    monkeypatch.setattr(kf, "_apply_scheduler", lambda pipe, cfg: None)
    monkeypatch.setattr(kf, "_bind_loras", lambda pipe, paths, scale, few_step=True: sorted(paths))
    monkeypatch.setattr(kf, "_ensure_ip_adapter", lambda pipe, n=1: None)
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count: [])  # no InstantID ref: scene lock
    crop = Image.new("RGB", (8, 8), (50, 50, 50))
    monkeypatch.setattr(kf, "_face_crops_for_slots",
                        lambda server, cast, slots, pad_ratio=0.35: ([crop for _ in slots], True))
    monkeypatch.setattr(kf, "encode_long_prompt",
                        lambda pipe, p, n, shot_id="shot": {"prompt_embeds": p, "negative_prompt_embeds": n})


def _render(tmp_path, slots, save, name):
    # The plate count guard (default on) has its own tests; these pin the guard off to test the flag alone.
    params = kf.KeyframeParams(width=64, height=48, identity_method="ip_adapter", plate_count_retries=0)
    pipe = _Pipe(params.width, params.height)
    out = tmp_path / name / "shot_03.png"
    res = kf.render_keyframe(_scene(slots), CAST, SB, _Server(pipe), out, params=params,
                             lora_paths={s: tmp_path / f"{s}.safetensors" for s in slots},
                             save_intermediates=save)
    return res, pipe


def _same(path, img):
    return Image.open(path).convert("RGB").tobytes() == img.convert("RGB").tobytes()


def _call_fingerprint(call):
    """Every input to one sampling call, images as bytes, so two runs can be compared exactly."""
    fp = {}
    for k, v in call.items():
        if isinstance(v, Image.Image):
            fp[k] = ("img", v.size, v.tobytes())
        elif isinstance(v, list):
            fp[k] = [("img", x.size, x.tobytes()) if isinstance(x, Image.Image) else repr(x) for x in v]
        elif isinstance(v, _Gen):
            fp[k] = ("gen", v.seed)
        else:
            fp[k] = repr(v)
    return fp


def test_two_person_saves_the_pose_the_plate_and_the_canny_pass_b_was_given(tmp_path, gpu_fakes):
    res, pipe = _render(tmp_path, ("A", "B"), True, "on")
    assert len(pipe.calls) == 2  # Pass A then Pass B
    assert list(res.intermediates) == ["pose", "plate", "canny"]
    for stage, p in res.intermediates.items():
        assert p == res.path.with_name(f"shot_03.{stage}.png") and p.is_file()
    assert _same(res.intermediates["pose"], pipe.calls[0]["image"])     # Pass A's control
    assert _same(res.intermediates["plate"], pipe.outputs[0])            # what Pass A drew
    assert _same(res.intermediates["canny"], pipe.calls[1]["image"])     # Pass B's control
    assert _same(res.path, pipe.outputs[1])                              # the still is still Pass B
    assert not _same(res.intermediates["plate"], pipe.outputs[1])


def test_one_person_has_no_pose_stage(tmp_path, gpu_fakes):
    res, pipe = _render(tmp_path, ("A",), True, "one")
    assert list(res.intermediates) == ["plate", "canny"]
    assert _same(res.intermediates["plate"], pipe.outputs[0])
    assert _same(res.intermediates["canny"], pipe.calls[1]["image"])


def test_off_saves_nothing(tmp_path, gpu_fakes):
    res, _ = _render(tmp_path, ("A", "B"), False, "off")
    assert res.intermediates == {}
    assert sorted(p.name for p in res.path.parent.iterdir()) == ["shot_03.png"]


def test_the_flag_changes_nothing_that_is_drawn(tmp_path, gpu_fakes):
    on, pipe_on = _render(tmp_path, ("A", "B"), True, "on")
    off, pipe_off = _render(tmp_path, ("A", "B"), False, "off")
    assert [_call_fingerprint(c) for c in pipe_on.calls] == [_call_fingerprint(c) for c in pipe_off.calls]
    assert on.path.read_bytes() == off.path.read_bytes()
    assert on.prompt == off.prompt and on.identity_path == off.identity_path


def test_save_refuses_an_unknown_stage(tmp_path):
    with pytest.raises(ValueError):
        kf.write_intermediates({"plate": Image.new("RGB", (2, 2)), "bogus": Image.new("RGB", (2, 2))},
                              tmp_path / "s.png")


# ------------------------------------------------------------------ the harness uploads and reports them



class _IntermediatesPipeline:
    """Draws every GENERATE still and keeps a plate and a canny for each, whatever the job asked."""
    def set_pretrained_loras(self, mapping):
        self.pretrained_loras = mapping

    def execute(self, plan, bundle, workdir):
        out = Outputs()
        for slot in plan.lora.train:
            p = workdir / f"lora_{slot}.safetensors"; p.write_bytes(b"x"); out.loras[slot] = p
        for s in plan.scenes:
            if s.keyframe_mode.value == "generate":
                p = workdir / f"{s.shot_id}.png"; p.write_bytes(b"x"); out.keyframes[s.shot_id] = p
                stages = {}
                for stage in ("plate", "canny"):
                    q = workdir / f"{s.shot_id}.{stage}.png"; q.write_bytes(b"y"); stages[stage] = q
                out.intermediates[s.shot_id] = stages
        return out


def _job(**over):
    return {"action": "preview", "project": "neon", "bundle_key": "bundles/neon.tar.gz", "quality_tier": "standard",
            **over}


def test_harness_uploads_and_lists_them_when_asked(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(debug_intermediates=True), pipeline=_IntermediatesPipeline(), store=store,
                  workdir=tmp_path / "w", job_id="job-9")
    items = res["debug_intermediates"]
    want = {(sh, st) for sh in ("shot_01", "shot_02") for st in ("plate", "canny")}
    assert {(i["shot_id"], i["stage"]) for i in items} == want
    for i in items:
        assert i["key"] == keys.job_keyframe_intermediate_key("neon", "job-9", i["shot_id"], i["stage"])
        assert i["key"] in store.puts
    # The stills are reported exactly as before: the intermediates are not keyframes.
    assert sorted(k["key"] for k in res["keyframes"]) == sorted(
        keys.job_keyframe_key("neon", "job-9", s) for s in ("shot_01", "shot_02"))


def test_harness_publishes_nothing_unasked(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(), pipeline=_IntermediatesPipeline(), store=store, workdir=tmp_path / "w", job_id="job-9")
    assert "debug_intermediates" not in res
    assert not [k for k in store.puts if any(k.endswith(f".{st}.png") for st in kf.INTERMEDIATE_STAGES)]
    assert json.dumps(res)  # still a plain JSON result


# ------------------------------------------------------------------ the GPU pipeline carries them through

@pytest.mark.parametrize("flag", [True, False])
def test_gpu_pipeline_asks_for_them_and_hands_them_to_the_harness(tmp_path, monkeypatch, flag):
    from vivijure_backend import pipeline as pl
    from vivijure_backend.contract import Bundle
    from vivijure_backend.orchestrator import plan as make_plan

    asked: list = []

    def fake_render(scene, cast, storyboard, server, out_path, *, params=None, lora_paths=None,
                    save_intermediates=False, **_):
        asked.append(save_intermediates)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_bytes(b"x")
        inter = {}
        if save_intermediates:
            inter = {"plate": out_path.with_name(f"{out_path.stem}.plate.png")}
            inter["plate"].write_bytes(b"y")
        return kf.KeyframeResult(shot_id=scene.id, path=out_path, slots=list(scene.character_slots),
                                 multi_char=False, prompt="p", seed=0, engine="regional",
                                 identity_requested="ip_adapter", identity_path="regional_ip_adapter",
                                 intermediates=inter)

    monkeypatch.setattr(pl._keyframe, "render_keyframe", fake_render)
    bundle = Bundle.extract(_bundle_tar(tmp_path / "b.tar.gz"), tmp_path / "project")
    req = _req(debug_intermediates=flag)
    plan = make_plan(req, bundle.storyboard, trained_slots={"A", "B"})
    pipe = pl.GpuPipeline(config=req.config, server=object(), debug_intermediates=req.debug_intermediates)
    monkeypatch.setattr(pipe, "_onnx_provider_facts", lambda: None)
    out = pipe.execute(plan, bundle, tmp_path / "w")
    assert asked and set(asked) == {flag}
    if flag:
        assert set(out.intermediates) == set(out.keyframes)
        assert all(set(v) == {"plate"} for v in out.intermediates.values())
    else:
        assert out.intermediates == {}
