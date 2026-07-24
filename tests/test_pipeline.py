"""CPU tests for the GpuPipeline convergence: the pure config->params mappers, and the
execute() orchestration with the three GPU stages stubbed (no torch, no R2). Mirrors the
fake-stage pattern in tests/test_harness.py."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from vivijure_backend.config import RenderConfig
from vivijure_backend.contract import Bundle, RenderRequest, Scene
from vivijure_backend.harness.handler import HarnessError, Outputs, run_job
from vivijure_backend.orchestrator import plan as make_plan
from vivijure_backend.pipeline import GpuPipeline, i2v_params_from, keyframe_params_from
from vivijure_backend.routing import QualityTier


# ----------------------------------------------------------------- pure config -> params

def test_keyframe_params_final_is_full_step():
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.FINAL))
    assert p.few_step is False and p.steps == 30
    assert p.scheduler == "dpmpp_2m_karras"      # full-step solver, distill adapter off


def test_keyframe_params_draft_is_few_step():
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.DRAFT))
    assert p.few_step is True and p.steps == 4   # distill_steps, not the full-path steps
    assert p.scheduler == "ddim_trailing"        # Hyper-SD few-step path pins DDIM trailing


def test_keyframe_params_default_identity_is_ip_adapter():
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.FINAL))
    assert p.identity_method == "ip_adapter"   # the default single-char identity path


def test_keyframe_params_thread_instantid():
    cfg = RenderConfig.from_request("final", {"keyframe": {
        "identity_method": "instantid",
        "instantid_ip_adapter_scale": 0.9}})
    p = keyframe_params_from(cfg)
    assert p.identity_method == "instantid"
    assert p.instantid_ip_adapter_scale == 0.9


def test_keyframe_params_pull_multichar_scales():
    cfg = RenderConfig.from_request("final", {"keyframe": {"multi_char": {
        "lora_scale_per_slot": 0.25, "ip_adapter_scale_per_slot": 0.6,
        "max_slots": 2, "pose_conditioning": False}}})
    p = keyframe_params_from(cfg)
    assert p.lora_scale == 0.25
    # ip_adapter_scale comes from kc.ip_adapter_scale (top-level), not mc.ip_adapter_scale_per_slot;
    # mc's per-slot value is consumed by the regional blending engine, not this params field.
    assert p.ip_adapter_scale == pytest.approx(0.65)  # KeyframeConfig default
    assert p.pose_conditioning is False
    assert p.max_slots == 2


def test_keyframe_params_default_size_is_square():
    # Regression: with no override the keyframe stays 1024x1024, so existing square renders are
    # unchanged.
    p = keyframe_params_from(RenderConfig.for_tier(QualityTier.FINAL))
    assert (p.width, p.height) == (1024, 1024)


def test_keyframe_params_thread_both_dims_for_non_square():
    # 16:9: BOTH width and height must reach the engine; before this fix only width survived and
    # every keyframe came out square.
    cfg = RenderConfig.from_request("final", {"keyframe": {"width": 1920, "height": 1080}})
    p = keyframe_params_from(cfg)
    assert (p.width, p.height) == (1920, 1080)


def test_keyframe_params_thread_dims_from_resolution_string():
    # The control plane's "WIDTHxHEIGHT" shape must survive config parse AND the engine mapping.
    cfg = RenderConfig.from_request("final", {"keyframe": {"resolution": "1344x768"}})
    p = keyframe_params_from(cfg)
    assert (p.width, p.height) == (1344, 768)


def test_keyframe_params_thread_a_vertical_size():
    cfg = RenderConfig.from_request("final", {"keyframe": {"resolution": "720x1280"}})
    p = keyframe_params_from(cfg)
    assert (p.width, p.height) == (720, 1280)


def test_i2v_params_track_tier_and_scene_duration():
    final = i2v_params_from(RenderConfig.for_tier(QualityTier.FINAL), Scene(prompt="x", target_seconds=4))
    assert final.distill is False and final.steps == 40
    assert final.num_frames == 65   # round(4*16)=64 -> snapped up to 4k+1
    draft = i2v_params_from(RenderConfig.for_tier(QualityTier.DRAFT), Scene(prompt="x", target_seconds=5))
    assert draft.distill is True and draft.steps == 4
    assert draft.num_frames == 81   # 5*16=80 -> snapped to 81 (and at the ceiling)


def test_i2v_params_carry_the_tier_feature_cache():
    from vivijure_backend.config import FeatureCache
    # The tier's denoise accelerator must reach the engine params (the gap item L closes).
    assert i2v_params_from(RenderConfig.for_tier(QualityTier.FINAL),
                           Scene(prompt="x")).feature_cache is FeatureCache.MIXCACHE
    assert i2v_params_from(RenderConfig.for_tier(QualityTier.STANDARD),
                           Scene(prompt="x")).feature_cache is FeatureCache.EASYCACHE
    # draft has distill on, so config forced cache to NONE -> nothing to cache at 4 steps
    assert i2v_params_from(RenderConfig.for_tier(QualityTier.DRAFT),
                           Scene(prompt="x")).feature_cache is FeatureCache.NONE


# ----------------------------------------------------------------- bundle + stub pipeline

STORYBOARD = {
    "title": "neon", "use_characters": ["A", "B"], "style_prefix": "anime,",
    "scenes": [
        {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
        {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
        {"id": "shot_03", "prompt": "A, authored", "character_slots": ["A"],
         "target_seconds": 3, "start_image": "injected/shot_03.png"},
    ],
}


def _extract_bundle(tmp_path: Path) -> Bundle:
    tarp = tmp_path / "b.tar.gz"
    members = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {
            "A": {"name": "Vesper", "prompt": "teal"}, "B": {"name": "Rhode", "prompt": "orange"}}}).encode(),
        "injected/shot_03.png": b"PNG-ish",   # the authored start_image for the INJECT shot
        "characters/refs/A/ref_01.png": b"PNG-ish",
        "characters/refs/B/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(tarp, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return Bundle.extract(tarp, tmp_path / "project")


class StubPipeline(GpuPipeline):
    """GpuPipeline with the three GPU stages replaced by recording stubs that write empty
    artifact files. Exercises the orchestration without torch."""
    def __init__(self, config, pretrained_loras=None):
        super().__init__(config=config, pretrained_loras=pretrained_loras or {}, server=object())
        self.trained: list[str] = []
        self.keyframed: list[str] = []
        self.animated: list[str] = []
        self.finished: list[str] = []
        self.keyframe_loras: dict[str, list[str]] = {}

    def _train_slot(self, char, out_dir):
        self.trained.append(char.slot)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / "lora.safetensors"; f.write_bytes(b"x"); return f

    def _render_keyframe(self, scene, cast, storyboard, out_path, lora_paths):
        self.keyframed.append(scene.id)
        self.keyframe_loras[scene.id] = sorted(lora_paths)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x"); return out_path

    def _animate(self, scene, keyframe_path, prompt, out_path):
        assert Path(keyframe_path).exists(), "animating from a keyframe that was never staged"
        self.animated.append(scene.id)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x"); return out_path

    def _finish_clip(self, shot_id, in_path, out_path):
        assert Path(in_path).exists(), "finishing a clip that was never animated"
        self.finished.append(shot_id)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x"); return out_path


# --------------------------------------------------------------------- cast-missing guard

def test_execute_raises_on_cast_missing_trained_slot(tmp_path):
    # If the plan lists a slot for training but the cast registry has no character for it,
    # execute must raise HarnessError immediately rather than silently skipping and producing
    # a keyframe with no identity (the old behaviour was a silent continue).
    from vivijure_backend.contract import Bundle, Cast, Storyboard
    from vivijure_backend.harness.handler import HarnessError
    bundle = _extract_bundle(tmp_path)
    # Remove B from the cast so the plan's "train B" slot has no character backing it.
    bundle.cast.characters.pop("B")
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "draft"})
    plan = make_plan(req, bundle.storyboard)
    assert "B" in plan.lora.train
    with pytest.raises(HarnessError, match="slot 'B'"):
        StubPipeline(req.config).execute(plan, bundle, tmp_path / "work")


# --------------------------------------------------------------------- execute orchestration

def test_execute_trains_keyframes_and_animates_the_whole_plan(tmp_path):
    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard)
    pipe = StubPipeline(req.config)
    out = pipe.execute(plan, bundle, tmp_path / "work")

    assert sorted(pipe.trained) == ["A", "B"]                 # nothing pretrained -> train both
    assert sorted(out.loras) == ["A", "B"]
    assert sorted(out.keyframes) == ["shot_01", "shot_02"]    # shot_03 is INJECT, not generated
    assert sorted(pipe.keyframed) == ["shot_01", "shot_02"]
    assert sorted(pipe.animated) == ["shot_01", "shot_02", "shot_03"]  # all needs_i2v, inject staged
    assert [s for s, _ in out.clips] == ["shot_01", "shot_02", "shot_03"]
    # final tier has finish enabled, so every animated clip is finished and the finished path replaces
    # the raw clip in the outputs (clip order preserved for the storyboard concat)
    assert pipe.finished == ["shot_01", "shot_02", "shot_03"]
    assert all("/finished/" in str(p) for _, p in out.clips)
    # keyframing saw the freshly trained adapters
    assert pipe.keyframe_loras["shot_02"] == ["A", "B"]


def test_execute_honors_reuse_and_pretrained(tmp_path):
    bundle = _extract_bundle(tmp_path)
    # A is pretrained (skip training, feed a staged adapter); shot_01 keyframe already exists.
    pre = tmp_path / "preA.safetensors"; pre.write_bytes(b"x")
    req = RenderRequest.from_dict({"action": "render", "project": "neon", "bundle_key": "x",
                                   "quality_tier": "final", "pretrained_loras": {"A": str(pre)}})
    plan = make_plan(req, bundle.storyboard,
                     trained_slots=set(req.pretrained_loras), existing_keyframes={"shot_01": None})
    work = tmp_path / "work"
    (work / "keyframes").mkdir(parents=True)
    (work / "keyframes" / "shot_01.png").write_bytes(b"x")   # stage the reused keyframe

    pipe = StubPipeline(req.config, pretrained_loras=req.pretrained_loras)
    out = pipe.execute(plan, bundle, work)

    assert pipe.trained == ["B"]                               # A pretrained -> only B trains
    assert "shot_01" not in pipe.keyframed                     # shot_01 reused, not regenerated
    assert sorted(out.keyframes) == ["shot_02"]
    assert "shot_01" in pipe.animated                          # reused keyframe still animates
    # the pretrained A adapter was wired into keyframing
    assert "A" in pipe.keyframe_loras["shot_02"]


def test_execute_fails_loud_when_reused_keyframe_is_missing(tmp_path):
    # A REUSE shot whose keyframe was never staged is a HARD per-shot error naming the shot --
    # never a silently shorter film under a success status (#245/#249: a degrade is never silent).
    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard, existing_keyframes={"shot_01": None})
    pipe = StubPipeline(req.config)
    with pytest.raises(HarnessError, match="shot_01"):
        pipe.execute(plan, bundle, tmp_path / "work")     # shot_01 keyframe not staged


def test_execute_fails_loud_when_injected_start_image_is_missing(tmp_path):
    # An INJECT shot whose authored start_image is absent from the bundle is a HARD per-shot
    # error naming the shot and the file, not a missing shot in a "successful" film.
    bundle = _extract_bundle(tmp_path)
    (bundle.root / "injected" / "shot_03.png").unlink()   # the authored keyframe vanishes
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard)
    pipe = StubPipeline(req.config)
    with pytest.raises(HarnessError, match="start_image"):
        pipe.execute(plan, bundle, tmp_path / "work")


def test_execute_finishes_clips_only_when_finish_is_enabled(tmp_path):
    # Draft tier finishes nothing (a fast preview): the raw i2v clips ship unchanged, _finish_clip
    # is never called. Final tier finishes every clip. This is the gate `config.finish.enabled`.
    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "draft"})
    plan = make_plan(req, bundle.storyboard)
    pipe = StubPipeline(req.config)
    assert pipe.config.finish.enabled is False         # draft baseline: both passes off
    out = pipe.execute(plan, bundle, tmp_path / "work-draft")
    assert pipe.finished == []                          # finish stage skipped entirely
    assert all("/finished/" not in str(p) for _, p in out.clips)   # raw i2v clips ship as-is

    # A draft render that explicitly turns interpolation on DOES finish (the override re-enables it).
    req2 = RenderRequest.from_dict({"action": "render", "project": "neon", "bundle_key": "x",
                                    "quality_tier": "draft",
                                    "render_overrides": {"finish": {"interpolate": True}}})
    pipe2 = StubPipeline(req2.config)
    assert pipe2.config.finish.enabled is True
    pipe2.execute(make_plan(req2, bundle.storyboard), bundle, tmp_path / "work-draft2")
    assert pipe2.finished == ["shot_01", "shot_02", "shot_03"]


# --------------------------------------------------------------------- run_job end to end

class FakeStore:
    def __init__(self, bundle_tar: Path):
        self.bundle_tar = bundle_tar; self.puts: list[str] = []; self.tars: list[str] = []

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest); return dest

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(); self.puts.append(key); return key

    def put_dir_as_tar(self, src_dir, key, *, metadata=None):
        self.tars.append(key); return key


def test_run_job_drives_gpu_pipeline_offloaded(tmp_path):
    # The whole harness flow on CPU with a stubbed GpuPipeline: plan -> execute -> finish.
    _extract_bundle(tmp_path)  # writes the bundle tar at tmp_path/b.tar.gz
    store = FakeStore(tmp_path / "b.tar.gz")
    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.FINAL))

    res = run_job(
        {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "quality_tier": "final", "render_overrides": {"finish_offloaded": True}},
        pipeline=pipe, store=store, workdir=tmp_path / "work")

    assert res["lora"]["A"]["lora_id"].endswith("A/pytorch_lora_weights.safetensors")
    assert [c["shot_id"] for c in res["clips"]] == ["shot_01", "shot_02", "shot_03"]
    assert {k["shot_id"] for k in res["keyframes"]} == {"shot_01", "shot_02", "shot_03"}
    assert any(k.endswith("manifest.json") for k in store.puts)
    assert res["state_key"] is None  # #112: per-artifact objects, no shared state tar


# ----------------------------------------------------- pretrained-LoRA R2 staging (item B)

class StagingStore(FakeStore):
    """FakeStore that records the keys it serves, so a test can assert a LoRA was fetched."""
    def __init__(self, bundle_tar):
        super().__init__(bundle_tar); self.gets: list[str] = []

    def get_file(self, key, dest):
        self.gets.append(key)
        return super().get_file(key, dest)


def test_run_job_stages_pretrained_lora_from_r2_and_skips_training(tmp_path):
    # A render that reuses a slot's R2 LoRA must NOT retrain it, and the adapter must be pulled
    # to local disk (the GPU layer never touches R2) and fed to keyframing.
    _extract_bundle(tmp_path)
    store = StagingStore(tmp_path / "b.tar.gz")
    LORA_KEY = "loras/neon/A/pytorch_lora_weights.safetensors"
    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.DRAFT), pretrained_loras={"A": LORA_KEY})

    res = run_job(
        {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "quality_tier": "draft", "pretrained_loras": {"A": LORA_KEY},
         "render_overrides": {"finish_offloaded": True}},  # skip the ffmpeg merge of stub clips
        pipeline=pipe, store=store, workdir=tmp_path / "work", job_id="j")

    assert pipe.trained == ["B"]                              # A reused -> only B trains
    assert LORA_KEY in store.gets                             # the adapter was actually fetched
    # the pipeline now holds a LOCAL staged path for A, not the R2 key, and it exists on disk
    assert "/pretrained/A/" in pipe.pretrained_loras["A"]
    assert Path(pipe.pretrained_loras["A"]).is_file()
    assert "A" in pipe.keyframe_loras["shot_01"]              # the staged LoRA reached keyframing
    assert res["lora"]["A"]["lora_id"] == LORA_KEY            # result still reports the durable R2 key


def test_run_job_fails_fast_when_a_reused_lora_cannot_be_staged(tmp_path):
    # A requested-but-unfetchable LoRA must fail the job BEFORE any GPU work, not silently render
    # the character without its identity.
    import pytest
    from vivijure_backend.harness.handler import HarnessError

    _extract_bundle(tmp_path)

    class MissingLoraStore(FakeStore):
        def get_file(self, key, dest):
            if str(key).endswith(".tar.gz"):
                return super().get_file(key, dest)
            raise FileNotFoundError(key)                      # the LoRA key is not in R2

    pipe = StubPipeline(RenderConfig.for_tier(QualityTier.DRAFT),
                        pretrained_loras={"A": "loras/neon/A.safetensors"})
    with pytest.raises(HarnessError, match="could not stage pretrained LoRA"):
        run_job({"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
                 "quality_tier": "draft", "pretrained_loras": {"A": "loras/neon/A.safetensors"}},
                pipeline=pipe, store=MissingLoraStore(tmp_path / "b.tar.gz"),
                workdir=tmp_path / "work", job_id="j")
    assert pipe.trained == []                                 # failed before training anything


def test_run_job_refuses_to_drop_staged_loras_a_pipeline_cannot_receive(tmp_path):
    # End-to-end guarantee: if staging succeeds but the pipeline has no way to receive the map,
    # raise rather than silently render the character without its LoRA (the asymmetry the review
    # flagged). A job with no reused LoRAs would be fine on such a pipeline; this one is not.
    import pytest
    from vivijure_backend.harness.handler import HarnessError

    _extract_bundle(tmp_path)

    class NoSetterPipeline:
        """A pipeline that forgot set_pretrained_loras."""
        def execute(self, plan, bundle, workdir):
            return Outputs()

    with pytest.raises(HarnessError, match="cannot receive staged reused LoRAs"):
        run_job({"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
                 "quality_tier": "draft", "pretrained_loras": {"A": "loras/neon/A/x.safetensors"},
                 "render_overrides": {"finish_offloaded": True}},
                pipeline=NoSetterPipeline(), store=StagingStore(tmp_path / "b.tar.gz"),
                workdir=tmp_path / "work", job_id="j")


# -------------------------------------------------- i2v per-step progress wiring (item M)

def test_animate_wires_per_step_i2v_progress(tmp_path, monkeypatch):
    # GpuPipeline._animate must hand i2v.animate a progress_cb bound to this shot that emits
    # i2v_step -- so the snapshot ticks step/total during the long full-step i2v.
    from vivijure_backend import i2v as i2v_mod
    from vivijure_backend.harness.progress import ProgressEmitter

    captured = {}

    def fake_animate(scene, keyframe, prompt, server, out_path, *, params=None, progress_cb=None):
        captured["cb"] = progress_cb
        class R: path = out_path
        return R()
    monkeypatch.setattr(i2v_mod, "animate", fake_animate)

    pipe = GpuPipeline(config=RenderConfig.for_tier(QualityTier.FINAL), server=object())
    emitter = ProgressEmitter(None, "neon", "j")     # store=None: emit accumulates in-memory
    pipe.set_progress(emitter)
    pipe._animate(Scene(prompt="x", id="shot_01"), tmp_path / "kf.png", "motion", tmp_path / "out.mp4")

    cb = captured["cb"]
    assert callable(cb)                              # a real per-step callback was wired
    cb(7, 40)                                        # diffusers would call this once per step
    last = emitter._events[-1]
    assert last["event"] == "i2v_step" and last["shot"] == "shot_01"
    assert last["step"] == 7 and last["total"] == 40


# ------------------------------------------------ i2v_params_from mapping (#15)

def _render_config_for_i2v(overrides=None):
    """Build a minimal RenderConfig with I2VConfig overrides for mapping tests."""
    from vivijure_backend.config import RenderConfig
    from vivijure_backend.routing import QualityTier
    return RenderConfig.from_request(QualityTier.STANDARD, {"i2v": overrides or {}})


def _scene(target=None):
    from vivijure_backend.contract import Scene
    d = {"id": "s1", "prompt": "motion"}
    if target is not None:
        d["target_seconds"] = target
    return Scene.from_dict(d, 0)


def test_i2v_params_from_maps_flow_shift():
    from vivijure_backend.pipeline import i2v_params_from
    cfg = _render_config_for_i2v({"flow_shift": 3.5})
    p = i2v_params_from(cfg, _scene(2))
    assert p.flow_shift == 3.5


def test_i2v_params_from_maps_i2v_seed_not_keyframe_seed():
    from vivijure_backend.pipeline import i2v_params_from
    cfg = _render_config_for_i2v({"seed": 777})
    p = i2v_params_from(cfg, _scene(2))
    assert p.seed == 777


def test_i2v_params_from_uses_seconds_per_shot_when_scene_has_no_duration():
    from vivijure_backend.pipeline import i2v_params_from
    from vivijure_backend.i2v import frames_for
    cfg = _render_config_for_i2v({"seconds_per_shot": 3.0, "fps": 16})
    p = i2v_params_from(cfg, _scene(None))
    # seconds_per_shot=3.0 at 16fps -> 48 frames -> snap_frames(48) = 49
    assert p.num_frames == 49


def test_i2v_params_from_scene_duration_wins_over_seconds_per_shot():
    from vivijure_backend.pipeline import i2v_params_from
    cfg = _render_config_for_i2v({"seconds_per_shot": 3.0, "fps": 16})
    p = i2v_params_from(cfg, _scene(2.0))
    # 2s at 16fps = 32 frames -> snap_frames(32) = 33
    assert p.num_frames == 33


def test_i2v_params_from_respects_config_num_frames_ceiling():
    from vivijure_backend.pipeline import i2v_params_from
    from vivijure_backend.i2v import snap_frames
    # The old code used the engine's frames_for (cap=81); now it uses ic.frames_for (cap=256).
    # Verify a 10-second scene at 16fps gets frames snapped up to 161, not capped at 81.
    cfg = _render_config_for_i2v({"fps": 16})
    p = i2v_params_from(cfg, _scene(10.0))  # 10s * 16fps = 160 -> snap to 161
    assert p.num_frames == 161  # not 81 (the old engine ceiling)
    assert (p.num_frames - 1) % 4 == 0  # is 4k+1


def test_execute_emits_informational_plan_tier_not_a_mismatch_warning(tmp_path):
    # #163: the old `tier_mismatch` warn false-fired on the by-design multi-arch pool (a single
    # planned-tier label always differs from two of three cards). It is downgraded to an
    # always-emitted informational `plan_tier {actual, planned}` trace -- no "mismatch" framing,
    # never a gate. The card-correlation value (actual card in the event) is preserved.
    from vivijure_backend.harness.progress import ProgressEmitter

    bundle = _extract_bundle(tmp_path)
    req = RenderRequest.from_dict({"action": "render", "project": "neon",
                                   "bundle_key": "x", "quality_tier": "final"})
    plan = make_plan(req, bundle.storyboard)
    pipe = StubPipeline(req.config)
    emitter = ProgressEmitter(None, "neon", "j")     # store=None: emit accumulates in-memory
    pipe.set_progress(emitter)
    pipe.execute(plan, bundle, tmp_path / "work")

    names = [e["event"] for e in emitter._events]
    assert "tier_mismatch" not in names              # the false-alarm warning is gone
    plan_tiers = [e for e in emitter._events if e["event"] == "plan_tier"]
    assert len(plan_tiers) == 1                       # informational, emitted once per render
    pt = plan_tiers[0]
    assert isinstance(pt["actual"], str) and pt["actual"]        # the card that actually ran
    assert isinstance(pt["planned"], list) and pt["planned"]     # the tier(s) the planner targeted
    assert pt["planned"] == sorted(pt["planned"])                # deterministic ordering
