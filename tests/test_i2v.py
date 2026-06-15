"""i2v's pure surface, tested on CPU: the frame-count math the temporal VAE constrains, the
duration it realizes, and the tier->profile decision. The Wan generation body needs torch and is
validated on a pod."""
from vivijure_backend.config import FeatureCache, I2VConfig
from vivijure_backend.contract import Scene
from vivijure_backend.i2v import (
    DEFAULT_FPS,
    MAX_FRAMES,
    I2VParams,
    animate,
    _set_feature_cache,
    _step_callback,
    clip_seconds,
    frames_for,
    snap_frames,
)
from vivijure_backend.routing import QualityTier


# ----------------------------------------------------------- feature cache install (item L)

class _FakeTransformer:
    """Mimics diffusers CacheMixin: enable_cache sets the flag, disable_cache clears it."""
    def __init__(self):
        self.disabled = 0
        self.enabled = []                 # thresholds passed to enable_cache, in order
        self.is_cache_enabled = False
    def enable_cache(self, config):
        self.enabled.append(config.threshold)
        self.is_cache_enabled = True
    def disable_cache(self):
        self.disabled += 1
        self.is_cache_enabled = False


class _FakePipe:
    def __init__(self, dual=True):
        self.transformer = _FakeTransformer()
        if dual:
            self.transformer_2 = _FakeTransformer()   # Wan 2.2 low-noise MoE expert


def _fake_diffusers_hooks(monkeypatch):
    import sys, types
    diffusers = types.ModuleType("diffusers")
    hooks = types.ModuleType("diffusers.hooks")

    class FirstBlockCacheConfig:
        def __init__(self, threshold):
            self.threshold = threshold

    hooks.FirstBlockCacheConfig = FirstBlockCacheConfig
    diffusers.hooks = hooks
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)


def test_feature_cache_mixcache_enables_fbcache_with_final_threshold(monkeypatch):
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)
    assert pipe.transformer.enabled == [0.20]        # final tier threshold via enable_cache
    assert pipe.transformer.is_cache_enabled is True
    assert pipe.transformer.disabled == 0            # nothing to reset on the first shot (silent)


def test_feature_cache_easycache_uses_a_more_conservative_threshold(monkeypatch):
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.EASYCACHE)
    assert pipe.transformer.enabled == [0.10]


def test_feature_cache_per_shot_reset_disables_prior_before_re_enabling(monkeypatch):
    # The matched pair: shot 2 must clear shot 1's cache before installing its own (no leak).
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)   # shot 1
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)   # shot 2
    assert pipe.transformer.disabled == 1            # shot 1's cache was disabled before shot 2's
    assert pipe.transformer.enabled == [0.20, 0.20]
    assert pipe.transformer.is_cache_enabled is True


def test_feature_cache_none_clears_a_prior_cache(monkeypatch):
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)   # enable
    _set_feature_cache(pipe, FeatureCache.NONE)       # then a NONE shot must turn it off
    assert pipe.transformer.disabled == 1
    assert pipe.transformer.is_cache_enabled is False


def test_feature_cache_none_on_a_fresh_pipe_is_silent():
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.NONE)
    assert pipe.transformer.disabled == 0            # no cache on -> no "nothing to disable" noise


def test_feature_cache_is_best_effort_when_the_cache_cannot_attach():
    # No diffusers in the test venv -> the FirstBlockCacheConfig import raises; must be swallowed
    # (run uncached) and never reach enable_cache.
    pipe = _FakePipe()
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)   # must not raise
    assert pipe.transformer.enabled == []


def test_feature_cache_tolerates_a_pipe_without_a_transformer():
    class Bare: pass
    _set_feature_cache(Bare(), FeatureCache.MIXCACHE)  # no transformer -> no-op, no raise


def test_feature_cache_enables_both_moe_experts(monkeypatch):
    # The bug: Wan 2.2 has two DiTs (transformer high-noise + transformer_2 low-noise); caching
    # only the first left the back ~70% of steps uncached. Both experts must get the cache.
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe(dual=True)
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)
    assert pipe.transformer.enabled == [0.20]
    assert pipe.transformer_2.enabled == [0.20]        # previously skipped -> the step-12 cliff


def test_feature_cache_per_shot_reset_clears_both_experts(monkeypatch):
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe(dual=True)
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)     # shot 1
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)     # shot 2
    assert pipe.transformer.disabled == 1 and pipe.transformer_2.disabled == 1
    assert pipe.transformer.enabled == [0.20, 0.20]
    assert pipe.transformer_2.enabled == [0.20, 0.20]


def test_feature_cache_single_dit_pipe_still_works(monkeypatch):
    # A pipe with no transformer_2 (draft / non-MoE / future) just caches the one it has.
    _fake_diffusers_hooks(monkeypatch)
    pipe = _FakePipe(dual=False)
    _set_feature_cache(pipe, FeatureCache.MIXCACHE)
    assert pipe.transformer.enabled == [0.20]
    assert not hasattr(pipe, "transformer_2")


# ----------------------------------------------------- non-square size honored (item B)

class _FakeImage:
    def __init__(self, width, height):
        self.width, self.height = width, height


class _RecordingI2VPipe:
    """Stand-in for the Wan i2v pipe: records the height/width it was called with so a test can
    assert the clip is animated at the keyframe's real (possibly non-square) size."""
    def __init__(self):
        self.called = None
        self.transformer = _FakeTransformer()
    def set_adapters(self, names, adapter_weights):
        pass
    def __call__(self, **kwargs):
        self.called = kwargs
        class _Out:
            frames = [["frame"]]
        return _Out()


def _stub_i2v_runtime(monkeypatch, image):
    """Stub the heavy imports animate() defers (torch + diffusers.utils) so the engine body runs on
    CPU. load_image returns our fake keyframe; export_to_video is a no-op recorder."""
    import sys, types
    torch = types.ModuleType("torch")

    class _Gen:
        def manual_seed(self, s):
            return self
    torch.Generator = lambda *a, **k: _Gen()
    monkeypatch.setitem(sys.modules, "torch", torch)

    diffusers = sys.modules.get("diffusers") or types.ModuleType("diffusers")
    utils = types.ModuleType("diffusers.utils")
    utils.load_image = lambda path: image
    utils.export_to_video = lambda frames, path, fps: None
    diffusers.utils = utils
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", utils)


class _Server:
    def __init__(self, pipe):
        self._pipe = pipe
    def i2v_pipeline(self):
        return self._pipe


def test_animate_honors_a_non_square_keyframe_size(tmp_path, monkeypatch):
    # The keyframe is now 16:9; i2v must animate at that size, not collapse to a square. With
    # height/width unset on the params, animate follows the loaded keyframe's real dimensions.
    from vivijure_backend.contract import Scene
    _stub_i2v_runtime(monkeypatch, _FakeImage(1920, 1080))
    pipe = _RecordingI2VPipe()
    animate(Scene(prompt="x", id="s"), tmp_path / "kf.png", "motion", _Server(pipe),
            tmp_path / "out.mp4", params=I2VParams(num_frames=5, steps=1))
    assert (pipe.called["width"], pipe.called["height"]) == (1920, 1080)


def test_animate_honors_a_vertical_keyframe_size(tmp_path, monkeypatch):
    from vivijure_backend.contract import Scene
    _stub_i2v_runtime(monkeypatch, _FakeImage(720, 1280))
    pipe = _RecordingI2VPipe()
    animate(Scene(prompt="x", id="s"), tmp_path / "kf.png", "motion", _Server(pipe),
            tmp_path / "out.mp4", params=I2VParams(num_frames=5, steps=1))
    assert (pipe.called["width"], pipe.called["height"]) == (720, 1280)


def test_animate_explicit_params_size_overrides_the_keyframe(tmp_path, monkeypatch):
    # An explicit non-square size on the params wins over the keyframe's own dimensions.
    from vivijure_backend.contract import Scene
    _stub_i2v_runtime(monkeypatch, _FakeImage(1024, 1024))
    pipe = _RecordingI2VPipe()
    animate(Scene(prompt="x", id="s"), tmp_path / "kf.png", "motion", _Server(pipe),
            tmp_path / "out.mp4", params=I2VParams(num_frames=5, steps=1, width=1280, height=720))
    assert (pipe.called["width"], pipe.called["height"]) == (1280, 720)


# ----------------------------------------------------- per-step progress callback (item M)

def test_step_callback_reports_one_based_step_and_passes_kwargs_through():
    seen = []
    cb = _step_callback(lambda step, total: seen.append((step, total)), 40)
    kw = {"latents": "opaque"}
    out = cb(None, 11, 0, kw)          # diffusers signature: (pipe, step_index, timestep, kwargs)
    assert seen == [(12, 40)]          # reported 1-based: step_index 11 -> step 12 of 40
    assert out is kw                   # callback_kwargs returned unchanged, loop unaffected


def test_step_callback_is_none_without_a_progress_cb():
    # No cb -> None, so animate omits callback_on_step_end entirely (zero overhead).
    assert _step_callback(None, 40) is None


def test_step_callback_swallows_a_failing_progress_cb():
    def boom(step, total):
        raise RuntimeError("progress write failed")
    cb = _step_callback(boom, 4)
    kw = {"latents": 1}
    assert cb(None, 0, 0, kw) is kw    # must not raise; render continues


def _scene(target=None):
    d = {"id": "s", "prompt": "x"}
    if target is not None:
        d["target_seconds"] = target
    return Scene.from_dict(d, 0)


# --------------------------------------------------------------------- frame math (4k+1)

def test_snap_frames_to_4k_plus_1():
    assert snap_frames(81) == 81   # already valid
    assert snap_frames(80) == 81   # round up to the next valid count
    assert snap_frames(5) == 5
    assert snap_frames(6) == 9
    assert snap_frames(4) == 5
    assert snap_frames(1) == 1


def test_snap_frames_does_not_exceed_ceiling():
    # snap(256) would round UP to 257; must clamp to 253 (the largest 4k+1 <= 256)
    result = snap_frames(256)
    assert result == 253
    assert (result - 1) % 4 == 0


def test_snap_frames_max_frames_param():
    assert snap_frames(81, max_frames=81) == 81   # exactly at ceiling, already valid
    # 82 snaps up to 85 > 81; step down: prev = 81 - (81-1)%4 = 81 (valid, 4*20+1)
    assert snap_frames(82, max_frames=81) == 81
    assert (81 - 1) % 4 == 0
    # with max=80: 80 snaps to 81 > 80; step down to 77 (4*19+1)
    assert snap_frames(80, max_frames=80) == 77
    assert (77 - 1) % 4 == 0


def test_frames_for_target_duration():
    assert frames_for(5, 16) == 81       # 80 -> snapped 81
    assert frames_for(4, 16) == 65       # 64 -> snapped 65
    assert frames_for(10, 16) == MAX_FRAMES  # capped at the model ceiling
    assert frames_for(None) == MAX_FRAMES    # no target -> ceiling
    assert frames_for(0) == MAX_FRAMES


def test_clip_seconds_is_frames_over_fps():
    # 81/16 = 5.0625; Python rounds half-to-even at 3 places -> 5.062
    assert clip_seconds(81, 16) == 5.062
    assert clip_seconds(65, 16) == 4.062


# ----------------------------------------------------------------------- tier profiles (I2VConfig)

def test_final_tier_is_full_step():
    cfg = I2VConfig.for_tier(QualityTier.FINAL)
    assert cfg.distill is False
    assert cfg.steps == 40
    assert cfg.guidance_scale == 3.5


def test_draft_tier_is_distilled():
    cfg = I2VConfig.for_tier(QualityTier.DRAFT)
    assert cfg.distill is True
    assert cfg.steps == 4
    assert cfg.guidance_scale == 1.0


def test_standard_tier_is_full_step_with_easycache():
    cfg = I2VConfig.for_tier(QualityTier.STANDARD)
    assert cfg.distill is False
    assert cfg.feature_cache is FeatureCache.EASYCACHE


def test_frames_for_derives_from_seconds():
    cfg = I2VConfig.for_tier(QualityTier.DRAFT)
    assert cfg.frames_for(2.0) == max(1, min(256, round(2.0 * cfg.fps)))
    assert cfg.frames_for(None) == cfg.num_frames


def test_default_params_are_the_distilled_path():
    p = I2VParams()
    assert p.distill is True
    assert p.steps == 4
    assert p.fps == DEFAULT_FPS
    assert p.num_frames == MAX_FRAMES


# ---------------------------------------------------------- flow_shift field + application

def test_i2v_params_has_flow_shift_field():
    p = I2VParams()
    assert hasattr(p, "flow_shift")
    assert p.flow_shift == 5.0


def test_i2v_params_flow_shift_is_independent():
    p = I2VParams(flow_shift=3.0)
    assert p.flow_shift == 3.0


# ---------------------------------------------------------- _set_distill guard (#16)

class _DistillAwarePipe:
    """Minimal stand-in for the Wan i2v pipe with distill state tracking."""
    def __init__(self, loaded=None, fused=False):
        if loaded is not None:
            self._vj_i2v_distill_loaded = loaded
        self._vj_i2v_distill_fused = fused
        self.transformer = _FakeTransformer()
        self.adapter_calls = []

    def set_adapters(self, names, adapter_weights):
        self.adapter_calls.append((names, adapter_weights))


def test_set_distill_raises_when_no_distill_loaded_and_distill_true():
    from vivijure_backend.i2v import _set_distill
    import pytest
    pipe = _DistillAwarePipe(loaded=False, fused=False)
    with pytest.raises(RuntimeError, match="4-step-no-distill"):
        _set_distill(pipe, True)


def test_set_distill_silent_when_no_distill_loaded_and_distill_false():
    from vivijure_backend.i2v import _set_distill
    pipe = _DistillAwarePipe(loaded=False, fused=False)
    _set_distill(pipe, False)   # must not raise; running full-step is fine
    assert pipe.adapter_calls == []


def test_set_distill_returns_early_when_fused_and_distill_true():
    from vivijure_backend.i2v import _set_distill
    pipe = _DistillAwarePipe(loaded=True, fused=True)
    _set_distill(pipe, True)    # fused = already distilled; no toggle needed
    assert pipe.adapter_calls == []


def test_set_distill_returns_early_when_fused_and_distill_false():
    from vivijure_backend.i2v import _set_distill
    pipe = _DistillAwarePipe(loaded=True, fused=True)
    _set_distill(pipe, False)   # can't un-fuse; caller controls step count
    assert pipe.adapter_calls == []


def test_set_distill_toggles_unfused_adapter_on():
    from vivijure_backend.i2v import _set_distill
    pipe = _DistillAwarePipe(loaded=True, fused=False)
    _set_distill(pipe, True)
    assert pipe.adapter_calls == [(["distill"], [1.0])]


def test_set_distill_toggles_unfused_adapter_off():
    from vivijure_backend.i2v import _set_distill
    pipe = _DistillAwarePipe(loaded=True, fused=False)
    _set_distill(pipe, False)
    assert pipe.adapter_calls == [(["distill"], [0.0])]


def test_set_distill_legacy_pipe_without_tags_is_tolerant():
    from vivijure_backend.i2v import _set_distill
    # Old pipe with no _vj_i2v_distill_loaded tag: treat as "unknown", best-effort toggle
    class LegacyPipe:
        adapter_calls = []
        transformer = _FakeTransformer()
        def set_adapters(self, names, adapter_weights):
            self.adapter_calls.append((names, adapter_weights))
    pipe = LegacyPipe()
    _set_distill(pipe, True)    # loaded is None (no tag) -> skip the guard, attempt toggle
    # Either succeeds (fine) or swallows (fine); must not raise


# ------------------------------------------------- frame ceiling reconciliation (#15)

def test_frames_for_respects_max_frames_parameter():
    # The engine ceiling is 81 (MAX_FRAMES) but the config can request higher durations.
    # frames_for with a custom ceiling must honor it.
    assert frames_for(10, 16, max_frames=161) == 161  # 160 -> snapped to 161
    assert frames_for(10, 16, max_frames=81) == 81    # capped at engine default


def test_frames_for_snaps_to_4k_plus_1_even_with_custom_ceiling():
    n = frames_for(10, 16, max_frames=200)
    assert (n - 1) % 4 == 0, f"{n} is not 4k+1"


def test_set_distill_fused_guard_covers_lightx2v_path():
    """When LightX2V baked the delta (loaded=True, fused=True), _set_distill must early-return
    regardless of the distill flag, not call set_adapters. This was the bug: LightX2V set
    fused=False so _set_distill tried set_adapters, raised, and re-raised 'no distill loaded'."""
    from vivijure_backend.i2v import _set_distill

    class BadSetAdaptersPipe:
        _vj_i2v_distill_loaded = True
        _vj_i2v_distill_fused = True  # LightX2V baked: treat as fused

        def set_adapters(self, *a, **kw):
            raise RuntimeError("set_adapters should not be called on a baked-delta pipe")

    _set_distill(BadSetAdaptersPipe(), True)   # must NOT raise
    _set_distill(BadSetAdaptersPipe(), False)  # must NOT raise
