"""CPU tests for the finishing stage (RIFE interpolation + face restore).

The pure frame/fps math is exercised directly. The GPU body (`finish_clip`) and the restorer
wrappers defer their heavy imports (torch / imageio / GFPGAN / CodeFormer), so they are tested
with fakes injected for `server`, the restorer/interpolator, and `imageio.v3` (stubbed into
`sys.modules`), no GPU and no real codec touched. The encode seam (`_encode_uniform`, an ffmpeg
rawvideo pipe) is monkeypatched with a recorder so nothing spawns ffmpeg or imports numpy. Frames
are plain sentinel objects, since the fakes never inspect pixels, so the suite needs no numpy/torch
(it runs on the CI CPU box with only pytest + PyYAML). The bug fixes the audit called out -- the
dead `only_faces` paste-back, the per-backend fidelity argument name, the silent missing-loader
skip, and the factor snapping -- each get an explicit test.
"""
import sys
import types

import pytest

from vivijure_backend import finish
from vivijure_backend.config import FaceRestore, FinishConfig, RenderConfig
from vivijure_backend.finish import (
    FinishParams,
    _encoder_argv,
    interpolated_fps,
    interpolated_frame_count,
    interpolation_passes,
    output_fps,
    snap_factor,
)
from vivijure_backend.pipeline import finish_params_from
from vivijure_backend.routing import QualityTier


# --------------------------------------------------------------------------- pure frame/fps math

def test_snap_factor_rounds_down_to_a_power_of_two():
    # The whole point of the snap: 3/5/6/7 are not valid RIFE factors and must land on a power of
    # two, rounding DOWN so a request never silently buys more interpolation than asked.
    assert snap_factor(1) == 1
    assert snap_factor(2) == 2
    assert snap_factor(3) == 2
    assert snap_factor(4) == 4
    assert snap_factor(5) == 4
    assert snap_factor(6) == 4
    assert snap_factor(7) == 4
    assert snap_factor(8) == 8


def test_snap_factor_clamps_and_defangs_junk():
    assert snap_factor(0) == 1          # off
    assert snap_factor(-3) == 1
    assert snap_factor(99) == 8         # capped at MAX_FACTOR then snapped
    assert snap_factor("nonsense") == 1
    assert snap_factor(None) == 1


def test_interpolation_passes_is_log2_of_the_snapped_factor():
    assert interpolation_passes(1) == 0
    assert interpolation_passes(2) == 1
    assert interpolation_passes(4) == 2
    assert interpolation_passes(8) == 3
    assert interpolation_passes(6) == 2   # 6 snaps to 4 -> 2 passes


def test_interpolated_frame_count_grows_per_pass():
    # N -> (N-1)*factor + 1; a 1-frame (or empty) clip is unchanged (no pair to interpolate).
    assert interpolated_frame_count(81, 2) == 161
    assert interpolated_frame_count(81, 4) == 321
    assert interpolated_frame_count(1, 8) == 1
    assert interpolated_frame_count(0, 8) == 0
    assert interpolated_frame_count(81, 1) == 81


def test_output_fps_respects_target_cap():
    p = FinishParams(interpolate=True, factor=4)            # 16 -> 64
    assert output_fps(16, p) == 64
    capped = FinishParams(interpolate=True, factor=4, target_fps=30)
    assert output_fps(16, capped) == 30                     # the hard cap wins
    assert interpolated_fps(16, 4) == 64
    off = FinishParams(interpolate=False)
    assert output_fps(16, off) == 16                        # no interpolation: source fps unchanged


# --------------------------------------------------------------- config snaps the factor (item 7)

def test_config_snaps_interpolation_factor_to_a_valid_power_of_two():
    # The [1,8] clamp would let 3/5/6/7 through and finish would silently round them down at render
    # time. Snapping at config validation makes the stored config the truth.
    assert FinishConfig.from_dict({"interpolation_factor": 3}).interpolation_factor == 2
    assert FinishConfig.from_dict({"interpolation_factor": 5}).interpolation_factor == 4
    assert FinishConfig.from_dict({"interpolation_factor": 6}).interpolation_factor == 4
    assert FinishConfig.from_dict({"interpolation_factor": 7}).interpolation_factor == 4
    assert FinishConfig.from_dict({"interpolation_factor": 8}).interpolation_factor == 8
    assert FinishConfig.from_dict({"interpolation_factor": 99}).interpolation_factor == 8  # clamp+snap


# --------------------------------------------------------------- config -> FinishParams mapper

def test_finish_params_from_maps_none_to_face_restore_off():
    cfg = RenderConfig.from_request("standard", {"finish": {"interpolate": True, "face_restore": "none"}})
    p = finish_params_from(cfg)
    assert p.interpolate is True
    assert p.face_restore is False                          # NONE -> off


def test_finish_params_from_carries_the_chosen_restorer_backend():
    for name in ("gfpgan", "codeformer"):
        cfg = RenderConfig.from_request("final", {"finish": {"face_restore": name}})
        p = finish_params_from(cfg)
        assert p.face_restore is True
        assert p.face_restore_backend == name


def test_finish_params_from_threads_factor_and_knobs():
    cfg = RenderConfig.from_request("final", {"finish": {
        "interpolate": True, "interpolation_factor": 6, "target_fps": 30,
        "face_restore": "gfpgan", "face_fidelity": 0.4, "only_faces": False}})
    p = finish_params_from(cfg)
    assert p.factor == 4                                    # 6 snapped at config -> 4 here
    assert p.target_fps == 30
    assert p.face_fidelity == 0.4
    assert p.only_faces is False


def test_finish_config_for_tier_baselines():
    assert FinishConfig.for_tier(QualityTier.DRAFT).enabled is False
    assert FinishConfig.for_tier(QualityTier.STANDARD).interpolate is True
    assert FinishConfig.for_tier(QualityTier.STANDARD).face_restore is FaceRestore.NONE
    assert FinishConfig.for_tier(QualityTier.FINAL).face_restore is FaceRestore.GFPGAN


# --------------------------------------------------------------- bug fix: only_faces is live (item 4)

class _RecordingRestorer:
    """A fake face restorer that records the kwargs each call receives, so a test can assert the
    finish stage passes the real, honest knobs (and no longer the dead `paste_back=... or True`)."""
    def __init__(self):
        self.calls = []

    def restore(self, frame, **kwargs):
        self.calls.append(kwargs)
        return frame


def test_restore_frame_passes_only_faces_through_honestly():
    # The bug: `paste_back=not cfg.only_faces or True` was ALWAYS True, so only_faces was dead and
    # paste_back/only_center_face leaked into the call. The fix passes only the uniform knobs and
    # lets only_faces actually vary.
    r = _RecordingRestorer()
    finish._restore_frame(r, object(), FinishParams(face_fidelity=0.3, only_faces=True))
    finish._restore_frame(r, object(), FinishParams(face_fidelity=0.3, only_faces=False))
    assert r.calls[0] == {"fidelity": 0.3, "only_faces": True}
    assert r.calls[1] == {"fidelity": 0.3, "only_faces": False}     # the flag is LIVE now
    # the dead arguments are gone
    assert "paste_back" not in r.calls[0]
    assert "only_center_face" not in r.calls[0]


def test_restore_frame_is_best_effort_per_frame():
    class Boom:
        def restore(self, frame, **k):
            raise RuntimeError("detector miss")
    f = object()
    out = finish._restore_frame(Boom(), f, FinishParams())
    assert out is f                                          # a bad frame passes through untouched


# --------------------------------------------------- bug fix: per-backend fidelity arg (item 4)

def test_gfpgan_restorer_maps_fidelity_to_weight():
    # GFPGAN's fidelity knob is `weight`. Build the wrapper without running __init__ (no GFPGAN
    # install on the CPU box) and inject a fake underlying restorer, then assert the mapping.
    from vivijure_backend.models import _GfpganRestorer

    class FakeGfpgan:
        def __init__(self): self.seen = {}
        def enhance(self, frame, **kw):
            self.seen = kw
            return None, None, frame

    r = _GfpganRestorer.__new__(_GfpganRestorer)
    r._restorer = FakeGfpgan()
    frame = object()
    r.restore(frame, fidelity=0.25, only_faces=True)
    assert r._restorer.seen["weight"] == 0.25               # GFPGAN: fidelity -> weight
    assert r._restorer.seen["paste_back"] is True           # restored faces are composited back


def test_codeformer_restorer_maps_fidelity_to_w():
    # CodeFormer's fidelity knob is `w`. Drive the wrapper's restore() with a fully faked helper +
    # net (no CodeFormer / basicsr / facexlib on the CPU box) and assert the mapping.
    from vivijure_backend.models import _CodeFormerRestorer

    seen = {}

    class FakeHelper:
        def __init__(self): self.cropped_faces = [_FakeTensor()]
        def clean_all(self): pass
        def read_image(self, f): pass
        def get_face_landmarks_5(self, only_center_face=False): pass
        def align_warp_face(self): pass
        def add_restored_face(self, f): pass
        def get_inverse_affine(self, x): pass
        def paste_faces_to_input_image(self, upsample_img=None): return "restored"

    class _FakeTensor:
        def unsqueeze(self, *a): return self
        def to(self, *a, **k): return self
        def astype(self, *a): return self
        def __truediv__(self, other): return self     # cropped / 255.0 in the wrapper

    def fake_net(t, w=None, adain=None):
        seen["w"] = w
        return [_FakeTensor()]

    # Stub the basicsr/torchvision helpers the restore() body defers-imports, so the call runs on CPU.
    fake_basicsr_utils = types.ModuleType("basicsr.utils")
    fake_basicsr_utils.img2tensor = lambda img, **k: _FakeTensor()
    fake_basicsr_utils.tensor2img = lambda t, **k: _FakeTensor()
    fake_tv = types.ModuleType("torchvision.transforms.functional")
    fake_tv.normalize = lambda *a, **k: None

    fake_torch = types.ModuleType("torch")

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake_torch.no_grad = lambda: _NoGrad()

    r = _CodeFormerRestorer.__new__(_CodeFormerRestorer)
    r._torch = fake_torch
    r._device = "cpu"
    r._net = fake_net
    r._helper = FakeHelper()

    with _patched_modules({
        "basicsr.utils": fake_basicsr_utils,
        "torchvision.transforms.functional": fake_tv,
    }):
        out = r.restore(object(), fidelity=0.8, only_faces=True)
    assert seen["w"] == 0.8                                  # CodeFormer: fidelity -> w
    assert out == "restored"


# --------------------------------------------------- finish_clip seam (order, fail-loud, encode)

class _FakeInterp:
    """A fake RIFE interpolator: returns a marker midpoint so a test can count inserted frames."""
    def interpolate(self, a, b):
        return ("mid", a, b)


class _FakeServer:
    """A ModelServer stand-in for finish_clip: hands back fakes (or raises) for each loader."""
    def __init__(self, *, interp=None, restorer=None, interp_raises=False, restore_raises=False):
        self._interp = interp
        self._restorer = restorer
        self._interp_raises = interp_raises
        self._restore_raises = restore_raises
        self.restorer_backend = None

    def frame_interpolator(self):
        if self._interp_raises:
            raise RuntimeError("RIFE weights missing")
        return self._interp

    def face_restorer(self, backend=None):
        self.restorer_backend = backend
        if self._restore_raises:
            raise RuntimeError("GFPGAN weights missing")
        return self._restorer


class _EncodeRecorder:
    """Stands in for finish._encode_uniform: consumes the streamed frame generator (so the finish
    pipeline actually runs) and records the frame list, the realized fps, and whether it was handed
    a lazy generator -- all without spawning ffmpeg or importing numpy on the CPU box."""
    def __init__(self):
        self.frames = None
        self.fps = None
        self.was_generator = None

    def __call__(self, frames, out_path, fps):
        self.was_generator = isinstance(frames, types.GeneratorType)
        self.frames = list(frames)
        self.fps = fps


class _patched_modules:
    """Context manager that installs fake modules into sys.modules and restores them after, so a
    deferred `import imageio.v3` (and friends) resolves to a CPU fake during the test."""
    def __init__(self, mapping):
        self.mapping = mapping
        self.saved = {}

    def __enter__(self):
        for name, mod in self.mapping.items():
            self.saved[name] = sys.modules.get(name)
            sys.modules[name] = mod
        return self

    def __exit__(self, *a):
        for name, prev in self.saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        return False


def _fake_imageio(frames_in):
    """A fake imageio.v3 module (plus its parent `imageio` package): imiter yields the given frames
    and immeta reports 16 fps. Returns the {module_name: module} mapping for `_patched_modules` --
    both `imageio` and `imageio.v3` are needed because `import imageio.v3` imports the parent first.
    Encoding no longer goes through imageio (it is an ffmpeg pipe now), so imwrite is absent; tests
    assert the encode via a patched `finish._encode_uniform` recorder instead."""
    v3 = types.ModuleType("imageio.v3")
    v3.immeta = lambda *a, **k: {"fps": 16}
    v3.imiter = lambda *a, **k: iter(frames_in)
    parent = types.ModuleType("imageio")
    parent.v3 = v3
    return {"imageio": parent, "imageio.v3": v3}


def _frames(n):
    return [("frame", i) for i in range(n)]


def test_encoder_argv_uses_nvenc_and_keeps_uniform_output():
    # NVENC path: GPU encoder selected, but the output stays H.264 / yuv420p at the realized fps and
    # frame size, so a finished clip is still concat-compatible with its siblings.
    argv = _encoder_argv("/tmp/out.mp4", 640, 480, 32, "h264_nvenc")
    assert "h264_nvenc" in argv and "libx264" not in argv
    assert argv[argv.index("-s") + 1] == "640x480"
    assert argv[argv.index("-framerate") + 1] == "32"
    assert argv[-3:] == ["-pix_fmt", "yuv420p", "/tmp/out.mp4"]     # output pixel format is uniform


def test_encoder_argv_falls_back_to_libx264():
    argv = _encoder_argv("/tmp/out.mp4", 320, 240, 16, "libx264")
    assert "libx264" in argv and "h264_nvenc" not in argv
    assert argv[-3:] == ["-pix_fmt", "yuv420p", "/tmp/out.mp4"]


def test_finish_clip_restores_then_interpolates_and_encodes_uniformly(tmp_path, monkeypatch):
    rec = _EncodeRecorder()
    monkeypatch.setattr(finish, "_encode_uniform", rec)
    restorer = _RecordingRestorer()
    server = _FakeServer(interp=_FakeInterp(), restorer=restorer)
    params = FinishParams(interpolate=True, factor=2, face_restore=True,
                          face_restore_backend="gfpgan", only_faces=False)
    with _patched_modules(_fake_imageio(_frames(3))):
        res = finish.finish_clip("shot_01", tmp_path / "in.mp4", tmp_path / "out.mp4",
                                 server, params=params)
    # restore ran on every input frame (3), THEN one interpolation pass streamed 3 -> 5 frames
    assert len(restorer.calls) == 3
    assert res.frames_in == 3 and res.frames_out == 5
    assert res.interpolated is True and res.face_restored is True
    assert res.out_fps == 32                                 # 16 * 2
    assert server.restorer_backend == "gfpgan"              # the chosen backend was requested
    # the encoder was fed the full 5-frame sequence lazily (streamed, not a materialized 8x list)
    assert rec.was_generator is True
    assert len(rec.frames) == 5
    assert rec.fps == 32


def test_finish_clip_encodes_uniformly_even_with_no_passes_run(tmp_path, monkeypatch):
    # A single-frame clip cannot interpolate and (here) is not restored, but it is STILL re-encoded
    # to the uniform form so the stream-copy concat survives it next to multi-frame finished clips.
    rec = _EncodeRecorder()
    monkeypatch.setattr(finish, "_encode_uniform", rec)
    server = _FakeServer(interp=_FakeInterp())
    params = FinishParams(interpolate=True, factor=4)        # interpolation requested...
    with _patched_modules(_fake_imageio(_frames(1))):
        res = finish.finish_clip("shot_x", tmp_path / "in.mp4", tmp_path / "out.mp4",
                                 server, params=params)
    assert res.interpolated is False                         # 1 frame: nothing to interpolate
    assert res.out_fps == 16                                 # source fps, untouched
    assert res.frames_out == 1
    assert len(rec.frames) == 1                              # but the single frame is still encoded
    assert rec.fps == 16


def test_finish_clip_fails_loud_when_a_configured_interpolator_cannot_load(tmp_path):
    # The audit fix: a CONFIGURED pass whose model cannot load must FAIL the render, not silently
    # downgrade to a no-op. The old `_safe` swallow is gone.
    server = _FakeServer(interp_raises=True)
    params = FinishParams(interpolate=True, factor=2)
    with _patched_modules(_fake_imageio(_frames(3))):
        with pytest.raises(RuntimeError, match="RIFE weights missing"):
            finish.finish_clip("shot_01", tmp_path / "in.mp4", tmp_path / "out.mp4",
                               server, params=params)


def test_finish_clip_fails_loud_when_a_configured_restorer_cannot_load(tmp_path):
    server = _FakeServer(restore_raises=True)
    params = FinishParams(face_restore=True, face_restore_backend="gfpgan")
    with _patched_modules(_fake_imageio(_frames(3))):
        with pytest.raises(RuntimeError, match="GFPGAN weights missing"):
            finish.finish_clip("shot_01", tmp_path / "in.mp4", tmp_path / "out.mp4",
                               server, params=params)
