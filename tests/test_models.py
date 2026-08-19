"""The quant matrix is the load-bearing claim: SDXL is a UNet and never reaches NVFP4 even
on a fp4-capable card, video stays fp8, and adapters load at base dtype. Asserted on CPU."""
from vivijure_backend.device import Device, Quant
from vivijure_backend.models import ModelFamily, ModelRole, ModelServer, quant_for

B200 = Device.classify((10, 0), "NVIDIA B200")
H200 = Device.classify((9, 0), "NVIDIA H200")
RTX6000 = Device.classify((12, 0), "NVIDIA RTX PRO 6000 Blackwell Server Edition")
CPU = Device.classify((0, 0), "cpu")


def test_sdxl_never_reaches_fp4_even_on_blackwell():
    assert quant_for(ModelFamily.SDXL_UNET, B200) is Quant.FP8
    assert quant_for(ModelFamily.SDXL_UNET, H200) is Quant.FP8


def test_dit_gets_fp4_only_where_the_card_supports_it():
    assert quant_for(ModelFamily.DIT, B200) is Quant.NVFP4
    assert quant_for(ModelFamily.DIT, H200) is Quant.FP8  # Hopper has no fp4 engine


def test_video_dit_is_fp8_on_both_archs():
    assert quant_for(ModelFamily.VIDEO_DIT, B200) is Quant.FP8
    assert quant_for(ModelFamily.VIDEO_DIT, H200) is Quant.FP8


def test_aux_always_loads_at_base_dtype():
    for dev in (B200, H200, CPU):
        assert quant_for(ModelFamily.AUX, dev) is Quant.BF16


def test_everything_falls_to_bf16_without_fp8():
    for fam in ModelFamily:
        assert quant_for(fam, CPU) is Quant.BF16


def test_only_dit_is_fp4_capable():
    assert ModelFamily.DIT.fp4_capable
    assert not ModelFamily.SDXL_UNET.fp4_capable
    assert not ModelFamily.VIDEO_DIT.fp4_capable


def test_model_server_plan_needs_no_gpu():
    # plan() reports what the LOADERS produce, not the card ceiling. keyframe_base was asserted here
    # as "fp8" for as long as this test existed; keyframe_pipeline loads bfloat16 and never calls a
    # quantizer, so that assertion encoded the #360 fiction rather than catching it.
    plan = ModelServer(device=B200).plan(env={})
    assert plan[ModelRole.KEYFRAME_BASE.value] == "bf16"   # SDXL: bf16 load, no quantize (#12535)
    assert plan[ModelRole.I2V.value] == "fp8"              # Wan video DiT, 192GB card, fp8 on
    assert plan[ModelRole.CONTROLNET_POSE.value] == "bf16"  # aux
    # every default role is represented
    assert set(plan) == {r.value for r in ModelRole}


def test_default_specs_cover_every_role():
    from vivijure_backend.models import DEFAULT_SPECS
    assert set(DEFAULT_SPECS) == set(ModelRole)
    for role, spec in DEFAULT_SPECS.items():
        assert spec.role is role
        assert spec.repo_id  # a real, non-empty HF id


def test_default_specs_include_controlnet_canny():
    from vivijure_backend.models import DEFAULT_SPECS
    spec = DEFAULT_SPECS[ModelRole.CONTROLNET_CANNY]
    assert spec.repo_id == "xinsir/controlnet-canny-sdxl-1.0"
    assert spec.family is ModelFamily.AUX


def test_bake_manifest_keep_set_includes_canny_controlnet():
    import json
    from pathlib import Path
    from vivijure_backend.models import DEFAULT_SPECS
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root / "deploy" / "bake-manifest.json").read_text())
    repos = {e["repo"]: e for e in m["keep_repos"]}
    canny = repos["models--xinsir--controlnet-canny-sdxl-1.0"]
    assert canny["role"] == "CONTROLNET"
    assert canny["rev"] == "1271357eda52d54b857c650cacb5b51144643ccb"
    assert canny["approx_gb_bf16"] == 2.5
    assert m["model_version"] == 1  # bump is the seed-build dispatch, not this src PR
    assert DEFAULT_SPECS[ModelRole.CONTROLNET_CANNY].repo_id == "xinsir/controlnet-canny-sdxl-1.0"


def test_missing_canny_weights_raise_harness_error_naming_the_role():
    from vivijure_backend.harness.handler import HarnessError
    from vivijure_backend.models import controlnet_load_failure
    err = controlnet_load_failure(ModelRole.CONTROLNET_CANNY, OSError("LocalEntryNotFoundError"))
    assert isinstance(err, HarnessError)
    assert "CONTROLNET_CANNY" in str(err)
    assert "xinsir/" not in str(err)  # role, not a raw diffusers path
    pose = controlnet_load_failure(ModelRole.CONTROLNET_POSE, OSError("missing pose"))
    assert isinstance(pose, OSError)


def test_controlnet_cache_keeps_both_and_swaps():
    from vivijure_backend.models import ModelServer
    server = ModelServer(device=H200)
    loaded = []

    def fake_load(spec):
        loaded.append(spec.role)
        return f"cn-{spec.role.value}"

    server._load_controlnet = fake_load
    pose = server._controlnet(ModelRole.CONTROLNET_POSE)
    canny = server._controlnet(ModelRole.CONTROLNET_CANNY)
    assert pose == "cn-controlnet_pose"
    assert canny == "cn-controlnet_canny"
    assert server._cache["controlnet_pose"] is pose
    assert server._cache["controlnet_canny"] is canny
    assert server._controlnet(ModelRole.CONTROLNET_POSE) is pose  # cache hit
    assert loaded == [ModelRole.CONTROLNET_POSE, ModelRole.CONTROLNET_CANNY]


def test_low_vram_unloads_idle_controlnet():
    from vivijure_backend.device import Arch, Device, Tier
    from vivijure_backend.models import ModelServer
    small = Device(name="local-16gb", capability=(8, 9), arch=Arch.OTHER,
                   tier=Tier.UNKNOWN, vram_gb=16, bandwidth_tbs=0)
    server = ModelServer(device=small)
    server._load_controlnet = lambda spec: f"cn-{spec.role.value}"
    server._controlnet(ModelRole.CONTROLNET_POSE)
    server._controlnet(ModelRole.CONTROLNET_CANNY)
    assert "controlnet_pose" not in server._cache  # idle dropped
    assert "controlnet_canny" in server._cache


# ------------------------------------------------------- repo_id allowlist (cold-start security)

from vivijure_backend.models import (
    ALLOWED_REPO_NAMESPACES,
    DEFAULT_SPECS as _DEFAULT_SPECS_FOR_ALLOWLIST,
    InvalidModelRepoId,
    validate_repo_id,
)
import pytest


def test_allowed_namespaces_are_derived_from_default_specs():
    expected = {spec.repo_id.split("/", 1)[0] for spec in _DEFAULT_SPECS_FOR_ALLOWLIST.values()}
    expected |= {
        spec.fp8_repo_id.split("/", 1)[0]
        for spec in _DEFAULT_SPECS_FOR_ALLOWLIST.values()
        if spec.fp8_repo_id
    }
    assert ALLOWED_REPO_NAMESPACES == frozenset(expected)
    assert "SG161222" in ALLOWED_REPO_NAMESPACES
    assert "Wan-AI" in ALLOWED_REPO_NAMESPACES


def test_validate_repo_id_accepts_default_specs_and_same_namespace_swaps():
    for spec in _DEFAULT_SPECS_FOR_ALLOWLIST.values():
        assert validate_repo_id(spec.repo_id) == spec.repo_id
        if spec.fp8_repo_id:
            assert validate_repo_id(spec.fp8_repo_id) == spec.fp8_repo_id
    # Same org, different name (documented deploy-time swap shape) stays allowed.
    assert validate_repo_id("SG161222/RealVisXL_V4.0") == "SG161222/RealVisXL_V4.0"
    assert validate_repo_id("  Wan-AI/Wan2.2-I2V-A14B-Diffusers  ") == (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers")


@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "/etc/passwd",
    "/tmp/models/weights",
    "\\Windows\\System32",
    "C:/Users/evil/model",
    "file:///tmp/x",
    "https://huggingface.co/evil/model",
    "../escape/repo",
    "org/../other",
    "just-a-name",
    "too/many/slashes",
    "evil-org/malware",  # HF-shaped but namespace not in DEFAULT_SPECS
    "cagliostrolab/animagine-xl-4.0",  # note-mentioned alt; not an allowlisted namespace
    "h94/IP-Adapter/extra",  # more than org/name
])
def test_validate_repo_id_rejects_paths_uris_and_foreign_namespaces(bad):
    with pytest.raises(InvalidModelRepoId):
        validate_repo_id(bad)


def test_validate_repo_id_honors_explicit_namespace_override():
    assert validate_repo_id("only-me/model", allowed_namespaces=frozenset({"only-me"})) == "only-me/model"
    with pytest.raises(InvalidModelRepoId):
        validate_repo_id("SG161222/RealVisXL_V5.0", allowed_namespaces=frozenset({"only-me"}))


# ----------------------------------------------------------------- RIFE 64-divisible padding (#245)

def test_pad_to_multiple_aligns_to_64():
    from vivijure_backend.models import _pad_to_multiple
    # already-aligned dims need no padding (multiples of 64)
    for n in (64, 128, 768, 960, 1280, 1344, 1920):
        assert n % 64 == 0 and _pad_to_multiple(n) == 0, n
    # the next-multiple-of-64 padding is in [1, 63]
    assert _pad_to_multiple(1) == 63
    assert _pad_to_multiple(65) == 63
    assert _pad_to_multiple(1080) == 8  # 1080 -> 1088 (not itself 64-aligned; pad+crop is a safe no-op visually)


def test_pad_to_multiple_rescues_every_non64_backend():
    """Each i2v backend that pulled for Monday (#246) emits a non-64 dim at 16:9; padding lifts it
    to the next multiple of 64 so RIFE stops crashing (#245)."""
    from vivijure_backend.models import _pad_to_multiple
    pad = _pad_to_multiple
    # alibaba-wan 1270x726 -> 1280x768 (the exact crash in #245: 1270 -> +10 -> 1280)
    assert (pad(726), pad(1270)) == (42, 10)
    # seedance / google-veo / vidu-q3 1280x720 -> width aligned, height 720 -> 768
    assert (pad(720), pad(1280)) == (48, 0)
    # minimax-hailuo 1364x768 -> width 1364 -> 1408, height aligned
    assert (pad(768), pad(1364)) == (0, 44)
    # and a square non-64 case stays handled (no aspect-ratio assumption)
    assert pad(726) == 42


# ------------------------------------------------------- i2v weight-source selection (offline-load gap)

from vivijure_backend.models import DEFAULT_SPECS, _select_i2v_weights

_I2V_SPEC = DEFAULT_SPECS[ModelRole.I2V]


def test_i2v_bf16_seed_loads_baked_bf16_not_the_unbaked_fp8_repo():
    # The sm_120 de-risk bug: a bf16-seed image has only the bf16 repo, but the runtime asked for the
    # -fp8 repo and crashed offline. With fp8 absent, every tier loads the baked bf16 repo, no R2 pull
    # (it is baked), and runtime-quantizes (weights_are_fp8 False).
    for final_tier in (False, True):
        repo, weights_are_fp8, pull = _select_i2v_weights(
            _I2V_SPEC, baked=True, final_tier=final_tier, fp8_present=False)
        assert repo == _I2V_SPEC.repo_id
        assert weights_are_fp8 is False
        assert pull is False


def test_i2v_fp8_bake_takes_the_prefp8_fast_path_on_draft_standard():
    repo, weights_are_fp8, pull = _select_i2v_weights(
        _I2V_SPEC, baked=True, final_tier=False, fp8_present=True)
    assert repo == _I2V_SPEC.fp8_repo_id
    assert weights_are_fp8 is True
    assert pull is False


def test_i2v_fp8_bake_final_tier_lazy_pulls_bf16_from_r2():
    repo, weights_are_fp8, pull = _select_i2v_weights(
        _I2V_SPEC, baked=True, final_tier=True, fp8_present=True)
    assert repo == _I2V_SPEC.repo_id
    assert weights_are_fp8 is False
    assert pull is True


def test_i2v_not_baked_pulls_bf16_and_quantizes():
    repo, weights_are_fp8, pull = _select_i2v_weights(
        _I2V_SPEC, baked=False, final_tier=False, fp8_present=False)
    assert repo == _I2V_SPEC.repo_id
    assert weights_are_fp8 is False
    assert pull is True


# ------------------------------------------------------- facexlib offline shim (finish-leg egress gap)

from vivijure_backend.models import _ensure_facexlib_offline, _FACEXLIB_WEIGHTS


def test_ensure_facexlib_offline_symlinks_baked_weights_where_facexlib_looks(tmp_path, monkeypatch):
    # GFPGAN 1.3.8 looks in cwd-relative gfpgan/weights; the shim must place the baked weights there
    # so facexlib's os.path.exists() check passes and it never fetches from github.
    models_root = tmp_path / "models"
    baked = models_root / "facexlib"
    baked.mkdir(parents=True)
    for fn in _FACEXLIB_WEIGHTS:
        (baked / fn).write_bytes(b"w")
    monkeypatch.chdir(tmp_path)
    _ensure_facexlib_offline(str(models_root))
    for fn in _FACEXLIB_WEIGHTS:
        link = tmp_path / "gfpgan" / "weights" / fn
        assert link.exists(), f"{fn} not resolvable where facexlib looks"
        assert link.resolve() == (baked / fn).resolve()


def test_ensure_facexlib_offline_is_a_noop_without_baked_weights(tmp_path, monkeypatch):
    # No baked dir -> nothing to link, must not raise and must not create empty link targets.
    monkeypatch.chdir(tmp_path)
    _ensure_facexlib_offline(str(tmp_path / "absent"))
    assert not (tmp_path / "gfpgan" / "weights").exists()


def test_ensure_facexlib_offline_idempotent(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    baked = models_root / "facexlib"
    baked.mkdir(parents=True)
    for fn in _FACEXLIB_WEIGHTS:
        (baked / fn).write_bytes(b"w")
    monkeypatch.chdir(tmp_path)
    _ensure_facexlib_offline(str(models_root))
    _ensure_facexlib_offline(str(models_root))  # second call must not raise on existing links
    for fn in _FACEXLIB_WEIGHTS:
        assert (tmp_path / "gfpgan" / "weights" / fn).exists()


def test_facexlib_manifest_pins_match_the_runtime_weight_set():
    """The manifest pins (what the bake stages + the gates assert) MUST be exactly the files the runtime
    shim resolves offline (_FACEXLIB_WEIGHTS). If they drift, the bake could carry the wrong set or miss
    one and face restore would phone github at render time."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root / "deploy" / "bake-manifest.json").read_text())
    fx = next(fd for fd in m["finish_dirs"] if fd["dir"] == "facexlib")
    pinned = {f["name"] for f in fx["files"]}
    assert pinned == set(_FACEXLIB_WEIGHTS), (pinned, set(_FACEXLIB_WEIGHTS))


# ------------------------------------------------- loaded precision vs card ceiling (#360 / #364)
#
# The card ceiling (quant_for) and what the loaders produce (loaded_quant_for) are separate
# questions, and conflating them is what let plan() report fp8 for roles that load bf16 out of an
# image holding zero fp8 weight files. These assert the SECOND question against the loader bodies.

from vivijure_backend.models import (  # noqa: E402 -- grouped with the block that uses them
    dominant_dtype, dtype_histogram, i2v_cpu_offload, i2v_fp8_active, i2v_precision_facts,
    loaded_quant_for, merge_histograms,
)


def test_sdxl_loads_bf16_on_every_card_despite_the_fp8_ceiling():
    # The ceiling is genuinely fp8 and is genuinely never used: peft cannot attach a per-scene
    # character LoRA to a torchao-quantized linear, so the keyframe pipes stay bf16.
    for dev in (B200, H200, RTX6000):
        assert quant_for(ModelFamily.SDXL_UNET, dev) is Quant.FP8       # what the card could do
        assert loaded_quant_for(ModelFamily.SDXL_UNET, dev, {}) is Quant.BF16  # what the loader does


def test_i2v_is_bf16_when_the_card_offloads():
    # A 96GB card CPU-offloads the inactive expert, and i2v_pipeline skips fp8 entirely on that path.
    # The card ceiling says fp8 on this card, so this is exactly where the two answers must differ.
    assert i2v_cpu_offload(RTX6000) is True
    assert quant_for(ModelFamily.VIDEO_DIT, RTX6000) is Quant.FP8
    assert loaded_quant_for(ModelFamily.VIDEO_DIT, RTX6000, {}) is Quant.BF16


def test_i2v_is_bf16_when_fp8_is_turned_off():
    assert loaded_quant_for(ModelFamily.VIDEO_DIT, H200, {}) is Quant.FP8
    assert loaded_quant_for(ModelFamily.VIDEO_DIT, H200, {"VJ_I2V_FP8": "0"}) is Quant.BF16
    assert i2v_fp8_active(H200, {"VJ_I2V_FP8": "0"}) is False
    assert i2v_fp8_active(CPU, {}) is False  # no fp8 engine at all


def test_plan_tracks_the_env_that_the_loader_reads():
    off = ModelServer(device=H200).plan(env={"VJ_I2V_FP8": "0"})
    assert off[ModelRole.I2V.value] == "bf16"
    on = ModelServer(device=H200).plan(env={})
    assert on[ModelRole.I2V.value] == "fp8"


# ------------------------------------------------------------- resident dtype measurement (#364)

class _Param:
    def __init__(self, dtype):
        self.dtype = dtype


class _Module:
    """Minimal stand-in for a diffusers module: just `.parameters()`."""

    def __init__(self, dtypes):
        self._params = [_Param(d) for d in dtypes]

    def parameters(self):
        return iter(self._params)


class _Pipe:
    def __init__(self, transformer=None, transformer_2=None):
        if transformer is not None:
            self.transformer = transformer
        if transformer_2 is not None:
            self.transformer_2 = transformer_2


def test_dtype_histogram_counts_every_parameter_not_just_the_first():
    # The first parameter is a keep-in-fp32 minority; a next(parameters()).dtype reading would
    # answer "float32" for a model that is overwhelmingly bf16. THAT is the failure mode.
    mod = _Module(["torch.float32"] + ["torch.bfloat16"] * 9)
    hist = dtype_histogram(mod)
    assert hist == {"float32": 1, "bfloat16": 9}
    assert dominant_dtype(hist) == "bfloat16"


def test_dominant_dtype_of_nothing_is_none_not_a_dtype():
    # An unmeasured model must not be indistinguishable from a correctly loaded one.
    assert dominant_dtype({}) is None
    assert dominant_dtype(merge_histograms([])) is None


def test_dominant_dtype_breaks_ties_deterministically():
    assert dominant_dtype({"float32": 4, "bfloat16": 4}) == "bfloat16"


def test_precision_facts_report_a_bf16_load_as_matching():
    pipe = _Pipe(_Module(["torch.bfloat16"] * 5), _Module(["torch.bfloat16"] * 5))
    facts = i2v_precision_facts(pipe, requested_dtype="bfloat16", repo_id="org/wan",
                                weights_are_fp8=False, runtime_quantized=True)
    assert facts["i2v_dtype"] == "bfloat16"
    assert facts["matches_request"] is True
    assert facts["runtime_quantized"] is True
    assert set(facts["experts"]) == {"transformer", "transformer_2"}


def test_precision_facts_catch_the_float8_request_that_silently_yields_fp32():
    # The latent trap: diffusers keeps WanTransformer3DModel out of the float8 cast, so the model
    # is instantiated at the process default and NOTHING raises. The measurement is the only thing
    # that can tell requested from resident.
    pipe = _Pipe(_Module(["torch.float32"] * 8), _Module(["torch.float32"] * 8))
    facts = i2v_precision_facts(pipe, requested_dtype="float8_e4m3fn", repo_id="org/wan-fp8",
                                weights_are_fp8=True, runtime_quantized=False)
    assert facts["i2v_dtype"] == "float32"
    assert facts["matches_request"] is False


def test_precision_facts_do_not_pass_when_nothing_was_measured():
    # A pipe with no transformer attribute at all: resident is None, so matches_request is False.
    # Absence must never read as agreement.
    facts = i2v_precision_facts(_Pipe(), requested_dtype="bfloat16", repo_id="org/wan",
                                weights_are_fp8=False, runtime_quantized=False)
    assert facts["i2v_dtype"] is None
    assert facts["matches_request"] is False
    assert facts["experts"] == {}


# ---------------------------------------------- ATTACHED onnx providers, never requested (#350)
#
# The InstantID face path ran on CPU through three green releases because the REQUESTED provider
# list said CUDA the whole time. These assert the read-back, and every one of them would pass
# against the defect if the requested list were recorded instead, which is the point.

from vivijure_backend.models import (  # noqa: E402
    attached_onnx_providers, onnx_provider_facts,
)


class _Session:
    def __init__(self, providers):
        self._providers = providers

    def get_providers(self):
        return list(self._providers)


class _OnnxModel:
    def __init__(self, session):
        self.session = session


class _Analyzer:
    def __init__(self, models):
        self.models = models


def test_attached_providers_read_the_session_not_the_request():
    app = _Analyzer({
        "detection": _OnnxModel(_Session(["CUDAExecutionProvider", "CPUExecutionProvider"])),
        "recognition": _OnnxModel(_Session(["CUDAExecutionProvider", "CPUExecutionProvider"])),
    })
    facts = onnx_provider_facts(attached_onnx_providers(app))
    assert facts["all_cuda"] is True
    assert facts["cuda_attached"] is True
    assert facts["per_model"]["detection"][0] == "CUDAExecutionProvider"


def test_the_cpu_bound_face_path_is_visible():
    # THE #346 shape: every session fell back to CPU while the requested list said CUDA. The render
    # would have succeeded (face_analyzer passes a CPU fallback by design), so nothing else in the
    # system could have caught this.
    app = _Analyzer({
        "detection": _OnnxModel(_Session(["CPUExecutionProvider"])),
        "recognition": _OnnxModel(_Session(["CPUExecutionProvider"])),
    })
    facts = onnx_provider_facts(attached_onnx_providers(app))
    assert facts["cuda_attached"] is False
    assert facts["all_cuda"] is False


def test_partial_cuda_is_not_healthy():
    # One session on CUDA and one on CPU. An any()-shaped summary would call this fine.
    app = _Analyzer({
        "detection": _OnnxModel(_Session(["CUDAExecutionProvider"])),
        "recognition": _OnnxModel(_Session(["CPUExecutionProvider"])),
    })
    facts = onnx_provider_facts(attached_onnx_providers(app))
    assert facts["cuda_attached"] is True   # something is on CUDA...
    assert facts["all_cuda"] is False       # ...but the path as a whole is not


def test_an_unreachable_session_is_nothing_attached_not_a_dropped_model():
    class _Broken:
        session = None

    class _Raises:
        class session:  # noqa: N801
            @staticmethod
            def get_providers():
                raise RuntimeError("session gone")

    app = _Analyzer({"detection": _Broken(), "recognition": _Raises()})
    per_model = attached_onnx_providers(app)
    assert per_model == {"detection": [], "recognition": []}   # present and empty, not absent
    assert onnx_provider_facts(per_model)["all_cuda"] is False


def test_no_analyzer_at_all_is_not_healthy():
    assert onnx_provider_facts({})["all_cuda"] is False


def test_model_server_reports_none_until_the_analyzer_loads():
    # None means the identity path did not run on this worker, which must stay distinct from a
    # loaded analyzer reporting CPU. Never triggers a load.
    server = ModelServer(device=H200)
    assert server.onnx_provider_facts() is None
    app = _Analyzer({"detection": _OnnxModel(_Session(["CPUExecutionProvider"]))})
    app._vj_onnx_providers = onnx_provider_facts(attached_onnx_providers(app))
    server._cache["face_analyzer"] = app
    assert server.onnx_provider_facts()["cuda_attached"] is False
