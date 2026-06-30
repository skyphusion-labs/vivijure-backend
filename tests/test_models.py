"""The quant matrix is the load-bearing claim: SDXL is a UNet and never reaches NVFP4 even
on a fp4-capable card, video stays fp8, and adapters load at base dtype. Asserted on CPU."""
from vivijure_backend.device import Device, Quant
from vivijure_backend.models import ModelFamily, ModelRole, ModelServer, quant_for

B200 = Device.classify((10, 0), "NVIDIA B200")
H200 = Device.classify((9, 0), "NVIDIA H200")
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
    plan = ModelServer(device=B200).plan()
    assert plan[ModelRole.KEYFRAME_BASE.value] == "fp8"   # SDXL
    assert plan[ModelRole.I2V.value] == "fp8"             # Wan video DiT
    assert plan[ModelRole.CONTROLNET_POSE.value] == "bf16"  # aux
    # every default role is represented
    assert set(plan) == {r.value for r in ModelRole}


def test_default_specs_cover_every_role():
    from vivijure_backend.models import DEFAULT_SPECS
    assert set(DEFAULT_SPECS) == set(ModelRole)
    for role, spec in DEFAULT_SPECS.items():
        assert spec.role is role
        assert spec.repo_id  # a real, non-empty HF id


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
