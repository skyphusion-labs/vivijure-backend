"""Device fingerprinting is pure given (capability, name), so the whole fleet matrix is
asserted on a CPU box with no GPU present."""
from vivijure_backend.device import Arch, Attention, Device, Quant, Tier


def test_b200_is_blackwell_fp4():
    d = Device.classify((10, 0), "NVIDIA B200")
    assert d.arch is Arch.BLACKWELL
    assert d.tier is Tier.B200
    assert d.supports_fp4 and d.supports_fp8
    assert d.dit_quant() is Quant.NVFP4
    assert d.video_quant() is Quant.FP8  # video stays fp8 even on a fp4-capable card
    assert d.attention() is Attention.FLASH3


def test_h200_is_hopper_fp8_only():
    d = Device.classify((9, 0), "NVIDIA H200")
    assert d.arch is Arch.HOPPER
    assert d.tier is Tier.H200
    assert d.supports_fp8 and not d.supports_fp4
    assert d.dit_quant() is Quant.FP8  # no fp4 engine on Hopper
    assert d.attention() is Attention.FLASH3


def test_rtx_pro_6000_is_blackwell():
    d = Device.classify((12, 0), "NVIDIA RTX PRO 6000 Blackwell")
    assert d.arch is Arch.BLACKWELL
    assert d.tier is Tier.RTX_PRO_6000
    assert d.supports_fp4


def test_tier_falls_back_to_capability_when_name_unhelpful():
    # An empty / unrecognized name must still resolve the tier from compute capability.
    assert Device.classify((10, 0), "").tier is Tier.B200
    assert Device.classify((12, 0), "some-blackwell").tier is Tier.RTX_PRO_6000
    assert Device.classify((9, 0), "").tier is Tier.H200


def test_unknown_arch_drops_to_bf16_and_sdpa():
    d = Device.classify((7, 0), "Tesla V100S")  # Volta: no fp8/fp4, no FA3
    assert d.arch is Arch.OTHER
    assert not d.supports_fp8 and not d.supports_fp4
    assert d.dit_quant() is Quant.BF16
    assert d.video_quant() is Quant.BF16
    assert d.attention() is Attention.SDPA


def test_classify_uses_reference_vram_when_not_probed():
    assert Device.classify((10, 0), "NVIDIA B200").vram_gb == 192
    # an explicit probe value wins over the reference table
    assert Device.classify((10, 0), "NVIDIA B200", vram_gb=180).vram_gb == 180


def test_detect_falls_back_to_cpu_without_gpu():
    # No torch/CUDA on the dev box: detect() must degrade to a usable CPU Device, not raise.
    d = Device.detect()
    assert d.tier is Tier.UNKNOWN
    assert d.name == "cpu"
    assert d.dit_quant() is Quant.BF16
