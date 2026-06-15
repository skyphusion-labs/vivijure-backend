"""Capability-aware device selection.

A render worker can land on any of the fleet's GPU tiers, and they do not share the
same fast paths. This module fingerprints the card the worker actually got and exposes
the optimal precision and attention backend for it, so the pipeline never hardcodes a
format a given card cannot accelerate.

Classification is by CUDA compute capability (the reliable signal); the product name is
a secondary hint. Everything is pure given a (capability, name), so it unit-tests on a
CPU box with no GPU present.

Fleet:
  RTX PRO 6000 (Blackwell, sm_120) -> NVFP4 + fp8
  B200         (Blackwell, sm_100) -> NVFP4 + fp8
  H200         (Hopper,    sm_90)  -> fp8 only (no native fp4)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Arch(str, Enum):
    """GPU microarchitecture, the axis that decides the fast path (precision + attention)."""
    BLACKWELL = "blackwell"
    HOPPER = "hopper"
    OTHER = "other"


class Tier(str, Enum):
    """A fleet GPU tier the worker can land on (or UNKNOWN off-GPU)."""
    B200 = "b200"
    H200 = "h200"
    RTX_PRO_6000 = "rtx_pro_6000"
    UNKNOWN = "unknown"


class Quant(str, Enum):
    """A weight precision the card can accelerate."""
    NVFP4 = "nvfp4"  # 4-bit float, Blackwell only
    FP8 = "fp8"      # Hopper + Blackwell
    BF16 = "bf16"    # universal fallback


class Attention(str, Enum):
    """The attention backend to use on this card."""
    FLASH3 = "flash_attn_3"  # Hopper + Blackwell
    SDPA = "sdpa"            # torch built-in fallback


# Reference specs, used when the live probe cannot report them (and for routing hints).
_VRAM_GB = {Tier.B200: 192, Tier.H200: 141, Tier.RTX_PRO_6000: 96}
_BW_TBS = {Tier.B200: 8.0, Tier.H200: 4.8, Tier.RTX_PRO_6000: 1.79}


@dataclass(frozen=True)
class Device:
    """A fingerprinted GPU (or a CPU fallback): its identity plus the precision and attention
    backend it can accelerate. Pure given (capability, name), so it builds and tests off-GPU."""
    name: str
    capability: tuple[int, int]  # (major, minor); B200 = (10, 0), RTX PRO 6000 = (12, 0), H200 = (9, 0)
    arch: Arch
    tier: Tier
    vram_gb: int
    bandwidth_tbs: float

    @property
    def supports_fp4(self) -> bool:
        return self.arch is Arch.BLACKWELL

    @property
    def supports_fp8(self) -> bool:
        return self.arch in (Arch.BLACKWELL, Arch.HOPPER)

    def dit_quant(self) -> Quant:
        """Card ceiling for FP4-capable image DiTs (FLUX / Qwen-Image / SANA): NVFP4 on
        Blackwell, fp8 on Hopper. SDXL is a UNet with no 4-bit engine and never reaches NVFP4;
        use `models.quant_for(family, device)` for the real per-model decision."""
        if self.supports_fp4:
            return Quant.NVFP4
        if self.supports_fp8:
            return Quant.FP8
        return Quant.BF16

    def video_quant(self) -> Quant:
        """Wan i2v: fp8 is the mature video path on both archs; fp4-for-video is young, so
        we do not reach for NVFP4 here even on Blackwell."""
        return Quant.FP8 if self.supports_fp8 else Quant.BF16

    def attention(self) -> Attention:
        """FlashAttention-3 covers Hopper (sm_90) and Blackwell (sm_100 / sm_120)."""
        return Attention.FLASH3 if self.capability[0] >= 9 else Attention.SDPA

    @classmethod
    def classify(cls, capability: tuple[int, int], name: str = "",
                 vram_gb: int = 0, bandwidth_tbs: float = 0.0) -> "Device":
        major = capability[0]
        arch = Arch.HOPPER if major == 9 else Arch.BLACKWELL if major in (10, 12) else Arch.OTHER
        tier = _tier_for(name, capability)
        return cls(
            name=name,
            capability=(capability[0], capability[1]),
            arch=arch,
            tier=tier,
            vram_gb=vram_gb or _VRAM_GB.get(tier, 0),
            bandwidth_tbs=bandwidth_tbs or _BW_TBS.get(tier, 0.0),
        )

    @classmethod
    def detect(cls) -> "Device":
        """Fingerprint the live CUDA device. Falls back to a CPU/unknown Device when no
        GPU is present, so import and tests work off-GPU."""
        try:
            import torch  # local import: the module must load on a CPU box

            if not torch.cuda.is_available():
                raise RuntimeError("cuda unavailable")
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            return cls.classify(
                (props.major, props.minor),
                name=props.name,
                vram_gb=round(props.total_memory / (1024 ** 3)),
            )
        except Exception:
            return cls(name="cpu", capability=(0, 0), arch=Arch.OTHER, tier=Tier.UNKNOWN,
                       vram_gb=0, bandwidth_tbs=0.0)

    def summary(self) -> str:
        return (f"{self.tier.value} ({self.arch.value}, sm_{self.capability[0]}{self.capability[1]}, "
                f"{self.vram_gb}GB): dit={self.dit_quant().value} video={self.video_quant().value} "
                f"attn={self.attention().value}")


def _tier_for(name: str, capability: tuple[int, int]) -> Tier:
    n = name.upper()
    if "B200" in n:
        return Tier.B200
    if "H200" in n:
        return Tier.H200
    if "PRO 6000" in n or "RTX 6000" in n:
        return Tier.RTX_PRO_6000
    # Name unhelpful: fall back to compute capability.
    return {10: Tier.B200, 12: Tier.RTX_PRO_6000, 9: Tier.H200}.get(capability[0], Tier.UNKNOWN)


# Detected once per process.
_CURRENT: "Device | None" = None


def current() -> Device:
    """The process-wide detected device, fingerprinted once and cached."""
    global _CURRENT
    if _CURRENT is None:
        _CURRENT = Device.detect()
    return _CURRENT
