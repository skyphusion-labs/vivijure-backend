"""Stage-to-GPU-tier routing policy.

`device.py` answers one axis: "what card am I on, and what is its fast path." This
answers the other: "for a given pipeline stage at a given quality tier, which GPU tier
*should* run it." They are independent. Device is an architecture call decided once the
work lands; routing is a cost / throughput call decided before it is scheduled.

How the policy is realized is a deploy choice, not this module's job. Either the control
plane fans stages out to tier-specific RunPod endpoints (cheap card for prep, B200 for
hero i2v), or a render simply runs whole on the tier its i2v demands. This module only
states the ideal target; tune the table without touching the pipeline.
"""
from __future__ import annotations

from enum import Enum

from .device import Tier


class Stage(str, Enum):
    """A pipeline stage the routing table places on a GPU tier (ASSEMBLE is off-GPU)."""
    LORA_TRAIN = "lora_train"
    KEYFRAME = "keyframe"
    I2V = "i2v"
    ASSEMBLE = "assemble"  # off-GPU (CPU container); never schedules to a GPU tier


class QualityTier(str, Enum):
    """The render quality tier: it sets every config baseline and drives stage-to-tier routing."""
    DRAFT = "draft"
    STANDARD = "standard"
    FINAL = "final"

    @classmethod
    def parse(cls, value: object) -> "QualityTier":
        try:
            return cls(str(value).lower())
        except ValueError:
            return cls.FINAL


# (Stage, QualityTier) -> target GPU Tier. ASSEMBLE is off-GPU and is never in the table.
_ROUTES: dict[tuple[Stage, QualityTier], Tier] = {
    # LoRA training is light for SDXL and not really quality-tier dependent (the LoRA is
    # the LoRA); keep it on the cheap card across the board.
    (Stage.LORA_TRAIN, QualityTier.DRAFT): Tier.RTX_PRO_6000,
    (Stage.LORA_TRAIN, QualityTier.STANDARD): Tier.RTX_PRO_6000,
    (Stage.LORA_TRAIN, QualityTier.FINAL): Tier.RTX_PRO_6000,
    # SDXL keyframes are cheap and NVFP4-accelerated on Blackwell -> the entry card is the
    # cost-optimal home regardless of tier.
    (Stage.KEYFRAME, QualityTier.DRAFT): Tier.RTX_PRO_6000,
    (Stage.KEYFRAME, QualityTier.STANDARD): Tier.RTX_PRO_6000,
    (Stage.KEYFRAME, QualityTier.FINAL): Tier.RTX_PRO_6000,
    # i2v is the long pole; it climbs the tiers with quality.
    (Stage.I2V, QualityTier.DRAFT): Tier.RTX_PRO_6000,   # 4-step distilled Wan, cheap
    (Stage.I2V, QualityTier.STANDARD): Tier.H200,        # fp8 workhorse, big VRAM batching
    (Stage.I2V, QualityTier.FINAL): Tier.B200,           # full-step + cache, max batch, hero
}


def gpu_for(stage: Stage, quality: QualityTier) -> Tier | None:
    """The GPU tier that should run `stage` at `quality`. None means off-GPU (assemble)."""
    if stage is Stage.ASSEMBLE:
        return None
    return _ROUTES.get((stage, quality), Tier.H200)  # mid-tier is the safe default


# RunPod GPU-type identifiers to request per tier. CONFIRM these against the live RunPod
# GPU-type ids before wiring submission; the display strings change.
RUNPOD_GPU_ID: dict[Tier, str] = {
    Tier.B200: "NVIDIA B200",
    Tier.H200: "NVIDIA H200",
    Tier.RTX_PRO_6000: "NVIDIA RTX PRO 6000 Blackwell",
}
