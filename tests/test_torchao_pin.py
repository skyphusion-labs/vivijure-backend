"""Hold torchao on 0.17.x while the runtime is torch 2.7.1.

0.18.0 does `from torch.nn.functional import ScalingType` at import (torch 2.10+).
That made `from diffusers import WanImageToVideoPipeline` fail on the 1.0.15 H200
smoke (s1-41aa1926802e) after dependabot #419. A pin bump that raises past 0.17
must fail CI until torch is bumped in the same runtime overlay."""
from __future__ import annotations

from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[1] / "deploy" / "requirements.txt"
CEILING = (0, 18, 0)


def _parse_torchao_pin(text: str) -> tuple[int, int, int]:
    for line in text.splitlines():
        raw = line.split("#", 1)[0].strip()
        if raw.startswith("torchao=="):
            ver = raw.split("==", 1)[1].strip()
            parts = tuple(int(p) for p in ver.split(".")[:3])
            return (parts + (0, 0, 0))[:3]
    raise AssertionError("deploy/requirements.txt has no torchao== pin")


def test_torchao_stays_below_scalingtype_break():
    pin = _parse_torchao_pin(REQUIREMENTS.read_text(encoding="utf-8"))
    assert pin < CEILING, (
        "torchao %s imports ScalingType from torch.nn.functional (torch 2.10+); "
        "runtime is torch 2.7.1. Bump torch in the same overlay, not torchao alone."
        % ".".join(map(str, pin))
    )
