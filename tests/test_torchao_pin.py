"""Hold torchao on 0.17.x while the runtime is torch 2.7.1.

0.18.0 does `from torch.nn.functional import ScalingType` at import (torch 2.10+)
in quantization/quantize_/workflows/float8/float8_tensor.py, reached from
`import torchao` via quantization/__init__.py. That made
`from diffusers import WanImageToVideoPipeline` fail on the 1.0.15 H200
smoke (s1-41aa1926802e) after dependabot #419. A pin bump that raises past 0.17
must fail CI until torch is bumped in the same runtime overlay."""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "deploy" / "requirements.txt"
DEPENDABOT = REPO / ".github" / "dependabot.yml"
SMOKE = REPO / "deploy" / "smoke_imports.py"
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


def test_parse_fails_when_pin_is_missing():
    with pytest.raises(AssertionError, match="no torchao== pin"):
        _parse_torchao_pin("diffusers==0.39.0\n")


def test_smoke_imports_covers_torchao_and_wan_i2v():
    """gpu_probe passed on 1.0.15 because it never imported these."""
    text = SMOKE.read_text(encoding="utf-8")
    assert '("torchao"' in text or '("torchao",' in text
    assert "diffusers.pipelines.wan.pipeline_wan_i2v" in text


def test_dependabot_ignores_torchao_018():
    text = DEPENDABOT.read_text(encoding="utf-8")
    assert "torchao" in text
    assert ">=0.18.0" in text
