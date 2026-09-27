"""kf_hash must change when ANY keyframe config field that changes the drawn image changes.

The planner reuses a cached keyframe when the stored hash equals the current one, so a field
missing from the hash means changing only that field silently reuses the old PNG."""
import pytest

from vivijure_backend.config import IdentityMethod, KeyframeConfig, QualityTier, Scheduler
from vivijure_backend.orchestrator import kf_hash


def _kc() -> KeyframeConfig:
    return KeyframeConfig.for_tier(QualityTier.FINAL)


@pytest.mark.parametrize("field,value", [
    ("scheduler", Scheduler.EULER),
    ("ip_adapter_scale", 0.31),
    ("instantid_ip_adapter_scale", 1.1),
    ("distill_model", "ByteDance/other-fewstep"),
])
def test_kf_hash_changes_when_only_this_field_changes(field, value):
    kc = _kc()
    assert getattr(kc, field) != value
    before = kf_hash(kc)
    setattr(kc, field, value)
    assert kf_hash(kc) != before, f"{field} is missing from the kf_hash payload"


def test_kf_hash_distill_flag_is_not_confused_with_a_matching_step_count():
    # distill=False steps=8 and distill=True distill_steps=8 both hash "steps": 8 today.
    a = _kc()
    a.distill, a.steps = False, 8
    b = _kc()
    b.distill, b.distill_steps = True, 8
    assert kf_hash(a) != kf_hash(b)


def test_kf_hash_is_stable_for_an_unchanged_config():
    assert kf_hash(_kc()) == kf_hash(_kc())
    kc = _kc()
    kc.identity_method = IdentityMethod.INSTANTID
    assert kf_hash(kc) != kf_hash(_kc())
