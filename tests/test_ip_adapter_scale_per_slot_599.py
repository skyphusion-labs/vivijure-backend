"""vivijure-backend#599: `multi_char.ip_adapter_scale_per_slot` reaches the two-person draw, and unset changes nothing.

Before, the field was parsed, documented (default 0.7) and hashed, but `KeyframeParams` had no such field, so both
two-person paths drew at `keyframe.ip_adapter_scale` (0.65). Ruled on the issue: wire it through, unset by default,
falling back to `keyframe.ip_adapter_scale`, so a job that does not set it draws and hashes exactly as before. The
render tests run the real scene-lock and plain regional bodies over the fakes from the #1143 tests.
"""
import pytest
from PIL import Image
from test_debug_intermediates_1143 import (  # noqa: F401
    CAST,
    SB,
    _Pipe,
    _scene,
    _Server,
    gpu_fakes,  # the fixture, by name
)

import vivijure_backend.keyframe as kf
from vivijure_backend.config import KeyframeConfig, MultiCharConfig
from vivijure_backend.contract import RenderRequest, Scene
from vivijure_backend.orchestrator import MULTI_CHAR_DEFAULTS, kf_hash
from vivijure_backend.pipeline import keyframe_params_from

# ------------------------------------------------------------------ config


@pytest.mark.parametrize("given,want", [({}, None), ({"ip_adapter_scale_per_slot": None}, None),
                                        ({"ip_adapter_scale_per_slot": 0.4}, 0.4),
                                        ({"ip_adapter_scale_per_slot": "0.4"}, 0.4),
                                        ({"ip_adapter_scale_per_slot": 2}, 1.0),
                                        ({"ip_adapter_scale_per_slot": -1}, 0.0),
                                        ({"ip_adapter_scale_per_slot": "lots"}, None)])
def test_unset_unless_a_number_is_given(given, want):
    assert MultiCharConfig.from_dict(given).ip_adapter_scale_per_slot == want


def test_the_default_is_unset_everywhere_and_the_planner_says_so():
    assert MultiCharConfig().ip_adapter_scale_per_slot is None
    assert kf.KeyframeParams().ip_adapter_scale_per_slot is None
    assert MULTI_CHAR_DEFAULTS["ip_adapter_scale_per_slot"] is None


def _params(mc=None, **kw):
    r = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                 "render_overrides": {"keyframe": {**kw, **({"multi_char": mc} if mc else {})}}})
    return keyframe_params_from(r.config)


def test_it_reaches_the_render_params():
    # The defect: this field stopped at the config and never reached KeyframeParams.
    assert _params({"ip_adapter_scale_per_slot": 0.4}).ip_adapter_scale_per_slot == 0.4
    assert _params().ip_adapter_scale_per_slot is None


# ------------------------------------------------------------------ the rule

def test_unset_falls_back_to_the_keyframe_scale():
    assert kf.regional_ip_adapter_scale(kf.KeyframeParams(ip_adapter_scale=0.65)) == 0.65
    assert kf.regional_ip_adapter_scale(kf.KeyframeParams(ip_adapter_scale=0.5)) == 0.5
    # The shipped request path: KeyframeConfig's 0.65, which is what both paths drew at before.
    assert kf.regional_ip_adapter_scale(_params()) == KeyframeConfig().ip_adapter_scale == 0.65


@pytest.mark.parametrize("per,want", [(0.4, 0.4), (0.0, 0.0), (1.0, 1.0), (1.7, 1.0), (-0.2, 0.0)])
def test_a_set_value_wins_and_is_clamped(per, want):
    assert kf.regional_ip_adapter_scale(kf.KeyframeParams(ip_adapter_scale=0.65, ip_adapter_scale_per_slot=per)) == want


# ------------------------------------------------------------------ the real render bodies

def _scene_lock(tmp_path, name, **kw):
    params = kf.KeyframeParams(width=64, height=48, identity_method="ip_adapter", plate_count_retries=0,
                               ip_adapter_scale=0.65, **kw)
    pipe = _Pipe(params.width, params.height)
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(pipe), tmp_path / name / "shot_03.png", params=params,
                       lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")})
    return pipe


@pytest.mark.usefixtures("gpu_fakes")
@pytest.mark.parametrize("kw,want", [({}, [0.65, 0.65]), ({"ip_adapter_scale_per_slot": 0.4}, [0.4, 0.4])])
def test_the_scene_lock_pass_b_draws_at_the_regional_scale(tmp_path, kw, want):
    # Pass B is the last draw, so the scale the fake pipe holds afterwards is the one Pass B used.
    assert _scene_lock(tmp_path, "s", **kw).ip_scale == want


@pytest.mark.usefixtures("gpu_fakes")
@pytest.mark.parametrize("kw,want", [({}, [0.65, 0.65]), ({"ip_adapter_scale_per_slot": 0.4}, [0.4, 0.4])])
def test_the_plain_regional_render_draws_at_the_regional_scale(tmp_path, monkeypatch, kw, want):
    monkeypatch.setattr(kf, "_ref_images", lambda cast, slot, count: [Image.new("RGB", (64, 64), (90, 90, 90))])
    params = kf.KeyframeParams(width=64, height=48, identity_method="ip_adapter", scene_lock=False,
                               ip_adapter_scale=0.65, **kw)
    pipe = _Pipe(params.width, params.height)
    kf.render_keyframe(_scene(("A", "B")), CAST, SB, _Server(pipe), tmp_path / "r" / "shot_03.png", params=params,
                       lora_paths={s: tmp_path / f"{s}.safetensors" for s in ("A", "B")})
    assert pipe.ip_scale == want


# ------------------------------------------------------------------ reuse hash, as in #589

# Computed with the code on main at 435e562, the commit before this change. A still drawn with the field unset (every
# still on record) must keep its exact hash; a set value now changes the drawing, so it must move.
BEFORE_599 = {
    "n1": "23b7e0c279e5ed2e",
    "n2": "ee77025e4bf000ed",
    "n2_lock_off": "34f6f9b2d77ff564",
    "n2_ip_0p5": "74fe7b6fcd263681",
    "n2_per_0p7": "ee77025e4bf000ed",
    "n2_per_0p5": "1c366caa1f80026c",
    "cfg_only": "ee77025e4bf000ed",
}


def _shot(n):
    return Scene.from_dict({"id": "s", "prompt": "at a workbench, looking at a laptop",
                            "character_slots": ["A", "B", "C", "D"][:n]}, 0)


def _hash(n, per=None, **kw):
    kc = KeyframeConfig()
    if per is not None:
        kc.multi_char.ip_adapter_scale_per_slot = per
    for k, v in kw.items():
        setattr(kc, k, v)
    return kf_hash(kc, scene=_shot(n))


@pytest.mark.parametrize("case,n,kw", [("n1", 1, {}), ("n2", 2, {}), ("n2_lock_off", 2, {"scene_lock": False}),
                                       ("n2_ip_0p5", 2, {"ip_adapter_scale": 0.5})])
def test_an_unset_still_keeps_its_stored_hash(case, n, kw):
    assert _hash(n, **kw) == BEFORE_599[case]
    assert kf_hash(KeyframeConfig()) == BEFORE_599["cfg_only"]


@pytest.mark.parametrize("per,case", [(0.7, "n2_per_0p7"), (0.5, "n2_per_0p5")])
def test_a_set_value_redraws(per, case):
    # Before, a set value drew exactly as unset (0.65); now it draws at its own scale, so a stored still must not be
    # reused. 0.7 is the sharp case: it used to hash the same as unset.
    assert _hash(2, per=per) != BEFORE_599[case]
    assert _hash(2, per=per) != _hash(2)


def test_two_set_values_hash_apart():
    assert _hash(2, per=0.4) != _hash(2, per=0.5)
