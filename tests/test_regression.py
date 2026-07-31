"""CPU unit tests for the FULL capability regression suite (deploy/runpod_verify.py:
evaluate_regression + RegressionConfig + REGRESSION_EVENTS + the CAP-3 structural RIFE probe). All
CPU-only: no GPU, no pod, no network -- same mock-first pattern as test_runpod_verify.py. Every
assertion targets a named field in a parsed @event payload, never English prose."""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "deploy" / "runpod_verify.py"
    spec = importlib.util.spec_from_file_location("runpod_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register so dataclass string annotations resolve at exec time
    spec.loader.exec_module(mod)
    return mod


rv = _load()


# ----------------------------------------------------------------- mock event streams

def _pass_lines():
    """A complete PASS-shaped regression stream: BAK-3/4 + base smoke + CAP-1..6, every gate green."""
    fe = rv.format_event
    return [
        fe("model_inventory", {"sdxl": True, "wan22": True, "rife_flownet": True,
                               "gfpgan": True, "all_present": True}),
        fe("model_precision", {"i2v_dtype": "bfloat16", "requested_dtype": "bfloat16",
                               "matches_request": True, "repo_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                               "weights_are_fp8": False, "runtime_quantized": True}),
        rv.emit_gpu_probe({"torch_cuda": True, "kernel_ok": True, "vj_baked": True,
                           "weights_on_disk": True, "vram_free_gb": 120.0, "vram_total_gb": 141.0,
                           "device_name": "NVIDIA H200"}),
        fe("mirror_skipped", {"reason": "baked"}),
        "Loading transformer ... (prose the parser must ignore)",
        fe("keyframe_done", {"shot_id": "s0", "key": "kf/s0.png", "width": 1280, "height": 720,
                             "format": "PNG", "bytes": 820_000, "elapsed_s": 31.0}),
        fe("clip_done", {"shot_id": "s0", "clip_key": "clips/s0.mp4", "num_frames": 49, "fps": 16,
                         "seconds": 3.0, "distilled": True, "elapsed_s": 78.0}),
        fe("rife_model_probe", {"block_count": 3, "c_per_block": 90,
                                "flownet_pkl_bytes": 23_400_000, "loaded": True}),
        fe("rife_done", {"shot_id": "s0", "input_frames": 2, "output_frames": 3, "factor": 2,
                         "h": 720, "w": 1280, "elapsed_s": 9.0}),
        fe("finish_done", {"shot_id": "s0", "clip_key": "clips/s0_finished.mp4", "interpolated": True,
                           "face_restored": True, "out_frames": 97, "out_fps": 32,
                           "bytes": 1_400_000, "elapsed_s": 52.0}),
        fe("first_frame", {"seconds": 41.0}),
        fe("sharpness", {"value": 0.99, "baseline": 1.0}),
        fe("e2e_done", {"shots": 2, "output_key": "renders/e2e/film.mp4", "has_audio": True,
                        "duration_s": 6.0, "bytes": 3_200_000, "elapsed_s": 420.0}),
        fe("complete", {"output_key": "renders/e2e/film.mp4", "clip_bytes": 3_200_000}),
    ]


def _without(lines, event_name):
    """Drop every `@event <event_name> ...` line."""
    return [ln for ln in lines if not ln.startswith(f"@event {event_name} ")]


def _replace(lines, event_name, payload):
    """Drop `event_name` then append a single replacement (last-writer-wins keeps it authoritative)."""
    out = _without(lines, event_name)
    out.append(rv.format_event(event_name, payload))
    return out


def _eval(lines, cfg=None):
    cfg = cfg or rv.RegressionConfig(image="img:test")
    return rv.evaluate_regression(rv.parse_events(lines), cfg)


# ----------------------------------------------------------------- the full pass

def test_evaluate_regression_pass():
    res = _eval(_pass_lines())
    assert res.passed is True
    assert all(res.checks.values()), [k for k, v in res.checks.items() if not v]
    # base smoke gates are present AND the new CAP/BAK gates are present (superset, one check dict).
    for k in ("kernel_ok", "sharpness_parity", "render_complete",        # base
              "bak3_all_models_present", "bak4_precision_valid",          # BAK
              "cap1_keyframe_format", "cap2_clip_fps", "cap3_rife_architecture",
              "cap4_finish_key", "cap6_e2e_audio"):                       # CAP
        assert k in res.checks, f"missing gate {k}"
    assert not res.warnings


# ----------------------------------------------------------------- CAP-3 RIFE (highest priority)

def test_evaluate_regression_rife_architecture_fail():
    # block_count=2 (a vendored-code drift from the flownet.pkl c=90/3-block weights) -> CAP-3 fail.
    lines = _replace(_pass_lines(), "rife_model_probe",
                     {"block_count": 2, "c_per_block": 90, "flownet_pkl_bytes": 23_400_000, "loaded": True})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap3_rife_architecture"] is False
    assert any("RIFE architecture" in r for r in res.reasons)


def test_evaluate_regression_rife_wrong_channels_fail():
    # c_per_block=64 (the stock RIFE default, not the c=90 these weights need) -> CAP-3 fail.
    lines = _replace(_pass_lines(), "rife_model_probe",
                     {"block_count": 3, "c_per_block": 64, "flownet_pkl_bytes": 23_400_000, "loaded": True})
    res = _eval(lines)
    assert res.checks["cap3_rife_architecture"] is False


def test_evaluate_regression_rife_output_frames_fail():
    # factor=2 over 2 input frames must yield 3 output frames (A, midpoint, B); 2 means no midpoint.
    lines = _replace(_pass_lines(), "rife_done",
                     {"shot_id": "s0", "input_frames": 2, "output_frames": 2, "factor": 2,
                      "h": 720, "w": 1280, "elapsed_s": 9.0})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap3_rife_output_frames"] is False


def test_evaluate_regression_rife_missing_event_fails_closed():
    # No rife_model_probe / rife_done at all -> CAP-3 gates fail (never silently pass).
    lines = _without(_without(_pass_lines(), "rife_model_probe"), "rife_done")
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap3_rife_loaded"] is False
    assert res.checks["cap3_rife_output_frames"] is False
    assert res.checks["cap3_rife_time"] is False


# ----------------------------------------------------------------- BAK-3 inventory / BAK-4 precision

def test_evaluate_regression_model_missing():
    lines = _replace(_pass_lines(), "model_inventory",
                     {"sdxl": True, "wan22": True, "rife_flownet": False, "gfpgan": True,
                      "all_present": False})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["bak3_all_models_present"] is False
    assert any("rife_flownet" in r for r in res.reasons)


def test_evaluate_regression_precision_absent_fails_with_its_own_reason():
    # An ABSENT model_precision event and a BAD one are different facts. Before this split, absence
    # was reported as "precision None is not a valid baked precision", which reads as a measured bad
    # value; nothing in src/ emitted the event at all, so that was the state the gate lived in.
    lines = _without(_pass_lines(), "model_precision")
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["bak4_precision_reported"] is False
    assert res.checks["bak4_precision_matches_request"] is False
    assert res.metrics["i2v_precision_reported"] is False
    assert any("never measured" in r for r in res.reasons), res.reasons


def test_evaluate_regression_precision_reported_gate_passes_when_it_is_reported():
    # POSITIVE CONTROL: the reported-gate is not stuck False. Without this, the test above would
    # pass against a gate hardcoded to fail.
    res = _eval(_pass_lines())
    assert res.checks["bak4_precision_reported"] is True
    assert res.checks["bak4_precision_matches_request"] is True


def test_evaluate_regression_precision_resident_not_matching_request_fails():
    # The keep-in-fp32 trap seen from the harness side: the load asked for float8 and the resident
    # model is fp32. Both fields are present and each looks plausible alone; only the comparison
    # catches it.
    lines = _replace(_pass_lines(), "model_precision",
                     {"i2v_dtype": "float32", "requested_dtype": "float8_e4m3fn",
                      "matches_request": False})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["bak4_precision_matches_request"] is False
    assert res.checks["bak4_precision_valid"] is False  # fp32 is never a valid bake either


def test_evaluate_regression_precision_stale_payload_without_the_field_fails_closed():
    # An emitter predating `matches_request` must not read as agreement. A missing field is an
    # unanswered question, and `.get()` returning None is exactly how that becomes a silent pass.
    lines = _replace(_pass_lines(), "model_precision", {"i2v_dtype": "bfloat16"})
    res = _eval(lines)
    assert res.checks["bak4_precision_reported"] is True   # the event WAS emitted
    assert res.checks["bak4_precision_valid"] is True      # and the dtype IS valid
    assert res.checks["bak4_precision_matches_request"] is False  # but nothing confirmed the match
    assert res.passed is False


def test_evaluate_regression_precision_fp8_warns_not_fails():
    # fp8 is a VALID baked precision; prod expects bf16, so fp8 WARNS but the run still passes.
    lines = _replace(_pass_lines(), "model_precision",
                     {"i2v_dtype": "float8_e4m3fn", "requested_dtype": "float8_e4m3fn",
                      "matches_request": True})
    res = _eval(lines)
    assert res.checks["bak4_precision_valid"] is True
    assert res.passed is True
    assert any("differs from the expected" in w for w in res.warnings)


def test_evaluate_regression_precision_fp32_fails():
    # fp32 is never a valid bake -> BAK-4 hard fail (not a warn).
    lines = _replace(_pass_lines(), "model_precision",
                     {"i2v_dtype": "float32", "requested_dtype": "float32",
                      "matches_request": True})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["bak4_precision_valid"] is False


# ----------------------------------------------------------------- baked-sentinel (base gate, regression path)

def test_evaluate_regression_baked_r2_pull():
    # A heavy mirror_complete leg on a baked image -> the base baked_no_r2_pull gate fails the regression.
    lines = _pass_lines() + [rv.format_event("mirror_complete", {"total_seconds": 360.0, "total_bytes": 1})]
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["baked_no_r2_pull"] is False
    assert any("R2 mirror leg" in r for r in res.reasons)


# ----------------------------------------------------------------- CAP-1/2/4/6 representative gates

def test_evaluate_regression_keyframe_blank_fails():
    lines = _replace(_pass_lines(), "keyframe_done",
                     {"shot_id": "s0", "key": "kf/s0.png", "width": 1280, "height": 720,
                      "format": "PNG", "bytes": 1_200, "elapsed_s": 31.0})  # below the 50k floor
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap1_keyframe_bytes"] is False


def test_evaluate_regression_clip_not_distilled_fails():
    lines = _replace(_pass_lines(), "clip_done",
                     {"shot_id": "s0", "clip_key": "clips/s0.mp4", "num_frames": 49, "fps": 16,
                      "seconds": 3.0, "distilled": False, "elapsed_s": 78.0})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap2_clip_distilled"] is False


def test_evaluate_regression_finish_no_face_fails():
    lines = _replace(_pass_lines(), "finish_done",
                     {"shot_id": "s0", "clip_key": "clips/s0_finished.mp4", "interpolated": True,
                      "face_restored": False, "out_frames": 97, "out_fps": 32,
                      "bytes": 1_400_000, "elapsed_s": 52.0})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap4_finish_face_restored"] is False


def test_evaluate_regression_finish_bad_key_fails():
    lines = _replace(_pass_lines(), "finish_done",
                     {"shot_id": "s0", "clip_key": "clips/s0.mp4", "interpolated": True,
                      "face_restored": True, "out_frames": 97, "out_fps": 32,
                      "bytes": 1_400_000, "elapsed_s": 52.0})  # not *_finished.mp4
    res = _eval(lines)
    assert res.checks["cap4_finish_key"] is False


def test_evaluate_regression_e2e_no_audio_fails():
    lines = _replace(_pass_lines(), "e2e_done",
                     {"shots": 2, "output_key": "renders/e2e/film.mp4", "has_audio": False,
                      "duration_s": 6.0, "bytes": 3_200_000, "elapsed_s": 420.0})
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap6_e2e_audio"] is False


def test_evaluate_regression_e2e_timeout_fails():
    lines = _replace(_pass_lines(), "e2e_done",
                     {"shots": 2, "output_key": "renders/e2e/film.mp4", "has_audio": True,
                      "duration_s": 6.0, "bytes": 3_200_000, "elapsed_s": 9999.0})  # over the 600s bound
    res = _eval(lines)
    assert res.passed is False
    assert res.checks["cap6_e2e_time"] is False


# ----------------------------------------------------------------- CAP-5 coverage gap

def test_evaluate_regression_coverage_gap_recorded_not_failing():
    res = _eval(_pass_lines())
    assert res.passed is True                                  # the gap does not flip passed
    assert "lora_apply" in res.coverage_gaps                   # but it IS recorded, never silently skipped
    assert "#280" in res.coverage_gaps["lora_apply"]


# ----------------------------------------------------------------- config + events contract

def test_regression_config_env_activates_regression_and_is_bf16():
    cfg = rv.RegressionConfig(image="img:test")
    assert cfg.env["VJ_REGRESSION"] == "1"
    assert cfg.env["VJ_VERIFY"] == "1"
    assert cfg.env["VJ_I2V_DISTILL"] == "1"
    assert "VJ_I2V_FP8" not in cfg.env          # prod is bf16-only; do NOT force a runtime fp8 quant
    assert cfg.expected_precision == "bfloat16"
    assert cfg.expect_baked is True             # inherits the smoke config's baked expectation


def test_regression_events_list_is_the_full_contract():
    # REGRESSION_EVENTS names every event a full run emits, including the base smoke events.
    for name in ("model_inventory", "model_precision", "gpu_probe", "mirror_skipped",
                 "keyframe_done", "clip_done", "rife_model_probe", "rife_done", "finish_done",
                 "first_frame", "sharpness", "e2e_done", "complete"):
        assert name in rv.REGRESSION_EVENTS
    # the mock pass stream emits exactly the contract's renderable events (sans the ignored prose).
    emitted = {n for n, _ in rv.parse_events(_pass_lines())}
    assert set(rv.REGRESSION_EVENTS) <= emitted


def test_evaluate_regression_accepts_plain_verifyconfig():
    # A caller passing a plain VerifyConfig still gets the full regression (defaults fill the bounds).
    res = rv.evaluate_regression(rv.parse_events(_pass_lines()), rv.VerifyConfig(image="img:test"))
    assert res.passed is True
    assert "cap3_rife_architecture" in res.checks


# ----------------------------------------------------------------- CAP-3 structural probe helper

class _FakeConv2d:
    def __init__(self, out_channels):
        self.out_channels = out_channels


class _FakeIFBlock:
    """Mimics rife IFBlock.conv0 = Sequential(conv(in, c//2), conv(c//2, c)); conv()=Seq(Conv2d,PReLU),
    so conv0[-1][0].out_channels == c -- exactly what rife_architecture_facts derives."""
    def __init__(self, c):
        self.conv0 = [[_FakeConv2d(c // 2), object()], [_FakeConv2d(c), object()]]


class _FakeFlownet:
    def __init__(self, c=90, n_blocks=3, with_teacher=True):
        for name in ("block0", "block1", "block2")[:n_blocks]:
            setattr(self, name, _FakeIFBlock(c))
        if with_teacher:
            self.block_tea = _FakeIFBlock(c)   # the teacher must be EXCLUDED from the count


def test_rife_architecture_facts_counts_three_blocks_at_c90():
    facts = rv.rife_architecture_facts(_FakeFlownet(c=90, n_blocks=3))
    assert facts == {"block_count": 3, "c_per_block": 90, "loaded": True}


def test_rife_architecture_facts_excludes_teacher_block():
    # block_tea is a 4th IFBlock(c=90) but is training-only; the count must stay 3, not 4.
    facts = rv.rife_architecture_facts(_FakeFlownet(c=90, n_blocks=3, with_teacher=True))
    assert facts["block_count"] == 3


def test_rife_architecture_facts_detects_wrong_arch():
    facts = rv.rife_architecture_facts(_FakeFlownet(c=64, n_blocks=3))
    assert facts["c_per_block"] == 64
    # a 2-block flownet is incomplete
    assert rv.rife_architecture_facts(_FakeFlownet(c=90, n_blocks=2))["block_count"] == 2


def test_rife_architecture_facts_best_effort_on_garbage():
    # A non-IFNet object must yield loaded=False, never raise (probe miss != crash).
    assert rv.rife_architecture_facts(object()) == {"block_count": 0, "c_per_block": 0, "loaded": False}


# ----------------------------------------------------------------- orchestration + CLI

def test_run_verify_regression_pass_promotes_and_records_gap():
    from test_runpod_verify import FakePodClient, _clock  # reuse the smoke harness fake
    client = FakePodClient(_pass_lines())
    cfg = rv.RegressionConfig(image="ghcr.io/skyphusion-labs/vivijure-backend:bf16", registry_auth_id="ra-1")
    report = rv.run_verify(client, cfg, clock=_clock(), evaluator=rv.evaluate_regression)
    assert report["passed"] is True
    assert report["signal"] == "promote"
    assert report["coverage_gaps"]["lora_apply"]            # the gap rides along in the report
    assert "cap3_rife_architecture" in report["checks"]


def test_main_regression_dry_run_passes_offline():
    # The CLI --regression default path is a mocked dry-run: exit 0, no GPU, no network.
    assert rv.main(["--image", "img:dry", "--regression"]) == 0
