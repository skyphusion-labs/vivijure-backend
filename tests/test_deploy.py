"""CPU tests for the deploy plumbing: the pure RunPod template-pin transform, and the worker
entry's per-job pipeline build over a shared model server. No Docker, no network, no GPU."""
import importlib.util
from pathlib import Path

from vivijure_backend import worker
from vivijure_backend.contract import RenderRequest
from vivijure_backend.pipeline import GpuPipeline


def _load_pin_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pin-runpod-template.py"
    spec = importlib.util.spec_from_file_location("pin_runpod_template", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- pin-runpod-template helpers

def test_prepare_template_splices_image_and_strips_typename():
    pin = _load_pin_script()
    fetched = {
        "__typename": "PodTemplate", "id": "tpl1", "name": "vj-backend",
        "imageName": "ghcr.io/skyphusion-labs/vivijure-backend:0.1.0",
        "env": [{"__typename": "EnvVar", "key": "HF_HOME", "value": "/opt/models/hf-cache"}],
    }
    out = pin.prepare_template(fetched, "ghcr.io/skyphusion-labs/vivijure-backend:0.2.0")
    assert out["imageName"] == "ghcr.io/skyphusion-labs/vivijure-backend:0.2.0"
    assert "__typename" not in out
    assert "__typename" not in out["env"][0]          # stripped recursively
    assert out["env"][0]["key"] == "HF_HOME"           # real fields preserved
    # the caller's dict is not mutated (image still the old tag, typename intact)
    assert fetched["imageName"].endswith(":0.1.0")
    assert fetched["__typename"] == "PodTemplate"


def test_prepare_template_preserves_registry_auth():
    # A pin must change ONLY imageName and carry containerRegistryAuthId (the GHCR pull cred, which
    # lives on the template and is already correct) through UNCHANGED. Clearing it (omit) OR changing
    # it (hardcode) breaks the pull: a present-but-wrong/empty cred makes RunPod fail the login even
    # for a public image.
    pin = _load_pin_script()
    fetched = {
        "id": "tpl1", "imageName": "ghcr.io/skyphusion-labs/vivijure-backend:0.2.2",
        "containerRegistryAuthId": "cmpjfaka40045l807oybv65gf",
    }
    out = pin.prepare_template(fetched, "ghcr.io/skyphusion-labs/vivijure-backend:0.2.3")
    assert out["imageName"].endswith(":0.2.3")                              # version updated
    assert out["containerRegistryAuthId"] == "cmpjfaka40045l807oybv65gf"    # cred untouched


def test_fetch_query_requests_registry_auth():
    # Regression guard for the prod bug: the fetch query OMITTED containerRegistryAuthId, so the
    # round-tripped saveTemplate dropped it and RunPod cleared the cred. It must be fetched so it
    # round-trips through prepare_template unchanged.
    path = Path(__file__).resolve().parents[1] / "scripts" / "pin-runpod-template.py"
    assert "containerRegistryAuthId" in path.read_text()


def test_strip_typename_handles_nested_lists_and_dicts():
    pin = _load_pin_script()
    o = {"__typename": "A", "xs": [{"__typename": "B", "k": 1}, {"k": 2}]}
    pin.strip_typename(o)
    assert "__typename" not in o and "__typename" not in o["xs"][0]


# ----------------------------------------------------------------------------- worker entry

def _req(**over):
    return RenderRequest.from_dict({"action": "render", "project": "neon", "bundle_key": "x",
                                    "quality_tier": "draft", **over})


def test_build_pipeline_carries_job_config_and_pretrained():
    req = _req(pretrained_loras={"A": "loras/ext/A.safetensors"})
    pipe = worker.build_pipeline(req)
    assert isinstance(pipe, GpuPipeline)
    assert pipe.config is req.config
    assert pipe.config.quality.value == "draft"
    assert pipe.pretrained_loras == {"A": "loras/ext/A.safetensors"}


def test_build_pipeline_shares_one_model_server_across_jobs():
    # Warm-worker reuse: every per-job pipeline wraps the SAME process-global ModelServer.
    a = worker.build_pipeline(_req())
    b = worker.build_pipeline(_req(quality_tier="final"))
    assert a.server is b.server is not None
    assert a.config is not b.config          # but each carries its own job config


def test_handler_skips_pipeline_build_for_finish_and_i2v_clip(monkeypatch):
    """finish_clip and i2v_clip use the ModelServer directly, so the worker entry must NOT build
    or register a render pipeline for them (build_pipeline touches RenderRequest fields they don't
    carry). render still builds + registers."""
    import vivijure_backend.harness.handler as hh
    calls: list = []
    monkeypatch.setattr(worker, "build_pipeline", lambda req: calls.append("build") or "PIPE")
    monkeypatch.setattr(worker, "register_pipeline", lambda p: calls.append(("register", p)))
    monkeypatch.setattr(hh, "handler", lambda job: {"action": job.get("input", job).get("action")})

    for action in ("finish_clip", "i2v_clip"):
        calls.clear()
        out = worker.handler({"input": {"action": action, "project": "p"}})
        assert out == {"action": action}
        assert calls == [], f"{action} must not build/register a pipeline"

    calls.clear()
    worker.handler({"input": {"action": "render", "project": "p", "bundle_key": "x",
                              "quality_tier": "draft"}})
    assert ("register", "PIPE") in calls and "build" in calls


def test_model_server_uses_job_config_specs(monkeypatch):
    """Cold-start: the first job's model fields must reach ModelServer.specs."""
    from vivijure_backend.models import ModelRole, DEFAULT_SPECS
    monkeypatch.setattr(worker, "_SERVER", None)
    req = _req(render_overrides={"keyframe": {"base_model": "custom/sdxl-base"}})
    pipe = worker.build_pipeline(req)
    assert worker._SERVER is not None
    assert worker._SERVER.specs[ModelRole.KEYFRAME_BASE].repo_id == "custom/sdxl-base"
    # weight_name and other non-repo fields must be preserved (regression: positional ModelSpec
    # construction dropped weight_name and broke the keyframe distill LoRA load)
    assert (worker._SERVER.specs[ModelRole.KEYFRAME_FEWSTEP].weight_name
            == DEFAULT_SPECS[ModelRole.KEYFRAME_FEWSTEP].weight_name)
    # warm-worker path: second job gets the SAME server (model already loaded)
    req2 = _req(render_overrides={"keyframe": {"base_model": "other/sdxl"}})
    pipe2 = worker.build_pipeline(req2)
    assert pipe2.server is pipe.server  # reused
    assert worker._SERVER.specs[ModelRole.KEYFRAME_BASE].repo_id == "custom/sdxl-base"  # unchanged


# ----------------------------------------------------------- de-risk driver sha <-> EXPECT_SHA guard

def test_derisk_expect_sha_matches_committed_driver():
    """The inner (derisk_pod_start.sh) injects deploy/vj_derisk.py and sha256-gates it against a
    hard-coded EXPECT_SHA before any GPU spend. If the driver changes but EXPECT_SHA does not, every
    de-risk fire would driver_corrupt at $0 -- so this asserts they stay in lockstep at commit time."""
    import hashlib, re
    root = Path(__file__).resolve().parents[1]
    driver = (root / "deploy" / "vj_derisk.py").read_bytes()
    got = hashlib.sha256(driver).hexdigest()
    inner = (root / "deploy" / "derisk_pod_start.sh").read_text()
    m = re.search(r"^EXPECT_SHA=([0-9a-f]{64})$", inner, re.MULTILINE)
    assert m, "EXPECT_SHA=<sha256> not found in derisk_pod_start.sh"
    assert m.group(1) == got, (
        f"EXPECT_SHA ({m.group(1)}) != sha256(vj_derisk.py) ({got}); "
        "bump EXPECT_SHA in derisk_pod_start.sh when the driver changes")


def test_derisk_driver_keeps_the_build_render_inputs_marker():
    """The inner's secondary gate greps for `def build_render_inputs`; keep the marker present so the
    integrity gate's discriminator stays valid."""
    root = Path(__file__).resolve().parents[1]
    assert "def build_render_inputs" in (root / "deploy" / "vj_derisk.py").read_text()
