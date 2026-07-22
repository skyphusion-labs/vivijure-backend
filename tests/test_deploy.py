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
    # Overrides must stay inside DEFAULT_SPECS namespaces (SG161222) so the allowlist accepts them.
    req = _req(render_overrides={"keyframe": {"base_model": "SG161222/custom-sdxl-base"}})
    pipe = worker.build_pipeline(req)
    assert worker._SERVER is not None
    assert worker._SERVER.specs[ModelRole.KEYFRAME_BASE].repo_id == "SG161222/custom-sdxl-base"
    # weight_name and other non-repo fields must be preserved (regression: positional ModelSpec
    # construction dropped weight_name and broke the keyframe distill LoRA load)
    assert (worker._SERVER.specs[ModelRole.KEYFRAME_FEWSTEP].weight_name
            == DEFAULT_SPECS[ModelRole.KEYFRAME_FEWSTEP].weight_name)
    # warm-worker path: a second job with the SAME models reuses the loaded server
    req_same = _req(render_overrides={"keyframe": {"base_model": "SG161222/custom-sdxl-base"}})
    pipe2 = worker.build_pipeline(req_same)
    assert pipe2.server is pipe.server  # reused
    # ...but a second job with DIFFERENT models is REFUSED, never silently rendered on the
    # previously-loaded (wrong) set -- the client resubmits and a fresh worker loads correctly.
    import pytest
    req_diverged = _req(render_overrides={"keyframe": {"base_model": "SG161222/other-sdxl"}})
    with pytest.raises(worker.ModelDivergenceError, match="SG161222/other-sdxl"):
        worker.build_pipeline(req_diverged)
    assert worker._SERVER.specs[ModelRole.KEYFRAME_BASE].repo_id == "SG161222/custom-sdxl-base"  # unchanged


def test_cold_start_rejects_disallowed_model_repo_id(monkeypatch):
    """Job-controlled repo_id must fail closed before ModelServer is built (path / foreign org)."""
    import pytest
    from vivijure_backend.models import InvalidModelRepoId
    monkeypatch.setattr(worker, "_SERVER", None)
    for bad in ("/etc/passwd", "evil-org/malware", "https://evil.example/model"):
        with pytest.raises(InvalidModelRepoId):
            worker.build_pipeline(_req(render_overrides={"keyframe": {"base_model": bad}}))
        assert worker._SERVER is None  # never constructed on reject


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


# ----------------------------------------------------------- facexlib baked as a finish_dir (offline)

def test_bake_manifest_declares_facexlib_finish_dir():
    """The face-restore finish leg pulls facexlib detection+parsing weights; they must be baked like
    the other finish_dirs (rife/GFPGANv1.4) so face restore is offline. Guards manifest drift."""
    import json
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root / "deploy" / "bake-manifest.json").read_text())
    dirs = {fd["dir"] for fd in m["finish_dirs"]}
    assert "facexlib" in dirs, "facexlib missing from bake-manifest finish_dirs"


def test_mirror_pulls_facexlib_finish_dir():
    """The R2 cold-mirror fallback must pull facexlib alongside rife/GFPGANv1.4, or a non-baked worker
    would fetch facexlib from github at render time."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "src" / "vivijure_backend" / "harness" / "models_mirror.py").read_text()
    assert "facexlib" in src, "models_mirror does not mirror facexlib"


# ----------------------------------------- facexlib sha256 PIN: single source of truth + both gates

import hashlib as _hashlib
import json as _json
import re as _re

import pytest


def _load_deploy_mod(name):
    path = Path(__file__).resolve().parents[1] / "deploy" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_manifest(tmp_path, files):
    """Write a throwaway bake-manifest.json pinning the given {name: bytes} under finish_dirs/facexlib,
    materialize each file under <tmp_path>/facexlib, and return the manifest path."""
    fx = tmp_path / "facexlib"
    fx.mkdir(exist_ok=True)
    pins = []
    for name, payload in files.items():
        (fx / name).write_bytes(payload)
        pins.append({"name": name, "url": "https://example/" + name,
                     "size": len(payload), "sha256": _hashlib.sha256(payload).hexdigest()})
    mp = tmp_path / "bake-manifest.json"
    mp.write_text(_json.dumps({"finish_dirs": [{"dir": "facexlib", "files": pins}]}))
    return mp


def test_facexlib_manifest_pins_are_complete_and_well_formed():
    """The pins ARE the source of truth, so each must carry a real 64-hex sha256, a positive int size,
    and the official facexlib release URL -- a band/magic gate is not enough for a public artifact."""
    pins = _load_deploy_mod("facexlib_pins").load_facexlib_pins()
    assert {p["name"] for p in pins} == {"detection_Resnet50_Final.pth", "parsing_parsenet.pth"}
    for p in pins:
        assert _re.fullmatch(r"[0-9a-f]{64}", p["sha256"]), p
        assert isinstance(p["size"], int) and p["size"] > 0
        assert p["url"].startswith("https://github.com/xinntao/facexlib/releases/")


def test_assert_finish_shas_passes_on_matching_bytes(tmp_path):
    bl = _load_deploy_mod("bake_layers")
    mp = _tiny_manifest(tmp_path, {"detection_Resnet50_Final.pth": b"\x80det-bytes",
                                   "parsing_parsenet.pth": b"\x80par-bytes"})
    out = bl.assert_finish_shas(tmp_path, mp, log=lambda *a, **k: None)
    assert sorted(out["verified"]) == ["detection_Resnet50_Final.pth", "parsing_parsenet.pth"]


def test_assert_finish_shas_fails_closed_on_corrupt_byte(tmp_path):
    bl = _load_deploy_mod("bake_layers")
    mp = _tiny_manifest(tmp_path, {"detection_Resnet50_Final.pth": b"\x80good-bytes"})
    (tmp_path / "facexlib" / "detection_Resnet50_Final.pth").write_bytes(b"\x80bAd!-bytes")  # same len, diff sha
    with pytest.raises(SystemExit):
        bl.assert_finish_shas(tmp_path, mp, log=lambda *a, **k: None)


def test_assert_finish_shas_fails_on_size_mismatch(tmp_path):
    bl = _load_deploy_mod("bake_layers")
    mp = _tiny_manifest(tmp_path, {"parsing_parsenet.pth": b"\x80abcdef"})
    (tmp_path / "facexlib" / "parsing_parsenet.pth").write_bytes(b"\x80abcdefEXTRA")  # wrong size
    with pytest.raises(SystemExit):
        bl.assert_finish_shas(tmp_path, mp, log=lambda *a, **k: None)


def test_assert_finish_shas_fails_when_file_missing(tmp_path):
    bl = _load_deploy_mod("bake_layers")
    mp = _tiny_manifest(tmp_path, {"detection_Resnet50_Final.pth": b"\x80present"})
    (tmp_path / "facexlib" / "detection_Resnet50_Final.pth").unlink()  # baked file absent
    with pytest.raises(SystemExit):
        bl.assert_finish_shas(tmp_path, mp, log=lambda *a, **k: None)


def test_dockerfile_runs_the_finish_sha_gate_before_the_sentinel():
    """assert-finish-shas must run in the same && chain that gates the .vj-baked write, so the sentinel
    can never be stamped over a corrupted/substituted facexlib weight. (#537 Shape Y: the weight bake +
    its gates live in the RUNTIME base, deploy/runtime.Dockerfile, not the slim consumer Dockerfile.)"""
    df = (Path(__file__).resolve().parents[1] / "deploy" / "runtime.Dockerfile").read_text()
    assert "assert-finish-shas" in df
    assert df.index("assert-finish-shas") < df.index("/opt/models/.vj-baked")
    # the union-keyed manifest sha256 gate must ALSO run before the sentinel (#537 byte-identity gate):
    # a stale/substituted weight layer fails LOUD before .vj-baked is ever trusted.
    assert "sha256sum -c weights-manifest.sha256" in df
    assert df.index("sha256sum -c weights-manifest.sha256") < df.index("/opt/models/.vj-baked")


def test_consumer_dockerfile_is_from_runtime_with_no_weight_bake():
    """#537 Shape Y: the consumer deploy/Dockerfile is FROM the pinned runtime base + COPY src only --
    it must NOT re-bake weights (no seed COPY, no assert chain, no .vj-baked stamp); those belong to
    the runtime base so a src-only release inherits + dedups them."""
    df = (Path(__file__).resolve().parents[1] / "deploy" / "Dockerfile").read_text()
    assert "vivijure-backend-runtime" in df, "consumer must FROM the runtime base"
    assert "COPY src/vivijure_backend" in df
    # no weight-bake in the consumer: the assert chain + weight COPY belong to the runtime base, so a
    # src-only release inherits + dedups them (a mention in a comment is fine; a RUN/COPY is not).
    assert "assert-weights" not in df, "the consumer must not run the weight-bake gate (the runtime does)"
    assert "seed-bins" not in df and "COPY --from=seed" not in df, "no weight bake in the consumer"


def test_stage_script_asserts_the_manifest_pin_not_a_literal():
    """The staging upload must assert the manifest sha256 pin (single source of truth), not a hardcoded
    literal or a size-band-only gate."""
    src = (Path(__file__).resolve().parents[1] / "deploy" / "stage_facexlib_to_r2.py").read_text()
    assert "load_facexlib_pins" in src and "verify_file" in src


def test_vj_derisk_probe_asserts_facexlib_sha_from_baked_manifest():
    """The de-risk runtime probe must upgrade from a presence check to an exact sha256 assert read from
    the baked manifest, and fail-closed on a mismatch."""
    src = (Path(__file__).resolve().parents[1] / "deploy" / "vj_derisk.py").read_text()
    assert "bake-manifest.json" in src
    assert "finish_weights_sha_mismatch" in src


# ---------------------------------------- de-risk egress guard (#17/#159, full-block socket-userspace)

import socket as _socket


def _load_vj_derisk():
    return _load_deploy_mod("vj_derisk")


def test_egress_allowed_full_block_only_loopback_and_afunix():
    vj = _load_vj_derisk()
    A = lambda fam, addr: vj._egress_allowed(fam, addr)[0]
    # ALLOW: AF_UNIX + every loopback form
    assert A(_socket.AF_UNIX, "/var/run/x.sock")
    for lb in ("127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"):
        assert A(_socket.AF_INET, (lb, 8000)), lb
    # BLOCK: everything non-loopback, regardless of port -- HF, github, R2-ish, DNS-over-TCP, arbitrary
    assert not A(_socket.AF_INET, ("140.82.112.3", 443))      # github
    assert not A(_socket.AF_INET, ("3.160.0.1", 443))         # huggingface-ish CDN
    assert not A(_socket.AF_INET, ("104.18.0.1", 443))        # r2-ish (R2 is NOT allowed under full block)
    assert not A(_socket.AF_INET, ("8.8.8.8", 53))            # no DNS-over-TCP allowance either
    assert not A(_socket.AF_INET, ("203.0.113.9", 8080))


def test_egress_allowed_non_inet_address_is_passthrough():
    vj = _load_vj_derisk()
    # a non-tuple address (defensive) is not an inet egress decision -> allowed (let the OS handle it)
    assert vj._egress_allowed(_socket.AF_INET, None)[0]


def test_flag_on_truthy_set():
    vj = _load_vj_derisk()
    import os
    for v in ("1", "true", "TRUE", "yes", "on"):
        os.environ["VJ_TEST_FLAG"] = v
        assert vj._flag_on("VJ_TEST_FLAG")
    for v in ("", "0", "false", "no", "off"):
        os.environ["VJ_TEST_FLAG"] = v
        assert not vj._flag_on("VJ_TEST_FLAG")
    os.environ.pop("VJ_TEST_FLAG", None)
    assert not vj._flag_on("VJ_TEST_FLAG")
