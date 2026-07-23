"""Wan 2.2 A14B character-LoRA training path (vivijure-cf #29).

CPU-only, no GPU and no ai-toolkit needed: the module's config generation, dataset prep, and
expert harvesting are pure/filesystem, and `train_slot_wan`'s ONE GPU touch (the ai-toolkit
`run.py` subprocess) is injected via `runner=`. Tests assert BOTH the happy path and the guards
(half-train fails loud; empty refs fail loud), each with a positive control.
"""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from vivijure_backend.contract import Character, RenderRequest
from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import Outputs, run_job
from vivijure_backend.orchestrator import plan
from vivijure_backend import wan_lora_train as W


# ------------------------------------------------------------------------------- keys

def test_wan_lora_key_layout_and_guard():
    assert keys.wan_lora_key("neon", "A", "high") == "loras/neon/A/wan_high_noise.safetensors"
    assert keys.wan_lora_key("neon", "A", "low") == "loras/neon/A/wan_low_noise.safetensors"
    assert keys.wan_lora_key("neon", "A", "HIGH") == "loras/neon/A/wan_high_noise.safetensors"
    # a name with slashes cannot scatter extra path segments
    seg = keys.wan_lora_key("neon", "../A", "high").split("neon/")[1].split("/wan")[0]
    assert "/" not in seg
    with pytest.raises(ValueError, match="high.*low"):
        keys.wan_lora_key("neon", "A", "middle")


# ---------------------------------------------------------------------- caption / config

def test_caption_carries_trigger_and_rejects_bad_template():
    ch = Character(slot="hero", name="chk_detective", prompt="weathered")
    assert W.caption_for(ch, "{name}") == "chk_detective"
    assert W.caption_for(ch, "{name}, {prompt}") == "chk_detective, weathered"
    # empty field collapses the dangling comma
    assert W.caption_for(Character(slot="h", name="t", prompt=""), "{name}, {prompt}") == "t"
    with pytest.raises(ValueError, match="unsupported placeholders"):
        W.caption_for(ch, "{name} {bogus}")


def test_build_config_is_dual_expert_bf16_and_wires_the_dataset():
    cfg = W.WanLoraTrainConfig(rank=32, steps=1500)
    c = W.build_aitoolkit_config("hero", Path("/ds"), Path("/out"), cfg)
    proc = c["config"]["process"][0]
    model = proc["model"]
    # the whole point: BOTH experts train -> two files
    assert model["model_kwargs"] == {"train_high_noise": True, "train_low_noise": True}
    assert model["arch"] == "wan22_14b"
    # spike-proven recipe: bf16, no quantization stall, both experts resident by default
    assert model["quantize"] is False and model["quantize_te"] is False
    assert model["low_vram"] is False
    assert proc["train"]["dtype"] == "bf16"
    # dataset + knobs threaded from the config
    assert proc["datasets"][0]["folder_path"] == "/ds"
    assert proc["train"]["steps"] == 1500
    assert proc["network"]["linear"] == 32 and proc["network"]["linear_alpha"] == 32
    assert proc["training_folder"] == "/out"


def test_build_config_low_vram_is_configurable():
    # control: low_vram flips when the deploy asks for a smaller card
    c = W.build_aitoolkit_config("h", Path("/ds"), Path("/o"), W.WanLoraTrainConfig(low_vram=True))
    assert c["config"]["process"][0]["model"]["low_vram"] is True


# ------------------------------------------------------------------------ dataset prep

def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return path


def test_prepare_dataset_copies_refs_and_captions(tmp_path):
    refs = [_png(tmp_path / f"src_{i}.png") for i in range(3)]
    ch = Character(slot="hero", name="chk_detective", prompt="", ref_paths=refs)
    ds = tmp_path / "ds"
    n = W.prepare_dataset(ch, ds, W.WanLoraTrainConfig())
    assert n == 3
    imgs = sorted(ds.glob("*.png"))
    txts = sorted(ds.glob("*.txt"))
    assert len(imgs) == 3 and len(txts) == 3
    # every caption carries the trigger token
    assert all("chk_detective" in t.read_text() for t in txts)


def test_prepare_dataset_no_refs_fails_loud(tmp_path):
    ch = Character(slot="hero", name="t", prompt="", ref_paths=[])
    with pytest.raises(ValueError, match="no reference images"):
        W.prepare_dataset(ch, tmp_path / "ds", W.WanLoraTrainConfig())


# --------------------------------------------------------------------------- harvest

def _touch(d: Path, name: str):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(b"LORA")


def test_harvest_takes_highest_step_of_each_expert(tmp_path):
    run = tmp_path / "output" / "hero"
    for step in ("000000250", "000001000"):
        _touch(run, f"hero_{step}_high_noise.safetensors")
        _touch(run, f"hero_{step}_low_noise.safetensors")
    high, low = W.harvest_experts(run, "hero")
    assert high.name == "hero_000001000_high_noise.safetensors"
    assert low.name == "hero_000001000_low_noise.safetensors"


def test_harvest_missing_expert_fails_loud(tmp_path):
    run = tmp_path / "output" / "hero"
    _touch(run, "hero_000001000_high_noise.safetensors")  # only high, no low
    with pytest.raises(FileNotFoundError, match="low-noise expert"):
        W.harvest_experts(run, "hero")


# ------------------------------------------------------- train_slot_wan (injected seam)

def _fake_runner_writes_experts(config_path, *, cwd, progress_cb=None):
    """A stand-in for the ai-toolkit subprocess: parse the REAL generated config (positive control
    that the seam receives a valid config), then write the two experts where harvest looks."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    proc = cfg["config"]["process"][0]
    name = cfg["config"]["name"]
    run_dir = Path(proc["training_folder"]) / name
    steps = proc["train"]["steps"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for e in ("high", "low"):
        (run_dir / f"{name}_{steps:09d}_{e}_noise.safetensors").write_bytes(b"LORA")
    if progress_cb:
        progress_cb("10/2000 loss=0.1 it/s")


def test_resolve_local_hf_snapshot_passthrough_for_dirs(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    assert W.resolve_local_hf_snapshot(str(d)) == str(d.resolve())


def test_resolve_local_hf_snapshot_missing_hub_id_fails_loud(monkeypatch):
    import sys
    import types

    def boom(*a, **k):
        raise RuntimeError("offline miss")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(FileNotFoundError, match="not in local cache"):
        W.resolve_local_hf_snapshot("ai-toolkit/does-not-exist")


def test_patch_aitoolkit_hub_ids_rewrites_umt5_and_vae(tmp_path, monkeypatch):
    te = tmp_path / "te"
    vae = tmp_path / "vae"
    te.mkdir()
    vae.mkdir()
    monkeypatch.setattr(W, "resolve_local_hf_snapshot", lambda s: str(te if "umt5" in s else vae))
    wan21 = tmp_path / "toolkit/models/wan21"
    wan22 = tmp_path / "extensions_built_in/diffusion_models/wan22"
    wan21.mkdir(parents=True)
    wan22.mkdir(parents=True)
    (wan21 / "wan21.py").write_text(f'te_path = "{W.AITOOLKIT_UMT5_REPO}"\n')
    (wan22 / "wan22_14b_model.py").write_text(f'_wan_vae_path = "{W.AITOOLKIT_VAE_REPO}"\n')
    out = W.patch_aitoolkit_hub_ids_for_offline(tmp_path)
    assert out["umt5"] == str(te) and out["vae"] == str(vae)
    assert str(te) in (wan21 / "wan21.py").read_text()
    assert str(vae) in (wan22 / "wan22_14b_model.py").read_text()
    # idempotent: second call leaves absolute paths alone
    W.patch_aitoolkit_hub_ids_for_offline(tmp_path)
    assert str(te) in (wan21 / "wan21.py").read_text()


def test_patch_aitoolkit_hub_ids_noop_on_stub_checkout(tmp_path):
    # CPU tests use a run.py-only stub; no rewrite targets -> no HF resolve
    assert W.patch_aitoolkit_hub_ids_for_offline(tmp_path) == {}


def test_train_slot_wan_end_to_end_with_injected_runner(tmp_path, monkeypatch):
    base = tmp_path / "wan-base"
    base.mkdir()
    monkeypatch.setattr(W, "resolve_local_hf_snapshot", lambda s: str(base))
    refs = [_png(tmp_path / f"src_{i}.png") for i in range(4)]
    ch = Character(slot="hero", name="chk_detective", prompt="", ref_paths=refs)
    seen = {}

    def runner(config_path, *, cwd, progress_cb=None):
        seen["config_path"] = Path(config_path)  # control: the seam WAS invoked
        _fake_runner_writes_experts(config_path, cwd=cwd, progress_cb=progress_cb)

    res = W.train_slot_wan(ch, tmp_path / "out", config=W.WanLoraTrainConfig(steps=1000), runner=runner)
    # both experts returned, both exist, distinct files
    assert res.high_path.is_file() and res.low_path.is_file()
    assert res.high_path != res.low_path
    assert "high_noise" in res.high_path.name and "low_noise" in res.low_path.name
    assert res.trigger == "chk_detective" and res.ref_count == 4 and res.steps == 1000
    # positive control: the config the seam consumed is the real generated recipe
    assert seen["config_path"].is_file()
    cfg = yaml.safe_load(seen["config_path"].read_text())
    assert cfg["config"]["process"][0]["model"]["model_kwargs"]["train_high_noise"] is True
    # offline-safe: name_or_path is the resolved local snapshot, not the hub id
    assert cfg["config"]["process"][0]["model"]["name_or_path"] == str(base)


def test_train_slot_wan_half_train_fails_loud(tmp_path, monkeypatch):
    base = tmp_path / "wan-base"
    base.mkdir()
    monkeypatch.setattr(W, "resolve_local_hf_snapshot", lambda s: str(base))
    refs = [_png(tmp_path / "src_0.png")]
    ch = Character(slot="hero", name="t", prompt="", ref_paths=refs)

    def half_runner(config_path, *, cwd, progress_cb=None):
        cfg = yaml.safe_load(Path(config_path).read_text())
        proc = cfg["config"]["process"][0]
        name = cfg["config"]["name"]
        run_dir = Path(proc["training_folder"]) / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"{name}_{proc['train']['steps']:09d}_high_noise.safetensors").write_bytes(b"x")

    with pytest.raises(FileNotFoundError, match="low-noise expert"):
        W.train_slot_wan(ch, tmp_path / "out", runner=half_runner)


# --------------------------------------------------------------- orchestrator cost

def _req(model_family=None):
    d = {"action": "train_lora", "project": "p", "bundle_key": "bundles/p.tar.gz"}
    if model_family is not None:
        d["model_family"] = model_family
    return RenderRequest.from_dict(d)


def _sb():
    from vivijure_backend.contract import Storyboard
    return Storyboard.from_dict({"use_characters": ["A", "B"], "scenes": [
        {"id": "s1", "prompt": "x", "character_slots": ["A"]}]})


def test_wan_family_uses_wan_cost():
    p = plan(_req(model_family="wan"), _sb())
    assert p.lora_family == "wan"
    assert sorted(p.lora.train) == ["A", "B"]
    assert p.estimated_gpu_seconds == 2 * 3600.0  # two slots, Wan per-slot cost


def test_default_family_is_sdxl_cost_on_cpu_dev():
    # train_lora with no model_family on a box without the train image stays SDXL (render EP compat).
    p = plan(_req(), _sb())
    assert p.lora_family == "sdxl"
    assert p.estimated_gpu_seconds == 2 * 180.0


def test_train_lora_defaults_wan_when_runtime_ready(monkeypatch):
    monkeypatch.setenv("VIVIJURE_WAN_BASE_PATH", "/opt/models/aitoolkit/wan-base")
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    p = plan(_req(), _sb())
    assert p.lora_family == "wan"
    assert p.estimated_gpu_seconds == 2 * 3600.0


def test_render_without_model_family_stays_sdxl_even_when_wan_ready(monkeypatch):
    monkeypatch.setattr(W, "wan_train_runtime_ready", lambda: True)
    d = {"action": "render", "project": "p", "bundle_key": "bundles/p.tar.gz"}
    p = plan(RenderRequest.from_dict(d), _sb())
    assert p.lora_family == "sdxl"


# --------------------------------------------------------- handler dual-expert upload

_WAN_STORYBOARD = {"title": "neon", "use_characters": ["A"],
                   "scenes": [{"id": "s1", "prompt": "A", "character_slots": ["A"]}]}


def _wan_bundle_tar(path: Path) -> Path:
    members = {
        "storyboard.yaml": yaml.safe_dump(_WAN_STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {"A": {"name": "Vesper", "prompt": "teal"}}}).encode(),
        "characters/refs/A/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


class _WanFakeStore:
    def __init__(self, bundle_tar: Path, existing: set[str] | None = None):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.existing = existing or set()

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest); return dest

    def exists(self, key):
        return key in self.existing

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists()
        self.puts.append(key); return key


class _WanFakePipeline:
    """Populates out.wan_loras for the slots a wan-family plan kept (two experts each)."""
    def execute(self, plan, bundle, workdir):
        out = Outputs()
        assert plan.lora_family == "wan"
        for slot in plan.lora.train:
            hi = workdir / f"{slot}_high.safetensors"; hi.write_bytes(b"h")
            lo = workdir / f"{slot}_low.safetensors"; lo.write_bytes(b"l")
            out.wan_loras[slot] = (hi, lo)
        return out


def test_wan_train_uploads_both_experts(tmp_path):
    store = _WanFakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(
        {"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
         "model_family": "wan"},
        pipeline=_WanFakePipeline(), store=store, workdir=tmp_path / "work")
    assert keys.wan_lora_key("neon", "A", "high") in store.puts
    assert keys.wan_lora_key("neon", "A", "low") in store.puts
    lora = res["lora"]["A"]
    assert lora["family"] == "wan"
    assert lora["lora_id_high"].endswith("wan_high_noise.safetensors")
    assert lora["lora_id_low"].endswith("wan_low_noise.safetensors")


def test_wan_restore_skips_when_both_experts_exist(tmp_path):
    # both experts already in R2 -> slot is "trained" -> the plan skips training it
    existing = {keys.wan_lora_key("neon", "A", "high"), keys.wan_lora_key("neon", "A", "low")}
    store = _WanFakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"), existing=existing)

    class _AssertNoTrain:
        def execute(self, plan, bundle, workdir):
            assert plan.lora.train == [], "a fully-trained wan slot must be skipped, not retrained"
            return Outputs()

    run_job({"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
             "model_family": "wan"},
            pipeline=_AssertNoTrain(), store=store, workdir=tmp_path / "work")


def test_wan_restore_half_upload_is_not_trained(tmp_path):
    # only the high expert exists -> NOT trained -> the plan still trains the slot
    existing = {keys.wan_lora_key("neon", "A", "high")}
    store = _WanFakeStore(_wan_bundle_tar(tmp_path / "b.tar.gz"), existing=existing)
    res = run_job({"action": "train_lora", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
                   "model_family": "wan"},
                  pipeline=_WanFakePipeline(), store=store, workdir=tmp_path / "work")
    # it retrained + uploaded both, proving the half-upload was not mistaken for trained
    assert keys.wan_lora_key("neon", "A", "low") in store.puts


# ------------------------------------------------ interpreter seam (cf#29 D1: env isolation)

def test_aitoolkit_python_defaults_to_sys_executable(monkeypatch):
    # default: the worker's own interpreter (pre-isolation behavior, unchanged)
    monkeypatch.delenv(W.AITOOLKIT_PYTHON_ENV, raising=False)
    import sys
    assert W.aitoolkit_python() == sys.executable


def test_aitoolkit_python_honors_env_override(monkeypatch):
    # the training image points this at the isolated aitoolkit conda env's python
    monkeypatch.setenv(W.AITOOLKIT_PYTHON_ENV, "/opt/conda/envs/aitoolkit/bin/python")
    assert W.aitoolkit_python() == "/opt/conda/envs/aitoolkit/bin/python"


def test_run_aitoolkit_launches_the_configured_interpreter(tmp_path, monkeypatch):
    """Real-seam positive control: _run_aitoolkit must launch the interpreter aitoolkit_python()
    resolves, not a hardcoded one. Point the env at a stub 'interpreter' that records it ran; the
    presence of its marker proves the configured interpreter was the one launched over the real
    subprocess seam (no Popen stubbing)."""
    (tmp_path / "run.py").write_text("# ai-toolkit entrypoint stub\n")
    marker = tmp_path / "interp_ran.marker"
    stub = tmp_path / "fake_interp.sh"
    stub.write_text(f"#!/usr/bin/env bash\ntouch {marker}\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv(W.AITOOLKIT_PYTHON_ENV, str(stub))

    cfg = tmp_path / "config.yaml"
    cfg.write_text("job: extension\n")
    W._run_aitoolkit(cfg, cwd=tmp_path)
    assert marker.is_file(), "the configured interpreter (VIVIJURE_AITOOLKIT_PYTHON) was not launched"


def test_run_aitoolkit_raises_on_nonzero_exit_of_configured_interpreter(tmp_path, monkeypatch):
    """Negative control: a non-zero exit from the configured interpreter fails loud (a broken
    training run must never be swallowed). Tail of stdout is attached for RunPod-empty-stream cases."""
    (tmp_path / "run.py").write_text("# stub\n")
    stub = tmp_path / "fail_interp.sh"
    stub.write_text("#!/usr/bin/env bash\necho boom-line\nexit 7\n")
    stub.chmod(0o755)
    monkeypatch.setenv(W.AITOOLKIT_PYTHON_ENV, str(stub))
    cfg = tmp_path / "config.yaml"
    cfg.write_text("job: extension\n")
    with pytest.raises(RuntimeError, match="exited 7") as ei:
        W._run_aitoolkit(cfg, cwd=tmp_path)
    assert "boom-line" in str(ei.value)
    assert "last" in str(ei.value)
