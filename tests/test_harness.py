"""The harness is CPU-only by design: keys/mirror/config are pure, and the whole job flow runs
against a fake pipeline + fake object store, so `run_job` is tested without a GPU, R2, or (on
the offloaded path) ffmpeg."""
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from vivijure_backend.harness import keys
from vivijure_backend.harness.handler import HarnessError, Outputs, run_job
from vivijure_backend.contract import RenderRequest
from vivijure_backend.harness.models_mirror import mirror_cmd, rclone_env
from vivijure_backend.harness.r2 import R2Config


# ------------------------------------------------------------------------------- keys

def test_keys_layout():
    assert keys.output_key("neon") == "renders/neon/full.mp4"
    assert keys.keyframe_hash_key("neon", "shot_01") == "renders/neon/keyframes/shot_01.hash"
    assert keys.lora_key("neon", "A") == "loras/neon/A/pytorch_lora_weights.safetensors"
    assert keys.keyframe_key("neon", "shot_01") == "renders/neon/keyframes/shot_01.png"
    assert keys.clip_key("neon", "shot_02") == "renders/neon/clips/shot_02.mp4"


def test_key_slug_is_path_safe():
    # a name with spaces/slashes must not smuggle extra path segments into a key
    assert keys.output_key("neon rain/standoff") == "renders/neon_rain_standoff/full.mp4"
    assert keys.output_key("   ") == "renders/untitled/full.mp4"

def test_key_slug_guards_shot_id():
    # A shot_id with slashes must not introduce extra segments into the key.
    # The threat is a raw "/" in the shot_id; _slug converts it to "_", neutralizing it.
    k = keys.keyframe_key("neon", "../evil")
    segment = k.split("keyframes/")[1]
    assert "/" not in segment, f"slash escaped into segment: {k}"
    assert keys.keyframe_key("neon", "shot 01") == "renders/neon/keyframes/shot_01.png"


def test_key_slug_guards_slot():
    k = keys.lora_key("neon", "../A")
    # slot segment is between the project slug and the filename; must contain no "/"
    segment = k.split("neon/")[1].split("/pytorch")[0]
    assert "/" not in segment, f"slash escaped into slot segment: {k}"



# --------------------------------------------------------------------- models mirror

def test_mirror_cmd_copies_with_links_and_excludes():
    cmd = mirror_cmd("r2:vivijure/models/hf-cache", Path("/hf"),
                     skip_repos=("models--X", "spaces--Y"))
    assert cmd[:4] == ["rclone", "copy", "--links", "--transfers"]
    assert "--config" not in cmd                       # remote resolves from env, no on-disk conf
    assert "--exclude" in cmd and "**/*.incomplete" in cmd
    assert "hub/models--X/**" in cmd and "hub/spaces--Y/**" in cmd
    assert cmd[-2:] == ["r2:vivijure/models/hf-cache", "/hf"]  # src, dst last


def test_rclone_env_carries_creds_in_env_only_and_rejects_partial():
    # The R2 secret must be configured for rclone via RCLONE_CONFIG_* env vars (never an on-disk
    # rclone.conf), so this returns a child env carrying the cred and writes nothing to disk.
    child = rclone_env({"R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
                        "R2_ENDPOINT": "https://x.r2"})
    assert child["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == "k"
    assert child["RCLONE_CONFIG_R2_SECRET_ACCESS_KEY"] == "s"
    assert child["RCLONE_CONFIG_R2_ENDPOINT"] == "https://x.r2"
    assert child["RCLONE_CONFIG_R2_TYPE"] == "s3" and child["RCLONE_CONFIG_R2_PROVIDER"] == "Cloudflare"
    assert child["RCLONE_CONFIG_R2_NO_CHECK_BUCKET"] == "true"
    with pytest.raises(RuntimeError, match="incomplete R2 creds"):
        rclone_env({"R2_ACCESS_KEY_ID": "k"})


def test_r2config_from_env_validates():
    cfg = R2Config.from_env({"R2_ENDPOINT": "e", "R2_ACCESS_KEY_ID": "k",
                             "R2_SECRET_ACCESS_KEY": "s", "R2_BUCKET": "vivijure"})
    assert cfg.bucket == "vivijure"
    with pytest.raises(RuntimeError, match="missing env"):
        R2Config.from_env({"R2_ENDPOINT": "e"})


# ----------------------------------------------------------- fakes for the job flow

STORYBOARD = {
    "title": "neon", "use_characters": ["A", "B"],
    "scenes": [
        {"id": "shot_01", "prompt": "A alone", "character_slots": ["A"], "target_seconds": 5},
        {"id": "shot_02", "prompt": "A and B", "character_slots": ["A", "B"], "target_seconds": 4},
    ],
}


def _bundle_tar(path: Path) -> Path:
    members = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
        "characters/registry.json": json.dumps({"characters": {
            "A": {"name": "Vesper", "prompt": "teal"}, "B": {"name": "Rhode", "prompt": "orange"}}}).encode(),
        "characters/refs/A/ref_01.png": b"PNG-ish",
        "characters/refs/B/ref_01.png": b"PNG-ish",
    }
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


class FakeStore:
    """Records puts; serves the prebuilt bundle for any get. No network."""
    def __init__(self, bundle_tar: Path):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.tars: list[str] = []
        self.meta: dict[str, dict | None] = {}  # key -> customMetadata recorded at put time

    def get_file(self, key, dest):
        shutil.copy(self.bundle_tar, dest)
        return dest

    def exists(self, key):
        return False  # default fake: FRESH project -- run_job's internal restore finds nothing

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(), f"uploading a nonexistent file: {path}"
        self.puts.append(key)
        self.meta[key] = metadata
        return key

    def put_dir_as_tar(self, src_dir, key, *, metadata=None):
        self.tars.append(key)
        self.meta[key] = metadata
        return key


class FakePipeline:
    """Produces empty artifact files for exactly the work the plan kept; no GPU."""
    def set_pretrained_loras(self, mapping):
        self.pretrained_loras = mapping   # a fake still accepts the staged map (it ignores it)

    def execute(self, plan, bundle, workdir):
        out = Outputs()
        for slot in plan.lora.train:
            p = workdir / f"lora_{slot}.safetensors"; p.write_bytes(b"x"); out.loras[slot] = p
        for s in plan.scenes:
            if s.keyframe_mode.value == "generate":
                p = workdir / f"{s.shot_id}.png"; p.write_bytes(b"x"); out.keyframes[s.shot_id] = p
            if s.needs_i2v:
                p = workdir / f"{s.shot_id}.mp4"; p.write_bytes(b"x"); out.clips.append((s.shot_id, p))
        return out


def _job(**over):
    return {"action": "render", "project": "neon", "bundle_key": "bundles/neon.tar.gz",
            "quality_tier": "final", **over}


# --------------------------------------------------------------------- job flow

def test_run_job_offloaded_emits_clips_and_manifest(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(
        _job(render_overrides={"finish_offloaded": True}, pretrained_loras={"A": "loras/ext/A.safetensors"}),
        pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")

    # pretrained A reused, B trained+uploaded
    assert res["lora"]["A"] == {"lora_id": "loras/ext/A.safetensors"}
    assert res["lora"]["B"]["lora_id"] == "loras/neon/B/pytorch_lora_weights.safetensors"
    # offloaded: per-shot clips in storyboard order, a manifest, and NO merged output
    assert [c["shot_id"] for c in res["clips"]] == ["shot_01", "shot_02"]
    assert res["output_key"] is None
    assert any(k.endswith("manifest.json") for k in store.puts)
    # No shared state object (#112): per-artifact keys only, state_key retired to None.
    assert res["state_key"] is None
    assert store.tars == []
    # keyframes uploaded for both generated shots
    assert {k["shot_id"] for k in res["keyframes"]} == {"shot_01", "shot_02"}


def test_run_job_never_stamps_submitter_identity(tmp_path):
    # Identity strip (#292): the backend must not parse or persist a submitter identity. Even if a
    # legacy/hostile job body injects `user_email`, from_dict drops it and NO artifact may carry it
    # as object metadata -- otherwise a stripped identity leaks back into R2 in a single-operator
    # self-host model. This locks the strip so it cannot silently regress.
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(user_email="director@example.com",
                       render_overrides={"finish_offloaded": True}),
                  pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    assert store.puts, "expected uploads"
    assert all(store.meta[k] is None for k in store.puts), \
        "no artifact may carry submitter identity metadata after the identity strip"
    # the injected user_email must not survive parsing onto the request
    assert not hasattr(RenderRequest.from_dict(_job(user_email="x@y.z")), "user_email")


def test_run_job_rejects_empty_bundle_key(tmp_path):
    # A render with no bundle_key must fail with a clear HarnessError, not a botocore
    # ParamValidationError from a head_object on an empty Key.
    with pytest.raises(HarnessError, match="bundle_key is required"):
        run_job(_job(bundle_key=""), pipeline=FakePipeline(), store=FakeStore(tmp_path / "x.tar.gz"),
                workdir=tmp_path / "w")


def test_run_job_rejects_invalid_storyboard(tmp_path):
    bad = dict(STORYBOARD, use_characters=["A"])  # shot_02 references B, not in use_characters
    tarp = tmp_path / "bad.tar.gz"
    with tarfile.open(tarp, "w:gz") as tf:
        for name, data in {"storyboard.yaml": yaml.safe_dump(bad).encode(),
                           "characters/registry.json": b'{"characters":{}}'}.items():
            info = tarfile.TarInfo(name=name); info.size = len(data); tf.addfile(info, io.BytesIO(data))
    with pytest.raises(HarnessError, match="invalid render job"):
        run_job(_job(), pipeline=FakePipeline(), store=FakeStore(tarp), workdir=tmp_path / "w")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_run_job_normal_merges_to_output_key(tmp_path):
    import subprocess

    class RealClipPipeline(FakePipeline):
        def execute(self, plan, bundle, workdir):
            out = super().execute(plan, bundle, workdir)
            real = []  # replace the empty stub clips with tiny real mp4s so assemble can merge
            for shot_id, _ in out.clips:
                p = workdir / f"{shot_id}_real.mp4"
                subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1:r=24",
                                "-pix_fmt", "yuv420p", "-t", "1", str(p)], capture_output=True, check=True)
                real.append((shot_id, p))
            out.clips = real
            return out

    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(), pipeline=RealClipPipeline(), store=store, workdir=tmp_path / "work")
    assert res["output_key"] == "renders/neon/full.mp4"
    assert res["seconds"] == pytest.approx(2.0, abs=0.4)  # two 1s clips merged
    assert "renders/neon/full.mp4" in store.puts


# ---------------------------------------------------- incremental reuse via per-artifact R2 (#112)

def _extract_bundle(tmp_path):
    """A real Bundle from the standard test tar, for driving _restore_prior_state."""
    from vivijure_backend.contract import Bundle
    tarp = _bundle_tar(tmp_path / "restore_b.tar.gz")
    return Bundle.extract(tarp, tmp_path / "restore_project")


class R2StateStore:
    """Per-artifact fake: `objects` maps key -> bytes; exists/get_file/get_bytes serve it."""
    def __init__(self, objects):
        self.objects = dict(objects)
        self.gets: list[str] = []

    def exists(self, key):
        return key in self.objects

    def get_file(self, key, dest):
        self.gets.append(key)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.objects[key])
        return dest

    def get_bytes(self, key):
        return self.objects[key]


def test_restore_derives_sets_from_per_artifact_r2_objects(tmp_path):
    """#112: trained_slots / existing_keyframes come straight from the per-identity R2 objects
    the prior render uploaded -- no shared state tar exists to read (or to race on)."""
    from vivijure_backend.harness.handler import _restore_prior_state

    bundle = _extract_bundle(tmp_path)
    store = R2StateStore({
        keys.lora_key("neon", "A"): b"weights",           # slot A trained; B never was
        keys.keyframe_key("neon", "shot_01"): b"PNG",     # shot_01 reusable; shot_02 not
    })
    trained, existing = _restore_prior_state(store, "neon", bundle)
    assert trained == {"A"}
    assert existing == {"shot_01": None}  # no .hash sidecar -> None ("reuse conservatively")
    # the reusable PNG was staged into the bundle tree for the pipeline to animate
    assert (bundle.root / "keyframes" / "shot_01.png").read_bytes() == b"PNG"


def test_restore_reads_hash_from_sidecar_object(tmp_path):
    """The .hash sidecar (#112) is the param hash the planner compares for reuse-vs-regen."""
    from vivijure_backend.harness.handler import _restore_prior_state

    bundle = _extract_bundle(tmp_path)
    store = R2StateStore({
        keys.keyframe_key("neon", "shot_01"): b"PNG",
        keys.keyframe_hash_key("neon", "shot_01"): b"abcdef1234567890\n",
    })
    trained, existing = _restore_prior_state(store, "neon", bundle)
    assert existing == {"shot_01": "abcdef1234567890"}
    # the hash is also staged locally alongside the PNG (warm-worker parity)
    assert (bundle.root / "keyframes" / "shot_01.hash").read_text() == "abcdef1234567890"


def test_restore_never_overwrites_a_bundle_provided_keyframe(tmp_path):
    """Hybrid-lane contract: a keyframe the BUNDLE carries wins over the R2-restored one
    (the control plane splices exact frames into the bundle; restore only fills gaps)."""
    from vivijure_backend.harness.handler import _restore_prior_state

    bundle = _extract_bundle(tmp_path)
    injected = bundle.root / "keyframes" / "shot_01.png"
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_bytes(b"BUNDLE-EXACT-FRAME")
    store = R2StateStore({keys.keyframe_key("neon", "shot_01"): b"R2-OLD-FRAME"})

    trained, existing = _restore_prior_state(store, "neon", bundle)
    assert injected.read_bytes() == b"BUNDLE-EXACT-FRAME"          # untouched
    assert keys.keyframe_key("neon", "shot_01") not in store.gets  # fetch skipped entirely
    assert "shot_01" in existing                                    # still known-reusable


def test_restore_skips_keyframe_absent_from_r2(tmp_path):
    """#108 semantics preserved: only an R2-present keyframe is reusable; an absent one is
    omitted so the planner GENERATEs it -- no phantom keys, and now no stale state object
    left to even claim one."""
    from vivijure_backend.harness.handler import _restore_prior_state

    bundle = _extract_bundle(tmp_path)
    store = R2StateStore({keys.keyframe_key("neon", "shot_01"): b"PNG"})
    trained, existing = _restore_prior_state(store, "neon", bundle)
    assert "shot_01" in existing
    assert "shot_02" not in existing


def test_restore_returns_empty_on_fresh_project_and_on_store_failure(tmp_path):
    """A fresh project (nothing in R2) and a store that throws both degrade to empty sets:
    the safe default is a full render, never an aborted job."""
    from vivijure_backend.harness.handler import _restore_prior_state

    bundle = _extract_bundle(tmp_path)
    trained, existing = _restore_prior_state(R2StateStore({}), "fresh", bundle)
    assert trained == set() and existing == {}

    class ExplodingStore:
        def exists(self, key):
            raise RuntimeError("store down")

    trained, existing = _restore_prior_state(ExplodingStore(), "fresh", bundle)
    assert trained == set() and existing == {}


def test_run_job_uploads_keyframe_hash_sidecars(tmp_path):
    """#112: each authored keyframe's param hash rides to R2 as its own sidecar object, so the
    NEXT render's restore can make the reuse decision without any shared state object."""
    class HashingPipeline(FakePipeline):
        def execute(self, plan, bundle, workdir):
            out = super().execute(plan, bundle, workdir)
            for shot_id, p in out.keyframes.items():
                Path(p).with_suffix(".hash").write_text(f"hash-{shot_id}")
            return out

    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    run_job(_job(render_overrides={"finish_offloaded": True}),
            pipeline=HashingPipeline(), store=store, workdir=tmp_path / "work")
    for shot_id in ("shot_01", "shot_02"):
        assert keys.keyframe_key("neon", shot_id) in store.puts
        assert keys.keyframe_hash_key("neon", shot_id) in store.puts


def test_run_job_writes_no_shared_state_object(tmp_path):
    """#112 acceptance: nothing a render persists is a shared mutable object. Every put is a
    per-identity artifact key; the old projects/<slug>/state.tar.gz is never written, so
    concurrent shards of a scattered render have nothing left to clobber."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(render_overrides={"finish_offloaded": True}),
                  pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    assert store.tars == []                                   # no put_dir_as_tar call at all
    assert not any(k.startswith("projects/") for k in store.puts)
    assert res["state_key"] is None


# --------------------------------------------------------- job-key guard + slug unification (S3)

def test_standalone_clip_keys_use_the_shared_slug():
    # One project = ONE slug spelling across the full-render and the standalone-job paths
    # ("My  Film" must never scatter clips across My_Film/ and My__Film/ phantom prefixes).
    assert keys.i2v_clip_key("My  Film/x", "sh ot") == "renders/My_Film_x/clips/sh_ot_i2v.mp4"
    assert keys.finished_clip_key("My  Film/x", "sh ot") == "renders/My_Film_x/clips/sh_ot_finished.mp4"
    full = keys.clip_key("My  Film/x", "s").rsplit("/", 1)[0]
    assert keys.i2v_clip_key("My  Film/x", "s").rsplit("/", 1)[0] == full


def test_check_job_key_accepts_a_key_under_its_prefix():
    k = "bundles/neon/b.tar.gz"
    assert keys.check_job_key(k, prefixes=("bundles/",), what="t") == k


def test_check_job_key_rejects_out_of_prefix_traversal_and_absolute():
    for bad in ("projects/neon/state.tar.gz",  # wrong prefix
                "/bundles/x",                   # absolute
                "bundles/../projects/x",        # traversal
                "bundles\\x",                   # backslash
                "",                             # empty
                " bundles/x"):                  # surrounding whitespace
        with pytest.raises(ValueError):
            keys.check_job_key(bad, prefixes=("bundles/",), what="t")


def test_run_job_rejects_a_bundle_key_outside_the_bundle_prefix(tmp_path):
    with pytest.raises(HarnessError, match="bundle_key"):
        run_job(_job(bundle_key="renders/neon/full.mp4"), pipeline=FakePipeline(),
                store=FakeStore(tmp_path / "x.tar.gz"), workdir=tmp_path / "w")


def test_run_job_rejects_a_pretrained_lora_ref_outside_loras(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    with pytest.raises(HarnessError, match="LoRA"):
        run_job(_job(pretrained_loras={"A": "bundles/evil.safetensors"},
                     render_overrides={"finish_offloaded": True}),
                pipeline=FakePipeline(), store=store, workdir=tmp_path / "w")


# --------------------------------------------------------- audio honesty (S3)

class NoAudioStore(FakeStore):
    """Serves the bundle for everything except audio keys, which fail like a missing object."""
    def get_file(self, key, dest):
        if key.startswith("audio/"):
            raise RuntimeError("NoSuchKey")
        return super().get_file(key, dest)


def test_requested_audio_that_cannot_be_fetched_fails_the_render(tmp_path):
    # A silent film under a success status is the dishonest degrade; the DEFAULT is a real failure.
    store = NoAudioStore(_bundle_tar(tmp_path / "b.tar.gz"))
    with pytest.raises(HarnessError, match="audio bed"):
        run_job(_job(audio_key="audio/bed.m4a", render_overrides={"finish_offloaded": True}),
                pipeline=FakePipeline(), store=store, workdir=tmp_path / "w")


def test_audio_optional_opt_in_degrades_loud_not_silent(tmp_path):
    # The EXPLICIT opt-in ships the film silent and says so in the TOP-LEVEL result, not just
    # the event stream (a degrade is never silent).
    store = NoAudioStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(audio_key="audio/bed.m4a",
                       render_overrides={"finish_offloaded": True, "audio_optional": True}),
                  pipeline=FakePipeline(), store=store, workdir=tmp_path / "w")
    assert res["audio_missing"] is True
    assert res["has_audio"] is False
    assert [c["shot_id"] for c in res["clips"]]       # the film itself still shipped


def test_result_reports_audio_missing_false_by_default(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(render_overrides={"finish_offloaded": True}),
                  pipeline=FakePipeline(), store=store, workdir=tmp_path / "w")
    assert res["audio_missing"] is False


def test_audio_key_outside_the_key_map_is_rejected(tmp_path):
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    with pytest.raises(HarnessError, match="audio_key"):
        run_job(_job(audio_key="projects/neon/state.tar.gz",
                     render_overrides={"finish_offloaded": True}),
                pipeline=FakePipeline(), store=store, workdir=tmp_path / "w")


# ------------------------------------------------------- F17: terminal snapshots never hit RunPod

def test_runpod_hook_never_mirrors_terminal_snapshots(monkeypatch):
    """F17: the SDK's progress_update posts {"status": "IN_PROGRESS", "output": snapshot} from a
    fresh daemon thread, racing the SDK's own terminal result POST on the same endpoint -- a
    mirrored error/complete snapshot can flip a FAILED/COMPLETED job back to IN_PROGRESS forever,
    holding the billed worker (observed: 344s on a 155ms config-error run). Terminal snapshots
    must never go through the hook; mid-flight ones still do."""
    import sys
    import types

    from vivijure_backend.harness.handler import _runpod_progress_hook

    sent = []
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(progress_update=lambda job, snap: sent.append(snap))
    monkeypatch.setitem(sys.modules, "runpod", fake)

    hook = _runpod_progress_hook({"id": "job-1"})
    hook({"status": "running", "counts": {"train_done": 1}})                       # mirrored
    hook({"status": "error", "error": {"stage": "config", "message": "boom"}})     # dropped
    hook({"status": "complete", "counts": {}})                                     # dropped
    assert [s["status"] for s in sent] == ["running"]


def test_emitter_error_path_sends_nothing_through_the_runpod_hook(monkeypatch):
    """The exact F17 shape end to end: ProgressEmitter.error() fans out the terminal snapshot to
    R2 + stdout but the RunPod hook drops it, so the handler's re-raise is what sets the job's
    terminal status (unclobbered)."""
    import sys
    import types

    from vivijure_backend.harness.handler import _runpod_progress_hook
    from vivijure_backend.harness.progress import ProgressEmitter

    sent = []
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(progress_update=lambda job, snap: sent.append(snap))
    monkeypatch.setitem(sys.modules, "runpod", fake)

    emitter = ProgressEmitter(None, "untitled", "job-1",
                              on_progress=_runpod_progress_hook({"id": "job-1"}), log=lambda _s: None)
    emitter.emit("started")
    emitter.error("config", "R2 config incomplete; missing env: R2_ACCESS_KEY_ID")
    assert [s["status"] for s in sent] == ["running"]


# ------------------------------------------------- #90: mirror posts quiesce before the result

def _fake_runpod_with_rp_progress(monkeypatch, target):
    """Install a fake runpod SDK whose rp_progress._thread_target is `target`, matching the
    import shape the hook uses (from runpod.serverless.modules import rp_progress)."""
    import sys
    import types

    fake = types.ModuleType("runpod")
    fake.serverless = types.ModuleType("runpod.serverless")
    fake.serverless.modules = types.ModuleType("runpod.serverless.modules")
    rp_progress = types.ModuleType("runpod.serverless.modules.rp_progress")
    rp_progress._thread_target = target
    fake.serverless.modules.rp_progress = rp_progress
    monkeypatch.setitem(sys.modules, "runpod", fake)
    monkeypatch.setitem(sys.modules, "runpod.serverless", fake.serverless)
    monkeypatch.setitem(sys.modules, "runpod.serverless.modules", fake.serverless.modules)
    monkeypatch.setitem(sys.modules, "runpod.serverless.modules.rp_progress", rp_progress)
    return fake


def test_runpod_hook_quiesce_joins_inflight_mirror_posts(monkeypatch):
    """#90: a slow in-flight mirror post must be DRAINED by hook.quiesce() before the handler
    returns -- otherwise it races the SDK's terminal result post on the same /job-done endpoint
    (the 400 'internal server error' noise, misattributed as 'Failed to return job results.')."""
    import threading
    import time

    from vivijure_backend.harness.handler import _runpod_progress_hook

    done = []

    def slow_post(job, snapshot):
        time.sleep(0.15)          # an HTTP post still in flight when the handler finishes
        done.append(snapshot)

    _fake_runpod_with_rp_progress(monkeypatch, slow_post)
    hook = _runpod_progress_hook({"id": "job-1"})
    hook({"status": "running", "counts": {"i2v_step": 1}})
    assert done == []             # post genuinely in flight
    hook.quiesce()
    assert len(done) == 1         # drained BEFORE the caller can post the terminal result


def test_runpod_hook_falls_back_to_sdk_progress_update_without_rp_progress(monkeypatch):
    """If the SDK internals move (no rp_progress module), the hook degrades to the untracked
    progress_update call -- the mirror keeps working, best-effort doctrine."""
    import sys
    import types

    from vivijure_backend.harness.handler import _runpod_progress_hook

    sent = []
    fake = types.ModuleType("runpod")
    fake.serverless = types.SimpleNamespace(progress_update=lambda job, snap: sent.append(snap))
    monkeypatch.setitem(sys.modules, "runpod", fake)

    hook = _runpod_progress_hook({"id": "job-1"})
    hook({"status": "running", "counts": {}})
    assert [s["status"] for s in sent] == ["running"]
    hook.quiesce()                # no tracked threads: a no-op, never an error
