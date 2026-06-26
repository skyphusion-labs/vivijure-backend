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
from vivijure_backend.harness.models_mirror import mirror_cmd, rclone_conf
from vivijure_backend.harness.r2 import R2Config


# ------------------------------------------------------------------------------- keys

def test_keys_layout():
    assert keys.output_key("neon") == "renders/neon/full.mp4"
    assert keys.state_key("neon") == "projects/neon/state.tar.gz"
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
    cmd = mirror_cmd(Path("/c.conf"), "r2:vivijure/models/hf-cache", Path("/hf"),
                     skip_repos=("models--X", "spaces--Y"))
    assert cmd[:5] == ["rclone", "--config", "/c.conf", "copy", "--links"]
    assert "--exclude" in cmd and "**/*.incomplete" in cmd
    assert "hub/models--X/**" in cmd and "hub/spaces--Y/**" in cmd
    assert cmd[-2:] == ["r2:vivijure/models/hf-cache", "/hf"]  # src, dst last


def test_rclone_conf_writes_creds_and_rejects_partial(tmp_path):
    conf = rclone_conf({"R2_ACCESS_KEY_ID": "k", "R2_SECRET_ACCESS_KEY": "s",
                        "R2_ENDPOINT": "https://x.r2"}, tmp_path)
    text = conf.read_text()
    assert "access_key_id = k" in text and "endpoint = https://x.r2" in text
    with pytest.raises(RuntimeError, match="incomplete R2 creds"):
        rclone_conf({"R2_ACCESS_KEY_ID": "k"}, tmp_path)


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
        return True  # default fake: any state-claimed artifact is treated as present in R2

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
    assert res["state_key"] == "projects/neon/state.tar.gz"
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
    assert store.meta[res["state_key"]] is None
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


# ---------------------------------------------------------------- incremental reuse

def _make_state_tar(path: Path, slot_markers: list[str], kf_ids: list[str]) -> Path:
    """Build a minimal state.tar.gz with lora markers and keyframe PNGs."""
    members: dict[str, bytes] = {
        "storyboard.yaml": yaml.safe_dump(STORYBOARD).encode(),
    }
    for slot in slot_markers:
        members[f"loras/{slot}/.trained"] = b""
    for shot_id in kf_ids:
        members[f"keyframes/{shot_id}.png"] = b"PNG"
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_run_job_copies_keyframes_into_bundle_root(tmp_path):
    """After a render, generated keyframes must appear in bundle.root/keyframes/ so the next
    incremental render can find them via _restore_prior_state."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    res = run_job(_job(action="preview"), pipeline=FakePipeline(), store=store,
                  workdir=tmp_path / "work")
    # FakePipeline generates keyframes for GENERATE shots; preview generates keyframes, no i2v
    work = tmp_path / "work" / "project"
    kf_dir = work / "keyframes"
    assert kf_dir.is_dir(), "keyframes/ must be written into bundle.root after render"
    kf_names = {p.stem for p in kf_dir.iterdir() if p.suffix == ".png"}
    assert "shot_01" in kf_names and "shot_02" in kf_names


def test_run_job_writes_lora_markers_into_bundle_root(tmp_path):
    """After a render, trained LoRA slots must have a .trained marker in bundle.root/loras/."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    # Use finish_offloaded to skip the ffmpeg merge step (FakePipeline produces dummy MP4 bytes)
    run_job(_job(render_overrides={"finish_offloaded": True}),
            pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    work = tmp_path / "work" / "project"
    for slot in ("A", "B"):
        marker = work / "loras" / slot / ".trained"
        assert marker.is_file(), f"lora marker missing for slot {slot}: {marker}"


def test_restore_prior_state_derives_sets_from_tar(tmp_path):
    """_restore_prior_state must extract the state tar and return the right sets."""
    from vivijure_backend.harness.handler import _restore_prior_state

    state_tar = _make_state_tar(tmp_path / "state.tar.gz",
                                slot_markers=["A"], kf_ids=["shot_01"])

    class StateFakeStore:
        def get_file(self, key, dest):
            shutil.copy(state_tar, dest)
            return dest
        def exists(self, key):
            return True  # keyframe is present in R2

    workdir = tmp_path / "work"
    workdir.mkdir()
    trained, existing = _restore_prior_state(StateFakeStore(), "neon", workdir)
    assert trained == {"A"}
    assert existing == {"shot_01": None}  # no .hash file in tar -> None value


def test_restore_prior_state_reads_hash_from_dot_hash_file(tmp_path):
    """A .hash file alongside the PNG in state is read back as the stored hash."""
    from vivijure_backend.harness.handler import _restore_prior_state

    import tarfile, io
    # Build a state tar with a keyframe PNG AND a .hash file alongside it
    tar_path = tmp_path / "state.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        # keyframe PNG (no project/ prefix -- matches _make_state_tar convention)
        png_data = b"PNG"
        ti = tarfile.TarInfo("keyframes/shot_01.png")
        ti.size = len(png_data)
        tf.addfile(ti, io.BytesIO(png_data))
        # .hash file alongside the PNG
        h = b"abcdef1234567890"
        hi = tarfile.TarInfo("keyframes/shot_01.hash")
        hi.size = len(h)
        tf.addfile(hi, io.BytesIO(h))

    class StateFakeStore:
        def get_file(self, key, dest):
            import shutil; shutil.copy(tar_path, dest); return dest
        def exists(self, key):
            return True  # keyframe is present in R2

    workdir = tmp_path / "work"
    workdir.mkdir()
    trained, existing = _restore_prior_state(StateFakeStore(), "neon", workdir)
    assert existing == {"shot_01": "abcdef1234567890"}


def test_restore_prior_state_skips_keyframe_absent_from_r2(tmp_path):
    """#108 regression: a stale state tar names a keyframe whose R2 object is GONE (e.g. the
    renders/ were cleared but the per-project state.tar.gz was not). That shot must NOT be marked
    REUSE off the stale state -- it's omitted from existing_keyframes so the planner GENERATEs it,
    instead of reporting a phantom keyframe key to a nonexistent object."""
    from vivijure_backend.harness import keys
    from vivijure_backend.harness.handler import _restore_prior_state

    # State tar claims BOTH shots have keyframes...
    state_tar = _make_state_tar(tmp_path / "state.tar.gz",
                                slot_markers=["A"], kf_ids=["shot_01", "shot_02"])

    class PartialR2FakeStore:
        """...but R2 only actually has shot_01 (shot_02's object was cleared)."""
        def get_file(self, key, dest):
            shutil.copy(state_tar, dest)
            return dest
        def exists(self, key):
            return key == keys.keyframe_key("neon", "shot_01")

    workdir = tmp_path / "work"
    workdir.mkdir()
    trained, existing = _restore_prior_state(PartialR2FakeStore(), "neon", workdir)
    assert trained == {"A"}
    assert "shot_01" in existing            # present in R2 -> reusable
    assert "shot_02" not in existing        # absent from R2 -> GENERATE, never a phantom


def test_restore_prior_state_returns_empty_on_missing_state(tmp_path):
    """A project with no prior state (KeyError / fetch failure) returns empty sets."""
    from vivijure_backend.harness.handler import _restore_prior_state

    class FailingStore:
        def get_file(self, key, dest):
            raise FileNotFoundError("no prior state")

    workdir = tmp_path / "work"
    workdir.mkdir()
    trained, existing = _restore_prior_state(FailingStore(), "fresh-project", workdir)
    assert trained == set() and existing == {}
