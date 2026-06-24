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
from vivijure_backend.harness.handler import HarnessError, Outputs, run_job, restore_from_r2
from vivijure_backend.contract import RenderRequest, Storyboard
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
    """In-memory R2 fake. Serves the prebuilt bundle for the bundle key, and models per-key PRESENCE
    for every object the worker puts (keyframes, hashes, loras) so the R2-authoritative restore (#112)
    sees a realistic store. `seed` pre-populates objects a prior render would have left. No network."""
    def __init__(self, bundle_tar: Path, *, seed: dict | None = None):
        self.bundle_tar = bundle_tar
        self.puts: list[str] = []
        self.tars: list[str] = []
        self.meta: dict[str, dict | None] = {}  # key -> customMetadata recorded at put time
        self.blobs: dict[str, bytes] = dict(seed or {})  # key -> bytes for objects present in R2

    def get_file(self, key, dest):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        if key in self.blobs:
            Path(dest).write_bytes(self.blobs[key])
        else:
            shutil.copy(self.bundle_tar, dest)  # the bundle key (only un-seeded get in these tests)
        return dest

    def get_bytes(self, key):
        return self.blobs[key]

    def exists(self, key):
        return key in self.blobs

    def put_file(self, path, key, *, content_type=None, metadata=None):
        assert Path(path).exists(), f"uploading a nonexistent file: {path}"
        self.puts.append(key)
        self.meta[key] = metadata
        self.blobs[key] = Path(path).read_bytes()
        return key

    def put_bytes(self, data, key, *, content_type=None, metadata=None):
        self.puts.append(key)
        self.meta[key] = metadata
        self.blobs[key] = data
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
                p.with_suffix(".hash").write_text("hash-" + s.shot_id)  # mirror pipeline._finish hash
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
    # #112: no shared state tar is written -- nothing for concurrent shards to clobber.
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
    assert res["state_key"] is None  # #112: no shared state object exists to stamp
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

# ----------------------------------------------- incremental reuse: R2-authoritative restore (#112)

def _sb():
    """The storyboard object restore_from_r2 walks (same STORYBOARD the fake bundle carries)."""
    return Storyboard.from_dict(STORYBOARD)


def test_run_job_uploads_keyframes_and_hashes_per_shot(tmp_path):
    """A render persists each keyframe AND its param-hash as its OWN per-shot R2 object (#112) -- the
    next render restores from these, no shared state tar. (The old behavior staged copies into
    bundle.root for a tarball; that shared object is gone.)"""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    run_job(_job(action="preview"), pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    for shot in ("shot_01", "shot_02"):
        assert keys.keyframe_key("neon", shot) in store.blobs
        assert keys.keyframe_hash_key("neon", shot) in store.blobs
    assert store.tars == []  # no shared mutable state object


def test_run_job_uploads_lora_adapter_as_trained_signal(tmp_path):
    """A trained slot's ADAPTER object in R2 (keys.lora_key) is the 'trained' signal the next render
    reads (#112) -- there is no .trained marker in a shared state tar anymore."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    run_job(_job(render_overrides={"finish_offloaded": True}),
            pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    for slot in ("A", "B"):
        assert keys.lora_key("neon", slot) in store.blobs, f"adapter missing for slot {slot}"


def test_restore_from_r2_happy_path_stages_keyframes_and_derives_sets(tmp_path):
    """The subtle part of #112: restore runs AFTER bundle-extract and must still STAGE prior keyframes
    on disk (so the pipeline can REUSE them) + derive trained_slots, with no shared state tar and no
    #108 hang. Seed R2 as a prior render left it; restore reuses both keyframes + skips both LoRAs."""
    seed = {
        keys.keyframe_key("neon", "shot_01"): b"PNG01",
        keys.keyframe_hash_key("neon", "shot_01"): b"hash-shot_01",
        keys.keyframe_key("neon", "shot_02"): b"PNG02",
        keys.keyframe_hash_key("neon", "shot_02"): b"hash-shot_02",
        keys.lora_key("neon", "A"): b"adapterA",
        keys.lora_key("neon", "B"): b"adapterB",
    }
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"), seed=seed)
    bundle_root = tmp_path / "project"; bundle_root.mkdir()
    trained, existing = restore_from_r2(store, "neon", _sb(), bundle_root)
    assert trained == {"A", "B"}
    assert existing == {"shot_01": "hash-shot_01", "shot_02": "hash-shot_02"}
    # keyframe PNGs STAGED on disk where the pipeline's _resolve_keyframe reads them
    assert (bundle_root / "keyframes" / "shot_01.png").read_bytes() == b"PNG01"
    assert (bundle_root / "keyframes" / "shot_02.png").read_bytes() == b"PNG02"


def test_restore_from_r2_present_keyframe_without_hash_reads_none(tmp_path):
    """A present keyframe with NO hash object reads None -> reuse conservatively (pre-hash state),
    exactly as the old state-tar path did."""
    seed = {keys.keyframe_key("neon", "shot_01"): b"PNG01"}  # no hash object
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"), seed=seed)
    bundle_root = tmp_path / "project"; bundle_root.mkdir()
    _, existing = restore_from_r2(store, "neon", _sb(), bundle_root)
    assert existing == {"shot_01": None}


def test_restore_from_r2_absent_keyframe_is_omitted_no_phantom(tmp_path):
    """#108 regression, R2-authoritative form: a storyboard shot whose keyframe is ABSENT from R2 is
    omitted (planner GENERATEs it) -- never a phantom reuse that hangs the shard. No state tar to be
    stale against; R2 presence is the only truth."""
    seed = {keys.keyframe_key("neon", "shot_01"): b"PNG01",
            keys.keyframe_hash_key("neon", "shot_01"): b"h1"}  # shot_02 absent from R2
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"), seed=seed)
    bundle_root = tmp_path / "project"; bundle_root.mkdir()
    trained, existing = restore_from_r2(store, "neon", _sb(), bundle_root)
    assert "shot_01" in existing and "shot_02" not in existing
    assert not (bundle_root / "keyframes" / "shot_02.png").exists()


def test_restore_from_r2_empty_on_fresh_project(tmp_path):
    """A fresh project (nothing in R2) restores empty sets -> a full render, no reuse."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))  # no seed
    bundle_root = tmp_path / "project"; bundle_root.mkdir()
    trained, existing = restore_from_r2(store, "fresh", _sb(), bundle_root)
    assert trained == set() and existing == {}


def test_concurrent_shards_do_not_clobber_state(tmp_path):
    """#112 -- the race the issue is about. Two concurrent scatter shards of ONE render, each scoped
    (process_shot_ids) to a DIFFERENT shot's keyframe, write to the SAME store. Both keyframes must
    survive and the next render must restore BOTH (regenerating neither).

    Under the OLD single shared projects/<slug>/state.tar.gz this was last-writer-wins: each shard
    tar'd only its own bundle.root, so the second write clobbered the first shard's keyframe out of
    the persisted state and the next render wastefully re-GENERATED it. This test would FAIL there
    (store.tars would hold two writes of the shared key, and a tar-based restore would see only the
    last shard's shot). With the R2-authoritative restore there is NO shared mutable object: each
    shard writes only its own stable per-shot key, so neither can clobber the other."""
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"))
    # Shard A renders only shot_01; shard B only shot_02 (concurrent, same project + store).
    run_job(_job(action="preview", process_shot_ids=["shot_01"]),
            pipeline=FakePipeline(), store=store, workdir=tmp_path / "shardA")
    run_job(_job(action="preview", process_shot_ids=["shot_02"]),
            pipeline=FakePipeline(), store=store, workdir=tmp_path / "shardB")

    assert store.tars == []  # no shared mutable state object was ever written
    assert keys.keyframe_key("neon", "shot_01") in store.blobs
    assert keys.keyframe_key("neon", "shot_02") in store.blobs  # the loser under the old shared key

    # The NEXT render restores from R2 and finds BOTH -> reuses both, regenerates neither.
    bundle_root = tmp_path / "next" / "project"; bundle_root.mkdir(parents=True)
    _, existing = restore_from_r2(store, "neon", _sb(), bundle_root)
    assert set(existing) == {"shot_01", "shot_02"}


def test_run_job_reuses_prior_keyframes_from_r2_no_regen(tmp_path):
    """End to end: a render whose project already has both keyframes in R2 (a prior render) REUSES
    them -- the planner generates no keyframe, so no fresh keyframe object is uploaded."""
    seed = {
        keys.keyframe_key("neon", "shot_01"): b"PNG01",  # no hash -> reuse conservatively (None)
        keys.keyframe_key("neon", "shot_02"): b"PNG02",
        keys.lora_key("neon", "A"): b"a", keys.lora_key("neon", "B"): b"b",
    }
    store = FakeStore(_bundle_tar(tmp_path / "b.tar.gz"), seed=seed)
    res = run_job(_job(action="preview"), pipeline=FakePipeline(), store=store, workdir=tmp_path / "work")
    assert {k["shot_id"] for k in res["keyframes"]} == {"shot_01", "shot_02"}  # both reported...
    # ...but neither was re-GENERATED: no fresh keyframe PNG was uploaded this run.
    assert [k for k in store.puts if k.startswith("renders/neon/keyframes/") and k.endswith(".png")] == []

