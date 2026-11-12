"""deploy/smoke_image.py (#645): smoke a worker on the stills base without pulling its weight layers.

A fake registry holds a toolchain, a stills base (the toolchain plus weight layers) and a worker (the
base plus thin layers), with real sha256 digests over real bytes. No network, no docker."""
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("smoke_image", ROOT / "deploy" / "smoke_image.py")
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)

REPO = "skyphusion-labs/vivijure-backend"
GHCR = "ghcr.io/" + REPO
GZ = "application/vnd.oci.image.layer.v1.tar+gzip"


def sha(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


class FakeRegistry:
    def __init__(self):
        self.blobs: dict[str, bytes] = {}
        self.manifests: dict[str, dict] = {}
        self.streamed: list[str] = []

    def put(self, data: bytes) -> str:
        d = sha(data)
        self.blobs[d] = data
        return d

    def image(self, layers: list[bytes], labels: dict | None = None, history: list | None = None,
              diff_ids: list[str] | None = None) -> str:
        """Push an image; each layer's diff_id is the sha of b'diff:' + its bytes (a stand-in)."""
        descs = [{"mediaType": GZ, "digest": self.put(x), "size": len(x)} for x in layers]
        cfg = {"architecture": "amd64", "os": "linux",
               "config": {"Entrypoint": ["/e"], "Env": ["A=1"], "Labels": dict(labels or {})},
               "rootfs": {"type": "layers",
                          "diff_ids": diff_ids if diff_ids is not None else [sha(b"diff:" + x) for x in layers]},
               "history": history if history is not None else
               [{"created_by": f"layer {i}"} for i in range(len(layers))] + [{"created_by": "ENV", "empty_layer": True}]}
        cb = json.dumps(cfg).encode()
        m = {"schemaVersion": 2, "mediaType": si.OCI_MANIFEST,
             "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": self.put(cb), "size": len(cb)},
             "layers": descs}
        mb = json.dumps(m).encode()
        d = sha(mb)
        self.manifests[d] = m
        return f"{GHCR}:t@{d}"

    # the three Registry methods
    def manifest(self, repo, digest):
        assert repo == REPO
        if digest not in self.manifests:
            raise si.Refused("manifest unknown")
        return self.manifests[digest]

    def blob_json(self, repo, digest):
        return json.loads(self.blobs[digest])

    def blob_stream(self, repo, digest):
        self.streamed.append(digest)
        return io.BytesIO(self.blobs[digest])


TOOL = [b"tool-a" * 50, b"tool-b" * 70]
WEIGHTS = [b"weights-1" * 400, b"weights-2" * 300]
THIN = [b"workdir", b"app-code" * 20]


@pytest.fixture
def stack():
    reg = FakeRegistry()
    tool = reg.image(TOOL)
    base = reg.image(TOOL + WEIGHTS, labels={si.TOOLCHAIN_LABEL: tool, "base": "1"})
    worker = reg.image(TOOL + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: tool, "base": "1"})
    return reg, tool, base, worker


def test_the_weight_layers_are_dropped_and_everything_else_kept_in_order(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    man = json.loads(p["manifest"])
    assert [x["digest"] for x in man["layers"]] == [sha(x) for x in TOOL + THIN]
    assert p["diff_ids"] == [sha(b"diff:" + x) for x in TOOL + THIN]
    assert [d["digest"] for d in p["dropped"]] == [sha(x) for x in WEIGHTS]
    assert p["dropped_bytes"] == sum(len(x) for x in WEIGHTS)
    assert p["kept"] == 4 and p["toolchain"] == tool
    assert p["bytes"] == len(p["config"]) + sum(len(x) for x in TOOL + THIN)


def test_the_config_is_the_workers_minus_the_dropped_layers(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    cfg = json.loads(p["config"])
    wm = reg.manifests[worker.split("@")[1]]
    wcfg = json.loads(reg.blobs[wm["config"]["digest"]])
    assert cfg["config"]["Entrypoint"] == wcfg["config"]["Entrypoint"]
    assert cfg["config"]["Env"] == wcfg["config"]["Env"]
    labels = dict(cfg["config"]["Labels"])
    assert labels.pop(si.SMOKE_OF_LABEL) == GHCR + "@" + worker.split("@")[1]
    assert labels == wcfg["config"]["Labels"]
    # history: the two weight layer entries are gone, the empty ENV entry stays
    assert [h["created_by"] for h in cfg["history"]] == ["layer 0", "layer 1", "layer 4", "layer 5", "ENV"]
    assert json.loads(p["manifest"])["config"]["digest"] == sha(p["config"].encode())


def test_without_a_base_nothing_is_dropped(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, None, reg)
    assert p["dropped"] == [] and p["kept"] == 6
    assert p["diff_ids"] == [sha(b"diff:" + x) for x in TOOL + WEIGHTS + THIN]


def test_a_plain_toolchain_base_drops_nothing():
    reg = FakeRegistry()
    tool = reg.image(TOOL)
    worker = reg.image(TOOL + THIN)
    p = si.plan(worker, tool, reg)
    assert p["dropped"] == [] and p["kept"] == 4


def test_a_worker_not_built_from_the_base_is_refused(stack):
    reg, tool, base, worker = stack
    other = reg.image([b"x"] + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: tool})
    with pytest.raises(si.Refused, match="not built FROM this BASE"):
        si.plan(other, base, reg)


def test_a_worker_whose_diff_ids_differ_from_the_base_is_refused(stack):
    reg, tool, base, _ = stack
    bad = reg.image(TOOL + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: tool},
                    diff_ids=[sha(b"other")] + [sha(b"diff:" + x) for x in (TOOL + WEIGHTS + THIN)[1:]])
    with pytest.raises(si.Refused, match="not built FROM this BASE"):
        si.plan(bad, base, reg)


def test_a_base_that_does_not_start_with_its_toolchain_is_refused():
    reg = FakeRegistry()
    tool = reg.image(TOOL)
    base = reg.image([b"other"] + TOOL[1:] + WEIGHTS, labels={si.TOOLCHAIN_LABEL: tool})
    worker = reg.image([b"other"] + TOOL[1:] + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: tool})
    with pytest.raises(si.Refused, match="does not start with its toolchain"):
        si.plan(worker, base, reg)


def test_a_worker_label_that_names_another_toolchain_is_refused(stack):
    reg, tool, base, _ = stack
    worker = reg.image(TOOL + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: GHCR + ":t@sha256:" + "0" * 64})
    with pytest.raises(si.Refused, match="label"):
        si.plan(worker, base, reg)


def test_a_toolchain_label_with_no_layers_over_it_is_refused():
    reg = FakeRegistry()
    tool = reg.image(TOOL)
    base = reg.image(TOOL, labels={si.TOOLCHAIN_LABEL: tool})
    worker = reg.image(TOOL + THIN, labels={si.TOOLCHAIN_LABEL: tool})
    with pytest.raises(si.Refused, match="adds no layers"):
        si.plan(worker, base, reg)


def test_history_that_does_not_match_the_layers_is_refused(stack):
    reg, tool, base, _ = stack
    worker = reg.image(TOOL + WEIGHTS + THIN, labels={si.TOOLCHAIN_LABEL: tool},
                       history=[{"created_by": "only one"}])
    with pytest.raises(si.Refused, match="history"):
        si.plan(worker, base, reg)


def test_diff_ids_that_do_not_match_the_layers_are_refused(stack):
    reg, tool, base, _ = stack
    worker = reg.image(TOOL + WEIGHTS + THIN, diff_ids=[sha(b"one")])
    with pytest.raises(si.Refused, match="diff_ids"):
        si.plan(worker, base, reg)


@pytest.mark.parametrize("ref", [
    GHCR + ":1.1.20",                                   # a tag, not a digest
    GHCR + "@sha256:" + "a" * 63,                       # short
    "docker.io/library/ubuntu@sha256:" + "a" * 64,      # another registry
    "ghcr.io/someone-else/x@sha256:" + "a" * 64,        # another owner
    GHCR + "@sha256:" + "A" * 64,                       # upper case
])
def test_a_ref_that_is_not_a_full_ghcr_digest_is_refused(stack, ref):
    reg, tool, base, worker = stack
    with pytest.raises(si.Refused, match="must be ghcr.io"):
        si.plan(ref, None, reg)
    with pytest.raises(si.Refused, match="must be ghcr.io"):
        si.plan(worker, ref, reg)


def _members(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        return {m.name: tf.extractfile(m).read() for m in tf.getmembers()}


def test_the_stream_is_a_loadable_archive_of_exactly_the_kept_blobs(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    out = io.BytesIO()
    si.stream(p, reg, out, "vj-smoke:run1")
    files = _members(out.getvalue())
    assert reg.streamed == [sha(x) for x in TOOL + THIN]          # weights never fetched
    man = json.loads(p["manifest"])
    for layer, data in zip(man["layers"], TOOL + THIN):
        assert files["blobs/sha256/" + layer["digest"][7:]] == data
    assert files["blobs/sha256/" + man["config"]["digest"][7:]] == p["config"].encode()
    assert json.loads(files["oci-layout"]) == {"imageLayoutVersion": "1.0.0"}
    index = json.loads(files["index.json"])
    (entry,) = index["manifests"]
    assert files["blobs/sha256/" + entry["digest"][7:]] == p["manifest"].encode()
    assert entry["annotations"]["io.containerd.image.name"] == "docker.io/library/vj-smoke:run1"
    (legacy,) = json.loads(files["manifest.json"])
    assert legacy["RepoTags"] == ["vj-smoke:run1"]
    assert legacy["Config"] == "blobs/sha256/" + man["config"]["digest"][7:]
    assert legacy["Layers"] == ["blobs/sha256/" + x["digest"][7:] for x in man["layers"]]
    assert not any(sha(x)[7:] in name for name in files for x in WEIGHTS)


def test_a_blob_that_does_not_hash_to_its_digest_stops_the_stream(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    reg.blobs[sha(THIN[1])] = THIN[1][:-1] + b"X"                # same size, different bytes
    with pytest.raises(si.Refused, match="hashing to"):
        si.stream(p, reg, io.BytesIO(), "vj-smoke:x")


def test_a_short_blob_stops_the_stream(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    reg.blobs[sha(TOOL[0])] = TOOL[0][:-5]
    with pytest.raises((si.Refused, tarfile.ReadError, OSError)):
        si.stream(p, reg, io.BytesIO(), "vj-smoke:x")


def test_a_long_blob_stops_the_stream(stack):
    reg, tool, base, worker = stack
    p = si.plan(worker, base, reg)
    reg.blobs[sha(TOOL[0])] = TOOL[0] + b"extra"
    with pytest.raises(si.Refused):
        si.stream(p, reg, io.BytesIO(), "vj-smoke:x")
