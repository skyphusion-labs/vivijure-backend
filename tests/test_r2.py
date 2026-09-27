"""CPU tests for the R2 client.

The real client needs boto3 + live credentials; those are never present in CI. Tests here cover
the pure logic that does not touch the network: config validation, and the `put_dir_as_tar`
tarball-content invariant (arcname="." means files land at root, not under a named directory).
"""
import tarfile
from pathlib import Path

import pytest

from vivijure_backend.harness.r2 import R2, R2Config


# --------------------------------------------------------------------------- config validation

def test_r2config_from_env_requires_all_four_keys():
    cfg = R2Config.from_env({
        "R2_ENDPOINT": "https://x.r2.dev",
        "R2_ACCESS_KEY_ID": "k",
        "R2_SECRET_ACCESS_KEY": "s",
        "R2_BUCKET": "vivijure",
    })
    assert cfg.bucket == "vivijure"
    with pytest.raises(RuntimeError, match="R2 config incomplete"):
        R2Config.from_env({"R2_ENDPOINT": "https://x.r2.dev"})


# ---------------------------------------------------------------- arcname="." tar invariant

def test_put_dir_as_tar_packs_contents_at_root(tmp_path, monkeypatch):
    """put_dir_as_tar uses arcname="." so the tar contains files at its root (no leading
    directory prefix). This is the invariant the bundle extractor relies on: it extracts INTO
    the project dir, so a name-rooted state tar would double-nest on the next render."""
    # Populate a source directory with a nested structure.
    src = tmp_path / "state"
    (src / "sub").mkdir(parents=True)
    (src / "top.txt").write_text("hello")
    (src / "sub" / "nested.txt").write_text("world")

    import io
    tar_bytes: list[bytes] = []

    class FakeBoto:
        def upload_file(self, local_path, bucket, key, ExtraArgs=None):
            # Read the tar bytes NOW (the caller deletes the temp file in its finally block).
            tar_bytes.append(Path(local_path).read_bytes())

    cfg = R2Config(endpoint="https://x", access_key_id="k", secret_access_key="s", bucket="b")
    client = R2(cfg)
    monkeypatch.setattr(client, "_client", lambda: FakeBoto())

    client.put_dir_as_tar(src, "state/latest.tar.gz")
    assert len(tar_bytes) == 1, "upload_file should be called exactly once"

    with tarfile.open(fileobj=io.BytesIO(tar_bytes[0]), mode="r:gz") as tf:
        names = {m.name for m in tf.getmembers()}

    # With arcname=".", members are ".", "./top.txt", "./sub", "./sub/nested.txt".
    # None should start with the source directory's basename.
    src_basename = src.name
    assert not any(n.startswith(src_basename) for n in names), (
        f"tar contains a path starting with {src_basename!r} -- arcname='.' was not applied: {names}"
    )
    # The actual content files must be present.
    assert "./top.txt" in names
    assert "./sub/nested.txt" in names


# ------------------------------------------------------------------------------ exists()

class _FakeClientError(Exception):
    """Same `.response` shape botocore's ClientError carries. CI does not install boto3, so the
    tests build the error from its documented shape rather than importing botocore."""

    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": "x"},
                         "ResponseMetadata": {"HTTPStatusCode": status}}


def _r2_with_head(monkeypatch, head):
    class FakeBoto:
        def head_object(self, Bucket, Key):
            return head()

    cfg = R2Config(endpoint="https://x", access_key_id="k", secret_access_key="s", bucket="b")
    client = R2(cfg)
    monkeypatch.setattr(client, "_client", lambda: FakeBoto())
    return client


def test_exists_true_when_object_present(monkeypatch):
    assert _r2_with_head(monkeypatch, lambda: {"ContentLength": 1}).exists("k") is True


@pytest.mark.parametrize("code", ["404", "NoSuchKey", "NotFound"])
def test_exists_false_only_for_a_real_not_found(monkeypatch, code):
    def head():
        raise _FakeClientError(code, 404)
    assert _r2_with_head(monkeypatch, head).exists("k") is False


@pytest.mark.parametrize("code,status", [
    ("403", 403), ("AccessDenied", 403), ("ExpiredToken", 400), ("InvalidAccessKeyId", 403),
    ("SlowDown", 503), ("InternalError", 500),
])
def test_exists_raises_on_non_404_errors(monkeypatch, code, status):
    """A credential or throttle failure is not proof of absence: it must surface, not read as
    'missing' (which would retrain every LoRA and redraw every keyframe on a paid GPU)."""
    def head():
        raise _FakeClientError(code, status)
    with pytest.raises(_FakeClientError):
        _r2_with_head(monkeypatch, head).exists("k")


def test_exists_with_real_botocore_client_error(monkeypatch):
    """The fake above mirrors botocore's shape; prove the real class classifies the same way
    wherever boto3 is installed (the GPU image; not the CI test env)."""
    botocore_exc = pytest.importorskip("botocore.exceptions")

    def err(code, status):
        return botocore_exc.ClientError(
            {"Error": {"Code": code, "Message": "x"},
             "ResponseMetadata": {"HTTPStatusCode": status}}, "HeadObject")

    def missing():
        raise err("404", 404)

    def denied():
        raise err("403", 403)

    assert _r2_with_head(monkeypatch, missing).exists("k") is False
    with pytest.raises(botocore_exc.ClientError):
        _r2_with_head(monkeypatch, denied).exists("k")
