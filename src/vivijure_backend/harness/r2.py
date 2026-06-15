"""R2 (S3-compatible) object I/O for the worker.

The worker holds exactly one credential: an R2 API token scoped to one bucket, delivered as
endpoint env vars, never baked into the image and never any skyphusion/Access secret. It pulls
the project bundle in at job start and pushes the rendered MP4 plus the project-state tarball
back out at the end. R2 speaks S3, so boto3 drives it directly.

boto3 is imported lazily inside `_client` so this module loads on a CPU box with no AWS deps;
the worker image installs boto3. Bundle *parsing* is not here (that is `contract.Bundle`); this
is just bytes in and out of the store.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class R2Config:
    """R2 connection settings: the endpoint, the access key pair, and the bucket. The worker's
    only credential, supplied through the environment."""
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_env(cls, env: dict | None = None) -> "R2Config":
        """Build from `R2_ENDPOINT` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET`;
        raise if any are missing."""
        e = env if env is not None else os.environ
        missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
                   if not e.get(k)]
        if missing:
            raise RuntimeError("R2 config incomplete; missing env: " + ", ".join(missing))
        return cls(e["R2_ENDPOINT"], e["R2_ACCESS_KEY_ID"], e["R2_SECRET_ACCESS_KEY"], e["R2_BUCKET"])


class R2:
    """A thin bucket-scoped object client. One per job is fine; boto3 clients are cheap."""

    def __init__(self, config: R2Config):
        self.config = config
        self._cli = None

    def _client(self):
        if self._cli is None:
            import boto3  # deferred: keep this module CPU/dep-light
            from botocore.config import Config

            self._cli = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                config=Config(signature_version="s3v4",
                              retries={"max_attempts": 5, "mode": "standard"}),
                region_name="auto",  # R2 ignores region; boto3 insists on one
            )
        return self._cli

    def get_file(self, key: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        head = self._client().head_object(Bucket=self.config.bucket, Key=key)
        expected = head.get("ContentLength")
        self._client().download_file(self.config.bucket, key, str(dest))
        actual = dest.stat().st_size
        if expected is not None and actual != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"R2 download truncated: {key!r} expected {expected} bytes, got {actual}")
        return dest

    def put_file(self, path: Path, key: str, *, content_type: str | None = None,
                 metadata: dict[str, str] | None = None) -> str:
        """Upload one file. `metadata` becomes S3 user metadata; the control plane's artifact
        route reads `user_email` from it to gate ownership, so a downloadable artifact must
        carry it."""
        extra: dict[str, object] = {}
        if content_type:
            extra["ContentType"] = content_type
        if metadata:
            extra["Metadata"] = metadata
        self._client().upload_file(str(path), self.config.bucket, key, ExtraArgs=extra or None)
        return key

    def put_bytes(self, data: bytes, key: str, *, content_type: str | None = None,
                  metadata: dict[str, str] | None = None) -> str:
        """Upload in-memory bytes with no temp file (for the small JSON / NDJSON the progress
        channel writes). put_object, not upload_file, since there is nothing on disk."""
        kwargs: dict[str, object] = {"Bucket": self.config.bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        if metadata:
            kwargs["Metadata"] = metadata
        self._client().put_object(**kwargs)
        return key

    def get_bytes(self, key: str) -> bytes:
        """Read one object into memory (for reading a progress snapshot back)."""
        return self._client().get_object(Bucket=self.config.bucket, Key=key)["Body"].read()

    def put_dir_as_tar(self, src_dir: Path, key: str, *, metadata: dict[str, str] | None = None) -> str:
        """Tar a directory contents-at-root (`arcname="."`) and upload it. Contents-at-root,
        not `<name>/`-rooted: the inbound bundle extracts INTO the project dir, so a name-rooted
        state tar would double-nest on the next incremental render."""
        extra: dict[str, object] = {"ContentType": "application/gzip"}
        if metadata:
            extra["Metadata"] = metadata
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                tar.add(str(src_dir), arcname=".")
            self._client().upload_file(str(tmp_path), self.config.bucket, key, ExtraArgs=extra)
        finally:
            tmp_path.unlink(missing_ok=True)
        return key
