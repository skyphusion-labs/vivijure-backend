"""R2 (S3-compatible) object I/O for the worker.

The worker holds exactly one TENANT credential: an R2 API token scoped to one bucket, carried in
the job payload (a pooled endpoint, one credential per tenant per job) or delivered as endpoint env
vars (a dedicated endpoint), never baked into the image and never any skyphusion/Access secret.
The shared models mirror is a SEPARATE credential on OUR bucket, read from the environment by
`models_mirror` and never routed through here. It pulls
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


# The optional per-job tenant R2 block in the job payload, and its required fields. Named here
# rather than inline so the handler, the tests, and the control plane all read one definition.
PAYLOAD_KEY = "r2"
PAYLOAD_REQUIRED = ("endpoint", "access_key_id", "secret_access_key", "bucket")


@dataclass(frozen=True)
class R2Config:
    """R2 connection settings for TENANT job I/O: the endpoint, the access key pair, the bucket,
    and an optional session token.

    Supplied either by the job payload (a pooled endpoint serving many tenants: each job carries
    its own tenant's credential) or by the endpoint environment (a dedicated endpoint: one tenant,
    one template). `session_token` is None for a static R2 API token and set for R2 temporary
    access credentials, which issue one.

    This config is for tenant job I/O ONLY. The models mirror (`models_mirror`) pulls shared
    weights from OUR bucket and reads its own credential straight from `os.environ`; it never
    receives this object, so a per-job tenant credential structurally cannot reach it."""
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    session_token: str | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> "R2Config":
        """Build from `R2_ENDPOINT` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` / `R2_BUCKET`;
        raise if any are missing. This is the dedicated-endpoint path, unchanged."""
        e = env if env is not None else os.environ
        missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
                   if not e.get(k)]
        if missing:
            raise RuntimeError("R2 config incomplete; missing env: " + ", ".join(missing))
        return cls(e["R2_ENDPOINT"], e["R2_ACCESS_KEY_ID"], e["R2_SECRET_ACCESS_KEY"], e["R2_BUCKET"])

    @classmethod
    def from_payload_block(cls, block: object) -> "R2Config":
        """Build from the payload's `r2` block. Raises on any malformation.

        Every message names FIELDS only, never values: this object is a credential, and the
        handler mirrors a config failure into the progress channel and stdout."""
        if not isinstance(block, dict):
            raise RuntimeError(
                f"job R2 config: {PAYLOAD_KEY!r} must be an object, got {type(block).__name__}")
        missing = [k for k in PAYLOAD_REQUIRED
                   if not (isinstance(block.get(k), str) and block[k].strip())]
        if missing:
            raise RuntimeError(
                "job R2 config incomplete; missing or blank fields: " + ", ".join(missing))
        token = block.get("session_token")
        if token is not None and not (isinstance(token, str) and token.strip()):
            raise RuntimeError(
                "job R2 config: session_token, when present, must be a non-empty string")
        return cls(
            endpoint=block["endpoint"].strip(),
            access_key_id=block["access_key_id"].strip(),
            secret_access_key=block["secret_access_key"].strip(),
            bucket=block["bucket"].strip(),
            session_token=token.strip() if isinstance(token, str) else None,
        )

    @classmethod
    def from_payload_or_env(cls, payload: dict | None, env: dict | None = None) -> "R2Config":
        """The tenant job-I/O credential for ONE job: the payload block when the job carries one,
        the endpoint environment when it does not.

        This is what lets one POOLED RunPod endpoint serve many tenants. A dedicated endpoint keeps
        working unchanged: no block in the payload means the four `R2_*` env vars, exactly as
        before.

        A PRESENT but malformed block FAILS the job; it never degrades to the environment. That is
        the load-bearing rule, not a style choice: falling back would run a tenant's job against
        OUR bucket under OUR credential, the precise failure this split exists to prevent. For the
        same reason ABSENT means the key is absent -- an explicit `"r2": null` is a producer
        defect and is refused, because the one thing a null must not do is silently select the
        shared credential."""
        if isinstance(payload, dict) and PAYLOAD_KEY in payload:
            return cls.from_payload_block(payload[PAYLOAD_KEY])
        return cls.from_env(env)

    @staticmethod
    def strip_from_payload(payload: dict) -> dict:
        """A COPY of `payload` with the credential block removed, for handing to everything
        downstream of the store.

        Nothing below the handler needs it (the store is dependency-injected), so removing it makes
        a leak structurally impossible rather than merely absent today: no future emitter, manifest,
        or error path can echo a field that is not in the dict it was given."""
        return {k: v for k, v in payload.items() if k != PAYLOAD_KEY}


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
                # None for a static R2 API token; set for R2 temporary access credentials.
                aws_session_token=self.config.session_token,
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

    def exists(self, key: str) -> bool:
        """True iff an object is actually present at `key`. Used to verify a state-claimed
        artifact really exists in R2 before trusting it: a stale/partial state tar can name a
        keyframe whose R2 object was since cleared, and reusing it ships a key to a nonexistent
        object (#108). Any head failure (404, transport, auth) returns False -- "absent" is the
        safe default here, since it triggers a (wasteful but correct) re-render rather than a
        phantom reuse."""
        from botocore.exceptions import ClientError
        try:
            self._client().head_object(Bucket=self.config.bucket, Key=key)
            return True
        except ClientError:
            return False

    def put_file(self, path: Path, key: str, *, content_type: str | None = None,
                 metadata: dict[str, str] | None = None) -> str:
        """Upload one file. `metadata`, when given, becomes generic S3 user metadata on the object.
        It carries NO submitter identity: the control plane completed the identity strip (#292), so
        the harness passes no owner tag (see handler._finish). The param is retained as a neutral
        passthrough for any future non-identity metadata need."""
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
