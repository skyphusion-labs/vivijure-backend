"""CPU tests for the per-job tenant R2 configuration (the pooled-endpoint credential split).

The backend uses R2 for two unrelated purposes under what used to be one credential, and only one
of them is per-tenant:

  1. the models mirror  -- shared weights, OUR bucket, identical bytes for every tenant;
  2. tenant job I/O     -- bundle in, film out, LoRAs, keyframes, clips, progress, THE tenant's
                           bucket.

These tests hold that split. The load-bearing one is
`test_malformed_block_fails_and_never_degrades_to_env`: a malformed block must FAIL the job, never
fall back to the environment, because falling back would run a tenant's job against our bucket
under our credential.
"""
from pathlib import Path

import pytest

from vivijure_backend.harness import models_mirror
from vivijure_backend.harness import r2 as r2_mod
from vivijure_backend.harness.r2 import (
    R2,
    R2Config,
    configured_r2_endpoint_hosts,
    validate_job_r2_endpoint,
)

# A complete, VALID environment, present in every test below. Malformed-block tests assert a
# refusal WHILE this is available: a negative test against an environment that could not have
# worked anyway would pass for the wrong reason.
ENV = {
    "R2_ENDPOINT": "https://env.r2.cloudflarestorage.com",
    "R2_ACCESS_KEY_ID": "env-key-id",
    "R2_SECRET_ACCESS_KEY": "env-secret-value",
    "R2_BUCKET": "vivijure",
}

TENANT_SECRET = "tenant-secret-value"
BLOCK = {
    "endpoint": "https://tenant.r2.cloudflarestorage.com",
    "access_key_id": "tenant-key-id",
    "secret_access_key": TENANT_SECRET,
    "bucket": "tenant-bucket",
}


# ------------------------------------------------------------------ fallback (dedicated endpoint)

def test_absent_block_falls_back_to_env():
    """No block in the payload is the dedicated-endpoint path: the four R2_* env vars, exactly as
    before this change. Backward compatibility is load-bearing -- prod runs dedicated endpoints
    today and must keep working unchanged."""
    cfg = R2Config.from_payload_or_env({"project": "p", "action": "render"}, ENV)
    assert cfg.bucket == "vivijure"
    assert cfg.access_key_id == "env-key-id"
    assert cfg.session_token is None


def test_absent_block_with_incomplete_env_still_raises_the_env_error():
    """The fallback path keeps its own failure mode; it is not swallowed by the new branch."""
    with pytest.raises(RuntimeError, match="R2 config incomplete; missing env"):
        R2Config.from_payload_or_env({"project": "p"}, {"R2_ENDPOINT": "https://x"})


# ------------------------------------------------------------------- preference (pooled endpoint)

def test_present_block_wins_over_a_fully_valid_env():
    """The POSITIVE control for the preference: the env here is complete and usable, so a test
    that merely 'succeeded' would prove nothing. Every field must come from the block."""
    cfg = R2Config.from_payload_or_env({"project": "p", "r2": BLOCK}, ENV)
    assert cfg.endpoint == "https://tenant.r2.cloudflarestorage.com"
    assert cfg.access_key_id == "tenant-key-id"
    assert cfg.secret_access_key == TENANT_SECRET
    assert cfg.bucket == "tenant-bucket"
    # Nothing from the environment leaked into the tenant config.
    assert cfg.bucket != ENV["R2_BUCKET"]
    assert cfg.access_key_id != ENV["R2_ACCESS_KEY_ID"]


def test_block_fields_are_stripped_of_surrounding_whitespace():
    cfg = R2Config.from_payload_or_env({"r2": {**BLOCK, "bucket": "  tenant-bucket  "}}, ENV)
    assert cfg.bucket == "tenant-bucket"


# ---------------------------------------------------------- job-supplied endpoint allowlist

def test_job_endpoint_accepts_cloudflare_r2_https_hosts():
    assert validate_job_r2_endpoint("https://tenant.r2.cloudflarestorage.com") == (
        "https://tenant.r2.cloudflarestorage.com")
    assert validate_job_r2_endpoint(
        "https://abc123.eu.r2.cloudflarestorage.com") == "https://abc123.eu.r2.cloudflarestorage.com"
    assert validate_job_r2_endpoint("https://tenant.r2.cloudflarestorage.com/") == (
        "https://tenant.r2.cloudflarestorage.com/")


@pytest.mark.parametrize("endpoint, why", [
    ("http://tenant.r2.cloudflarestorage.com", "http"),
    ("https://169.254.169.254/", "link-local metadata ip"),
    ("https://127.0.0.1/", "loopback ip"),
    ("https://[::1]/", "loopback ipv6"),
    ("file:///etc/passwd", "file scheme"),
    ("https://metadata.google.internal/", "gcp metadata host"),
    ("https://metadata/", "short metadata host"),
    ("https://evil.example/", "unrelated host"),
    ("https://tenant.r2.cloudflarestorage.com.evil.example/", "suffix spoof"),
    ("ftp://tenant.r2.cloudflarestorage.com", "ftp"),
    ("//tenant.r2.cloudflarestorage.com", "scheme-relative"),
    ("https://user:pass@tenant.r2.cloudflarestorage.com", "userinfo"),
    ("https://tenant.r2.cloudflarestorage.com:8443", "non-443 port"),
    ("https://tenant.r2.cloudflarestorage.com/bucket", "path"),
    ("https://tenant.r2.cloudflarestorage.com?x=1", "query"),
])
def test_job_endpoint_rejects_non_r2_and_ssrf_shapes(endpoint, why):
    with pytest.raises(RuntimeError, match="job R2 config: endpoint"):
        validate_job_r2_endpoint(endpoint)


def test_job_endpoint_configured_account_pattern_is_accepted():
    extras = configured_r2_endpoint_hosts({"CLOUDFLARE_ACCOUNT_ID": "acct01deadbeef"})
    assert "acct01deadbeef.r2.cloudflarestorage.com" in extras
    assert validate_job_r2_endpoint(
        "https://acct01deadbeef.r2.cloudflarestorage.com", extra_hosts=extras)


def test_job_endpoint_allowed_hosts_env_is_the_operator_override(monkeypatch):
    monkeypatch.setenv("R2_ALLOWED_ENDPOINT_HOSTS", "r2.example.test, metadata.google.internal")
    extras = configured_r2_endpoint_hosts()
    assert "r2.example.test" in extras
    assert "metadata.google.internal" not in extras, "metadata hosts cannot be opted in"
    assert validate_job_r2_endpoint(
        "https://r2.example.test", extra_hosts=extras) == "https://r2.example.test"


def test_job_endpoint_ip_cannot_be_opted_in():
    extras = configured_r2_endpoint_hosts({"R2_ALLOWED_ENDPOINT_HOSTS": "169.254.169.254"})
    assert extras == frozenset()
    with pytest.raises(RuntimeError, match="job R2 config: endpoint"):
        validate_job_r2_endpoint("https://169.254.169.254/", extra_hosts=extras)


def test_payload_block_refuses_a_non_r2_endpoint_and_never_degrades_to_env():
    with pytest.raises(RuntimeError, match="job R2 config: endpoint"):
        R2Config.from_payload_or_env(
            {"r2": {**BLOCK, "endpoint": "http://169.254.169.254/latest/meta-data/"}}, ENV)


def test_env_path_is_not_gated_by_the_job_allowlist():
    """Dedicated-endpoint R2_ENDPOINT is operator-set. The allowlist is for a job-supplied
    endpoint only; applying it here would break the from_env path that still accepts the
    historical r2.dev shape used in tests and some operator files."""
    cfg = R2Config.from_env({
        "R2_ENDPOINT": "https://x.r2.dev",
        "R2_ACCESS_KEY_ID": "k",
        "R2_SECRET_ACCESS_KEY": "s",
        "R2_BUCKET": "vivijure",
    })
    assert cfg.endpoint == "https://x.r2.dev"


# ------------------------------------------------------------- refusal (the load-bearing property)

@pytest.mark.parametrize("block, why", [
    ({}, "empty object"),
    ({k: v for k, v in BLOCK.items() if k != "bucket"}, "missing bucket"),
    ({k: v for k, v in BLOCK.items() if k != "secret_access_key"}, "missing secret"),
    ({k: v for k, v in BLOCK.items() if k != "endpoint"}, "missing endpoint"),
    ({k: v for k, v in BLOCK.items() if k != "access_key_id"}, "missing access key id"),
    ({**BLOCK, "bucket": ""}, "blank bucket"),
    ({**BLOCK, "bucket": "   "}, "whitespace-only bucket"),
    ({**BLOCK, "access_key_id": None}, "null field"),
    ({**BLOCK, "bucket": 7}, "non-string field"),
    ("not-an-object", "string instead of an object"),
    ([BLOCK], "list instead of an object"),
    (None, "explicit null block"),
])
def test_malformed_block_fails_and_never_degrades_to_env(block, why):
    """A PRESENT but malformed block must fail the job. It must NOT fall back to the environment.

    This is the whole safety property of the split: a silent fallback would put a tenant's film in
    OUR bucket under OUR credential, which is the precise failure the per-job credential exists to
    prevent. The valid ENV is passed in on purpose -- the refusal has to hold when the fallback
    would have worked.

    `None` is in this list deliberately. An explicit `"r2": null` is a producer defect, and the one
    thing a null must not do is quietly select the shared credential."""
    with pytest.raises(RuntimeError) as exc:
        R2Config.from_payload_or_env({"project": "p", "r2": block}, ENV)
    # Refused for being malformed, not by silently building the env config and failing later.
    assert "job R2 config" in str(exc.value), why


def test_a_valid_block_is_accepted_by_the_same_call_the_malformed_ones_hit():
    """Control for the parametrized refusals above: the code path they exercise CAN succeed, so a
    raise there is a real refusal and not a function that rejects everything."""
    cfg = R2Config.from_payload_or_env({"r2": dict(BLOCK)}, ENV)
    assert cfg.bucket == "tenant-bucket"


# --------------------------------------------------------------------------- secret hygiene

@pytest.mark.parametrize("block", [
    {**BLOCK, "bucket": ""},
    {**BLOCK, "endpoint": None},
    {**BLOCK, "session_token": ""},
])
def test_refusal_messages_name_fields_never_values(block):
    """The handler mirrors a config failure into the R2 progress channel and stdout, so the
    message is a published surface. It may name fields; it may never carry a credential."""
    with pytest.raises(RuntimeError) as exc:
        R2Config.from_payload_or_env({"r2": block}, ENV)
    msg = str(exc.value)
    for secret in (TENANT_SECRET, BLOCK["access_key_id"], ENV["R2_SECRET_ACCESS_KEY"]):
        assert secret not in msg, f"a credential value reached the error message: {msg!r}"


# --------------------------------------------------------------------------- session token

def test_session_token_is_optional_and_defaults_to_none():
    assert R2Config.from_payload_or_env({"r2": dict(BLOCK)}, ENV).session_token is None


def test_session_token_is_carried_when_present():
    """R2 temporary access credentials issue a session token; a static R2 API token does not."""
    cfg = R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": "tok"}}, ENV)
    assert cfg.session_token == "tok"


@pytest.mark.parametrize("token", ["", "   ", 7, []])
def test_blank_or_non_string_session_token_is_refused(token):
    with pytest.raises(RuntimeError, match="session_token"):
        R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": token}}, ENV)


def test_session_token_reaches_the_boto3_client(monkeypatch):
    """A carried token that never reaches boto3 would authenticate as a plain key pair and fail at
    the first call, so assert it lands in the client kwargs. boto3 is a GPU-image dependency and is
    absent from the test environment, so it is faked here rather than skipped -- a skip would make
    this a check that cannot fail."""
    import sys
    import types

    seen: dict = {}

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **kw: seen.update(kw) or object()
    fake_botocore = types.ModuleType("botocore")
    fake_config_mod = types.ModuleType("botocore.config")
    fake_config_mod.Config = lambda **kw: kw
    fake_botocore.config = fake_config_mod
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", fake_config_mod)

    R2(R2Config.from_payload_or_env({"r2": {**BLOCK, "session_token": "tok"}}, ENV))._client()
    assert seen["aws_session_token"] == "tok"
    assert seen["aws_access_key_id"] == "tenant-key-id"

    seen.clear()
    R2(R2Config.from_env(ENV))._client()
    assert seen["aws_session_token"] is None, "the env path must pass no session token"


# --------------------------------------------------------------------------- payload stripping

def test_strip_removes_only_the_credential_block():
    payload = {"project": "p", "action": "render", "bundle_key": "bundles/p.tar.gz", "r2": BLOCK}
    stripped = R2Config.strip_from_payload(payload)
    assert "r2" not in stripped
    assert stripped == {"project": "p", "action": "render", "bundle_key": "bundles/p.tar.gz"}
    # The caller's dict is untouched (the handler reads `payload` again for `action`).
    assert "r2" in payload


def test_strip_is_a_noop_when_there_is_no_block():
    payload = {"project": "p"}
    assert R2Config.strip_from_payload(payload) == payload


# ------------------------------------------------- the handler boundary (the real effect test)

def _stub_handler_deps(monkeypatch):
    """Neutralize everything in `handler` below the store so the test exercises the credential
    boundary and nothing else. Returns the module plus the list the stubbed run_job records into."""
    from vivijure_backend.harness import handler as handler_mod
    from vivijure_backend.harness import job_done_diag, pipeline_registry

    seen_payloads: list[dict] = []

    monkeypatch.setattr(models_mirror, "ensure_models", lambda *a, **k: True)
    monkeypatch.setattr(models_mirror, "start_i2v_prefetch", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_registry, "get_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(job_done_diag, "register", lambda *a, **k: None)
    monkeypatch.setattr(handler_mod, "_runpod_progress_hook", lambda job: None)

    def fake_run_job(payload, **kwargs):
        seen_payloads.append(payload)
        return {"ok": True}

    monkeypatch.setattr(handler_mod, "run_job", fake_run_job)
    return handler_mod, seen_payloads


def test_handler_uses_the_payload_credential_and_strips_it_before_the_pipeline(monkeypatch):
    """The end-to-end boundary property: the tenant credential builds the store and then does not
    exist as far as anything downstream is concerned. Stripping it makes a leak structurally
    impossible rather than merely absent today -- no future emitter, manifest, or error path can
    echo a field that is not in the dict it was handed."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    monkeypatch.setattr(R2Config, "from_env",
                        classmethod(lambda cls, env=None: R2Config("no", "no", "no", "no")))

    captured: list[R2Config] = []
    real_init = R2.__init__

    def spy_init(self, config):
        captured.append(config)
        real_init(self, config)

    monkeypatch.setattr(R2, "__init__", spy_init)

    out = handler_mod.handler({"id": "job-1", "input": {
        "project": "p", "action": "render", "bundle_key": "bundles/p.tar.gz", "r2": dict(BLOCK),
    }})

    assert out == {"ok": True}
    assert captured[0].bucket == "tenant-bucket", "the store was not built from the payload block"
    assert len(seen) == 1
    assert "r2" not in seen[0], "the credential block reached the pipeline payload"
    assert seen[0]["project"] == "p", "stripping removed more than the credential block"


def test_handler_refuses_a_malformed_block_rather_than_running_on_our_bucket(monkeypatch):
    """The refusal at the real entry point, not just at the config helper: a malformed block must
    stop the job before any store exists, with a valid environment sitting right there."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    monkeypatch.setattr(R2Config, "from_env", classmethod(lambda cls, env=None: R2Config(
        ENV["R2_ENDPOINT"], ENV["R2_ACCESS_KEY_ID"], ENV["R2_SECRET_ACCESS_KEY"], ENV["R2_BUCKET"])))

    with pytest.raises(RuntimeError, match="job R2 config"):
        handler_mod.handler({"id": "job-1", "input": {
            "project": "p", "action": "render", "r2": {"bucket": "tenant-bucket"},
        }})
    assert seen == [], "a job ran despite a malformed credential block"


# ------------------------------------------------- the models mirror stays on OUR credential

def test_models_mirror_reads_its_credential_from_the_environment():
    child = models_mirror.rclone_env(ENV)
    assert child["RCLONE_CONFIG_R2_ACCESS_KEY_ID"] == ENV["R2_ACCESS_KEY_ID"]
    assert child["RCLONE_CONFIG_R2_ENDPOINT"] == ENV["R2_ENDPOINT"]


def test_models_mirror_has_no_code_path_to_the_per_job_credential():
    """The mirror must not be able to REACH a tenant credential, however the handler changes.

    Checked on the parsed AST, not on the source text. The first version of this test matched raw
    tokens and went red the moment `models_mirror` gained a COMMENT naming
    `R2Config.from_payload_or_env` as a cross-reference. A doc reference is not a code path, and a
    check that cannot tell the two apart either blocks correct documentation or, worse, gets relaxed
    until it stops testing anything. The AST does not see comments or docstrings at all."""
    import ast

    def code_references(source: str) -> set:
        """Every name actually referenced in code: imports, attributes, and bare names."""
        found = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                found.add(node.module or "")
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.Attribute):
                found.add(node.attr)
            elif isinstance(node, ast.Name):
                found.add(node.id)
        return found

    # CONTROL: the instrument must actually detect what it is looking for. Without this the
    # assertion below would pass just as happily against a broken parser or a typo'd name.
    control = code_references(
        "from .r2 import R2Config\ncfg = R2Config.from_payload_or_env(payload)\n")
    assert {"R2Config", "from_payload_or_env", "r2"} <= control, (
        f"CONTROL FAILED: the AST scan missed references it must catch: {control}")

    mirror_refs = code_references(Path(models_mirror.__file__).read_text())
    for token in ("R2Config", "from_payload_block", "from_payload_or_env"):
        assert token not in mirror_refs, (
            f"models_mirror references {token!r} in code: the shared-weights pull must never see "
            f"a per-job tenant credential")
    assert "r2" not in mirror_refs, "models_mirror imports the tenant job-I/O module"

    # And the dependency runs one way only: r2 reads the marker from the mirror, never the reverse.
    r2_refs = code_references(Path(r2_mod.__file__).read_text())
    assert "uses_namespaced_mirror_creds" in r2_refs


# ============================================================ the mirror credential is its OWN
#
# The section above proves the mirror has no code path to the PAYLOAD credential. That is true and
# it is not the whole claim: until the mirror had its own environment names, the mirror and tenant
# job I/O read the SAME four variables, so the code paths were separate while the credential and the
# bucket were one. These tests hold the credential separation itself.

MIRROR_ENV_FULL = {
    "MODELS_R2_ENDPOINT": "https://models.r2.cloudflarestorage.com",
    "MODELS_R2_ACCESS_KEY_ID": "models-key-id",
    "MODELS_R2_SECRET_ACCESS_KEY": "models-secret-value",
    "MODELS_R2_BUCKET": "vivijure",
}


def test_mirror_prefers_its_own_namespaced_credential():
    resolved = models_mirror.mirror_env({**ENV, **MIRROR_ENV_FULL})
    assert resolved["R2_ACCESS_KEY_ID"] == "models-key-id"
    assert resolved["R2_ENDPOINT"] == "https://models.r2.cloudflarestorage.com"
    assert resolved["R2_ACCESS_KEY_ID"] != ENV["R2_ACCESS_KEY_ID"]


def test_mirror_bucket_comes_from_its_own_name_not_the_tenant_bucket():
    """The latent defect this namespacing fixes. `templateEnv` (control plane, src/runpod.ts) sets
    `R2_BUCKET` to the TENANT's bucket, so before the split a cold non-baked worker on a provisioned
    tenant endpoint would mirror weights from `r2:<tenant-bucket>/models/...`, a prefix that does not
    exist there. It was masked by the baked-image and volume sentinels short-circuiting before the
    pull, not by being correct."""
    resolved = models_mirror.mirror_env({**MIRROR_ENV_FULL, "R2_BUCKET": "some-tenant-bucket"})
    assert resolved["R2_BUCKET"] == "vivijure"


def test_mirror_falls_back_to_the_legacy_names_for_un_repinned_endpoints():
    """Transitional, and deliberately preserves today's behaviour rather than silently repointing a
    live endpoint's weights pull. Its removal is tracked and gating; see the PR."""
    resolved = models_mirror.mirror_env(dict(ENV))
    assert resolved["R2_ACCESS_KEY_ID"] == ENV["R2_ACCESS_KEY_ID"]
    assert resolved["R2_BUCKET"] == ENV["R2_BUCKET"]


def test_mirror_and_job_io_resolve_to_DIFFERENT_credentials_from_one_environment():
    """The claim the code-path test does not make: one environment plus one job payload yields two
    credentials against two buckets, neither borrowing from the other."""
    env = {**ENV, **MIRROR_ENV_FULL}
    job = R2Config.from_payload_or_env({"r2": dict(BLOCK)}, env)
    mirror = models_mirror.mirror_env(env)

    assert job.bucket == "tenant-bucket"
    assert mirror["R2_BUCKET"] == "vivijure"
    assert job.access_key_id != mirror["R2_ACCESS_KEY_ID"]
    assert job.secret_access_key != mirror["R2_SECRET_ACCESS_KEY"]
    # And neither is the legacy shared credential that used to serve both.
    assert job.access_key_id != ENV["R2_ACCESS_KEY_ID"]
    assert mirror["R2_ACCESS_KEY_ID"] != ENV["R2_ACCESS_KEY_ID"]


# ==================================================== an endpoint that may serve more than one tenant

def test_absent_block_is_REFUSED_on_an_endpoint_carrying_the_mirror_marker():
    """The hole the payload block alone does not close.

    On a POOLED endpoint an absent block must not fall back to the environment: the render would
    succeed into whatever bucket the shared template names, the endpoint would report healthy, and
    isolation would be gone with nothing failing.

    The legacy `R2_*` set is COMPLETE and usable in this env on purpose. A pooled template is
    supposed to carry none, which would make `from_env` raise by itself, but that is a convention a
    maintainer can undo by adding the names back. The refusal has to hold when the fallback would
    have worked."""
    env = {**ENV, **MIRROR_ENV_FULL}
    with pytest.raises(RuntimeError, match="job R2 config required"):
        R2Config.from_payload_or_env({"project": "p", "action": "render"}, env)


def test_the_same_env_accepts_a_job_that_carries_its_block():
    """Positive control for the refusal above: the endpoint is not simply broken."""
    env = {**ENV, **MIRROR_ENV_FULL}
    assert R2Config.from_payload_or_env({"r2": dict(BLOCK)}, env).bucket == "tenant-bucket"


def test_a_legacy_endpoint_still_falls_back(monkeypatch):
    """Backward compatibility, stated as a test rather than an intention: an endpoint provisioned
    before the split has no marker, so the environment fallback stays live and prod keeps working
    through this change unchanged."""
    assert not models_mirror.uses_namespaced_mirror_creds(ENV)
    assert R2Config.from_payload_or_env({"project": "p"}, ENV).bucket == "vivijure"


def test_the_two_refusals_are_distinguishable():
    """A malformed block on a pooled endpoint must report the MALFORMATION, not the missing block.
    One error string for two different operator mistakes would send the reader to the wrong repo."""
    env = {**ENV, **MIRROR_ENV_FULL}
    with pytest.raises(RuntimeError) as exc:
        R2Config.from_payload_or_env({"r2": {"bucket": "tenant-bucket"}}, env)
    msg = str(exc.value)
    assert "job R2 config incomplete" in msg
    assert "job R2 config required" not in msg


def test_handler_refuses_an_absent_block_on_a_pooled_endpoint(monkeypatch):
    """The refusal at the real entry point, with the marker in the process environment where a
    pooled worker would actually find it."""
    handler_mod, seen = _stub_handler_deps(monkeypatch)
    for k, v in MIRROR_ENV_FULL.items():
        monkeypatch.setenv(k, v)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)

    with pytest.raises(RuntimeError, match="job R2 config required"):
        handler_mod.handler({"id": "job-1", "input": {
            "project": "p", "action": "render", "bundle_key": "bundles/p.tar.gz",
        }})
    assert seen == [], "a job ran on a pooled endpoint without a tenant credential"
