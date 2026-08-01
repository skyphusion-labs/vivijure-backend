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
from vivijure_backend.harness.r2 import R2, R2Config

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
    """The split is STRUCTURAL, not policy: `models_mirror` builds its own rclone environment from
    `os.environ` and is never handed the store or the payload, so a tenant credential cannot reach
    the shared-weights pull however the handler changes.

    A grep that finds nothing proves nothing on its own, so each token is first asserted PRESENT in
    `r2.py`, where it genuinely lives. Without that control this test would pass against a typo."""
    mirror_src = Path(models_mirror.__file__).read_text()
    r2_src = Path(r2_mod.__file__).read_text()
    for token in ("PAYLOAD_KEY", "from_payload_block", "from_payload_or_env", "R2Config"):
        assert token in r2_src, f"CONTROL FAILED: {token!r} absent from r2.py, so the check below is vacuous"
        assert token not in mirror_src, (
            f"models_mirror references {token!r}: the shared-weights pull must never see a "
            f"per-job tenant credential")
