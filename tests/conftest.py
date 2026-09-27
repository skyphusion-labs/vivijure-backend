"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _fresh_model_server(monkeypatch):
    """`worker._SERVER` is process-global (warm workers reuse it). A test that builds or fakes one
    must not leak it into the next test, so every test starts cold and monkeypatch restores it."""
    from vivijure_backend import worker
    monkeypatch.setattr(worker, "_SERVER", None)


@pytest.fixture(autouse=True)
def _fresh_volume_repoint_state(monkeypatch):
    """`models_mirror._PRISTINE_ROOTS` is process-global: it records the roots the ENDPOINT
    configured, captured once before any volume repoint, so a later job can be handed them back
    (backend#453). One process is one endpoint in production; in a test run it is one pytest
    process for the whole suite, so it must not leak from one test's fake environment into the
    next."""
    from vivijure_backend.harness import models_mirror
    monkeypatch.setattr(models_mirror, "_PRISTINE_ROOTS", None)
