"""Shared test fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _fresh_model_server(monkeypatch):
    """`worker._SERVER` is process-global (warm workers reuse it). A test that builds or fakes one
    must not leak it into the next test, so every test starts cold and monkeypatch restores it."""
    from vivijure_backend import worker
    monkeypatch.setattr(worker, "_SERVER", None)
