"""vivijure-backend#559: a warm worker serves a job whose keyframe base differs from the one it loaded.

`style` picks the keyframe base per job (anime: animagine-xl-4.0, photoreal: RealVisXL_V5.0), but the
process-global ModelServer froze the first job's base and `_check_model_divergence` refused the other
style for the life of the worker. A cold anime job that failed on missing weights also left the
worker refusing every photoreal job after it. Now a keyframe base on local disk swaps in on the warm
server (the two SDXL pipes are dropped and reload on the next draw), a base not on disk fails with the
missing-weights error and changes nothing (anime is not on the GPU image, #557), and i2v still refuses.

`ModelServer` is built for real (its constructor touches no GPU); only its cache is seeded with
stand-in objects, so the test reads which entries a swap drops."""
import pytest

import vivijure_backend.harness.models_mirror as mirror

from vivijure_backend import worker
from vivijure_backend.contract import RenderRequest
from vivijure_backend.models import DEFAULT_SPECS, KEYFRAME_ANIME_REPO, InvalidModelRepoId, ModelRole

PHOTOREAL = DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id
LOADED = ("keyframe", "instantid", "controlnet_pose", "controlnet_canny", "face_analyzer", "i2v")


@pytest.fixture(autouse=True)
def on_disk(monkeypatch):
    """Which keyframe bases this worker's HF cache holds. Default: both, so a swap can happen."""
    held = {PHOTOREAL, KEYFRAME_ANIME_REPO}
    monkeypatch.setattr(mirror, "repo_in_hf_cache", lambda repo_id, env=None: repo_id in held)
    return held


def _req(**keyframe):
    return RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                    "render_overrides": {"keyframe": keyframe}})


def _warm(**keyframe):
    """A warm worker: built by a first job, with every model it can hold already cached."""
    worker.build_pipeline(_req(**keyframe))
    for key in LOADED:
        worker._SERVER._cache[key] = object()
    return worker._SERVER


def _events(capsys, name):
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(f"@event {name} ")]


def test_the_two_styles_resolve_to_different_bases():
    """The premise: style is a per-job choice that changes KEYFRAME_BASE."""
    assert _req(style="anime").config.keyframe.base_model == KEYFRAME_ANIME_REPO
    assert _req(style="photoreal").config.keyframe.base_model == PHOTOREAL
    assert KEYFRAME_ANIME_REPO != PHOTOREAL


@pytest.mark.parametrize("first,then", [("photoreal", "anime"), ("anime", "photoreal")])
def test_a_warm_worker_serves_the_other_style(first, then, capsys):
    server = _warm(style=first)
    capsys.readouterr()
    pipe = worker.build_pipeline(_req(style=then))
    assert pipe.server is server  # the same warm server, not a new one
    assert server.specs[ModelRole.KEYFRAME_BASE].repo_id == _req(style=then).config.keyframe.base_model
    # Only the pipes built from the keyframe base are dropped; everything else stays loaded.
    assert "keyframe" not in server._cache and "instantid" not in server._cache
    assert set(server._cache) == set(LOADED) - {"keyframe", "instantid"}
    [ev] = _events(capsys, "keyframe_model_swap")
    assert "KEYFRAME_BASE" in ev and KEYFRAME_ANIME_REPO in ev and PHOTOREAL in ev


def test_the_same_base_drops_nothing_and_says_nothing(capsys):
    server = _warm(style="photoreal")
    capsys.readouterr()
    worker.build_pipeline(_req(style="photoreal"))
    assert set(server._cache) == set(LOADED)
    assert _events(capsys, "keyframe_model_swap") == []


def test_the_swap_keeps_the_distill_weight_name():
    """dataclasses.replace swaps only the repo id; dropping weight_name breaks the distill LoRA load."""
    server = _warm(style="photoreal")
    worker.build_pipeline(_req(style="anime"))
    assert (server.specs[ModelRole.KEYFRAME_FEWSTEP].weight_name
            == DEFAULT_SPECS[ModelRole.KEYFRAME_FEWSTEP].weight_name)
    assert server.specs[ModelRole.KEYFRAME_BASE].family == DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].family


def test_a_disallowed_base_is_rejected_before_the_warm_server_changes():
    server = _warm(style="photoreal")
    with pytest.raises(InvalidModelRepoId):
        worker.build_pipeline(_req(base_model="evil-org/malware"))
    assert server.specs[ModelRole.KEYFRAME_BASE].repo_id == PHOTOREAL
    assert set(server._cache) == set(LOADED)


def test_an_i2v_divergence_still_refuses():
    server = _warm(style="photoreal")
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "bundles/p.tar.gz",
                                   "render_overrides": {"i2v": {"model": "Wan-AI/custom-i2v"}}})
    with pytest.raises(worker.ModelDivergenceError, match="Wan-AI/custom-i2v"):
        worker.build_pipeline(req)
    assert "i2v" in server._cache


def test_the_swap_takes_only_keyframe_roles():
    server = _warm(style="photoreal")
    with pytest.raises(ValueError, match="I2V"):
        server.swap_keyframe_specs({ModelRole.I2V: DEFAULT_SPECS[ModelRole.I2V]})


def test_a_base_not_on_disk_fails_honestly_and_keeps_the_warm_pipes(on_disk, capsys):
    """The anime-on-GPU case after #557: the job fails with the missing-weights error, and the
    photoreal pipes stay loaded, so the next photoreal job pays no reload."""
    on_disk.discard(KEYFRAME_ANIME_REPO)
    server = _warm(style="photoreal")
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="is not on this worker"):
        worker.build_pipeline(_req(style="anime"))
    assert server.specs[ModelRole.KEYFRAME_BASE].repo_id == PHOTOREAL
    assert set(server._cache) == set(LOADED)
    assert _events(capsys, "keyframe_model_swap") == []
    worker.build_pipeline(_req(style="photoreal"))  # served warm, nothing dropped
    assert set(server._cache) == set(LOADED)


def test_a_worker_left_on_a_missing_base_recovers(on_disk):
    """The poison case: a cold anime job built the server with the anime spec, then failed on load.
    Before #559 that worker refused every photoreal job after it."""
    on_disk.discard(KEYFRAME_ANIME_REPO)
    server = _warm(style="anime")  # cold start takes the job's spec; the load would fail later
    assert server.specs[ModelRole.KEYFRAME_BASE].repo_id == KEYFRAME_ANIME_REPO
    worker.build_pipeline(_req(style="photoreal"))
    assert server.specs[ModelRole.KEYFRAME_BASE].repo_id == PHOTOREAL
