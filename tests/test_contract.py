import io
import json
import tarfile

import pytest

from vivijure_backend.contract import (
    _safe_extract,
    Bundle,
    Cast,
    RenderRequest,
    RenderResult,
    Scene,
    Storyboard,
    Keyframe,
    Clip,
)

SAMPLE = {
    "title": "neon rain standoff",
    "full_prompt": "a tense rooftop standoff in neon rain",
    "duration_seconds": 18,
    "clip_seconds": 5,
    "style_prefix": "cyberpunk anime,",
    "use_characters": ["A", "B"],
    "scenes": [
        {"id": "shot_01", "prompt": "Vesper alone", "character_slots": ["A"], "target_seconds": 5},
        {"prompt": "Vesper and Rhode face off", "character_slots": ["A", "B"], "target_seconds": 4,
         "start_image": "clips/shot_02_keyframe.png"},
        {"prompt": "wide skyline", "character_slots": [], "act": "III"},
    ],
}


def test_scene_defaults_and_id_autogen():
    sb = Storyboard.from_dict(SAMPLE)
    assert sb.scenes[0].id == "shot_01"
    # second scene had no id -> generated from its 0-based index (1) as shot_02
    assert sb.scenes[1].id == "shot_02"
    assert sb.scenes[1].is_multi_character
    assert not sb.scenes[0].is_multi_character
    assert sb.scenes[1].start_image == "clips/shot_02_keyframe.png"
    assert sb.scenes[0].start_image is None


def test_unknown_slots_are_filtered():
    sc = Scene.from_dict({"prompt": "x", "character_slots": ["A", "Z", "B", 3]}, 0)
    assert sc.character_slots == ["A", "B"]


def test_style_fields_normalize_to_none_string():
    sb = Storyboard.from_dict({"scenes": [{"prompt": "x"}]})
    assert sb.style_category == "None"
    assert sb.style_preset == "None"
    assert sb.title == "untitled"


def test_use_characters_filtered_to_known_slots():
    sb = Storyboard.from_dict({"use_characters": ["A", "B", "Q"], "scenes": [{"prompt": "x"}]})
    assert sb.use_characters == ["A", "B"]


def test_empty_storyboard_rejected():
    with pytest.raises(ValueError):
        Storyboard.from_dict({"scenes": []})


def test_from_yaml_accepts_json_too():
    # YAML is a JSON superset; from_yaml must parse a JSON document verbatim.
    sb = Storyboard.from_yaml(json.dumps(SAMPLE))
    assert sb.title == "neon rain standoff"
    assert len(sb.scenes) == 3


def test_render_request_maps_overrides_key():
    req = RenderRequest.from_dict({
        "action": "finalize",
        "project": "neon",
        "bundle_key": "bundles/neon.tar.gz",
        "quality_tier": "standard",
        "render_overrides": {"finish_offloaded": True},
        "pretrained_loras": {"A": "loras/A.safetensors"},
        "process_shot_ids": ["shot_02"],
    })
    assert req.action == "finalize"
    assert req.overrides == {"finish_offloaded": True}  # render_overrides -> overrides
    assert req.pretrained_loras == {"A": "loras/A.safetensors"}
    assert req.process_shot_ids == ["shot_02"]


def test_render_request_defaults_and_type_guards():
    req = RenderRequest.from_dict({"bundle_key": "x"})
    assert req.action == "render"
    assert req.quality_tier == "draft"          # wallet-safe default; the studio always sends it
    assert req.overrides == {}          # missing -> {}
    assert req.process_shot_ids is None  # absent -> None, not []
    # wrong types are coerced to safe empties, never crash
    bad = RenderRequest.from_dict({"render_overrides": "nope", "process_shot_ids": "nope"})
    assert bad.overrides == {}
    assert bad.process_shot_ids is None


def test_render_request_builds_typed_config_from_tier_and_overrides():
    # The typed generation config is parsed from quality_tier + the namespaced render_overrides,
    # while the raw overrides dict is still kept for routing flags the harness reads.
    req = RenderRequest.from_dict({
        "bundle_key": "x",
        "quality_tier": "draft",
        "render_overrides": {"keyframe": {"steps": 6}, "i2v": {"num_frames": 49},
                             "finish_offloaded": True},
    })
    assert req.config.quality.value == "draft"
    assert req.config.keyframe.steps == 6           # override over the draft baseline
    assert req.config.keyframe.distill is True       # draft tier baseline
    assert req.config.i2v.num_frames == 49
    assert req.overrides.get("finish_offloaded") is True   # routing flag preserved on raw dict


def test_render_request_default_config_is_a_draft_render_config():
    # An omitted quality_tier must never silently buy the most expensive render: the default is
    # DRAFT (the wallet-safe floor). The studio always sends the tier explicitly.
    req = RenderRequest.from_dict({"bundle_key": "x"})  # quality_tier defaults to "draft"
    assert req.config.quality.value == "draft"
    assert req.config.keyframe.distill is True         # draft = few-step distilled


def test_cast_from_registry_filters_bad_slots():
    cast = Cast.from_registry({"characters": {
        "A": {"name": "Vesper", "prompt": "teal-haired"},
        "Z": {"name": "Ghost", "prompt": "ignored"},  # not a real slot
        "B": "not-a-dict",                              # malformed
    }})
    assert set(cast.characters) == {"A"}
    assert cast.characters["A"].name == "Vesper"


def test_render_result_to_dict_shape():
    res = RenderResult(
        project="neon",
        output_key="renders/neon.mp4",
        seconds=18.0,
        has_audio=True,
        keyframes=[Keyframe("shot_01", "kf/shot_01.png")],
        clips=[Clip("shot_01", "clips/shot_01.mp4", target_seconds=5.0)],
        lora={"A": {"lora_id": "loras/A"}},
        state_key="state/neon.tar.gz",
    )
    d = res.to_dict()
    assert d["output_key"] == "renders/neon.mp4"
    assert d["keyframes"] == [{"shot_id": "shot_01", "key": "kf/shot_01.png"}]
    assert d["clips"][0]["target_seconds"] == 5.0
    assert d["lora"] == {"A": {"lora_id": "loras/A"}}


def test_render_result_omits_clips_when_empty():
    assert "clips" not in RenderResult(project="x").to_dict()


# ------------------------------------------------------------------ bundle extraction

def _make_bundle(tmp_path, members: dict[str, bytes]) -> "Path":  # noqa: F821
    tar_path = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return tar_path


def test_bundle_extract_resolves_storyboard_and_refs(tmp_path):
    import yaml
    sb_yaml = yaml.safe_dump(SAMPLE).encode()
    registry = json.dumps({"characters": {
        "A": {"name": "Vesper", "prompt": "teal"},
        "B": {"name": "Rhode", "prompt": "orange"},
    }}).encode()
    tar = _make_bundle(tmp_path, {
        "storyboard.yaml": sb_yaml,
        "characters/registry.json": registry,
        "characters/refs/A/ref_01.png": b"\x89PNG-fake",
        "characters/refs/A/ref_02.jpg": b"jpeg-fake",
        "characters/refs/B/ref_01.webp": b"webp-fake",
        "characters/refs/A/notes.txt": b"ignored non-image",
    })
    bundle = Bundle.extract(tar, tmp_path / "out")
    assert bundle.storyboard.title == "neon rain standoff"
    assert set(bundle.cast.characters) == {"A", "B"}
    a_refs = [p.name for p in bundle.cast.characters["A"].ref_paths]
    assert a_refs == ["ref_01.png", "ref_02.jpg"]  # sorted, non-images dropped
    assert [p.name for p in bundle.cast.characters["B"].ref_paths] == ["ref_01.webp"]


def test_bundle_missing_storyboard_raises(tmp_path):
    tar = _make_bundle(tmp_path, {"characters/registry.json": b"{}"})
    with pytest.raises(FileNotFoundError):
        Bundle.extract(tar, tmp_path / "out")


def test_bundle_rejects_path_traversal(tmp_path):
    tar = _make_bundle(tmp_path, {"../escape.txt": b"pwned", "storyboard.yaml": b"{}"})
    with pytest.raises(ValueError):
        Bundle.extract(tar, tmp_path / "out")




def test_bundle_rejects_symlink(tmp_path):
    """Symlinks must be rejected even when the stored name is safe."""
    tar = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        # Add storyboard so extraction doesn't fail on missing storyboard
        info = tarfile.TarInfo("storyboard.yaml")
        data = b"{}"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        # Add a symlink member
        sym = tarfile.TarInfo("link.txt")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "/etc/passwd"
        tf.addfile(sym)
    with pytest.raises(ValueError, match="unsafe link"):
        from vivijure_backend.contract import _safe_extract
        import tarfile as tf_mod
        dest = tmp_path / "out"
        dest.mkdir()
        with tf_mod.open(tar, "r:gz") as tff:
            _safe_extract(tff, dest)


def test_bundle_rejects_sibling_prefix(tmp_path):
    """Path traversal via sibling-dir prefix (/tmp/proj vs /tmp/projEVIL) must be rejected."""
    tar = _make_bundle(tmp_path, {"../sibling.txt": b"escape", "storyboard.yaml": b"{}"})
    with pytest.raises(ValueError, match="unsafe path"):
        Bundle.extract(tar, tmp_path / "out")

def test_render_request_parses_audio_key():
    req = RenderRequest.from_dict(
        {"action": "render", "project": "p", "bundle_key": "b", "audio_key": "audio/x.m4a"})
    assert req.audio_key == "audio/x.m4a"
    assert RenderRequest.from_dict({"project": "p"}).audio_key is None


def test_bundle_rejects_hardlink(tmp_path):
    """Hardlinks must be rejected alongside symlinks."""
    tar = tmp_path / "bundle.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        info = tarfile.TarInfo("storyboard.yaml")
        data = b"{}"
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        lnk = tarfile.TarInfo("hardlink.txt")
        lnk.type = tarfile.LNKTYPE
        lnk.linkname = "/etc/passwd"
        tf.addfile(lnk)
    with pytest.raises(ValueError, match="unsafe link"):
        from vivijure_backend.contract import _safe_extract
        import tarfile as tf_mod
        dest = tmp_path / "out"
        dest.mkdir()
        with tf_mod.open(tar, "r:gz") as tff:
            _safe_extract(tff, dest)
