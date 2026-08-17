"""The contract between the Vivijure control plane and this render backend.

Everything the backend reads (the storyboard, the cast, the render-job request) and
everything it returns (keyframes, clips, the final video, trained-LoRA ids, state) is
typed here. These shapes mirror the control plane's own storyboard schema and render-job
API; they are a data contract, not borrowed implementation.

Parsing is deliberately forgiving: unknown keys are ignored, and only the fields the
renderer actually consumes are surfaced, so the control plane can add authored fields
without breaking an older backend.
"""
from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import RenderConfig

# Character slots. The control plane allocates A..D; the renderer treats them as opaque
# identifiers for a region / LoRA / ref directory.
SLOT_IDS = ("A", "B", "C", "D")


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _str(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _bool(v: Any, default: bool = False) -> bool:
    """Coerce a request flag to bool. Accepts a real bool, a number (nonzero => True), or the
    JSON-ish strings "true"/"1"/"yes"/"on" (case-insensitive); everything else falls back to
    `default`. Used for the small set of routing flags a caller may send loosely typed."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return default


def validate_bundle_relpath(rel: str, *, what: str) -> str:
    """A job-authored path that must stay inside the extracted bundle.

    Syntax gate only: rejects absolute paths (posix, Windows drive, UNC), URI schemes,
    backslashes, NUL, and any `..` component. `resolve_inside` is the filesystem follow-up
    that confirms the realpath is still under the bundle root.
    """
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"{what} must be a non-empty relative path inside the bundle")
    p = rel.strip()
    if "\x00" in p:
        raise ValueError(f"{what} must not contain a NUL byte")
    if "\\" in p:
        raise ValueError(f"{what} must use forward slashes and stay inside the bundle")
    if "://" in p or p.startswith("file:"):
        raise ValueError(f"{what} must be a relative path, not a URI")
    if p.startswith("/"):
        raise ValueError(f"{what} must be a relative path inside the bundle, not absolute")
    if len(p) >= 2 and p[1] == ":":
        raise ValueError(f"{what} must be a relative path inside the bundle, not absolute")
    if ".." in p.split("/"):
        raise ValueError(f"{what} must not contain '..' components")
    parts = [part for part in p.split("/") if part not in ("", ".")]
    if not parts:
        raise ValueError(f"{what} must be a non-empty relative path inside the bundle")
    return p


def _optional_bundle_relpath(d: dict[str, Any], key: str) -> str | None:
    """Parse an optional storyboard path field; blank/absent is None, anything else is gated."""
    if key not in d:
        return None
    raw = _str(d[key]) or None
    return validate_bundle_relpath(raw, what=key) if raw else None


def resolve_inside(root: Path, rel: str, *, what: str) -> Path:
    """Join `rel` onto `root`, resolve, and refuse anything that leaves `root`.

    `validate_bundle_relpath` is the syntax gate. This is the filesystem gate: a symlink or
    remaining relative trick that resolves outside the bundle is rejected even when the
    authored string looked safe.
    """
    rel = validate_bundle_relpath(rel, what=what)
    base = Path(root).resolve()
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{what} resolves outside the job bundle")
    return target


# --------------------------------------------------------------------------- storyboard

@dataclass
class Scene:
    """One shot. Only `prompt` is required; the rest are authored hints."""
    prompt: str
    id: str | None = None
    character_slots: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    target_seconds: float | None = None
    act: str | None = None
    start_image: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], index: int) -> "Scene":
        slots = [s for s in (d.get("character_slots") or []) if s in SLOT_IDS]
        start = _num(d.get("start"))
        end = _num(d.get("end"))
        target_seconds = _num(d.get("target_seconds"))
        if target_seconds is None and start is not None and end is not None and end > start:
            target_seconds = end - start
        return cls(
            prompt=_str(d.get("prompt")),
            id=_str(d.get("id")) or f"shot_{index + 1:02d}",
            character_slots=slots,
            start=start,
            end=end,
            target_seconds=target_seconds,
            act=(_str(d["act"]) or None) if "act" in d else None,
            start_image=_optional_bundle_relpath(d, "start_image"),
        )

    @property
    def is_multi_character(self) -> bool:
        return len(self.character_slots) >= 2


@dataclass
class Storyboard:
    """The validated storyboard.json. `title` and `scenes` are the load-bearing fields;
    the style block is applied uniformly to every scene's prompt."""
    title: str
    scenes: list[Scene]
    full_prompt: str = ""
    duration_seconds: float | None = None
    clip_seconds: float | None = None
    style_prefix: str = ""
    style_category: str = "None"
    style_preset: str = "None"
    use_characters: list[str] = field(default_factory=list)
    cast_rules: str = ""
    refs_dir: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Storyboard":
        scenes = [Scene.from_dict(s, i) for i, s in enumerate(d.get("scenes") or []) if isinstance(s, dict)]
        if not scenes:
            raise ValueError("storyboard.json has no scenes")
        seen: dict[str, int] = {}
        for scene in scenes:
            if scene.id in seen:
                seen[scene.id] += 1
                scene.id = f"{scene.id}_{seen[scene.id]}"
            else:
                seen[scene.id] = 1
        use_chars = [s for s in (d.get("use_characters") or []) if s in SLOT_IDS]
        # style_category / style_preset normalize to the literal "None" when absent, so a
        # downstream "is it None?" check tests the string, never a null.
        cat = _str(d.get("style_category")).strip() or "None"
        preset = _str(d.get("style_preset")).strip() or "None"
        return cls(
            title=_str(d.get("title"), "untitled"),
            scenes=scenes,
            full_prompt=_str(d.get("full_prompt")),
            duration_seconds=_num(d.get("duration_seconds")),
            clip_seconds=_num(d.get("clip_seconds")),
            style_prefix=_str(d.get("style_prefix")),
            style_category=cat,
            style_preset=preset,
            use_characters=use_chars,
            cast_rules=_str(d.get("cast_rules")),
            refs_dir=_optional_bundle_relpath(d, "refs_dir"),
        )

    @classmethod
    def from_yaml(cls, text: str) -> "Storyboard":
        # The bundle ships storyboard.yaml. YAML is a superset of JSON, so this also
        # accepts a storyboard.json verbatim.
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def from_json(cls, text: str) -> "Storyboard":
        return cls.from_dict(json.loads(text))


# --------------------------------------------------------------------------------- cast

@dataclass
class Character:
    """One cast member: the slot it occupies, its name (the keyframe trigger token), its
    identity prompt, and the reference images the backend resolved from the bundle's refs dir."""
    slot: str
    name: str
    prompt: str
    ref_paths: list[Path] = field(default_factory=list)  # training / IP-Adapter references


@dataclass
class Cast:
    """The project's characters keyed by slot, parsed from `characters/registry.json`."""
    characters: dict[str, Character]  # slot -> Character

    @classmethod
    def from_registry(cls, registry: dict[str, Any]) -> "Cast":
        raw = registry.get("characters") or {}
        out: dict[str, Character] = {}
        for slot, c in raw.items():
            if slot not in SLOT_IDS or not isinstance(c, dict):
                continue
            out[slot] = Character(slot=slot, name=_str(c.get("name"), slot), prompt=_str(c.get("prompt")))
        return cls(characters=out)


# ------------------------------------------------------------------------------- bundle

@dataclass
class Bundle:
    """An extracted project bundle: the storyboard, the cast (with resolved ref paths),
    and the root the renderer writes its working tree under."""
    root: Path
    storyboard: Storyboard
    cast: Cast

    @classmethod
    def extract(cls, tar_path: Path, dest: Path) -> "Bundle":
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extract(tf, dest)

        sb_path = dest / "storyboard.yaml"
        if not sb_path.is_file():
            raise FileNotFoundError(f"bundle is missing storyboard.yaml at {sb_path}")
        storyboard = Storyboard.from_yaml(sb_path.read_text(encoding="utf-8"))

        reg_path = dest / "characters" / "registry.json"
        cast = Cast.from_registry(json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else {})

        refs_root = resolve_inside(dest, storyboard.refs_dir or "characters/refs", what="refs_dir")
        for slot, char in cast.characters.items():
            slot_dir = refs_root / slot
            if slot_dir.is_dir():
                char.ref_paths = sorted(p for p in slot_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))

        return cls(root=dest, storyboard=storyboard, cast=cast)


def _safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    """Reject path traversal and symlink/hardlink escapes before extracting an untrusted tar.

    `str.startswith` has a sibling-prefix bug (/tmp/proj matches /tmp/proj-evil); use
    Path.is_relative_to (Python 3.9+) for a correct parent-dir check. Symlinks and hardlinks
    are rejected outright: they can escape dest even when the stored name is safe.
    """
    dest = dest.resolve()
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"unsafe link in bundle: {member.name}")
        target = (dest / member.name).resolve()
        if not target.is_relative_to(dest):
            raise ValueError(f"unsafe path in bundle: {member.name}")
    tf.extractall(dest)


# --------------------------------------------------------------------------- render job

@dataclass
class RenderRequest:
    """What the control plane submits. `action` selects the pipeline path; `config` is the
    typed generation config (keyframe / i2v / lora) parsed from `render_overrides`, the single
    source of truth for what drives generation; `pretrained_loras` maps a slot to an
    already-trained LoRA so its training is skipped.

    `overrides` is kept as the raw `render_overrides` dict for the small set of non-generation
    *routing* flags the pipeline still reads off it (e.g. `finish_offloaded` for the off-GPU
    finish path); every actual generation knob now lives typed under `config`.

    NOTE: the standalone `finish_clip` and `i2v_clip` actions are sibling job types, NOT
    RenderRequests: the harness routes them directly to `run_finish_job` / `run_i2v_clip_job`
    (no bundle / planner) before any RenderRequest is built, so each carries its own
    input/output shape (see docs/contract.md). i2v_clip drives one shot's Wan image-to-video;
    finish_clip runs RIFE/face-restore."""
    action: str  # "render" | "preview" | "regen_shot" | "finalize" | "train_lora"
    # (the standalone "finish_clip"/"i2v_clip" actions are handled by the harness, not RenderRequest)
    project: str
    bundle_key: str
    # Default tier is DRAFT, the wallet-safe floor: an omitted tier must never silently buy the
    # most expensive render. The studio always sends the tier explicitly (runpod-submit.ts), so
    # this default only governs direct callers that omit the field.
    quality_tier: str = "draft"
    config: RenderConfig = field(default_factory=RenderConfig)
    overrides: dict[str, Any] = field(default_factory=dict)  # raw render_overrides; routing flags only
    pretrained_loras: dict[str, str] = field(default_factory=dict)
    process_shot_ids: list[str] | None = None  # finalize / regen subset
    # Cost door: when set on a `render`, the pipeline draws keyframes (training the LoRAs they
    # need) then SHORT-CIRCUITS before any i2v/finish GPU-seconds -- identical motion-skip to the
    # `preview` action, but keeping the `render` action intent. Honored in orchestrator.plan().
    # A caller asking for a cheap draft keyframe pass must never silently pay for a full render.
    # DEPRECATED, kept for compat: the control plane stopped sending this flag in studio v0.160.0
    # and sends action:"preview" for the cheap keyframe pass instead (vivijure runpod-submit.ts).
    # Direct/legacy callers still work; do not build new callers on it. See docs/contract.md.
    keyframes_only: bool = False
    # Optional audio bed: an R2 key the control plane staged (e.g. audio/<uuid>.m4a). The harness
    # fetches it and muxes it under the final video; None leaves the render silent.
    audio_key: str | None = None
    # Which LoRA family a train job fits / a render consumes: "sdxl" (in-process SDXL adapter) or
    # "wan" (Wan 2.2 A14B two-expert adapter via ai-toolkit, cf#29). Empty string means auto (the
    # orchestrator resolves: train_lora on the train image -> wan; render inline train -> sdxl).
    # Unknown non-empty values fall back to "sdxl".
    model_family: str = ""
    # NOTE: there is deliberately NO submitter-identity field. The control plane completed the
    # anti-SaaS identity strip (vivijure #292): the studio is single-operator, sends no user_email,
    # and serves /api/artifact by key with no per-row ownership check. A legacy/hostile job body that
    # still carries `user_email` is IGNORED here (from_dict never reads it), so a stripped identity
    # cannot resurface as artifact object metadata. See SECURITY.md.

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RenderRequest":
        from .config import RenderConfig
        overrides = d.get("render_overrides") if isinstance(d.get("render_overrides"), dict) else {}
        quality_tier = _str(d.get("quality_tier"), "draft")  # wallet-safe default; see the field note
        return cls(
            action=_str(d.get("action"), "render"),
            project=_str(d.get("project"), "untitled"),
            bundle_key=_str(d.get("bundle_key")),
            quality_tier=quality_tier,
            config=RenderConfig.from_request(quality_tier, overrides),
            overrides=overrides,
            pretrained_loras=d.get("pretrained_loras") if isinstance(d.get("pretrained_loras"), dict) else {},
            process_shot_ids=d.get("process_shot_ids") if isinstance(d.get("process_shot_ids"), list) else None,
            audio_key=_str(d.get("audio_key")) or None,
            keyframes_only=_bool(d.get("keyframes_only")),
            model_family=_str(d["model_family"], "sdxl") if "model_family" in d else "",
        )


@dataclass
class Keyframe:
    """A rendered keyframe in the result: the shot it belongs to and its object-store key."""
    shot_id: str
    key: str  # object-store key of the PNG


@dataclass
class Clip:
    """A per-shot i2v clip in the result: the shot, its object-store key, and the clip length."""
    shot_id: str
    key: str
    target_seconds: float | None = None


@dataclass
class RenderResult:
    """What the backend returns. The control plane polls for `output_key` (the final
    MP4) plus the per-shot `keyframes` and `clips`. `state_key` is always None since #112
    (kept for wire-compat): incremental state lives as per-artifact R2 objects now, not a
    shared tarball."""
    project: str
    output_key: str | None = None
    seconds: float | None = None
    has_audio: bool = False
    # True ONLY on the explicit audio_optional soft-degrade: a requested bed could not be fetched
    # and the job opted in, so the film shipped silent. Surfaced top-level (not just as an event)
    # so a poller never has to read the event stream to learn its film lost its audio.
    audio_missing: bool = False
    keyframes: list[Keyframe] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    lora: dict[str, Any] = field(default_factory=dict)  # slot -> {lora_id}
    state_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "project": self.project,
            "output_key": self.output_key,
            "seconds": self.seconds,
            "has_audio": self.has_audio,
            "audio_missing": self.audio_missing,
            "keyframes": [{"shot_id": k.shot_id, "key": k.key} for k in self.keyframes],
            "lora": self.lora,
            "state_key": self.state_key,
        }
        if self.clips:
            out["clips"] = [
                {"shot_id": c.shot_id, "key": c.key, **({"target_seconds": c.target_seconds} if c.target_seconds else {})}
                for c in self.clips
            ]
        return out
