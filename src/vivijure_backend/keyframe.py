"""Per-scene SDXL keyframe: the still each shot's i2v animates from.

One scene, one image. For a single character the job is ordinary: SDXL with the slot's trained
LoRA and an IP-Adapter pulling identity from the bundle refs. The hard case is two characters
in one frame, where the naive result is *bleed*: one identity's features smear onto the other
(both faces drift toward an average, hair color crosses over). The fix this module implements
is regional: each character is bound to its own half of the canvas with a per-region IP-Adapter
mask and its own LoRA, at deliberately moderate scales (a strong per-slot scale is what causes
the cross-contamination), with optional OpenPose conditioning to plant two distinct bodies so
the regions have something to attach to.

Clean-room: the SDXL + IP-Adapter masking + ControlNet wiring is built from diffusers' own
documented interfaces (the IP-Adapter regional/masking guide, the SDXL ControlNet pipeline), not
any prior pipeline. The prompt building, region geometry, and engine-path decision are pure and
unit-tested on a CPU box; the generation body defers torch/diffusers/PIL and is validated on a
pod. Engine knobs live in `KeyframeParams` here; the control plane's typed `KeyframeConfig`
(separate work) maps into them, it does not replace them.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from pathlib import Path

from .contract import Cast, Scene, Storyboard
from .prompt_chunks import CLIP_CHUNK_TOKENS, PROMPT_CHUNKS, encode_long_prompt

# Anti-bleed defaults: a character LoRA pushed hard in a shared frame is exactly what bleeds, so
# the per-slot scales are deliberately moderate and the identity comes mostly from the masked
# IP-Adapter. Mirrors orchestrator.MULTI_CHAR_DEFAULTS (the planner's view of the same numbers).
DEFAULT_LORA_SCALE = 0.3
DEFAULT_LORA_SCALE_PER_SLOT = 0.35
DEFAULT_IP_ADAPTER_SCALE = 0.7
DEFAULT_CANNY_SCALE = 0.70
# Pass B identity crop padding, a fraction of the face box on each side. 0.35 is the face with little hair, which
# is what lost Mara's curls in the two-shot (vivijure-cf#1143): the shared text cannot bind hair to one person,
# and this crop does not carry it. Kept as the default so behaviour is unchanged until a live sweep picks one.
DEFAULT_FACE_CROP_PAD = 0.35
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, extra limbs, fused faces, two heads, deformed, blurry, watermark, text"
)
# Appended to the negative when scene.prompt is non-empty (defense in depth; the two-pass is the fix).
SCENE_LOCK_NEGATIVE = (
    "studio backdrop, grey background, seamless paper, cyclorama, passport photo, "
    "headshot on backdrop, portrait studio, empty soundstage"
)
PASS_B_LOCK_CLAUSE = "in this location, not a studio portrait"
# The persistent base adapter name the ModelServer loads the Hyper-SD few-step distill LoRA under
# (models.ModelServer.keyframe_pipeline). _bind_loras keeps it active at 1.0 on the few-step path and
# 0.0 on the full-step (final) path, so one warm pipe renders every tier.
DISTILL_ADAPTER = "distill"

# A few-step distilled keyframe (Hyper-SD/DMD2) is trained to be (near-)guidance-free; driving it at the
# full-step CFG over-saturates and tears the frame (RGB channel split + grain). Mirrors the i2v distill
# (guidance_scale 1.0). Enforced at the source in `_guidance` so a caller that forgets to lower CFG on
# the few-step path can't break the render; the full-step (final) tier keeps its real CFG.
DISTILL_GUIDANCE = 1.0


def _guidance(cfg: "KeyframeParams") -> float:
    """The guidance scale to actually sample at: clamped to the distilled value on the few-step path,
    the configured CFG on the full-step (final) path."""
    return DISTILL_GUIDANCE if cfg.few_step else cfg.guidance_scale


@dataclass
class KeyframeParams:
    """Engine knobs for one keyframe render.

    The hosted default is the frame the movie keeps: full steps, guidance 5.0, 1280x720,
    InstantID for one person. `few_step` is an explicit opt-in (the draft tier), not the
    default. The control plane's KeyframeConfig fills these per job.
    """
    steps: int = 30
    guidance_scale: float = 5.0
    width: int = 1280                # 16:9 film frame; non-square is honored, not squared
    height: int = 720
    seed: int = 0
    few_step: bool = False           # Hyper-SD/DMD2. Off unless a caller opts in.
    scheduler: str = "dpmpp_2m_karras"  # config.Scheduler value; few-step wants ddim_trailing
    lora_scale: float = DEFAULT_LORA_SCALE          # single-char; regional uses lora_scale_per_slot
    lora_scale_per_slot: float = DEFAULT_LORA_SCALE_PER_SLOT
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE
    scene_lock: bool = True          # Pass A scene plate, Pass B canny + face-crop; off is a debug hatch
    canny_scale: float = DEFAULT_CANNY_SCALE
    face_crop_pad: float = DEFAULT_FACE_CROP_PAD  # Pass B identity crop padding (cf#1143)
    plate_ip_adapter_scale: float = 0.0  # Pass A per-region portrait IP-Adapter on a two-person shot; 0 = off (cf#1143)
    # Pass A per-region descriptions on a two-person shot (cf#1143). None = the per-shot default (cf#1281), see
    # plate_regional_text_default_on; True or False sets it.
    plate_regional_text: bool | None = None
    plate_people_free_background: bool = False  # with regional text: the band names no one (cf#1143 option c)
    plate_count_retries: int = 3     # 0..3 plate redraws while it has too many faces; 0 = off (cf#1143)
    # Identity method for the SINGLE-character path. InstantID is the default: insightface
    # face-embed through InstantID's image projection. "ip_adapter" is the h94 fallback.
    # Two-person shots ignore this and stay on the masked IP-Adapter regional path.
    identity_method: str = "instantid"
    instantid_ip_adapter_scale: float = 0.8  # InstantID face-embedding (image-proj) scale
    # ControlNet-OpenPose to separate bodies in multi-char shots. None = each path's default (backend#596):
    # on for the plain regional path, off for a guarded scene-lock plate. See plate_pose_on / regional_pose_on.
    pose_conditioning: bool | None = None
    controlnet_pose_scale: float = 0.55  # OpenPose ControlNet conditioning scale (regional path)
    region_gutter: int = 64          # px dead band between regional masks so they do not seam-blend
    pose_spread: float = 0.0         # 0 = figures at region centres; else centre-to-centre spacing (cf#1143)
    negative_prompt: str = DEFAULT_NEGATIVE
    max_slots: int = 2               # regional path is tuned for 2; more is future work


@dataclass
class KeyframeResult:
    """The outcome of rendering one shot's keyframe: where the PNG landed, which slots and engine
    path it used, and the prompt and seed that produced it.

    `identity_requested` and `identity_path` are recorded SEPARATELY and on purpose. The configured
    identity method is a request, and the InstantID path silently falls back to the shared
    IP-Adapter pipe whenever insightface finds no face in the reference, so the two genuinely
    differ on real renders. A record holding only one of them cannot answer which code path ran,
    and a record that cannot say which path ran cannot answer whether that path is healthy (#350)."""
    shot_id: str
    path: Path
    slots: list[str]
    multi_char: bool
    prompt: str
    seed: int
    engine: str                      # "single" | "regional"
    identity_requested: str = ""     # what the config ASKED for: "instantid" | "ip_adapter"
    identity_path: str = ""          # what actually RAN: see IDENTITY_PATHS
    # stage -> PNG beside `path`, only when the render was asked to save intermediates. See
    # INTERMEDIATE_STAGES. Empty on every path that has no such stage (no plate, no Pass B).
    intermediates: dict[str, Path] = field(default_factory=dict)
    # The plate count guard's record when it ran, else None. See plate_count_record.
    plate_count: dict | None = None


# The diagnostic intermediates a scene-lock keyframe can save (vivijure-cf#1143), in draw order:
# the Pass A pose control (only when the plate draws under pose), the Pass A plate, and the Pass B
# canny map of that plate (only when Pass B runs). The names are the file and key suffixes.
# With the plate count guard on, every plate draw is also kept as plate_try0.. (try0 is the first draw); "plate"
# is always the one Pass B was given.
INTERMEDIATE_STAGES: tuple[str, ...] = ("pose", "plate", "plate_try0", "plate_try1", "plate_try2", "plate_try3",
                                        "canny")


# Plate redraw seeds: far from the base seed so a retry never repeats a neighbouring shot's or seed's draw.
PLATE_RETRY_SEED_STRIDE = 1_000_003


def plate_count_retries_for(scene: Scene, cfg: KeyframeParams) -> int:
    """How many plate redraws the count guard may make for this shot (vivijure-cf#1143). Two-person-plus shots
    only: a one-person or empty shot's plate has no third person to catch. Clamped to 0..3; 0 = off."""
    if engine_for(scene, cfg) != "regional":
        return 0
    return max(0, min(3, int(cfg.plate_count_retries)))


def plate_pose_default_off(people: int | None, *, max_slots: int, pose_conditioning: bool | None,
                           plate_count_retries: int) -> bool:
    """True iff the backend#596 default takes the pose skeleton OFF a scene-lock plate that drew under it before.

    Measured on vivijure-cf#1281 (18 jobs, one board, 6 seeds): under the fixed standing skeleton the pair posed
    for the camera instead of acting, 2 of 6 held the action with the right people; with no skeleton, 5 of 6,
    and the plate count guard caught every third person. So the default holds only while the guard runs: with
    the guard off and the pose unset, the plate keeps its skeleton, and "pose off, guard off" is never a default.
    An explicit `pose_conditioning` is the job's own choice and never counts as the default.

    `people` is the shot's character count, or None for a config-only reuse hash (then it applies whenever the
    rest holds, as for plate_count_retries). Only 2..max_slots people draw a pose at all. Pure."""
    if pose_conditioning is not None:
        return False
    if people is not None and not (2 <= people <= max_slots):
        return False
    return max(0, min(3, int(plate_count_retries))) > 0


def plate_regional_text_default_on(people: int | None, *, max_slots: int, plate_regional_text: bool | None,
                                   pose_conditioning: bool | None, plate_count_retries: int) -> bool:
    """True iff the vivijure-cf#1281 default turns per-region text ON for a scene-lock plate that drew without it.

    Measured on vivijure-cf#1281, board 2 (6 seeds, keyframes-only through the studio): with the backend#596
    default (no pose skeleton, the plate count guard on) the plate drew each person's look from the whole shot
    text, and Dev came out as a woman or long-haired on 3 of 6; with per-region text on top, 6 of 6 held the
    action with the right people and no third person. With the skeleton back on, 5 of 6. So the default is
    coupled to the measured cell: it applies only where the pose is ALSO left to its default and that default
    takes the skeleton off, which already requires the guard (plate_pose_default_off). An explicit
    `plate_regional_text`, or an explicit `pose_conditioning`, is the job's own choice and never the default.

    `people` is the shot's character count, or None for a config-only reuse hash. Pure."""
    if plate_regional_text is not None:
        return False
    return plate_pose_default_off(people, max_slots=max_slots, pose_conditioning=pose_conditioning,
                                  plate_count_retries=plate_count_retries)


def plate_pose_on(scene: Scene, cfg: KeyframeParams) -> bool:
    """Whether a scene-lock Pass A plate draws under the pose skeleton. Only a two-plus-person (regional) shot
    ever does; an explicit `pose_conditioning` wins; unset follows `plate_pose_default_off`. Pure."""
    if engine_for(scene, cfg) != "regional":
        return False
    if cfg.pose_conditioning is not None:
        return bool(cfg.pose_conditioning)
    return not plate_pose_default_off(len(scene.character_slots), max_slots=cfg.max_slots,
                                      pose_conditioning=None, plate_count_retries=cfg.plate_count_retries)


def regional_pose_on(cfg: KeyframeParams) -> bool:
    """Whether the plain regional path (scene lock off) draws under the pose skeleton: on unless a job turns it
    off. backend#596 changed only the scene-lock plate; this path was not measured. Pure."""
    return cfg.pose_conditioning is not False


def plate_retry_seed(seed: int, attempt: int) -> int:
    """The seed of plate redraw `attempt` (1-based). Deterministic, so a rerun reproduces the same redraws."""
    return (int(seed) + attempt * PLATE_RETRY_SEED_STRIDE) % 2**31


def count_faces(analyzer, image) -> int:
    """Faces insightface finds in a PIL image (the analyser the scene-lock path already loads for face crops).
    Measured offline on 19 saved plates: it caught 11 of 12 plates with a third person and read every clean plate
    as 2. It misses a person facing away from the camera; that is the guard's known limit."""
    import numpy as np
    arr = np.array(image.convert("RGB"))[:, :, ::-1]  # insightface wants BGR
    return len(analyzer.get(arr) or [])


def choose_plate(counts: list[int], expected: int) -> int:
    """The draw Pass B gets: the first that holds at most `expected` faces, else the one with the fewest (the
    earliest on a tie). Fewer faces than people is accepted: a person can face away, and the guard is for extras."""
    if not counts:
        raise ValueError("choose_plate needs at least one draw")
    for i, n in enumerate(counts):
        if n <= expected:
            return i
    return min(range(len(counts)), key=lambda i: (counts[i], i))


def plate_count_record(counts: list[int], expected: int, retries: int) -> dict:
    """The guard's honest record, for the result and the event. `clean` false means the still was made from a
    plate that still has too many faces, and says so."""
    chosen = choose_plate(counts, expected)
    return {"expected": expected, "faces": list(counts), "chosen": chosen,
            "clean": counts[chosen] <= expected, "retries": retries}


def intermediate_path(out_path: Path, stage: str) -> Path:
    """Where one intermediate lands: beside the keyframe, `<shot>.<stage>.png`."""
    if stage not in INTERMEDIATE_STAGES:
        raise ValueError(f"unknown keyframe intermediate stage {stage!r}")
    return Path(out_path).with_name(f"{Path(out_path).stem}.{stage}.png")


def write_intermediates(images: dict, out_path: Path) -> dict[str, Path]:
    """Write each collected intermediate image beside the keyframe and return stage -> path, in
    INTERMEDIATE_STAGES order. An unknown stage is a programming error and raises before anything
    is written."""
    unknown = set(images) - set(INTERMEDIATE_STAGES)
    if unknown:
        raise ValueError(f"unknown keyframe intermediate stage(s) {sorted(unknown)!r}")
    saved: dict[str, Path] = {}
    for stage in INTERMEDIATE_STAGES:
        if stage in images:
            p = intermediate_path(out_path, stage)
            images[stage].save(p)
            saved[stage] = p
    return saved


# The identity paths a keyframe render can actually take. Named here so the render record, the
# progress channel and any history query share one vocabulary.
IDENTITY_PATHS: tuple[str, ...] = (
    "instantid",            # insightface embedded a real face and drove the InstantID IP attn
    "ip_adapter",           # single subject, a reference image conditioned the shared pipe
    "regional_ip_adapter",  # multi-character masked path, one reference per region
    "lora_only",            # a character LoRA bound, no reference image
    "none",                 # no character identity in this shot at all
)


# ------------------------------------------------------------------------- prompt building

def slot_trigger(cast: Cast, slot: str) -> str:
    """The token that summons a slot's identity in the prompt: its trained LoRA trigger, which
    is the character's name (see lora_train.TrainedLora.trigger), falling back to the slot id."""
    char = cast.characters.get(slot)
    return (char.name if char and char.name else slot)


# SDXL's text encoders read 77 tokens including start and end, so 75 are ours per chunk. The prompt is
# encoded in PROMPT_CHUNKS chunks (prompt_chunks.py, vivijure-cf#1143), so this is the whole budget.
# Past it, the encoder drops the tail and says so in a keyframe.prompt_cut event.
PROMPT_TOKEN_BUDGET = CLIP_CHUNK_TOKENS * PROMPT_CHUNKS
# The estimate below is not the real tokenizer, which is not on this CPU path. This margin absorbs the
# difference. Measured against CLIP on the vivijure-cf#1147 boards: it estimated low by at most 7 in 75,
# so the margin scales with the budget.
PROMPT_TOKEN_MARGIN = 8 * PROMPT_CHUNKS


def estimate_tokens(text: str) -> int:
    """Rough CLIP token count: one per word and one per punctuation mark."""
    return len(re.findall(r"\w+|[^\w\s]", text or ""))


def slot_description(cast: Cast, slot: str, scene_prompt: str = "") -> str:
    """The slot's character description from the cast registry, or "" when it has none or the
    scene prompt already contains it. This is what holds hair and wardrobe on a shot prompt that
    names only the character: a trained LoRA carries the face, not the outfit."""
    char = cast.characters.get(slot)
    text = (char.prompt if char else "").strip()
    if not text or text.lower() in (scene_prompt or "").lower():
        return ""
    return text


# What a description clause is, for the token budget (vivijure-cf#1147). The order is the priority:
# a stated adult age first, then what a viewer tells people apart by, then other physical traits.
# Mood goes last. A portrait direction is not a trait at all, and it fights the shot's location.
_ADULT_CLAUSE = re.compile(r"\b(adult|grown|(?:twenties|thirties|forties|fifties|sixties|seventies|eighties))\b", re.I)
_LOOK_CLAUSE = re.compile(
    r"\b(hair|haired|curls?|curly|ringlets?|braids?|ponytail|bun|bangs|fringe|undercut|mohawk|dreadlocks|bald|buzz|"
    r"buzz-cut|crew-cut|shaved|beard|bearded|stubble|mustache|moustache|goatee|sideburns|eye\s?patch|goggles|glasses|scar|"
    r"jacket|coat|hoodie|shirt|t-shirt|top|tee|blouse|sweater|cardigan|vest|dress|skirt|overalls|apron|uniform|robe|"
    r"trousers|pants|jeans|shorts|boots|shoes|gloves|hat|cap|helmet|scarf|armband|tie|suit|plating|armor|armour)\b", re.I)
_MOOD_CLAUSE = re.compile(
    r"\b(calm|presence|composed|restless|alert|energy|demeanou?r|expression|attitude|mood|vibe|watchful|"
    r"gaze|quiet|stern|resolute|gentle|confident|tired|magnetic)\b", re.I)
_DIRECTION_CLAUSE = re.compile(r"\b(portrait|head-and-shoulders|background|focus|studio|headshot)\b", re.I)


# A garment named after a place draws the place. "charcoal field jacket" put a shot that says "in the
# dim workshop" in a field of grass, 4 of 4 at one seed, and "charcoal jacket" put the same 4 indoors
# (vivijure-cf#1147, and the likely cause of the field seen on vivijure-cf#1145).
_PLACE_GARMENT = re.compile(
    r"\b(?:field|beach|desert|forest|safari|street|ski|mountain|garden)\s+(jacket|coat|vest|dress|boots|hat|shirt|cap)\b", re.I)
# Said before the description on a one-person plate. Saying the action again after the description is
# what keeps it, and saying it twice can draw the person twice (vivijure-cf#1147).
PLATE_SOLO = "solo"


_COUNT_WORDS = {2: "two", 3: "three", 4: "four"}


def adult_clause(names: list[str], *, named: bool) -> str:
    """The shot's statement that every person in it is an adult (vivijure-cf#1143). It is the safety
    clause, so it is said ONCE per shot for everyone, placed before anything the budget can cut, and never
    fitted away. Stating it per shot instead of per description is also what frees the two-person budget.

    `named` is the identity pass, which names each person (the names are its LoRA triggers). The plate has
    no triggers, so it counts them. An empty shot has no clause."""
    n = len(names)
    if n == 0:
        return ""
    if not named:
        return "an adult" if n == 1 else f"{_COUNT_WORDS.get(n, str(n))} adults"
    if n == 1:
        return f"{names[0]}, an adult"
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    return f"{listed}, both adults" if n == 2 else f"{listed}, all adults"


def setting_for(scene: Scene, storyboard: Storyboard) -> str:
    """The film's setting for this shot, or "" when there is none, or the shot prompt or the style prefix
    already says it (vivijure-cf#1145). A shot that names its own place keeps its own place."""
    setting = (storyboard.setting or "").strip()
    if not setting:
        return ""
    low = setting.lower()
    if low in (scene.prompt or "").lower() or low in (storyboard.style_prefix or "").lower():
        return ""
    return setting


def drop_place_from_garment(text: str) -> str:
    """"charcoal field jacket" -> "charcoal jacket". A place the prompt means to be in is untouched."""
    return _PLACE_GARMENT.sub(r"\1", text or "")


def clause_rank(clause: str, *, adult_stated: bool = False) -> int:
    """0 is kept first. A clause naming a garment or hair outranks a mood word in the same clause.

    `adult_stated`: the shot already says everyone is an adult (adult_clause), so a description's own age
    clause is a look detail, ranked after hair and clothes (vivijure-cf#1143)."""
    if _ADULT_CLAUSE.search(clause):
        return 2 if adult_stated else 0
    if _LOOK_CLAUSE.search(clause):
        return 1
    return 3 if _MOOD_CLAUSE.search(clause) else 2


def fit_description(text: str, budget: int, *, name: str = "", scene_prompt: str = "", max_rank: int = 3,
                    adult_stated: bool = False) -> str:
    """The clauses of a description that fit `budget` estimated tokens, in their written order.

    The budget is spent by clause_rank, so a mood phrase can never cost a garment. A clause that
    does not fit is skipped and a lower one may still fit. Dropped outright: a clause that is only
    the character's name, because the trigger says it; one the scene prompt already says, because
    tokens are scarce; and a portrait direction, because it pulls the shot out of its location.
    A clause ranked below `max_rank` is left out however much room there is."""
    clauses = [c.strip() for c in re.split(r"[,;]|\.\s", text.strip().rstrip(". "))]
    usable = [
        (i, c) for i, c in enumerate(clauses)
        if c and c.lower() != name.lower() and c.lower() not in (scene_prompt or "").lower()
        and not _DIRECTION_CLAUSE.search(c) and clause_rank(c, adult_stated=adult_stated) <= max_rank
    ]
    kept: list[tuple[int, str]] = []
    used = 0
    for i, clause in sorted(usable, key=lambda item: (clause_rank(item[1], adult_stated=adult_stated), item[0])):
        for form in (clause, clause.split(" over ")[0]):  # a jacket without what is under it still reads
            cost = estimate_tokens(form) + 1  # the joining comma
            if used + cost <= budget:
                kept.append((i, form))
                used += cost
                break
    return ", ".join(c for _, c in sorted(kept))


def _join_prompt(parts: list[str]) -> str:
    # Fragments may carry their own edge commas (the control plane's style_prefix ends in ","),
    # so strip leading/trailing commas per fragment before joining; internal commas (the trigger
    # list) are untouched, so two characters never collapse into one.
    cleaned = [c for c in (p.strip().strip(",").strip() for p in parts if p) if c]
    return ", ".join(cleaned)


def build_prompt(scene: Scene, cast: Cast, storyboard: Storyboard, *,
                 include_triggers: bool = True, lock_clause: bool = False) -> str:
    """Compose the SDXL prompt: the storyboard style prefix, the scene's own prompt, optionally the
    LoRA trigger tokens and the Pass B location-lock clause, and each slot's character description.
    Pass A has no triggers and no lock, but it draws the person whose outline Pass B then keeps
    through canny, so a plate with no description fixes the hair and the cut of the clothes before
    identity runs (vivijure-cf#1147, seen by rendering the plate alone at the same seed). It gets
    only the age, hair and garments, because the whole description makes a head-and-shoulders
    portrait, and the scene is said again after them, because a person described last poses for the
    camera instead of doing the action. The repeat comes last, so the budget only ever cuts the
    repeat. Pieces are comma-joined and de-duplicated of empty fragments so a missing style or
    prompt never leaves a dangling comma.

    Every shot with people says once that they are all adults (adult_clause, vivijure-cf#1143). In
    the identity pass that clause names them, and the names are the LoRA triggers, so it stands where
    the triggers stood. It is never fitted away: if a long style and scene would push it past the
    budget, it moves ahead of the scene.

    Descriptions come LAST and are FITTED to what is left of the token budget (two 75-token chunks,
    prompt_chunks.py), split evenly across the shot's slots. Everything before them outranks them,
    and the lock clause above all: it is what keeps a person in the scene (vivijure-cf#1147). The even
    split is what stops the first slot from taking the whole budget on a two-person shot. On a
    two-person shot a description's own age clause ranks after hair and clothes, because the shot
    already says both are adults.

    The film's `setting` (vivijure-cf#1145) follows the shot's own prompt when the shot does not say
    it. It is the lowest priority: said only if it still fits after the descriptions."""
    names = [slot_trigger(cast, s) for s in scene.character_slots]
    multi = len(names) >= 2
    # The identity pass names each person once, in the clause that also says they are adults; the names are
    # the LoRA triggers. The plate has no triggers, so its clause counts people instead.
    adult = adult_clause(names, named=include_triggers)
    # The clause sits where the triggers always sat, after the scene. If a long style and a long scene would
    # push it past the budget, it moves ahead of the scene instead: the safety clause is never the part cut.
    early = estimate_tokens(_join_prompt([storyboard.style_prefix, scene.prompt, adult])) > (
        PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN)
    tail = [] if early else [adult]
    if storyboard.style_preset and storyboard.style_preset != "None":
        tail.append(storyboard.style_preset)
    if lock_clause:
        tail.append(PASS_B_LOCK_CLAUSE)
    if not include_triggers and len(scene.character_slots) == 1:
        tail.append(PLATE_SOLO)
    setting = setting_for(scene, storyboard)

    def head(with_setting: bool) -> str:
        parts = [storyboard.style_prefix, adult if early else "", scene.prompt, setting if with_setting else "", *tail]
        return drop_place_from_garment(_join_prompt(parts))

    prompt = head(False)
    wanted = [(s, drop_place_from_garment(slot_description(cast, s, scene.prompt))) for s in scene.character_slots]
    wanted = [(s, d) for s, d in wanted if d]
    fitted: list[str] = []
    if wanted:
        room = PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN - estimate_tokens(prompt)
        share = room // len(wanted)
        if share > 0:
            # In a plate, rank 1 and above only (age, hair, garments). With the adult clause said once for
            # the shot, a two-person plate drops the per-person age for hair and clothes.
            max_rank = 3 if include_triggers else 1
            fitted = [fit_description(d, share, name=slot_trigger(cast, s), scene_prompt=scene.prompt,
                                      max_rank=max_rank, adult_stated=multi)
                      for s, d in wanted]
            fitted = [f for f in fitted if f]
    # The setting is the lowest priority after the scene and the descriptions: it is said only if it
    # still fits, so it can never cost a person their hair or clothes (vivijure-cf#1145).
    if setting:
        with_setting = head(True)
        if estimate_tokens(_join_prompt([with_setting, *fitted])) <= PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN:
            prompt = with_setting
    if not include_triggers and scene.character_slots:
        return drop_place_from_garment(_join_prompt([prompt, *fitted, scene.prompt]))
    return _join_prompt([prompt, *fitted])


def scene_lock_negative(scene: Scene, base: str) -> str:
    """Anatomy negative plus the studio-lock tail, but only when the scene actually has a
    location prompt. An empty scene prompt must not grow a studio-negative that then fights
    an empty positive."""
    if not (scene.prompt and scene.prompt.strip()):
        return base
    if SCENE_LOCK_NEGATIVE in (base or ""):
        return base
    return f"{base}, {SCENE_LOCK_NEGATIVE}" if base else SCENE_LOCK_NEGATIVE


@dataclass(frozen=True)
class KeyframePass:
    """One t2i call in the scene-lock two-pass. CPU-testable plan shape; the GPU body executes it."""
    name: str                    # "plate" | "identity"
    prompt: str
    negative_prompt: str
    bind_loras: bool
    lora_scale: float
    use_ip_adapter: bool
    face_crop: bool
    controlnet: str              # "none" | "pose" | "canny"
    canny_from_plate: bool


def plan_keyframe_passes(scene: Scene, cast: Cast, storyboard: Storyboard,
                         cfg: KeyframeParams) -> list[KeyframePass]:
    """The two-pass plan (or Pass A only for no-character shots). Pure."""
    engine = engine_for(scene, cfg)
    negative = scene_lock_negative(scene, cfg.negative_prompt)
    plate_prompt = build_prompt(scene, cast, storyboard, include_triggers=False)
    plate_cn = "pose" if plate_pose_on(scene, cfg) else "none"
    plate = KeyframePass(
        name="plate", prompt=plate_prompt, negative_prompt=negative,
        bind_loras=False, lora_scale=0.0, use_ip_adapter=False, face_crop=False,
        controlnet=plate_cn, canny_from_plate=False,
    )
    if not scene.character_slots:
        return [plate]
    ident_scale = cfg.lora_scale_per_slot if engine == "regional" else cfg.lora_scale
    identity = KeyframePass(
        name="identity",
        prompt=build_prompt(scene, cast, storyboard, include_triggers=True, lock_clause=True),
        negative_prompt=negative,
        bind_loras=True, lora_scale=ident_scale, use_ip_adapter=True, face_crop=True,
        controlnet="canny", canny_from_plate=True,
    )
    return [plate, identity]


def plate_ip_adapter_scale_for(scene: Scene, cfg: KeyframeParams) -> float:
    """The scale at which Pass A takes each person's whole cast portrait, confined to their region, or 0.0 for off.

    vivijure-cf#1143: on a two-person shot the plate draws hair from shared text that SDXL does not bind to a
    person, and Pass B keeps the plate's outline (measured: the bun survived canny 0.4 and every identity-crop
    padding). Only the no-plate path, which takes the whole portrait, restored the hair. This gives the plate the
    portrait. Two-person shots only: a one-person plate holds its hair from its own description, and an empty shot
    has no one to condition. Off unless set, so the plate is unchanged by default."""
    if engine_for(scene, cfg) != "regional":
        return 0.0
    return max(0.0, min(1.0, float(cfg.plate_ip_adapter_scale)))


def plate_regional_text_on(scene: Scene, cfg: KeyframeParams) -> bool:
    """Whether Pass A gives each person's description to their own region (vivijure-cf#1143 option a). Two-person
    shots within the regional slot cap only: when set, or when unset and plate_regional_text_default_on holds."""
    if engine_for(scene, cfg) != "regional":
        return False
    if cfg.plate_regional_text is not None:
        return bool(cfg.plate_regional_text)
    return plate_regional_text_default_on(len(scene.character_slots), max_slots=cfg.max_slots,
                                          plate_regional_text=None, pose_conditioning=cfg.pose_conditioning,
                                          plate_count_retries=cfg.plate_count_retries)


def plate_region_prompts(scene: Scene, cast: Cast, storyboard: Storyboard) -> tuple[str, list[str]]:
    """The two-person plate's prompts for per-region text: (shared, [one per person, in slot order]).

    `shared` is the plate prompt with NO person's description: the style, the scene and its action, the setting,
    the adult clause and the scene repeat. The background keeps it, and it opens every person's prompt, so none of
    it can be masked away. Each person's prompt is `shared` plus that person's own description, fitted to what is
    left of the budget for one person (not split, since each prompt holds one), with hair and clothes ahead of age.
    No other person's description is in it: that is the whole point."""
    shared = build_prompt(scene, Cast(characters={}), storyboard, include_triggers=False)
    out = []
    for s in scene.character_slots:
        desc = drop_place_from_garment(slot_description(cast, s, scene.prompt))
        room = PROMPT_TOKEN_BUDGET - PROMPT_TOKEN_MARGIN - estimate_tokens(shared)
        fitted = fit_description(desc, room, name=slot_trigger(cast, s), scene_prompt=scene.prompt, max_rank=1,
                                 adult_stated=True) if desc and room > 0 else ""
        out.append(_join_prompt([shared, fitted]))
    return shared, out


def plate_people_free_background_on(scene: Scene, cfg: KeyframeParams) -> bool:
    """Whether the band outside both people reads a people-free prompt (vivijure-cf#1143 option c). Only with
    per-region text on, which is what gives the band a prompt of its own."""
    return plate_regional_text_on(scene, cfg) and bool(cfg.plate_people_free_background)


def plate_background_prompt(storyboard: Storyboard) -> str:
    """The prompt the band outside both people reads under option (c): the style and the setting, and nothing else.

    Measured on 1.1.9 (vivijure-cf#1143): with per-region text on, the band read the shot text "Mara and Dev
    stand ...", and drew a third person there at 2 of 2 seeds. So it gets no shot text, no name, no description and
    no people count. It draws no person, so it carries no adult clause; every region that draws a person still
    does (plate_region_prompts). With no style and no setting it is empty, which is a valid prompt."""
    return _join_prompt([storyboard.style_prefix, (storyboard.setting or "").strip()])


def keyframe_render_path(engine: str, cfg: KeyframeParams, *, has_instantid_ref: bool) -> str:
    """Which GPU body `render_keyframe` takes.

    Two people stay regional (scene_lock when it is on). One person with InstantID and a
    reference still takes InstantID even when scene_lock is on. Scene lock covers the
    other single-person shots.
    """
    if engine == "regional":
        return "scene_lock" if cfg.scene_lock else "regional"
    if engine == "single" and cfg.identity_method == "instantid" and has_instantid_ref:
        return "instantid"
    if cfg.scene_lock:
        return "scene_lock"
    return engine


class LoraOnlyKeyframe(RuntimeError):
    """A still with a LoRA and no reference image is not a frame a clip model may animate."""


def refuse_lora_only_still(identity_path: str) -> None:
    """Fail the still before it is saved. `lora_only` must not leave the keyframe endpoint."""
    if identity_path == "lora_only":
        raise LoraOnlyKeyframe(
            "keyframe identity_path is lora_only; the still has no reference image and is not returned"
        )


# ------------------------------------------------------------------------- region geometry

def region_boxes(width: int, height: int, n: int, *, orientation: str = "vertical",
                 gutter: int = 0) -> list[tuple[int, int, int, int]]:
    """Split the canvas into `n` regions, one per character, as (left, top, right, bottom) boxes.
    Vertical orientation (side-by-side strips) is the default for the common two-shot; horizontal
    stacks them. `gutter` carves a dead band of that many pixels between adjacent regions (the
    interior edges are inset by gutter//2; the outer canvas edges stay flush) so the IP-Adapter
    masks do not abut and blend at the seam. Pure geometry, so the mask generation that consumes it
    is testable without an image library."""
    if n <= 1:
        return [(0, 0, width, height)]
    g = max(0, gutter) // 2
    boxes = []
    if orientation == "vertical":
        step = width // n
        for i in range(n):
            left = i * step + (g if i > 0 else 0)
            right = (width if i == n - 1 else (i + 1) * step) - (g if i < n - 1 else 0)
            boxes.append((left, 0, right, height))
    else:
        step = height // n
        for i in range(n):
            top = i * step + (g if i > 0 else 0)
            bottom = (height if i == n - 1 else (i + 1) * step) - (g if i < n - 1 else 0)
            boxes.append((0, top, width, bottom))
    return boxes


def single_identity_path(slot: "str | None", *, has_ref_image: bool, has_lora: bool) -> str:
    """Which identity path a single-subject render actually took. PURE, so the classification is
    asserted on CPU rather than inferred from a GPU body: a slot with no usable reference image
    renders from its LoRA alone and produces a perfectly good keyframe, so the difference is
    invisible from the outside unless it is recorded (#350)."""
    if slot and has_ref_image:
        return "ip_adapter"
    if slot and has_lora:
        return "lora_only"
    return "none"


def engine_for(scene: Scene, params: KeyframeParams) -> str:
    """Which path this scene takes: 'regional' (the masked multi-identity, anti-bleed path) for a
    two-plus-character shot within the slot cap, else 'single'."""
    n = len(scene.character_slots)
    return "regional" if 2 <= n <= params.max_slots else "single"


# --------------------------------------------------------------------------- render (GPU)

def render_keyframe(
    scene: Scene,
    cast: Cast,
    storyboard: Storyboard,
    server,
    out_path: Path,
    *,
    params: KeyframeParams | None = None,
    lora_paths: dict[str, Path] | None = None,
    pose_image: "Path | None" = None,
    save_intermediates: bool = False,
) -> KeyframeResult:
    """Render one scene's keyframe to `out_path`.

    `server` is a `models.ModelServer` (provides the fp8 SDXL pipeline with the distill LoRA);
    `lora_paths` maps slot -> trained adapter file. Single-character shots take the ordinary
    identity path; two-character shots take the regional anti-bleed path. Heavy imports are
    deferred; the body is validated on a pod.

    `save_intermediates` (diagnostic, vivijure-cf#1143) also writes the scene-lock path's pose
    control, plate and canny map beside `out_path` and lists them in `KeyframeResult.intermediates`.
    It changes nothing that is drawn. Other paths have no plate and save nothing.
    """
    cfg = params or KeyframeParams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slots = list(scene.character_slots)
    engine = engine_for(scene, cfg)

    import torch  # deferred: keep this module CPU-importable

    loras = {s: Path(p) for s, p in (lora_paths or {}).items()}
    single_slot = scene.character_slots[0] if scene.character_slots else None
    has_instantid_ref = bool(single_slot and _ref_images(cast, single_slot, count=1))
    path = keyframe_render_path(engine, cfg, has_instantid_ref=has_instantid_ref)

    # One person with an InstantID ref takes InstantID even when scene_lock is on.
    # Two people stay on the regional body. scene_lock is plate then canny + face-crop
    # IP-Adapter for every other shot.
    collected: dict = {}
    report: dict = {}
    if path == "scene_lock":
        image, identity_path, prompt = _render_scene_lock(
            server, scene, cast, storyboard, cfg, loras, pose_image,
            sink=collected if save_intermediates else None, report=report)
    elif path == "instantid":
        prompt = build_prompt(scene, cast, storyboard)
        image, identity_path = _render_instantid(server, prompt, scene, cast, cfg, loras, single_slot)
    else:
        prompt = build_prompt(scene, cast, storyboard)
        pipe = server.keyframe_pipeline()
        _apply_scheduler(pipe, cfg)
        generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
        if engine == "single":
            image, identity_path = _render_single(pipe, prompt, scene, cast, cfg, loras, generator)
        else:
            image, identity_path = _render_regional(
                pipe, prompt, scene, cast, cfg, loras, generator, pose_image)

    refuse_lora_only_still(identity_path)
    image.save(out_path)
    intermediates = write_intermediates(collected, out_path) if collected else {}
    return KeyframeResult(shot_id=scene.id or "shot", path=out_path, slots=slots,
                          multi_char=(engine == "regional"), prompt=prompt, seed=cfg.seed,
                          engine=engine, identity_requested=cfg.identity_method,
                          identity_path=identity_path, intermediates=intermediates,
                          plate_count=report.get("plate_count"))


def _render_scene_lock(server, scene, cast, storyboard, cfg, loras, pose_image, sink=None, report=None):
    """Pass A: scene plate (no LoRA, no IP-Adapter). Pass B: canny of that plate + face-crop
    IP-Adapter + split LoRA. No-character shots skip Pass B. Few-step tiers still run Pass B
    (t2i). InstantID is not this path. Returns `(image, identity_path, prompt)`.

    `sink`, when a dict, collects the intermediates by INTERMEDIATE_STAGES name. Collecting only
    keeps a reference; it changes no input to any pass.

    The plate count guard (`plate_count_retries`, two-person-plus shots): after the first plate, count its faces and
    redraw ONLY the plate with plate_retry_seed while it has more faces than people, up to the limit. Pass B then
    runs once, on the chosen plate, with the original generator, so a first plate that is clean gives exactly the
    still the guard-off path draws. `report["plate_count"]` gets the record whenever the guard ran."""
    import json
    import torch
    from .models import ModelRole

    engine = engine_for(scene, cfg)
    passes = plan_keyframe_passes(scene, cast, storyboard, cfg)
    pipe = server.keyframe_pipeline()
    _apply_scheduler(pipe, cfg)
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    # Distill adapter tracks the tier; character LoRAs stay off for Pass A.
    _bind_loras(pipe, {}, cfg.lora_scale, few_step=cfg.few_step)
    plate_ip_scale = plate_ip_adapter_scale_for(scene, cfg)
    plate_ip_kw = _plate_ip_kwargs(pipe, cast, scene, cfg, plate_ip_scale) if plate_ip_scale > 0 else {}
    if not plate_ip_kw:
        _ensure_ip_adapter(pipe, 0)

    plate_spec = passes[0]
    if plate_spec.controlnet == "pose":
        server.set_keyframe_controlnet(ModelRole.CONTROLNET_POSE)
        slots = scene.character_slots[:cfg.max_slots]
        boxes = region_boxes(cfg.width, cfg.height, len(slots), orientation="vertical",
                             gutter=cfg.region_gutter)
        if pose_image is not None:
            from PIL import Image
            control = Image.open(pose_image).convert("RGB").resize((cfg.width, cfg.height))
        else:
            control = _pose_skeleton(cfg.width, cfg.height, boxes, spread=cfg.pose_spread)
        cn_scale = cfg.controlnet_pose_scale
        if sink is not None:
            sink["pose"] = control
    else:
        server.set_keyframe_controlnet(ModelRole.CONTROLNET_POSE)
        control = _blank_control(cfg.width, cfg.height)
        cn_scale = 0.0
    from . import regional_text
    plate_prompt = plate_spec.prompt
    regions: list = []
    if plate_regional_text_on(scene, cfg):
        shared, persons = plate_region_prompts(scene, cast, storyboard)
        plate_prompt = shared  # the background's prompt; each person's region gets its own below
        slots = scene.character_slots[:cfg.max_slots]
        boxes = region_boxes(cfg.width, cfg.height, len(slots), orientation="vertical", gutter=cfg.region_gutter)
        regions = [(encode_long_prompt(pipe, p, plate_spec.negative_prompt, shot_id=scene.id or "shot")["prompt_embeds"], b)
                   for p, b in zip(persons, boxes)]
    background = None
    if regions and plate_people_free_background_on(scene, cfg):
        background = encode_long_prompt(pipe, plate_background_prompt(storyboard), plate_spec.negative_prompt,
                                        shot_id=scene.id or "shot")["prompt_embeds"]
    def draw_plate(gen):
        with regional_text.applied(pipe, regions, width=cfg.width, height=cfg.height,
                                   cfg_batch=_guidance(cfg) > 1.0, background=background):
            return pipe(
                **encode_long_prompt(pipe, plate_prompt, plate_spec.negative_prompt, shot_id=scene.id or "shot"),
                num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
                height=cfg.height, width=cfg.width, generator=gen,
                image=control, controlnet_conditioning_scale=cn_scale,
                **plate_ip_kw,
            ).images[0]

    plate = draw_plate(generator)
    retries = plate_count_retries_for(scene, cfg)
    if retries:
        expected = len(scene.character_slots[:cfg.max_slots])
        analyzer = server.face_analyzer()
        draws = [plate]
        counts = [count_faces(analyzer, plate)]
        while counts[-1] > expected and len(draws) <= retries:
            retry_gen = torch.Generator(device="cuda").manual_seed(plate_retry_seed(cfg.seed, len(draws)))
            draws.append(draw_plate(retry_gen))
            counts.append(count_faces(analyzer, draws[-1]))
        record = plate_count_record(counts, expected, retries)
        plate = draws[record["chosen"]]
        if report is not None:
            report["plate_count"] = record
        if sink is not None:
            for i, d in enumerate(draws):
                sink[f"plate_try{i}"] = d
        print("@event keyframe.plate_count " + json.dumps(
            {"shot_id": scene.id or "shot", "seed": cfg.seed, **record}, sort_keys=True), flush=True)
    print("@event keyframe.scene_plate " + json.dumps(
        {"shot_id": scene.id or "shot", "seed": cfg.seed, "plate_ip_adapter_scale": plate_ip_scale if plate_ip_kw else 0.0,
         "plate_regional_text": bool(regions), "plate_people_free_background": background is not None},
        sort_keys=True), flush=True)
    if sink is not None:
        sink["plate"] = plate

    if len(passes) == 1:
        return plate, "none", plate_spec.prompt

    ident = passes[1]
    server.set_keyframe_controlnet(ModelRole.CONTROLNET_CANNY)
    canny = _canny_preprocess(plate)
    if sink is not None:
        sink["canny"] = canny
    scale = ident.lora_scale
    if engine == "regional":
        slots = scene.character_slots[:cfg.max_slots]
        crops, any_face = _face_crops_for_slots(server, cast, slots, pad_ratio=cfg.face_crop_pad)
        image, identity_path = _pass_b_regional(
            pipe, ident, scene, cfg, loras, generator, canny, scale, crops, any_face)
    else:
        slot = scene.character_slots[0] if scene.character_slots else None
        crops, any_face = _face_crops_for_slots(server, cast, [slot] if slot else [], pad_ratio=cfg.face_crop_pad)
        image, identity_path = _pass_b_single(
            pipe, ident, scene, cfg, loras, generator, canny, scale, crops, any_face)
    # Next shot's Pass A expects pose on the shared pipe.
    server.set_keyframe_controlnet(ModelRole.CONTROLNET_POSE)
    return image, identity_path, ident.prompt


def _pass_b_single(pipe, ident, scene, cfg, loras, generator, canny, scale, crops, any_face):
    slot = scene.character_slots[0] if scene.character_slots else None
    _bind_loras(pipe, {slot: loras[slot]} if (slot and slot in loras) else {}, scale,
                few_step=cfg.few_step)
    if any_face and crops and crops[0] is not None:
        _ensure_ip_adapter(pipe, 1)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
        ip_kw = {"ip_adapter_image": crops[0]}
        identity_path = single_identity_path(slot, has_ref_image=True,
                                             has_lora=bool(slot and slot in loras))
    else:
        _ensure_ip_adapter(pipe, 0)
        ip_kw = {}
        identity_path = single_identity_path(slot, has_ref_image=False,
                                             has_lora=bool(slot and slot in loras))
    image = pipe(
        **encode_long_prompt(pipe, ident.prompt, ident.negative_prompt, shot_id=scene.id or "shot"),
        num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
        height=cfg.height, width=cfg.width, generator=generator,
        image=canny, controlnet_conditioning_scale=cfg.canny_scale,
        **ip_kw,
    ).images[0]
    return image, identity_path


def _pass_b_regional(pipe, ident, scene, cfg, loras, generator, canny, scale, crops, any_face):
    from diffusers.image_processor import IPAdapterMaskProcessor

    slots = scene.character_slots[:cfg.max_slots]
    w, h = cfg.width, cfg.height
    boxes = region_boxes(w, h, len(slots), orientation="vertical", gutter=cfg.region_gutter)
    _bind_loras(pipe, {s: loras[s] for s in slots if s in loras}, scale, few_step=cfg.few_step)
    if any_face:
        _ensure_ip_adapter(pipe, n=len(slots))
        scales = [cfg.ip_adapter_scale if c is not None else 0.0 for c in crops]
        pipe.set_ip_adapter_scale(scales)
        from PIL import Image
        ip_images = [c if c is not None else Image.new("RGB", (64, 64), (0, 0, 0)) for c in crops]
        masks = IPAdapterMaskProcessor().preprocess(
            [_box_mask(w, h, b) for b in boxes], height=h, width=w)
        ip_kw = {"ip_adapter_image": ip_images, "cross_attention_kwargs": {"ip_adapter_masks": masks}}
        identity_path = "regional_ip_adapter"
    else:
        _ensure_ip_adapter(pipe, 0)
        ip_kw = {}
        identity_path = "lora_only" if any(s in loras for s in slots) else "none"
    image = pipe(
        **encode_long_prompt(pipe, ident.prompt, ident.negative_prompt, shot_id=scene.id or "shot"),
        num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
        height=h, width=w, generator=generator,
        image=canny, controlnet_conditioning_scale=cfg.canny_scale,
        **ip_kw,
    ).images[0]
    return image, identity_path


def _plate_ip_kwargs(pipe, cast, scene, cfg, scale):
    """Pass A keyword arguments for the per-region portrait IP-Adapter (vivijure-cf#1143 option b): one whole cast
    portrait per person, each confined to the same region box the pose skeleton and Pass B use. A person with no
    portrait gets a blank image at scale 0. Returns {} when no one in the shot has a portrait, so the caller draws
    the plate as before. GPU path: validated on a pod."""
    from diffusers.image_processor import IPAdapterMaskProcessor
    from PIL import Image

    slots = scene.character_slots[:cfg.max_slots]
    w, h = cfg.width, cfg.height
    boxes = region_boxes(w, h, len(slots), orientation="vertical", gutter=cfg.region_gutter)
    refs = [_ref_images(cast, s, count=1) for s in slots]
    if not any(refs):
        return {}
    images = [r[0] if r else Image.new("RGB", (64, 64), (0, 0, 0)) for r in refs]
    _ensure_ip_adapter(pipe, n=len(slots))
    pipe.set_ip_adapter_scale([scale if r else 0.0 for r in refs])
    masks = IPAdapterMaskProcessor().preprocess([_box_mask(w, h, b) for b in boxes], height=h, width=w)
    return {"ip_adapter_image": images, "cross_attention_kwargs": {"ip_adapter_masks": masks}}


def _face_crops_for_slots(server, cast, slots, *, pad_ratio: float = DEFAULT_FACE_CROP_PAD):
    """Per-slot identity crop (or None), padded by `pad_ratio` of the face box on each side. No-face slots skip
    IP-Adapter (blank, scale 0)."""
    from . import instantid as _iid
    crops = []
    for slot in slots:
        refs = _ref_images(cast, slot, count=1)
        crop = None
        if refs:
            try:
                crop = _iid.crop_face(server.face_analyzer(), refs[0], pad_ratio=pad_ratio)
            except Exception:  # noqa: BLE001 -- no-face / analyzer miss is a skip, not a crash
                crop = None
        crops.append(crop)
    return crops, any(c is not None for c in crops)


def _canny_preprocess(image, *, low: int = 100, high: int = 200):
    """Canny edges of a PIL RGB image as RGB (xinsir ControlNet conditioning).

    Prefers OpenCV (the baked image; insightface pulls it) matching the model card.
    Falls back to a Sobel magnitude so CPU tests never import cv2."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(image.convert("RGB"))
    try:
        import cv2
        edges = cv2.Canny(arr, low, high)
    except Exception:  # noqa: BLE001 -- CPU tests / missing cv2
        gray = arr.mean(axis=2).astype(np.float32)
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        hi = (mag.max() * (high / 255.0)) if mag.max() else 1.0
        edges = (mag >= hi).astype(np.uint8) * 255
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb)


def _render_single(pipe, prompt, scene, cast, cfg, loras, generator):
    """One character (or none): the slot's LoRA bound, identity from a single IP-Adapter image.
    The plain SDXL identity path. Returns `(image, identity_path)`: which of the three single-subject
    identity paths this render actually took is not derivable from the outside, because a slot with
    no usable reference image silently renders from the LoRA alone."""
    slot = scene.character_slots[0] if scene.character_slots else None
    _bind_loras(pipe, {slot: loras[slot]} if (slot and slot in loras) else {}, cfg.lora_scale,
                few_step=cfg.few_step)

    ip_images = _ref_images(cast, slot, count=1) if slot else None
    if ip_images:
        _ensure_ip_adapter(pipe, 1)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
    else:
        _ensure_ip_adapter(pipe, 0)  # clear any IP-Adapter a prior scene left on the shared pipe
    # The shared keyframe pipe is a ControlNet pipeline; a single subject wants no pose control, so
    # hand it a blank control image at conditioning_scale 0.0 -> the ControlNet is inert (zero
    # residual) and this path renders exactly like plain SDXL.
    identity_path = single_identity_path(slot, has_ref_image=bool(ip_images),
                                         has_lora=bool(slot and slot in loras))
    image = pipe(
        **encode_long_prompt(pipe, prompt, cfg.negative_prompt, shot_id=scene.id or "shot"),
        num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
        height=cfg.height, width=cfg.width, generator=generator,
        image=_blank_control(cfg.width, cfg.height),
        controlnet_conditioning_scale=0.0,
        **({"ip_adapter_image": ip_images[0]} if ip_images else {}),
    ).images[0]
    return image, identity_path


def _render_regional(pipe, prompt, scene, cast, cfg, loras, generator, pose_image):
    """Two characters, no bleed: each slot's identity is confined to its own region with a
    per-region IP-Adapter mask, each LoRA bound at the moderate per-slot scale, and (when on)
    OpenPose conditioning planting two bodies. The masks are what keep one face off the other.
    Returns `(image, identity_path)`."""
    from diffusers.image_processor import IPAdapterMaskProcessor

    slots = scene.character_slots[:cfg.max_slots]
    w, h = cfg.width, cfg.height
    boxes = region_boxes(w, h, len(slots), orientation="vertical", gutter=cfg.region_gutter)

    _bind_loras(pipe, {s: loras[s] for s in slots if s in loras}, cfg.lora_scale_per_slot,
                few_step=cfg.few_step)

    # One IP-Adapter image per slot (its refs), each confined to its region's mask.
    _ensure_ip_adapter(pipe, n=len(slots))
    pipe.set_ip_adapter_scale([cfg.ip_adapter_scale] * len(slots))
    ip_images = [_ref_images(cast, s, count=1)[0] for s in slots]
    masks = IPAdapterMaskProcessor().preprocess(
        [_box_mask(w, h, b) for b in boxes], height=h, width=w)

    # Plant two distinct bodies: an OpenPose skeleton with one standing figure per region box, so
    # the masked IP-Adapter identities land on SEPARATE people instead of merging into one (the
    # "older Aria" collapse). An explicit pose_image overrides the generated skeleton. With
    # pose_conditioning off, a blank control image at scale 0.0 makes the ControlNet inert and the
    # path degrades to masked-IP-Adapter only.
    if regional_pose_on(cfg):
        if pose_image is not None:
            from PIL import Image
            control = Image.open(pose_image).convert("RGB").resize((w, h))
        else:
            control = _pose_skeleton(w, h, boxes, spread=cfg.pose_spread)
        cn_scale = cfg.controlnet_pose_scale
    else:
        control = _blank_control(w, h)
        cn_scale = 0.0
    image = pipe(
        **encode_long_prompt(pipe, prompt, cfg.negative_prompt, shot_id=scene.id or "shot"),
        num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
        height=h, width=w, generator=generator,
        ip_adapter_image=ip_images,
        cross_attention_kwargs={"ip_adapter_masks": masks},
        image=control,
        controlnet_conditioning_scale=cn_scale,
    ).images[0]
    return image, "regional_ip_adapter"


def _render_instantid(server, prompt, scene, cast, cfg, loras, slot):
    """Single-character InstantID keyframe: the slot's reference face is embedded by insightface,
    projected to identity tokens, and injected into the UNet cross-attention (InstantID's face IP-
    Adapter) on a plain SDXL pipe. The slot's character LoRA still binds for body/style.

    Identity tokens reach the UNet through the IP attn processors' side channel (`proc.id_embeds`), so
    the prompt embeds stay clean. Uses the identity IP-Adapter only (no IdentityNet ControlNet -- see
    models.instantid_pipeline). GPU body, validated on a pod.

    Returns `(image, identity_path)`. The no-face branch below falls back to the shared IP-Adapter
    pipe and returns THAT path, so a render whose config asked for InstantID and did not get it says
    so in its own record instead of being indistinguishable from one that did (#350)."""
    import torch
    from . import instantid as _iid

    pipe = server.instantid_pipeline()
    _apply_scheduler(pipe, cfg)
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
    _bind_loras(pipe, {slot: loras[slot]} if slot in loras else {}, cfg.lora_scale,
                few_step=cfg.few_step)

    ref = _ref_images(cast, slot, count=1)[0]
    analyzed = _iid.analyze_face(server.face_analyzer(), ref)
    if analyzed is None:  # no face detected: fall back to the shared keyframe pipe's IP-Adapter path
        kpipe = server.keyframe_pipeline()
        _apply_scheduler(kpipe, cfg)
        return _render_single(kpipe, prompt, scene, cast, cfg, loras, generator)  # reports its own path
    embedding = analyzed[0]
    id_tokens = _iid.faceid_tokens(pipe._vj_image_proj, embedding)  # (1, num_tokens, 2048)

    # Feed identity through the IP attn SIDE channel. CFG batch is [uncond; cond]: zero identity for
    # the uncond pass so guidance does not invent a face, the real tokens for cond.
    id_embeds = torch.cat([torch.zeros_like(id_tokens), id_tokens], dim=0)
    for proc in pipe._vj_id_attn.values():
        proc.scale = cfg.instantid_ip_adapter_scale
        proc.id_embeds = id_embeds
    try:
        return pipe(
            **encode_long_prompt(pipe, prompt, cfg.negative_prompt, shot_id=scene.id or "shot"),
            num_inference_steps=cfg.steps, guidance_scale=_guidance(cfg),
            height=cfg.height, width=cfg.width, generator=generator,
        ).images[0], "instantid"
    finally:
        for proc in pipe._vj_id_attn.values():
            proc.id_embeds = None  # clear so a later render on the warm pipe never reuses them


# --------------------------------------------------------------------------- internals (GPU)

def _apply_scheduler(pipe, cfg: KeyframeParams) -> None:
    """Pin the SDXL sampler for this render on the warm shared pipe. The Hyper-SD few-step LoRA
    needs DDIM with timestep_spacing="trailing" (cfg=0); the full-step final tier wants a higher-
    order solver. Every scheduler is rebuilt from the pristine full-step config stashed by the
    ModelServer (`_vj_default_scheduler`), so repeated draft->final->draft swaps on the warm pipe
    never compound. Unrecognized values fall back to that pristine scheduler. GPU path: diffusers is
    imported lazily so this module stays CPU-importable; the mapping is validated on a pod."""
    from diffusers import (
        DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
    )
    base = getattr(pipe, "_vj_default_scheduler", None) or pipe.scheduler
    base_cfg = base.config
    sched = getattr(cfg, "scheduler", "ddim_trailing")
    if sched == "ddim_trailing":
        pipe.scheduler = DDIMScheduler.from_config(base_cfg, timestep_spacing="trailing")
    elif sched == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(base_cfg)
    elif sched == "dpmpp_2m_karras":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_cfg, use_karras_sigmas=True)
    elif sched == "dpmpp_2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_cfg)
    elif sched == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(base_cfg)
    elif sched == "euler_ancestral":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(base_cfg)
    elif sched == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(base_cfg)
    else:
        pipe.scheduler = base  # tcd is handled by the unified-LoRA path (not wired here); restore base


def _normalize_lora_state_dict(sd: dict) -> dict:
    """Convert a raw-PEFT LoRA state dict to the diffusers SDXL convention.

    Raw-PEFT format (save_file(get_peft_model_state_dict(unet)) without convert_state_dict_to_diffusers):
    keys have a `base_model.model.` prefix (or none), no `unet.` scope, and use `lora_A`/`lora_B`
    weight names. Diffusers format: `unet.<layer>.lora.down.weight` / `unet.<layer>.lora.up.weight`.

    Returns sd unchanged when it is already in diffusers format (no `.lora_A.` keys present).
    Pure key remapping -- no tensors allocated, CPU-safe.
    """
    if not any(".lora_A." in k or ".lora_B." in k for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith("base_model.model."):
            k = k[len("base_model.model."):]
        k = f"unet.{k}"
        k = k.replace(".lora_A.", ".lora.down.")
        k = k.replace(".lora_B.", ".lora.up.")
        out[k] = v
    return out


def _bind_loras(pipe, slot_paths: dict, scale: float, *, few_step: bool = True) -> list[str]:
    """Load each slot's character LoRA and activate it alongside whatever base adapter is already
    on the pipe (a few-step distill LoRA, if the ModelServer loaded one). Character LoRAs go on at
    the moderate per-slot `scale` (the anti-bleed weight); base adapters stay at 1.0, EXCEPT the
    distill adapter, whose weight is gated by `few_step`: 1.0 on the draft/standard few-step path,
    0.0 on the full-step (final) path, so the one warm pipe renders every tier without reloading.
    Discovering the base adapter instead of hardcoding a name means this is correct whether or not a
    distill LoRA is present, and a missing one never silently drops the character identity.

    The keyframe pipeline is a process-global shared across every scene on a warm worker, so the
    previous scene's character adapters are still attached on entry. Record the true base adapters
    once (whatever is present before any character is bound), then drop the prior scene's character
    adapters before rebinding; otherwise peft rejects the duplicate name ("Adapter name A already
    in use"). Clearing them also stops a no-character scene inheriting the last scene's identity."""
    if not hasattr(pipe, "_vj_base_adapters"):
        pipe._vj_base_adapters = _adapter_names(pipe)
    base = pipe._vj_base_adapters  # persistent adapters (e.g. a distill LoRA); never a character
    stale = [n for n in _adapter_names(pipe) if n not in base]
    if stale:
        pipe.delete_adapters(stale)
    loaded = []
    for slot, path in slot_paths.items():
        p = Path(path)
        if p.is_file():
            # On the GPU path the LoRA always exists on disk; load it so we can detect and convert
            # raw-PEFT format (cast-builder writes via save_file(get_peft_model_state_dict()) without
            # convert_state_dict_to_diffusers). CPU unit-test stubs use fictional paths that are
            # never on disk, so they take the else branch and the fake pipe handles them as before.
            from safetensors.torch import load_file as _sf_load
            state_dict = _normalize_lora_state_dict(_sf_load(str(p)))
            pipe.load_lora_weights(state_dict, adapter_name=slot)
        else:
            pipe.load_lora_weights(str(path), adapter_name=slot)
        # diffusers silently loads ZERO modules when the safetensors keys do not match the pipe's
        # convention. Left unchecked the slot never registers and set_adapters below explodes with
        # an opaque "not in the list of present adapters: set()". The normalization above should
        # handle all known raw-PEFT variants; this guard catches anything genuinely unrecognized.
        if slot not in _adapter_names(pipe):
            raise ValueError(
                f"LoRA for slot {slot!r} ({path}) registered no adapter: its safetensors keys did "
                f"not match the diffusers convention after normalization. Refusing to render the "
                f"character without its identity adapter.")
        loaded.append(slot)
    # The distill base adapter rides at 1.0 on the few-step path and 0.0 (inert) on the full-step
    # path; any other base adapter always stays at 1.0. Set the active set whenever ANYTHING is on
    # the pipe (base and/or characters), so the distill weight tracks the tier even on a scene with
    # no character. A pipe with no base and no character (the bare no-identity scene) sets nothing.
    base_weights = [(1.0 if few_step else 0.0) if n == DISTILL_ADAPTER else 1.0 for n in base]
    names = base + loaded
    if names:
        pipe.set_adapters(names, adapter_weights=base_weights + [scale] * len(loaded))
    return loaded


def _adapter_names(pipe) -> list[str]:
    """Names of the LoRA adapters currently loaded on the pipeline, de-duplicated across its
    components. Empty when none are loaded."""
    try:
        listed = pipe.get_list_adapters()  # {component: [adapter_name, ...]}
    except Exception:
        return []
    seen: list[str] = []
    for names in listed.values():
        for n in names:
            if n not in seen:
                seen.append(n)
    return seen


def _ensure_ip_adapter(pipe, n: int = 1):
    """Bring the shared keyframe pipe to EXACTLY n IP-Adapter encoders (n=0 clears it).

    The pipe is process-global across scenes, so a prior regional scene can leave 2 encoders
    attached while a later single-character scene needs 1. set_ip_adapter_scale and the call-time
    ip_adapter_image must both match the loaded count, or diffusers raises ("Cannot assign 1
    scale_configs to 2 IP-Adapter"). So reload whenever the count differs in either direction (not
    only when fewer are present, the old `>= n` bug), and unload entirely for a no-character scene."""
    if getattr(pipe, "_vj_ip_loaded", 0) == n:
        return
    if getattr(pipe, "_vj_ip_loaded", 0):
        pipe.unload_ip_adapter()
        pipe._vj_ip_loaded = 0
    if n:
        from .models import DEFAULT_SPECS, ModelRole
        repo = DEFAULT_SPECS[ModelRole.IP_ADAPTER].repo_id
        from .load_events import timed_load
        with timed_load("keyframe.ip_adapter"):
            pipe.load_ip_adapter(repo, subfolder="sdxl_models",
                                 weight_name=["ip-adapter_sdxl.safetensors"] * n if n > 1 else "ip-adapter_sdxl.safetensors")
        pipe._vj_ip_loaded = n


def _ref_images(cast, slot, count: int):
    """The first `count` reference images for a slot, as PIL images (the IP-Adapter identity)."""
    from PIL import Image
    char = cast.characters.get(slot)
    if not char or not char.ref_paths:
        return []
    return [Image.open(p).convert("RGB") for p in char.ref_paths[:count]]


def _box_mask(width, height, box):
    """A white-in-box, black-elsewhere mask image for one region (the IP-Adapter attends only
    where the mask is white)."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def _blank_control(width, height):
    """Black control image. A ControlNet at conditioning_scale 0 ignores it, so the single /
    no-pose paths keep the shared ControlNet pipe behaving like plain SDXL."""
    from PIL import Image
    return Image.new("RGB", (width, height), (0, 0, 0))


# COCO-18 OpenPose layout (the format the xinsir SDXL OpenPose ControlNet follows). Keypoint order
# 0..17: nose, neck, R-shoulder, R-elbow, R-wrist, L-shoulder, L-elbow, L-wrist, R-hip, R-knee,
# R-ankle, L-hip, L-knee, L-ankle, R-eye, L-eye, R-ear, L-ear. `_POSE_KP` is one upright,
# front-facing standing figure as (x, y) normalized within a region box.
_POSE_KP = [
    (0.50, 0.10), (0.50, 0.20), (0.40, 0.22), (0.34, 0.38), (0.32, 0.54),
    (0.60, 0.22), (0.66, 0.38), (0.68, 0.54), (0.44, 0.54), (0.43, 0.74),
    (0.43, 0.94), (0.56, 0.54), (0.57, 0.74), (0.57, 0.94), (0.47, 0.085),
    (0.53, 0.085), (0.44, 0.095), (0.56, 0.095),
]
_POSE_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
               (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_POSE_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
                (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
                (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
                (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def pose_figure_shifts(width: int, boxes, spread: float) -> list[float]:
    """Horizontal shift of each region's pose figure, in pixels (vivijure-cf#1143).

    With `spread` 0 (or one box) every shift is 0: each figure stands at the centre of its region, as it always
    has. Otherwise the figures' centres are placed `spread * width` apart in total, evenly around the frame
    centre, which closes the empty band a wide frame leaves between two people (the plate drew a third person
    there). Each shift is clamped so the figure, which spans _POSE_KP's x range of its box width, stays inside
    its own region box: the masks, LoRAs and per-region text that key on those boxes still match the bodies."""
    n = len(boxes)
    if n < 2 or spread <= 0:
        return [0.0] * n
    xs = [x for x, _ in _POSE_KP]
    lo, hi = min(xs), max(xs)
    shifts = []
    for i, (l, _t, r, _b) in enumerate(boxes):
        bw = r - l
        target = width * (0.5 + spread * (i / (n - 1) - 0.5))
        centre = l + 0.5 * bw
        dx = target - centre
        dx = max(l - (l + lo * bw), min(r - (l + hi * bw), dx))  # keep [l+lo*bw+dx, l+hi*bw+dx] in [l, r]
        shifts.append(dx)
    return shifts


def _pose_skeleton(width, height, boxes, *, spread: float = 0.0):
    """Render an OpenPose skeleton with ONE standing COCO-18 figure per region box on black, so the
    ControlNet plants exactly that many distinct bodies (the fix for two characters merging into
    one). Each figure scales into its box with a small vertical margin, shifted sideways by
    `pose_figure_shifts` (none at `spread` 0). Geometry is deterministic; visual fidelity to the
    ControlNet's training distribution is validated on the pod."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (l, t, r, b), dx in zip(boxes, pose_figure_shifts(width, boxes, spread)):
        bw, bh = r - l, b - t
        my = int(bh * 0.04)  # top/bottom margin so head + ankles are not clipped at the box edge
        pts = [(l + x * bw + dx, t + my + y * (bh - 2 * my)) for (x, y) in _POSE_KP]
        lw = max(2, bw // 70)
        for i, (a, c) in enumerate(_POSE_LIMBS):
            draw.line([pts[a], pts[c]], fill=_POSE_COLORS[i % len(_POSE_COLORS)], width=lw)
        rad = max(2, bw // 90)
        for i, (px, py) in enumerate(pts):
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=_POSE_COLORS[i % len(_POSE_COLORS)])
    return img
