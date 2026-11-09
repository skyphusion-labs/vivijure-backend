"""vivijure-backend#478: the keyframe reuse hash covers which character LoRA each slot binds.

Lead ruling (option 2): `kf_hash` takes `bound_loras` ({slot: adapter key}), so a repointed, added or
removed LoRA changes the hash and the still is redrawn. One helper (`bound_lora_keys`) builds the map; the
planner hashes with it and stores it on the RenderPlan, and the pipeline's sidecar writer reads it from the
plan. The default hash (no LoRA bound) is pinned against main, so stills drawn without a LoRA stay reusable.
Accepted limit: a retrain at the same key is not caught."""
import pathlib
import re

from vivijure_backend.contract import Cast, RenderRequest, Storyboard
from vivijure_backend.harness.keys import lora_key
from vivijure_backend.orchestrator import KeyframeMode, LoraPlan, bound_lora_keys, kf_hash, plan

CAST = Cast.from_registry({"characters": {"A": {"name": "Mara", "prompt": "dark curls"}, "B": {"name": "Ode", "prompt": "grey beard"}}})
SB = Storyboard.from_dict({"title": "p", "use_characters": ["A", "B"], "scenes": [
    {"id": "shot_01", "prompt": "Mara frowns at a gauge.", "character_slots": ["A"]},
    {"id": "shot_02", "prompt": "Mara and Ode argue.", "character_slots": ["A", "B"]},
    {"id": "shot_03", "prompt": "An empty corridor.", "character_slots": []},
]})
SC = {s.id: s for s in SB.scenes}
KA = "loras/p/A/pytorch_lora_weights.safetensors"


def _kc():
    return RenderRequest.from_dict({"action": "render", "project": "p", "bundle_key": "b"}).config.keyframe


def _h(shot, bound=None):
    return kf_hash(_kc(), scene=SC[shot], cast=CAST, storyboard=SB, bound_loras=bound)


# Read from main at 6476358 (pre-#478) with the same inputs. If one of these moves, every stored still
# drawn without a LoRA redraws fleet-wide.
PINNED_MAIN = {"shot_01": "e21b0738ed5966c7", "shot_02": "9a38ccd7d2575efb", "shot_03": "5adba591d8de7da1"}


def test_the_default_hash_is_pinned_against_main():
    assert kf_hash(_kc()) == "bc7df897df3558e8"
    for shot, want in PINNED_MAIN.items():
        assert _h(shot) == want
        assert _h(shot, {}) == want


def test_a_lora_for_a_slot_the_shot_does_not_hold_keeps_its_hash():
    assert _h("shot_03", {"A": KA, "B": "loras/p/B/x.safetensors"}) == PINNED_MAIN["shot_03"]
    assert _h("shot_01", {"B": "loras/p/B/x.safetensors"}) == PINNED_MAIN["shot_01"]


def test_binding_a_lora_changes_the_hash_of_the_shots_that_slot_is_in():
    assert _h("shot_01", {"A": KA}) != PINNED_MAIN["shot_01"]
    assert _h("shot_02", {"A": KA}) != PINNED_MAIN["shot_02"]


def test_a_repointed_or_removed_lora_changes_it():
    a = _h("shot_02", {"A": KA, "B": "loras/p/B/one.safetensors"})
    assert _h("shot_02", {"A": KA, "B": "loras/p/B/two.safetensors"}) != a   # repointed
    assert _h("shot_02", {"A": KA}) != a                                       # removed
    assert _h("shot_02", {"A": KA, "B": "loras/p/B/one.safetensors"}) == a     # same map, same hash


def test_limit_a_retrain_at_the_same_key_is_not_seen():
    # Accepted on #478: the key is the identity, so a new adapter at the same key hashes the same.
    assert _h("shot_01", {"A": KA}) == _h("shot_01", {"A": KA})


def test_bound_lora_keys_binds_what_the_draw_binds():
    lora = LoraPlan(train=["A"], reuse=["B", "C"])
    got = bound_lora_keys("my proj", lora, trained_slots={"B"}, pretrained_loras={"C": "loras/my_proj/C/custom.safetensors"})
    assert got == {
        "A": lora_key("my proj", "A"),                 # trained this run: uploaded to the project key
        "B": lora_key("my proj", "B"),                 # reused from R2
        "C": "loras/my_proj/C/custom.safetensors",     # pretrained, as staged
    }


def test_a_pretrained_entry_supersedes_the_stored_adapter():
    got = bound_lora_keys("p", LoraPlan(train=[], reuse=["A"]), trained_slots={"A"}, pretrained_loras={"A": "loras/p/A/v2.safetensors"})
    assert got == {"A": "loras/p/A/v2.safetensors"}


def test_a_wan_job_binds_only_pretrained_adapters():
    got = bound_lora_keys("p", LoraPlan(train=["A"], reuse=["B"]), trained_slots={"B"}, pretrained_loras={"C": "loras/p/C/x.safetensors"}, lora_family="wan")
    assert got == {"C": "loras/p/C/x.safetensors"}


def _plan(existing, trained=frozenset(), pretrained=None):
    req = RenderRequest.from_dict({"action": "preview", "project": "p", "bundle_key": "b", "pretrained_loras": pretrained or {}})
    return plan(req, SB, trained_slots=set(trained) | set(req.pretrained_loras), existing_keyframes=existing, cast=CAST)


def test_the_plan_stores_the_map_and_reuses_only_a_still_hashed_with_it():
    p0 = _plan({}, trained={"A", "B"})
    assert p0.bound_loras == {"A": lora_key("p", "A"), "B": lora_key("p", "B")}
    # The hash the pipeline writes beside a drawn still: the same function over the plan's own map.
    written = {s: kf_hash(_kc(), scene=SC[s], cast=CAST, storyboard=SB, bound_loras=p0.bound_loras) for s in SC}
    p1 = _plan(written, trained={"A", "B"})
    assert {s.shot_id: s.keyframe_mode for s in p1.scenes} == {s: KeyframeMode.REUSE for s in SC}


def test_a_still_from_before_478_redraws_once_where_a_lora_is_bound_and_is_kept_where_none_is():
    old = dict(PINNED_MAIN)   # what main wrote: no LoRA in the hash
    p = _plan(old, trained={"A", "B"})
    assert {s.shot_id: s.keyframe_mode for s in p.scenes} == {
        "shot_01": KeyframeMode.GENERATE, "shot_02": KeyframeMode.GENERATE, "shot_03": KeyframeMode.REUSE}


def test_repointing_a_pretrained_lora_redraws_its_shots_only():
    p0 = _plan({}, pretrained={"B": "loras/p/B/one.safetensors"})
    written = {s: kf_hash(_kc(), scene=SC[s], cast=CAST, storyboard=SB, bound_loras=p0.bound_loras) for s in SC}
    p1 = _plan(written, pretrained={"B": "loras/p/B/two.safetensors"})
    assert {s.shot_id: s.keyframe_mode for s in p1.scenes} == {
        "shot_01": KeyframeMode.REUSE, "shot_02": KeyframeMode.GENERATE, "shot_03": KeyframeMode.REUSE}


def _calls(text: str, name: str):
    """Each call of `name(` in `text`, as its full argument text (balanced parentheses)."""
    out = []
    for m in re.finditer(rf"(?<!def )\b{name}\(", text):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        out.append(text[m.end():i - 1])
    return out


def test_every_kf_hash_call_in_src_passes_the_one_map():
    """The writer and the reader must compute the identical value, so both pass bound_loras, and the
    pipeline's writer passes the plan's map rather than building its own."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "vivijure_backend"
    calls = [(f.name, args) for f in sorted(src.rglob("*.py")) for args in _calls(f.read_text(), "kf_hash")]
    assert sorted(n for n, _ in calls) == ["orchestrator.py", "pipeline.py"]
    assert all("bound_loras=" in args for _, args in calls), calls
    assert [args for n, args in calls if n == "pipeline.py"][0].rstrip().endswith("bound_loras=plan.bound_loras")


def test_control_the_call_scanner_sees_a_call_without_the_map():
    text = "x = kf_hash(kc, scene=s)\ny = kf_hash(kc, scene=f(s), bound_loras=b)\ndef kf_hash(kc):\n"
    assert _calls(text, "kf_hash") == ["kc, scene=s", "kc, scene=f(s), bound_loras=b"]
