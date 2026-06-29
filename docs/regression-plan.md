# Vivijure Baked-Image Pod Regression Suite -- Design Plan

**Status:** PLAN -- awaiting Conrad/Mackaye sign-off before implementation.
**Scope:** extends `deploy/runpod_verify.py` (#131 harness) from a single-render smoke
to a FULL capability regression that gates both Phase C (serverless promote) and any
public "it works" claim.

---

## Why this suite exists

The current harness (`runpod_verify.py`) proves: GPU is visible, cu128 kernel loads on
this card, `.vj-baked` sentinel is present, weights are on disk, a single i2v draft render
completes within time bounds, and the output clip is non-empty. That is a SMOKE, not a
regression. It does NOT verify:

- That SDXL keyframe generation produces a valid PNG at the right dimensions.
- That the Wan2.2 i2v clip pointer carries all 6 required fields.
- That the RIFE interpolator (freshly vendored in C2) actually LOADS and interpolates
  with the correct c=90 IFNet architecture.
- That the finish path (GFPGAN face restore, audio mux) runs end-to-end.
- That a multi-shot storyboard completes without ordering or assembly defects.
- That the baked weights are at the correct precision (fp8).
- That no model file is missing from the baked layer.

Conrad's hard standard: no promote, no public claim, until all of these are exercised.
This document defines the test contract that satisfies it.

---

## 1. Capability Matrix

Each capability is a named check (`checks["CAP_N_..."]`). Every assertion is on the
structured `@event` channel -- NOT on English prose -- per the GMCP-style testability
philosophy. Wall-clock bounds are hard: if a capability exceeds its bound, the run fails
and the pod is stopped (not deleted) for SSH debug.

### CAP-1: Keyframe generation (SDXL)

**Input:** single-scene storyboard, minimal prompt ("cinematic mountain at dawn, 16:9").
**Path:** `keyframe.run_scene()` -> SDXL pipeline -> writes PNG to local temp.
**New event the pod entrypoint emits:**
```
@event keyframe_done {"shot_id": str, "key": str, "width": int, "height": int,
                      "format": "PNG", "bytes": int, "elapsed_s": float}
```
**Assertions:**
- `format == "PNG"`
- `width` and `height` match the configured SDXL aspect (e.g. 1280x720 or 1024x576);
  neither dimension is 0.
- `bytes >= 50_000` (not blank/degenerate)
- `elapsed_s <= 120.0`

**Failure path:** a blank or wrong-dimension keyframe fails the check; the SDXL pipeline
may silently produce a black frame on a misconfigured VAE, so byte count matters.

### CAP-2: i2v clip (Wan2.2 datacenter)

**Input:** keyframe from CAP-1.
**Path:** `i2v.run_scene()` with `distill=True, fp8=True` (draft tier, spend-bounded).
**New event:**
```
@event clip_done {"shot_id": str, "clip_key": str, "num_frames": int, "fps": int,
                  "seconds": float, "distilled": bool, "elapsed_s": float}
```
**Assertions (the 6-field pointer contract):**
- `clip_key` non-empty
- `num_frames >= 17` (minimum non-trivial clip; 1 frame = likely an encode error)
- `fps == 16` (Wan2.2 documented default)
- `seconds >= 1.0`
- `distilled == True` (proves the draft-tier path ran, not the full slow model)
- `elapsed_s <= 300.0`

### CAP-3: RIFE interpolation (the C2 vendored package) -- MUST exercise

This is the highest-priority check. RIFE HDv3 was just re-hosted from a canonical licensed
source; it MUST be exercised to prove the vendored package loads and interpolates correctly
against the flownet.pkl weights that live in R2 (or in the baked layer if pre-baked there).

**Input:** any two adjacent frames extracted from the CAP-2 clip.
**Path:** `models.RIFEInterpolator.__init__()` -> `interpolate(frame_a, frame_b)`.

**Two events the pod entrypoint emits:**

Architecture probe (emitted at model load time):
```
@event rife_model_probe {"block_count": int, "c_per_block": int,
                         "flownet_pkl_bytes": int, "loaded": bool}
```
Interpolation result:
```
@event rife_done {"shot_id": str, "input_frames": 2, "output_frames": int,
                  "factor": int, "h": int, "w": int, "elapsed_s": float}
```

**Assertions:**
- `rife_model_probe.loaded == True`
- `rife_model_probe.block_count == 3` (IFNet has block0/block1/block2, all IFBlock c=90)
- `rife_model_probe.c_per_block == 90` (the architecture the flownet.pkl weights expect)
- `rife_model_probe.flownet_pkl_bytes > 0`
- `rife_done.output_frames == 3` (factor=2, 2 input frames -> 3 output: A, midpoint, B)
- `rife_done.h` and `rife_done.w` match the CAP-2 clip dimensions (padding stripped)
- `rife_done.elapsed_s <= 60.0`

**How to emit `rife_model_probe`:** after `Model.load_model()`, introspect
`model.flownet` (the `IFNet` instance): count `IFBlock` children whose constructor
parameter `c` equals 90. This is a structural assertion on the vendored code, not a
string match on a file name.

### CAP-4: Finish path (RIFE + GFPGAN + encode)

**Input:** clip from CAP-2.
**Path:** `finish.finish_clip()` with `FinishParams(interpolate=True, factor=2,
face_restore=True, face_restore_backend="gfpgan")`.
**New event:**
```
@event finish_done {"shot_id": str, "clip_key": str, "interpolated": bool,
                    "face_restored": bool, "out_frames": int, "out_fps": int,
                    "bytes": int, "elapsed_s": float}
```
**Assertions:**
- `interpolated == True`
- `face_restored == True`
- `out_fps == 32` (16fps input, factor=2 -> 32fps output)
- `out_frames >= 33` (17 input frames * 2 - 1 minimum)
- `bytes >= 100_000`
- `clip_key` ends in `_finished.mp4`
- `elapsed_s <= 180.0`

**Note on "16:9 framing":** the assemble/crop step targets 16:9 output; assert
`out_w / out_h` is within 2% of 1.778 if the codec probe is practical to add. Flag
as a "nice-to-have assertion" -- not a hard gate blocker if aspect probing adds >10s.

### CAP-5: LoRA apply (#280 Wan2.2 adapter)

**COVERAGE GAP -- skipped for Phase C gate.**

The baked image carries no LoRA adapter (they are user-supplied at runtime). Issue #280
tracks the Wan2.2 LoRA adapter; until a validated adapter is part of the baked layer,
this capability cannot be tested on the pod. The regression report records:
```json
{"coverage_gap": {"lora_apply": "skipped -- #280 open, no baked adapter"}}
```
This gap does NOT block Phase C. It IS a blocker for any public claim that "LoRA works."

### CAP-6: End-to-end multi-shot

**Input:** 2-shot storyboard, each shot a distinct simple prompt, 2s duration (distill tier).
**Path:** full pipeline: keyframe -> i2v -> finish -> assemble (with a short synthetic audio
track or null audio). The assemble step exercises shot ordering + ffmpeg concat + audio mux.
**New event:**
```
@event e2e_done {"shots": int, "output_key": str, "has_audio": bool,
                 "duration_s": float, "bytes": int, "elapsed_s": float}
```
**Assertions:**
- `shots == 2`
- `bytes >= 200_000`
- `has_audio == True` (even a synthetic 440Hz sine is sufficient; proves the mux path)
- `duration_s >= 2.0`
- `elapsed_s <= 600.0` (hard 10-min bound for the full 2-shot run)

---

## 2. Baked-image Correctness Checks

These run BEFORE any capability test (order matters: a baked-sentinel miss means weights
came from R2 and the pod should be stopped immediately before spending GPU time).

### BAK-1: Sentinel present (`vj_baked`)
Already in the existing harness via `@event gpu_probe`. No change needed.

### BAK-2: No R2 mirror leg at runtime
Already in the existing harness: `mirror_skipped` event with `reason == "baked"` must
appear; `mirror_complete` must NOT (or its `total_seconds` must be <= the staging bound).
No change needed.

### BAK-3: Model inventory -- all files present
**New event (emitted by the pod entrypoint before any render):**
```
@event model_inventory {"sdxl": bool, "wan22": bool, "rife_flownet": bool,
                        "gfpgan": bool, "all_present": bool}
```
Expected paths (resolved from `VJ_MODELS_ROOT`):
- `sdxl`: `hf-cache/hub/` directory non-empty (SDXL model files present)
- `wan22`: the Wan2.2 model directory under `hf-cache/`
- `rife_flownet`: `rife/flownet.pkl` exists and `size > 0`
- `gfpgan`: GFPGAN weight file under the expected path

**Assertions:** `all_present == True`. Any `False` field is named in the failure reason.

### BAK-4: FP8 precision
**New event:**
```
@event model_precision {"i2v_dtype": str}
```
Where `i2v_dtype` is the string representation of the loaded i2v model's parameter dtype.
**Assertion:** `i2v_dtype in {"float8_e4m3fn", "bfloat16"}` -- fp8 or the bf16 fallback
(both are valid baked precisions; pure fp32 is not). If fp8 is expected and bfloat16 is
seen, log a warning but do not fail (precision fallback is non-fatal; a hard fp8-only
assertion would be wrong if the baked image legitimately ships bf16 weights).

### BAK-5: VRAM headroom
Already in the existing harness: `vram_free_gb >= 8.0` after model load. No change needed.

---

## 3. Assertion Discipline

- **Every assertion targets a named field in a parsed `@event` payload.** The harness
  calls `find_event(events, name)` and checks specific keys -- never `"passed" in log_text`.
- **Artifact size checks** (`bytes > threshold`) are the backstop for blank/degenerate output
  where the pipeline does not error but produces nothing useful. They are NOT a proxy for
  quality; sharpness_parity (already in the harness) handles that.
- **Timing bounds** are HARD: exceed them and the check fails. They are sized at 3x the
  expected median on an H100 to absorb normal cold-start variance without being trivially
  loose. A pod that times out at the TTL wall (`timed_out == True`) also fails regardless
  of which checks passed.
- **No silent skip:** every coverage gap is recorded in the report (`coverage_gaps` dict).
  A gap that silently passes as "untested" would violate the #249/#77 degrade discipline.

---

## 4. Pass/Fail Criteria + Cost/Time Budget

### Pass criteria (ALL must hold)
1. `timed_out == False`
2. All `checks` entries are `True`
3. `model_inventory.all_present == True` (fail-fast: emit this before any render)
4. `mirror_skipped` event with `reason == "baked"` present; `mirror_complete` absent or
   staging within bound.

Coverage gaps (CAP-5 LoRA) are RECORDED in the report but do NOT flip `passed` to False.

### Cost / time budget

| Phase | Expected | Hard bound |
|---|---|---|
| Pod cold start + CUDA init | 2-5 min | -- |
| BAK checks (inventory + probe) | 30s | -- |
| CAP-1 keyframe (SDXL) | 20-45s | 120s |
| CAP-2 i2v clip (distilled) | 45-90s | 300s |
| CAP-3 RIFE load + interpolate | 5-15s | 60s |
| CAP-4 finish (interp + GFPGAN) | 30-60s | 180s |
| CAP-6 e2e 2-shot (full pipeline) | 4-8 min | 600s |
| **Total expected** | **12-18 min** | -- |
| **Hard TTL (pod wall-clock)** | -- | **30 min (1800s)** |

GPU tier: H100 80GB HBM3 (first preference) or H200 (if available; same `i2v` tier as
the current harness). Cost estimate at the hard TTL:
- H100 @ $2.69/hr: 30-min cap = **$1.35 max**; expected run ~**$0.90**

These estimates are printed in the run report (conservative-high, same as the existing
`cost_estimate_usd` function). If no H100/H200 is available at spin time, the harness
aborts before spending anything and emits `spun: false`.

---

## 5. Extension to `deploy/runpod_verify.py`

The regression suite extends the #131 harness. No forking -- one file, one contract.

### New exports

**`RegressionConfig(VerifyConfig)`** -- subclass adding:
```python
max_keyframe_seconds: float = 120.0
max_clip_seconds: float = 300.0
max_rife_seconds: float = 60.0
max_finish_seconds: float = 180.0
max_e2e_seconds: float = 600.0
min_keyframe_bytes: int = 50_000
min_clip_bytes: int = 100_000   # finish output
min_e2e_bytes: int = 200_000
expected_rife_block_count: int = 3
expected_rife_c: int = 90
```

**`evaluate_regression(events, cfg)`** -- calls `evaluate(events, cfg)` for the existing
checks, then adds the CAP/BAK checks. Returns an extended `VerifyResult`. Every new check
is added to the same `checks` dict under a stable key (e.g. `"cap1_keyframe_format"`,
`"cap3_rife_architecture"`, `"bak3_all_models_present"`).

**`REGRESSION_EVENTS`** -- the complete ordered list of event names a full regression
run must emit. Used by tests to build a minimal mock stream.

### Pod-side entrypoint changes

The pod entrypoint (currently in the worker module or `smoke_imports.py`) gains a
`VJ_REGRESSION=1` env flag. When set:
1. Emits `model_inventory` and `model_precision` before any render.
2. Runs the existing gpu_probe + mirror check.
3. Runs CAP-1 through CAP-4 sequentially (each emitting its `@event`).
4. Runs CAP-6 (2-shot e2e).
5. Emits the final `@event complete` the existing harness waits for.

The `VerifyConfig.env` dict gains `"VJ_REGRESSION": "1"` when a `RegressionConfig` is
used, so the harness activates it automatically.

### Test additions

`tests/test_regression.py` (new file):
- `test_evaluate_regression_pass`: full mock event stream -> all checks True.
- `test_evaluate_regression_rife_architecture_fail`: `block_count=2` -> `cap3_rife_architecture` False.
- `test_evaluate_regression_baked_r2_pull`: `mirror_complete` present with high staging -> `baked_no_r2_pull` False.
- `test_evaluate_regression_model_missing`: `model_inventory.all_present=False` -> `bak3_all_models_present` False.
- `test_evaluate_regression_coverage_gap`: `coverage_gaps.lora_apply` recorded; `passed` still True.

All tests are CPU-only (no GPU, no pod, no network) -- same pattern as the existing
`tests/test_runpod_verify.py`.

---

## 6. What the Suite Gates

### Phase C: serverless promote (HARD GATE)
`report["signal"] == "promote"` (i.e. `passed == True`) on the baked image tag is
required before Strummer's promote script runs. No other document or human judgment
substitutes. The report JSON is the artifact.

### Public "it works" claim (HARD GATE)
No release notes, documentation, Discord announcement, or marketing text may claim
"vivijure works" / "renders work" / "RIFE interpolation works" for a given image tag
unless that tag's regression report shows `passed == True` with zero unresolved
failures. Coverage gaps (LoRA) are explicitly called out as "not yet verified."

### LoRA claim (SEPARATE GATE, blocked by #280)
Any claim that LoRA adapters work requires a regression run that passes CAP-5 with a
validated adapter. CAP-5 is not in scope for the current baked-image sprint.

---

## Open questions / items requiring Conrad sign-off

1. **16:9 framing assertion**: is a codec-level aspect-ratio probe (ffprobe on the
   finished clip) required for Phase C, or is it deferred? Adds ~10s of probe work per clip.
2. **FP8 hard-fail vs warn**: should `i2v_dtype != "float8_e4m3fn"` fail the run or
   warn? Currently proposed as warn. If Conrad wants a hard fp8 gate, change to fail.
3. **Audio in CAP-6**: is a synthetic 440Hz sine track acceptable as the audio source to
   prove the mux path, or does the regression need a real audio file asset?
4. **CAP-4 GFPGAN face restore**: GFPGAN requires a face in the frame to restore. If the
   draft-tier i2v clip produces no detectable face (abstract prompt), `face_restored` may
   be False not due to a bug but due to no face. Mitigation: use a portrait prompt for
   CAP-4 specifically ("close-up portrait of a person, cinematic"). Flag for decision.
