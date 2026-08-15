# The finish_clip wall-clock guard (#422)

One monotonic budget bounds a whole `finish_clip` invocation on the RunPod worker. This page is the
coverage claim: which steps are inside the budget, which are not, and what the number was argued
against. A guard whose coverage is not written down invites someone downstream to relay a partial
number as an invocation ceiling, which is an under-claim nothing detects.

Code: [`finish.py`](../src/vivijure_backend/finish.py) (`Deadline`, `FinishDeadlineExceeded`,
`max_finish_seconds`) and [`harness/handler.py`](../src/vivijure_backend/harness/handler.py)
(where the budget is opened and where the expiry is caught). Tests:
[`tests/test_finish_deadline.py`](../tests/test_finish_deadline.py).

## What it does

`Deadline.start()` opens the budget on the FIRST line of a `finish_clip` invocation, in
`harness.handler.handler`, so the R2 client build and the cold-start model mirror are inside it and
the number is an invocation ceiling rather than a stage total. One shared deadline, not a timeout
per step: per-step timeouts multiply, and the invocation bound is then a sum nobody computes.

On expiry the handler RETURNS

```json
{"ok": false, "detail": "finish_clip deadline: 431.2s of 420s budget at stage encode"}
```

and never raises. A raise leaves no structured output, classifies deterministic in `vivijure-core`
and fails the WHOLE render after the GPU spend is banked -- strictly worse than the unbounded hang
it replaces, which the phase ceiling at least recovers.

The shape is the consumer contract, `vivijure-cf` `modules/_shared/finish-soft-degrade.ts` (read at
`219cd6b6352a0535a64d76fc228c41fd6c43ee06`):

- `ok: false` is the entire discriminator. `softDegradeInCompletedOutput` requires a structured
  object output whose `ok` is exactly `false`. A genuine crash leaves no structured output and keeps
  failing loud, so this widens nothing.
- `detail` is the FIRST key its `degradeReason` reads, and it slices to 120 characters there.
- NO top-level `error`. RunPod lifts a top-level `error` in a handler RETURN into a job-level FAILED
  envelope; `detail` keeps the envelope COMPLETED and the output arrives intact.

## Configuring it

`VJ_FINISH_MAX_SECONDS`, seconds, default `420`. Junk, an empty string, zero and negatives all fall
back to the default. There is deliberately NO value that turns the guard off: a guard an operator
can disable by typo is the defect this closes.

## Coverage: 13 of 14 compute steps are inside the budget

Enumerated end to end, in execution order. "Interruptible" is the honest column: a step can be
INSIDE the budget and still not be cuttable mid-flight, and the difference is what decides whether
the ceiling is a guarantee or a strong bound.

| # | Step | Where | In budget | Interruptible mid-step |
|---|------|-------|-----------|------------------------|
| 1 | R2 client build | `handler` | yes | edge only (local, no network) |
| 2 | Cold-start model mirror `ensure_models` | `handler` | yes | edge only (no-ops on the baked image) |
| 3 | R2 GET of the input clip | `run_finish_job` | yes | edge only (botocore socket timeouts) |
| 4 | Container probe `iio.immeta` | `finish_clip` | yes | edge only (header read) |
| 5 | Frame decode `iio.imiter` | `finish_clip` | yes | YES, per decoded frame |
| 6 | Face-restorer model load | `finish_clip` | yes | edge only |
| 7 | Face restore loop | `_restore_clip` | yes | YES, per frame |
| 8 | RIFE interpolator model load | `finish_clip` | yes | edge only |
| 9 | Audio probe (ffprobe) | `_source_has_audio` | yes | YES, subprocess timeout |
| 10 | RIFE interpolation | `_finished_stream` | yes | YES, per source pair |
| 11 | ffmpeg encode (pipe + wait) | `_encode_uniform` | yes | YES, watchdog kills the process |
| 12 | ffmpeg audio mux | `_mux_audio` | yes | YES, subprocess timeout |
| 13 | R2 PUT of the finished clip | `run_finish_job` | yes | edge only (botocore socket timeouts) |
| 14 | R2 PUT of the `.hash` sidecar | `run_finish_job` | **NO** | deliberately outside |

**13 of 14 inside. 6 of those 13 are interruptible mid-step** (3 hard-killed subprocesses, 3
per-iteration loops); the other 7 are checked at their edges only.

Step 14 is the one deliberate exclusion: the provenance sidecar is already best-effort, it runs
AFTER the artifact is uploaded, and a miss only disables reuse (the core re-runs). Bounding it would
buy nothing and could only turn a free miss into a degrade.

### The residual, stated plainly

The seven edge-only rows mean the budget is a STRONG BOUND and not a hard guarantee. Two of them
matter in practice:

- **R2 GET and PUT** are bounded by botocore socket timeouts, not by this guard.
  [`harness/r2.py`](../src/vivijure_backend/harness/r2.py) sets `retries={"max_attempts": 5}` and
  leaves connect/read timeouts at the botocore defaults. Tightening them would change the render,
  train and i2v paths too, so it is out of scope here.
- **The two model loads** read baked weights into the GPU. A wedged CUDA context inside one is not
  interruptible from Python.

There are also two hangs the per-iteration rows do NOT cover, because a Python-level check only sits
BETWEEN units of work: a decoder that stalls inside a single frame (row 5), and a single
`interp.interpolate` call that never returns (row 10).

**Consequence: this door must NOT declare `max_invocation_seconds` on the strength of this guard
alone.** That field lives in the module MANIFEST in `vivijure-cf`, not in this repo, so this change
could not declare it in any case; the point is that a future declaration has to answer the rows
above first (`vivijure-core#223`).

## Why 420 seconds

Three bounds. The default clears all three, and the first two are executable assertions in
`tests/test_finish_deadline.py` so a later bump cannot quietly break either.

### 1. The platform ceiling, which is NOT settled

Measured 2026-08-15 through the RunPod API. Endpoint `t9wcvlxh8rc5la` (`vivijure-backend`,
`type: QUEUE`) reports `timeout: 0`. The v2 schema calls that field "Per-request execution timeout
in milliseconds" and declares no minimum and no default, so the schema does not settle it either.

The two docs pages disagree, and they describe DIFFERENT product surfaces:

- `serverless/endpoints/endpoint-configurations`: execution timeout default 600s, range 5s to 7
  days. This is the queue-endpoint setting, which is what this endpoint is.
- `flash/configuration/parameters`: `execution_timeout_ms` default `0` = no limit. This is the Flash
  SDK, a different surface; this endpoint was not created through it.

`0` is below the documented 5s floor, so it cannot be a value set through the console; it reads as
"never configured". Whether unset resolves to the 600s default or to no enforcement is the part
that is open.

**Control for the reading** (so the `0` is a measurement and not a dead field): all 6 endpoints on
the account were read in one call. Five report `timeout: 0`; `vivijure-blender`
(`0uc4dmpxmn8jop`, created 2026-08-07) reports `timeout: 600000`. So `0` is a distinct stored state
and the field is capable of reporting a real value. That control does NOT settle the semantics.

**Settling it empirically would need a deliberately hanging GPU job on a live endpoint**, which is
spend and an observed firing. Not done here, and not needed: staying strictly under 600s makes the
guard fire under BOTH readings. Under the pessimistic reading a guard at or above 600 could never
fire and would be decoration, and a platform kill is a FAILED envelope with no structured output --
a failed render rather than a degrade. 420 leaves 180s of margin.

### 2. The core retry arithmetic

`vivijure-core` (`src/film-model.ts`, read at `1efaae3aac0886e8ef4cc7607c9e58082f9038cb`):
`FINISH_STEP_MAX_ATTEMPTS = 3` and `PHASE_HARD_DEADLINE_SECONDS = 5400`. A retry moves the
`attempts` counter and never the `idx` progress marker, so a guard of G costs up to `3 * G` of wall
clock with no marker movement.

Core does not CAP at 5400: `film-model.ts` takes
`max(PHASE_HARD_DEADLINE_SECONDS, FINISH_STEP_MAX_ATTEMPTS * longest)`, so 5400 is a FLOOR that a
large declared ceiling RAISES. `3 * 420 = 1260s`, which is 23% of the floor and leaves it untouched;
a default above 1800 would silently extend every film phase containing this door.

### 3. The work itself

The largest clip that can reach this stage is 256 frames (`config.py` clamps `num_frames` to
1..256; the default is 81, five seconds at 16 fps). The worst legal configuration is 8x
interpolation with face restore: 256 restores plus `255 * 7 = 1785` RIFE midpoints, on the
`BLACKWELL_180` / `HOPPER_141` pools this endpoint runs. That is minutes, not hours. 420s is
multiples of the worst legal case and roughly an order of magnitude above the default one.

## Known limits

- **The passthrough tag is generic.** The cf module renders every backend soft degrade as the single
  literal `passthrough:backend-soft-degrade`, so the CAUSE is not machine-visible downstream; the
  reason string only becomes the human note in `FinishOutput.degraded`, which nothing reads at the
  clip level. Tracked as `vivijure-core#226` (which is what the cf source cites) and `cf#595`. Not
  fixable from this repo.
- **The platform ceiling is unsettled**, as above. Recorded as a measured field, not a conclusion.
- **No `max_invocation_seconds` declaration** follows from this change; see the residual section.
