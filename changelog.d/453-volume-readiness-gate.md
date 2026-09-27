**Fix: the network-volume readiness gate asks what THIS job needs, and names what it found (#453).**

`_resolve_volume` required both the base and the i2v sentinel unconditionally. Conrad retired the
local i2v path on 2026-09-26, so a volume preloaded with the honest post-ruling corpus (keyframes,
no Wan; the i2v repos were 60% of the 107.8 GB weight set) has no i2v sentinel. That volume was
therefore refused on every call, forever: no error, every worker falling back to the R2 mirror, and
the only symptom a cold-start cost that reads as a volume nobody configured.

The i2v sentinel is now required only when the job's action can actually reach the Wan i2v stage
(`models_mirror.i2v_stage_possible`, fed by `handler` from the job's own action; no new endpoint
knob). The base sentinel is still always required, and a volume missing a sentinel the job NEEDS
still degrades WHOLESALE to the R2 mirror rather than being repointed -- the guard against a
half-preloaded volume producing mysterious i2v failures is unchanged, only its premise moved.

`_resolve_volume` now returns a named `VolumeState` (`ABSENT`, `READY`, `READY_NO_I2V`,
`INCOMPLETE`) instead of a bare bool, so a monitor can tell "the i2v corpus was deliberately
retired" from "the preload did not finish" without parsing the log line; the two are identical on
disk and used to render as one reassuring reading. The structured skip event carries the same
split (`"reason": "volume"` vs `"reason": "volume_no_i2v"`).

Because readiness is now a per-job question while the `HF_HOME` / `VJ_MODELS_ROOT` repoint is
process-global, a decline also takes back a repoint an earlier job on the same warm worker made;
otherwise a `render` following a `preview` would inherit the keyframe-only volume, skip the R2 pull
and try to mirror Wan onto the shared volume at i2v time.

NOT settled here, and it needs spend, not code: whether a RunPod global network volume preserves
the symlinks the HF cache corpus is built from (`deploy/bake_layers.py`). The vendor docs enumerate
the POSIX semantics a network volume lacks and say nothing about symlinks. That needs one real
volume and one probe; tracked on `skyphusion-labs/fleet-chezmoi#2087`.
