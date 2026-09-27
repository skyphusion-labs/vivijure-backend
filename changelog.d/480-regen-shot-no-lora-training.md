**Fix: `regen_shot` trains no LoRAs, per the contract (#482).**

`docs/contract.md` says `regen_shot` trains no LoRAs, but the planner trained every slot that was
neither pretrained nor already trained, on a paid GPU, before redrawing keyframes. It now zeroes
`to_train` for `regen_shot` (as it already does for `finalize`) and records the skip.

**Behaviour change.** A `regen_shot` on a project whose slot has no adapter available now draws
that shot without the character LoRA (IP-Adapter identity fallback, recorded on the
`keyframe_done` event) instead of training one first. A LoRA trained by an earlier render is used
as before.
