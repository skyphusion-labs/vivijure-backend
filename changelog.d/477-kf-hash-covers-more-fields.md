**Fix: `kf_hash` covers scheduler, identity scales, distill and distill_model (#479).**

`kf_hash` omitted `scheduler`, `ip_adapter_scale`, `instantid_ip_adapter_scale`, `distill` and
`distill_model`. Each changes the drawn keyframe (or the few-step path that draws it), so changing
only one of them reused the old PNG; `distill=False, steps=8` and `distill=True, distill_steps=8`
also hashed the same. All five are now in the hash.

**One-time cost.** The hash payload changed, so every keyframe cached before this change no longer
matches and is redrawn once on its next render (GPU seconds, once per cached shot). The bound
character LoRA is still not in the hash (#478).
