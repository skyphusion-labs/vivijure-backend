**Fix: a stored LoRA the planner counts as trained is now staged for keyframing (#464).**

The planner skips training a slot whose adapter exists at `lora_key(project, slot)`, but only
`pretrained_loras` were staged into the pipeline, so on a later render the stored adapter was
counted as reused and never loaded. `run_job` now stages those stored adapters (SDXL family only)
when a keyframe will be drawn; an unfetchable adapter fails fast. A `pretrained_loras` entry for
the same slot supersedes the stored one.
