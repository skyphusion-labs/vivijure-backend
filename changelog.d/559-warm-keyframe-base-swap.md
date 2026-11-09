**Fix: a warm worker swaps to a job's keyframe base instead of refusing it, and a base not on disk no longer poisons the worker (#559).**

The process-global `ModelServer` froze the first job's keyframe base, so `_check_model_divergence`
refused every later job with another `style` or `base_model` for the life of the worker. A cold
anime job (animagine is not on the GPU image, #557) also failed on missing weights while leaving
its spec loaded, so that worker then refused every photoreal job after it.

`build_pipeline` now calls `ModelServer.swap_keyframe_specs` first. A new base that is on local disk
swaps in: only the `keyframe` and `instantid` pipes are dropped, they reload on the next draw, and
`@event keyframe_model_swap` names the change. A new base that is not on disk raises the existing
missing-weights error before anything changes, so the loaded pipes stay warm. The repo id is
validated before either. An i2v divergence is still refused with `ModelDivergenceError`.

**Cost:** one SDXL base reload from local disk per change of base on a warm worker, where the job
used to be refused.
