**Fix: `finish_clip` and `i2v_clip` use the process-global ModelServer; a diverging `i2v_clip` model config is now refused (#476).**

`run_finish_job` and `run_i2v_clip_job` built a fresh `ModelServer()` per job, so a warm worker
reloaded its models on every standalone job (paid GPU time), and `i2v_clip` ignored its
`config.model` / `distill_model`. Both now take the warm process-global server via
`worker.standalone_server()`. `i2v_clip`'s model config now takes effect on a cold worker.

**Behaviour change, an `i2v_clip` that used to succeed now fails.** On a warm worker whose loaded
i2v models differ from the job's `config.model` / `distill_model`, the job is refused with
`ModelDivergenceError`, the same refusal a render already got. Previously the foreign
`config.model` was silently ignored and the clip rendered on the loaded set. A refusal the caller
sees is deliberate: it beats a substitution the caller cannot see. Likewise a job-supplied i2v repo
id now passes `validate_repo_id` before use, so an invalid id is rejected where it used to be
parsed and dropped. `finish_clip` names no models and just reuses the singleton.
