**Fix: `ensure_models` carries the network-volume repoint to the real environment (#484).**

`ensure_models` repointed `VJ_MODELS_ROOT` and `HF_HOME` on a copy of the environment, so the
deferred model loads (which read `HF_HOME` at call time) never used the volume. When a volume
resolves, both keys are now written back to the environment it was asked to use (the caller's dict
when `env` is passed, `os.environ` otherwise). No change when no volume resolves; latent today
because no endpoint sets `VJ_VOLUME_ROOT`.
