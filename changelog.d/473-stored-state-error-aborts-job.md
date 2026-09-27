**Fix: a stored-state existence error aborts the job instead of retraining (#475).**

Since #460 `R2.exists()` raises on anything but a real not-found, but `_restore_prior_state` still
absorbed the error and treated it as "nothing stored", so every LoRA was retrained and every
keyframe redrawn on a paid GPU before the same error failed the final upload. Any store error
during the existence probes now fails the job with a `HarnessError` before GPU work starts; a
genuine not-found still restores to empty and renders fresh. Failing to fetch a keyframe or its
hash sidecar after existence is confirmed keeps its best-effort behaviour (regenerate).
`docs/operations.md` and `docs/architecture.md` updated to match.
