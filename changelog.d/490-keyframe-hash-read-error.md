**Fix: an unreadable keyframe hash sidecar no longer reads as "no sidecar" (#491).**

`_restore_prior_state` read each stored keyframe's `.hash` sidecar inside a bare `except`, so a 403,
an expired credential, a throttle or a transport error stored `None`, which the planner reads as
"reuse the cached keyframe conservatively": a keyframe drawn with different parameters was served
as current. Only a real not-found (the classification `R2.exists()` uses, now `is_not_found`)
still means "no sidecar". Any other error now raises `HarnessError`, so the job fails before GPU
work, as the LoRA and keyframe existence checks already do (#460, #475). Behaviour change: a render
whose sidecar read fails with anything but not-found previously succeeded (with a possibly stale
keyframe) and now fails.
