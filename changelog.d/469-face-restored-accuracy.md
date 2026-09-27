**Fix: `face_restored` is reported only when a frame was actually restored (#470).**

`finish_clip` set it True unconditionally, even when the restorer failed on every frame (each
failure passes the raw frame through by design), so the result, `applied` list and regression
checks claimed a restore that did not happen. `face_restored` is now true only if at least one
frame was restored, and an all-failed clip prints an `@event face_restore_degraded` line. Per-frame
best-effort behaviour is unchanged.
