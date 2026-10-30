**Fix: a still with no hash is redrawn, never reused (#488).**

A job used to reuse any still at the flat key that had no hash sidecar.
film-a9939630 asked for SDXL, drew nothing, and returned a cloud door's stills as COMPLETED.
Reuse now needs a matching hash.
Old GPU stills with no sidecar are redrawn once.
A reused still is reported with `reused: true`, so the studio can stop stamping it.
