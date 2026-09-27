**Fix: `face_restore: true` selects the default backend in `finish_clip` (#466).**

The backend name was derived with `str(cfg.get("face_restore") or "gfpgan")`, so `true` became
`"True"` and `FaceRestore("true")` raised `ValueError`. Only a string now names a backend; `true`
selects the default (`gfpgan`). The off values (`None`, `False`, `"none"`, `""`) still disable it.
