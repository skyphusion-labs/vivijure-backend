**Fix: `finish_clip` writes to the `output_key` it is given (vivijure-core#379).**

One film can no longer replace another film's finished clip, once core sends a film-scoped key. With no `output_key` the shared per-project key is used, as before. A key that is not an `.mp4` under this project's `clips/` is logged and the shared key is used, so the job does not fail.
