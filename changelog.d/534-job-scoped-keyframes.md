**Fix: concurrent renders of one project keep their own keyframes (#534).**

A render now writes `renders/<project>/keyframes/<job_id>/<shot>.png`.
Reuse now reads `<shot>.latest.json`, one object with the key and the param hash.
A reused still is copied to the new job's key.
A project from before this change is still read from the flat key.
Nothing new is written at the flat key.
This needs vivijure-core 1.27.0 for `own-gpu` clip submits and lost-poll recovery.
