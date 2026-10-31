**Fix: an anime keyframe job gets its checkpoint on a thin worker (#557).**

Anime jobs failed with `keyframe base cagliostrolab/animagine-xl-4.0 is not on this worker`, because the cold pull and the bake both skipped the repo and nothing else pulled it. The job's own load now pulls it from the R2 mirror, about 6.5 GiB, the first time that worker needs it. A photoreal job never pulls it. A pull that fails fails the job.
