**Feat: the plate count guard is on by default (`keyframe.plate_count_retries` 3) for two-person-plus shots.**

Measured live on 1.1.13 at six seeds (vivijure-cf#1143): clean two-person stills rose from 2 of 6 to 5 of 6. Every still marked clean had exactly two people, and the one failure was reported as `clean: false`. The cost is about 1.7 extra plates per two-person keyframe, about 23 s on average, at most 3 plates. The value now enters the keyframe reuse hash for two-person-plus shots only, so each existing two-person still redraws once, and one-person and empty shots keep their hash. Set it to 0 to turn the guard off.

**Docs: the cold-start rule after a repin is corrected.** A warm-up batch does not keep workers warm after a new image: every new worker pulls about 24 GB. Keep about 4 to 6 cold workers at once by submitting in small batches, and budget one cold pull per worker (`docs/runpod-endpoint-config.md`, `docs/bake-box.md`).
