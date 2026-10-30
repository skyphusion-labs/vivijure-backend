**Fix: a cold `finish_clip` job pulls only the models its step loads, and a finished pull no longer costs the job.**

A RIFE step now pulls the 11.6 MiB `rife` leg instead of 53 GB. Each pulled leg has its own sentinel, so a warm worker does not copy it again, and a later job that needs the rest still pulls the rest.
The mirror has its own budget, `VJ_FINISH_MIRROR_MAX_SECONDS` (default 120). The 420 s budget starts when the mirror returns.
A hung `rclone` is killed at the mirror budget and the job soft-degrades at stage `model_mirror`.
