### fix(mirror): a model pull is killed on a stall, and reports progress while it runs

The pull budget (#564) bounds a whole pull, so it cannot tell a slow pull from a hung one: a hung pull held its GPU until the 3000 s budget, about $2.90. Every model pull now also ends when rclone's transferred bytes have not moved for `VJ_MIRROR_STALL_SECONDS`, default 300 s (about $0.29 of GPU). A slow pull that still moves is left alone. A stall fails a job by name (`model pull stalled: ...`), and a finish job soft-degrades as it does on its budget.

This covers every pull: the job pull, `ensure_i2v_models`, `ensure_hub_repos`, and the volume self-preload, which falls back to the job's own pull on a stall. While a pull runs, each rclone stats line becomes a `mirror_progress` event with the leg and the bytes moved, so a cold pull no longer looks like a silent job.
