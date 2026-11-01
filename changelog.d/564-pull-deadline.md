### fix(mirror): the main model pull has a deadline on every action

A cold worker pulled 26 GB for up to 2618 s on the GPU clock, with only the endpoint's 3600 s job timeout to end it. The pull in `ensure_models` outside `finish_clip` now has a budget, `VJ_MIRROR_MAX_SECONDS`, default 3000 s. Past it rclone is killed, the job fails by name, and the worker goes free.

Not covered: `ensure_i2v_models`, `ensure_hub_repos` and the volume self-preload have no deadline.
