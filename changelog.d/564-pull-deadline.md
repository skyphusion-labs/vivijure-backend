### fix(mirror): a model pull has a deadline on every action

A cold worker pulled 26 GB for up to 2618 s on the GPU clock, with only the endpoint's 3600 s job timeout to end it. A pull outside `finish_clip` now has a budget, `VJ_MIRROR_MAX_SECONDS`, default 1800 s. Past it rclone is killed, the job fails by name, and the worker goes free. A film's retry then starts the job again.
