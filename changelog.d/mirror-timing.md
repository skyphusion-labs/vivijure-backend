**Fix: a slow cold start now names its own stage in the job's progress object.**

`mirror_done` carries the seconds per leg, total seconds, bytes, and the rclone passes that failed and were retried. A failed pull writes `mirror_failed` with the legs it finished. Before this, the numbers were only in the worker's log, which goes when the worker does.
