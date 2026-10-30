**Fix: a global-volume worker never pulls weights after a test (#530).**

`VJ_VOLUME_KIND=global` refuses rclone and Hub.
Preview no longer starts the Wan prefetch.
Stage the full R2 models tree onto the volume first.
An incomplete volume fails the job.
The worker does not fall back to R2.
