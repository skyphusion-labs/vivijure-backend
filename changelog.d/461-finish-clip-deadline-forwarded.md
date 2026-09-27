**Fix: `finish_clip` forwards its Deadline to `run_finish_job` (#462).**

`handler()` built the Deadline and checked it before dispatch but did not pass it on, so the
fetch, upload and in-engine checks in `run_finish_job` were skipped and a stalled encode was bounded
only by the platform ceiling. The Deadline is now forwarded.
