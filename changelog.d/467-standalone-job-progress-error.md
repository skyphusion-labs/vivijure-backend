**Fix: a failed standalone job is recorded on the progress channel (#468).**

`run_finish_job` and `run_i2v_clip_job` now record the failure (`progress.error(<action>, e)`, then
re-raise), matching `run_job`. Previously a failed standalone job left the snapshot at `running`.
The wall-clock degrade path (`FinishDeadlineExceeded`) is unchanged and still ends `complete`.
