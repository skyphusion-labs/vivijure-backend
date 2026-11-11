**Fix: a finish job no longer writes `<shot>_finished.latest.json` (#613).**

Nothing read that object: core does not reuse a finished clip through a pointer (vivijure-core#382). The clip and its `.hash` sidecar are written as before.
