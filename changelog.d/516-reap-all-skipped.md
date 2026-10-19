### Fixed

- `bake`: the pod reap no longer reports success when every pod it was handed is
  rejected by the name check. "Nothing is running" and "pods came back and none
  of them matched" had the same shape and the same exit code, and only the second
  one can hide a billing orphan. It now warns and fails, which also turns the
  first real dispatch into the measurement of whether RunPod's `name` filter is
  exact, a question a live read could not settle against an empty account.
- `bake`: a pods listing that is a bare `null`, number or boolean no longer raises
  `TypeError` out of the reap. Inside `run()`'s `finally` that exception replaced
  the one that sent us there, so a create timeout was reported as a complaint
  about iteration.
