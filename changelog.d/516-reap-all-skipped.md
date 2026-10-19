### Fixed

- `bake`: the pod reap no longer reports success for any listing that cannot
  establish what is running. "Nothing is running", "pods came back and none
  matched", and "the listing failed" had the same shape and the same exit code;
  only the first may report success. A failed listing is the likelier of the two
  unanswerable cases, since a 429 or 5xx on a list endpoint is ordinary.
- `bake`: each rejection reason is reported separately, because one counter made
  four different facts print one sentence and in one reachable case that sentence
  was false. A pod carrying the name we asked for but no usable id is not a
  name-filter problem; it is the pod we wanted, arriving unaddressable.
- `bake`: a row with a real, different name is reported as the measurement it is.
  It can only come back if RunPod's `name` filter is not exact, which is the
  question a live read could not settle against an empty account.
- `bake`: a pods listing that is a bare `null`, number, boolean or non-JSON body
  no longer raises `TypeError` out of the reap. Inside `run()`'s `finally` that
  exception replaced the one that sent us there, so a create timeout was reported
  as a complaint about iteration.
- `bake`: the reap never renders a pod row. A Pod response carries 34 properties
  including `env`, and the branch that printed a row fires only for a row whose
  name is ours, so it was the one branch guaranteed to print our own bake pod and
  its `GHCR_TOKEN`. It now prints an allowlist of identifying fields, so a new
  response property cannot leak by default.
- `bake`: a dict listing body with no `pods` list is no longer read as an empty
  account. That branch exists to survive a response-shape change, and reading a
  missing key as `[]` was the one way it would survive one by reporting that
  nothing is running.
