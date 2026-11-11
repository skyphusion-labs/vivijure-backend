**CI: dispatch inputs go through env and are validated; thin-image publishes only from main.**

`runtime-build.yml` and `thin-image.yml` read their `workflow_dispatch` inputs through `env` instead of
placing them in the script text, and check each against its shape: integers for the runtime
versions, a digest-pinned `ghcr.io/skyphusion-labs/vivijure-backend:runtime-*` ref for
`overlay_from`, `X.Y.Z` and `t<N>` for the thin image. `thin-image.yml` refuses a commit that is not
on `main`, as `runtime-build.yml` already does, and its push trigger on the deleted
`ci/530-thin-gha` branch is removed, so no branch build publishes a release-shaped tag.
