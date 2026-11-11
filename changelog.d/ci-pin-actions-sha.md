**CI: every third-party Action is pinned by commit SHA.**

Each `uses:` reference in `.github/` now names a full commit SHA with the exact release tag as a
comment (for example `actions/checkout@<sha> # v7.0.1`), resolved from the tag the line used before.
A moved tag can no longer change what a workflow runs. Dependabot (`github-actions`) keeps proposing
updates against the pinned form.
