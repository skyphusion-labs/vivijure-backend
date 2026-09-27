# changelog.d

One file per change, so two PRs never touch the same hunk of `CHANGELOG.md` (backend#471).
Ported from `vivijure-cf` / `vivijure-core`.

**Filename:** `<issue>-<short-slug>.md` (for example `460-r2-exists-errors.md`), issue number first
so a directory listing sorts by issue. No issue number: `pr<N>-<slug>.md`.

**Content:** exactly the block that would have gone under `## Unreleased` today, in the same shape
as the existing entries (a `**Fix: ...**` or `**Feat: ...**` line, a blank line, then the prose).
No new syntax, no front matter.

**Required** for a PR that touches `src/` (CI job `changelog`), unless it carries the
`no-changelog` label. Adding a direct `CHANGELOG.md` edit still satisfies the check during the
migration window, but a fragment is preferred. `README.md` and `.gitkeep` never count as a fragment.

**Release:** `python3 scripts/changelog-assemble.py <version> <date>` (for example
`1.0.13 2026-09-30`) writes the `## [<version>] -- <date>` section from every fragment plus any body
still under `## Unreleased`, deletes the consumed fragments, and refuses if the version already has
a heading. Run it in the release commit, before tagging.
