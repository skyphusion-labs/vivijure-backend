# Contributing

Thanks for your interest. A few things to know before you open an issue or PR.

## Project posture

This is a labor of love, maintained as time allows. Response times on issues and PRs may
vary. If you find it useful and want to make it better, you are welcome here.

## A clean-room project (please read)

This backend is an **independent, clean-room reimplementation**, written from the
control-plane API contract and the underlying models' own public documentation. It
deliberately shares no source with any prior or third-party render pipeline. That
independence is a feature of the project, and we keep it intact:

- **Do not paste, attach, or describe code, diffs, file layouts, or implementation
  details from any other render pipeline** in issues, PRs, comments, or commit messages.
  If your suggestion is "do it the way project X does," describe the *behavior or the
  public model/API docs*, never their source.
- The pipeline **core is maintainer-authored** and stays that way. Code contributions are
  reviewed and curated carefully; some will be reimplemented by the maintainer rather than
  merged verbatim, specifically to preserve the provenance. This is not a comment on your
  work, it is how the clean room stays clean.
- By submitting a contribution you affirm it is **your own original work** (or
  appropriately licensed), that you have the right to contribute it, and that it carries
  no third-party render-pipeline code. For code PRs, sign your commits off
  (`git commit -s`, a [DCO](https://developercertificate.org/) affirmation).

## Where contributions fit best

Most welcome, lowest friction:

- **Issues and bug reports** with a clear repro (a minimal bundle / job input is gold).
- **Documentation** fixes and clarifications.
- **Tests** that pin existing behavior (CPU-testable; see below).
- Small, self-contained fixes (a crash, an off-by-one, a config edge case) described
  from observed behavior.

Larger feature work is best discussed in an issue first, so we can agree on the shape
(and on how to keep it independent) before you invest time.

## House rules

- **No em-dashes (U+2014) or en-dashes (U+2013) anywhere** in source, comments, docs, or
  commit messages. Use commas, semicolons, parentheses, or a double hyphen (`--`).
- **Conventional Commits**: `fix(scope): ...`, `feat(scope): ...`, `docs: ...`, `ci: ...`.
  The body explains the *why*.
- Releases are SemVer-style `backend-vX.Y.Z` tags (PATCH for fixes, MINOR for features,
  pre-1.0).
- License: contributions are accepted under the project's **AGPL-3.0-only** license.

## Testing

CPU-testable logic (the contract, bundle reading, routing, config, assembly planning,
keyframe/pose setup) is covered by the suite in `tests/` and runs in CI with no GPU. Run
it locally:

```bash
pip install -r requirements-dev.txt
pytest
```

The GPU render path (SDXL keyframes, LoRA training, Wan i2v) is **validated by the
maintainer on real hardware**, gated and tagged; it is not exercised in CI and PRs are not
blind-merged on it. Keep new logic CPU-testable where you can, and call out clearly in the
PR anything that needs a GPU-validation pass, so it can be scheduled rather than assumed.

Before you start, [docs/development.md](docs/development.md) explains the CPU/GPU split and the
test layout, and [docs/architecture.md](docs/architecture.md) maps how the pieces interface.

## Pull requests

- Branch from `main`; CI (tests) must pass.
- `main` is protected and changes land by review. Open the PR, keep it focused, and tag
  the maintainer.
