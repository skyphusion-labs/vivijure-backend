# Contributing

Thanks for your interest. A few things to know before you open an issue or PR.

## Project posture

This is a labor of love, maintained as time allows. Response times on issues and PRs may
vary. If you find it useful and want to make it better, you are welcome here.

## Contributing code

This backend is an independent, built-from-scratch implementation, written against the
control-plane API contract and the underlying models' own public documentation. It is a
clean-sheet codebase, which is exactly what makes it pleasant to extend. A couple of
standard hygiene points keep it that way:

- By submitting a contribution you affirm it is **your own original work** (or
  appropriately licensed), and that you have the right to contribute it. Please do not
  paste code or diffs you do not have the rights to.
- For code PRs, **sign your commits off** (`git commit -s`, a
  [DCO](https://developercertificate.org/) affirmation).

## Where contributions fit best

Most welcome, lowest friction:

- **Issues and bug reports** with a clear repro (a minimal bundle / job input is gold).
- **Documentation** fixes and clarifications.
- **Tests** that pin existing behavior (CPU-testable; see below).
- Small, self-contained fixes (a crash, an off-by-one, a config edge case) described
  from observed behavior.

Larger feature work is best discussed in an issue first, so we can agree on the shape
before you invest time.

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
