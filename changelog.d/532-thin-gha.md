**Feat: thin keyframe image builds on GitHub Actions (#530).**

The bake job stays disabled.
`thin-image.yml` builds `--target toolchain` and `Dockerfile.thin`.
It pushes to GHCR.
It does not pull the seed.
It does not need a GPU runner.
