**Docs: the backend endpoint runs 1.1.12; the repin cold-start rule; the 1.1.x tag decision.**

`docs/runpod-endpoint-config.md`:
- records the 1.1.11 (#584) and 1.1.12 (#585) pins, with the 1.1.11 rollback;
- adds a standing repin procedure: wait 65 s after the release, then submit 2 or 3 jobs first because every worker starts cold and pulls about 24 GB.

`docs/bake-box.md` points at that procedure. `RELEASES.md` records that the 1.1.x line is GHCR tags only, with no git tag.
