**Docs: the hosted backend endpoint drops the B200 and B300 pools (#625, #606).**

`docs/runpod-endpoint-config.md` now records the backend pools as `BLACKWELL_96`, `ADA_80_PRO` and `HOPPER_141`, with the before list and the rollback. The current-state table is re-read on 2026-11-11, and the image row moves from 1.1.17 to 1.1.18@sha256:3ec38492. The Recommendation for the render backend no longer says B200 / H200: RunPod serves SDXL keyframes only, and keyframes are proven on 96 GB and 80 GB cards. This records a live endpoint change that the lead applies under CR #625. No code changes.
