# Wan LoRA training (decoupled, then retired)

> **Retired.** The `vivijure-wan-train` repo is archived and hosted Wan training is retired; Wan
> training is self-host only. This page records the decouple.

Wan 2.2 A14B character LoRA training moved to the **`vivijure-wan-train`** satellite repo
(Conrad ruling 2026-07-23). This backend image trains **SDXL** adapters inline only.

- Satellite: https://github.com/skyphusion-labs/vivijure-wan-train
- Studio binding: `RUNPOD_WAN_TRAIN_ENDPOINT_ID` (an operator's own endpoint now)
- Migration: see `vivijure-wan-train/docs/migration-from-backend-train-image.md`

The retired `:train-*` tags on `ghcr.io/skyphusion-labs/vivijure-backend` remain on GHCR for
rollback until retention expires; do not build new train images from this repo.
