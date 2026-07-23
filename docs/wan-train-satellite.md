# Wan LoRA training (decoupled)

Wan 2.2 A14B character LoRA training moved to the **`vivijure-wan-train`** satellite repo
(Conrad ruling 2026-07-23). This backend image trains **SDXL** adapters inline only.

- Satellite: https://github.com/skyphusion-labs/vivijure-wan-train
- Control plane: unchanged (`RUNPOD_WAN_TRAIN_ENDPOINT_ID`, same job payload and R2 keys)
- Migration: see `vivijure-wan-train/docs/migration-from-backend-train-image.md`

The retired `:train-*` tags on `ghcr.io/skyphusion-labs/vivijure-backend` remain on GHCR for
rollback until retention expires; do not build new train images from this repo.
