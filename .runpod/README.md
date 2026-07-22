# RunPod Hub -- Vivijure Backend

This directory configures the [RunPod Hub](https://console.runpod.io/hub/skyphusion-labs/vivijure-backend)
listing for the main Vivijure GPU render worker.

## Required environment (shared Studio bucket)

Hub deployers fill these when they deploy from the listing. They are the same four names the
worker reads at runtime (`src/vivijure_backend/harness/r2.py`):

| Env key | Hub UI label | What to put |
| --- | --- | --- |
| `R2_ENDPOINT` | R2 S3 endpoint | `https://<account-id>.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 access key ID | Public half of an R2 API token with read/write on the bucket |
| `R2_SECRET_ACCESS_KEY` | R2 secret access key | Secret half of that token |
| `R2_BUCKET` | R2 bucket | Bucket name shared with Vivijure Studio (preset default: `vivijure`) |

**Preset:** "Standard (shared Studio bucket)" sets `R2_BUCKET=vivijure`. Override if your Studio
uses another bucket name; the Studio and this worker must agree.

**Not the satellite name.** Finish satellites (`vivijure-musetalk`, `vivijure-upscale`,
`vivijure-audio-upscale`) read `R2_ENDPOINT_URL`. This backend reads `R2_ENDPOINT` (no `_URL`).
Copy-paste from a satellite endpoint will miss the bucket.

## Required files (Hub probe)

Hub requires `handler.py`, `Dockerfile`, and `README.md` in `.runpod/` or the repo root.
This package keeps the production entry as git symlinks at the repo root
(`handler.py` -> `src/vivijure_backend/worker.py`, `Dockerfile` -> `deploy/Dockerfile`).
`.runpod/handler.py` and `.runpod/Dockerfile` are **real files** (not symlinks) so the Hub
listing checklist detects them. The image `CMD` still runs `python -m vivijure_backend.worker`.

## Hub test

`.runpod/tests.json` sends `{ "action": "health" }` on an H200. That probe does not need R2
credentials. A full render still needs the four R2 vars above.

## Operator checklist (listing status)

1. GitHub release exists for the image you want Hub to index (currently `backend-v1.0.5` / `:1.0.5`).
2. In the RunPod console Hub page, confirm the listing build/test is green (Pending vs Live).
3. After any Hub-facing change under `.runpod/`, cut a new GitHub release so Hub re-indexes.

Full deploy path for operators outside the Hub UI: [docs/deploy.md](../docs/deploy.md).
