# Incident: `1.0.1` / t2 weight-digest rewrite broke RunPod pulls (2026-07-15)

## Symptom

After pinning `ghcr.io/skyphusion-labs/vivijure-backend:1.0.1`, RunPod workers failed
during image load with **`unexpected EOF`**. Rolling back to `:1.0.0` loaded fine.
Concurrent worker count was not the cause (16-wide pulls of `1.0.0` had worked before).

## Root cause

A **full** `runtime-build.yml` rebuild produced `runtime-1-bf16-t2` and re-ran
`COPY --from=seed` for the weight bins. BuildKit emitted **new layer digests** for
near-identical bytes (example: **+6 B** on an 8.23 GB layer). Measured vs `1.0.0` / t1:

| | |
| --- | --- |
| Shared compressed layers | ~2.9 GB |
| New digests RunPod had to cold-pull | ~**101 GB** |

GHCR itself was fine (large new blobs full-`sha256` verified). The failure was the
forced first-ever pull of those new multi-GB blobs on RunPod hosts. `:1.0.0` "working"
after rollback was consistent with **host cache** of the old digests.

The docs previously claimed `COPY --from=seed` was digest-stable across toolchain
rebuilds. That claim is **false in practice** on the bake lane we use.

## Fix shipped

1. **`deploy/runtime-overlay.Dockerfile`** -- for deps-only bumps, `FROM` a known-good
   prior runtime and only refresh pip + finish-dep patches. Weight layers inherit by
   blob identity.
2. **`runtime-build.yml` input `overlay_from`** -- when set, build the overlay path;
   when empty, full FROM-cuda + seed COPY (CUDA/torch/apt/seed changes only).
3. **Gate:** `python deploy/bake_layers.py assert-shared-diff-ids --image … --base …`
   fails the bake if RootFS layers are not mostly shared with the overlay base.
4. **`backend-v1.0.2`** -- overlay `runtime-1-bf16-t3` FROM t1; ~**104 GB** shared with
   `1.0.0`, ~**96 MB** new. Digest
   `sha256:ea38bc9db5c9b538fa9448b54fbf74501635422511584d419d6435eb5ef5cddd`.

## Standing rules

- **Do not pin `:1.0.1` on RunPod.** Prefer `:1.0.2` or later overlay-based releases.
- **Deps-only toolchain bumps** (safetensors / tokenizers / transformers / …): always
  use `overlay_from=<prior runtime@digest>`, never a full seed re-COPY.
- **Full rebuilds** remain for CUDA / torch / apt / seed changes; treat weight digests
  as **not** guaranteed across those rebuilds until a stronger determinism fix lands.
- Keep `requirements.txt` installable on the bake **Python 3.11** conda line
  (`numpy` 2.5.x needs >=3.12; `transformers` / `tokenizers` upper bounds must resolve).

## Pointers

- Architecture + overlay recipe: [`weights-base-and-snapshots.md`](weights-base-and-snapshots.md)
- Release table / pin notes: [`../RELEASES.md`](../RELEASES.md)
- Fleet handoff + both endpoint IDs:
  `fleet-chezmoi` `docs/runlog/2026-07-15-backend-v1.0.2-ready.md` and
  `docs/runbooks/vivijure-runpod-endpoints.md`
- PRs: vivijure-backend #273 (overlay path), #277 (t3 repin), #278 (RELEASES)
