# Backend image chain: seed -> runtime -> backend + runner snapshots (#537)

How the backend bake stopped re-downloading ~87 GB of weights from R2 on every release, and how a
custom runner snapshot makes a release bake assemble-and-push only. ICD-grade: reproducible from this
doc alone.

> **Status (2026-07-05, S19, Shape Y):** the SEED image + its build (`seed-build.yml`) and the RUNTIME
> base + its build (`runtime-build.yml`) land first; the backend `deploy/Dockerfile` + `release.yml`
> slim to `FROM runtime` + `COPY src` (follow-up PR, with the release-contract doc). The runner
> snapshot pre-pulls the RUNTIME image. The image-generation runner is real
> (`ubuntu-latest-32c-128gb-1200-gen`); enterprise scoping items are tracked in fleet-chezmoi #377.

## Why (the problem)

`release.yml` used to, on EVERY `backend-v*` tag, `rclone` the curated ~87 GB (bf16) / ~90 GB (fp8)
weight seed from R2, bin-pack it, and bake it. That stage + bin-pack + the ~87 GB image push repeat
unchanged every release even though the weights change rarely. The fix splits the image into a chain
so a src-only release re-pushes ONLY the app layer and stages R2 exactly once per weight version.

## The chain (three images, one responsibility each)

```
  seed-build.yml (dispatch, R2)                 runtime-build.yml (dispatch, NO R2)        release.yml (tag)
  --------------------------------              -----------------------------------        -----------------
  stage R2 seed -> symlinks -> bin-pack         FROM nvidia/cuda + toolchain               FROM runtime@digest
   -> sha256 manifest -> assert non-hollow        + hf-configs (before weights, #206)        COPY src + smoke + CMD
   -> debian-slim carrier                         + COPY --from=seed 24 bin layers           push  (only the app
   -> push                                        + sha256sum -c manifest gate                     layers upload;
      vivijure-backend-seed                        + assert-weights/finish/no-tree             the runtime + weight
        :<modelver>-<precision>                    + .vj-baked stamp                           layers dedup)
              |                                    -> push
              |  COPY --from (24 bins,             vivijure-backend-runtime
              |  deterministic -> dedup)             :<modelver>-<precision>-t<toolchainver>
              +----------------------------------------->  |
                                                           |  FROM @digest
                                                           +----------------->  vivijure-backend:<X.Y.Z>
```

### 1. SEED image -- `ghcr.io/skyphusion-labs/vivijure-backend-seed:<modelver>-<precision>`

The immutable, versioned carrier for the curated model seed (a tiny `debian:bookworm-slim` holding
`/seed-bins/bin-00..23` + the union-keyed `weights-manifest.sha256` + the model licenses). Built by
`.github/workflows/seed-build.yml` (dispatch-only, on `vivijure-bake`): stage R2 seed -> reconstruct
HF-cache symlinks -> `assert-weights` (non-hollow) -> bin-pack (`deploy/bake_layers.py bin`) ->
`manifest` -> build `deploy/seed.Dockerfile` -> per-layer gate -> push. **This is the ONLY image that
stages from R2**, and only on a weight-set change. `deploy/seed.Dockerfile`.

### 2. RUNTIME base -- `ghcr.io/skyphusion-labs/vivijure-backend-runtime:<modelver>-<precision>-t<toolchainver>`

The full validated runtime + baked weights: `FROM nvidia/cuda` + apt + conda + torch(cu128) + render
deps + finish-dep patches + RIFE + the HF config bake + the weights `COPY --from=seed` (24 `<10 GB`
bin layers) + the `sha256sum -c` manifest gate + `assert-weights`/`assert-finish-shas`/
`assert-no-tree-cache` + the `.vj-baked` stamp. Built by `.github/workflows/runtime-build.yml`
(dispatch-only). **NO R2**: the weights come from the pinned seed. `deploy/runtime.Dockerfile`.

- The toolchain + assert/stamp sections are byte-identical to the pre-split monolith (verified), so
  the proven CUDA/torch layout and the `#206` hf-configs->weights order are preserved exactly
  (hf-configs runs BEFORE the weight `COPY --from`, both inside this image).
- Two precisions from one Dockerfile via `FROM seed-${VJ_BAKE_PRECISION}` selection; both seed pins
  are authoritative in `runtime.Dockerfile` (tag + `@sha256`).
- `COPY --from=seed@digest` is **deterministic** (fixed source bytes + metadata), so the 24 weight
  layers reuse identical blobs across runtime rebuilds -> they dedup on GHCR. A toolchain/CUDA bump
  rebuilds this image but pulls the seed (no R2) and re-pushes only the changed toolchain layers.

### 3. BACKEND (consumer) -- `ghcr.io/skyphusion-labs/vivijure-backend:<X.Y.Z>`

`FROM runtime@digest` + `WORKDIR` + `COPY src` + `COPY smoke_imports.py` + `CMD`. Built by
`release.yml` on a `backend-v*` tag. Every layer below `COPY src` is inherited from the runtime base
by blob identity, so a src-only release **re-pushes only the app layers** ("layer already exists" on
the runtime + weight blobs). Lands in the follow-up PR with the release-contract doc.

## Why this shape (Shape Y), not the alternatives

- **`hf-configs` is load-bearing and constrains the layering.** `deploy/bake_hf_configs.py` runs
  ONLINE (needs the conda/HF stack) and MUST run BEFORE the weight COPY (the `#206` tree-cache /
  curated-subset prod bug). So the weights cannot sit in a weights-only base with the toolchain on
  top (hf-configs would run after them, inverting the order), and "plain base + apt-install CUDA"
  abandons the validated `nvidia/cuda` sm_120 layout. The only correctness-preserving FROM-base is
  one that carries the full runtime with hf-configs before the weights -> the RUNTIME image.
- **Dedup vs COPY --from.** `FROM runtime` inherits the weight layers as identical blobs, so a
  release push dedups them (near-zero upload). A consumer that `COPY --from`'d the weights would
  re-create them as new blobs and re-push ~87 GB every release.
- **R2 exactly once.** The immutable SEED image is the weight source the runtime `COPY --from`s, so a
  toolchain rebuild never re-stages R2; R2 runs only when the weight set itself changes.

## Runner snapshot (`runner-snapshot.yml`)

Pre-pulls the pinned RUNTIME image + buildx + HF CLI into a larger-runner image via GitHub's
job-level `snapshot:` keyword (custom images for larger runners, GA 2026-03), on the image-generation
runner `ubuntu-latest-32c-128gb-1200-gen`. A release bake on a runner built from that image reads the
runtime base cache-warm (a fast local `FROM`) instead of pulling it. **ACCELERATOR ONLY** (design law
#3): the consumer pins the runtime by digest, so a stale snapshot degrades to a slow pull, never to a
wrong image. Enterprise scoping (group grant + tag/workflow allowlist + the second, image-gen-OFF
consuming runner) is tracked in fleet-chezmoi #377.

## Rebuild triggers (summary; the full release contract lands with `release.yml`)

- **src-only change:** tag `backend-vX.Y.Z` -> `release.yml` (`FROM runtime` + `COPY src`). Fast.
- **weight-set change:** `seed-build.yml` -> repin `SEED_REF_*` in `runtime.Dockerfile` ->
  `runtime-build.yml` -> repin `RUNTIME_REF_*` in `deploy/Dockerfile` -> tag.
- **toolchain/deps/CUDA change:** `runtime-build.yml` (bump `-t<N>`, no R2) -> repin `RUNTIME_REF_*`
  in `deploy/Dockerfile` -> tag.

## Acceptance

- Seed builds + pushes (R2); runtime builds from the pinned seed; the `sha256sum -c` gate passes on a
  good seed and FAILS LOUD on a tampered one (proven with real local images: gate PASS / tamper exit 1
  / symlink survives / precision selection).
- Dedup: a 2nd consecutive src-only release build shows "layer already exists" on the inherited
  runtime + weight blobs.
- Timed: a release bake on the snapshot runner vs the pre-split R2-stage bake.

## Re-bake cadence (CVE freshness)

Baked images rot: CVEs and stale toolchains freeze in at bake time. Post-Shape-Y this matters most for
the RUNTIME base, because every released image inherits its layers, so **runtime age = shipped CVE
posture**. The cadence policy (wired as `cron` triggers alongside `workflow_dispatch` on the existing
dispatch workflows, no new machinery):

| Artifact | Cadence | Mechanism |
|---|---|---|
| **RUNTIME base** | monthly floor + on-demand | `runtime-build.yml` cron (`0 6 1 * *`) reads the currently shipped `RUNTIME_REF_BF16` tag from `deploy/Dockerfile` and rebuilds at the SAME tag (fresh base/apt/pip-patch layers, new digest -- NO R2, weights come from the seed). It then auto-opens a `RUNTIME_REF` digest-bump PR (`auto/runtime-repin-<prec>`, force-updated, human merges through the gate) and re-triggers the snapshot. Deliberate toolchain bumps use the `workflow_dispatch` inputs (bump `-t<N>`). |
| **RUNNER snapshot** | event-coupled + monthly backstop | `runtime-build.yml` dispatches `runner-snapshot.yml` on every successful re-bake (its whole job is pre-pulling the runtime). `runner-snapshot.yml` also has a monthly cron (`0 7 1 * *`) that reads the exact shipped pin as a safety net. Snapshot age as a health signal is infra's half (fleet-chezmoi #370). |
| **SEED** | EXEMPT | The seed is content-addressed weight DATA, not software -- no CVE surface, rebuilt ONLY on a weight-set change. Do NOT add a periodic 87 GB restage. |

Drift guard: the scheduled paths READ the current pin rather than carrying a frozen input, so a cron can
never refresh a config nobody ships (that would LOOK like hygiene while doing nothing). The auto-repin
PR degrades gracefully: if the org "Actions can create/approve PRs" setting is ever off, the branch is
still pushed and the run logs a warning -- never a hard fail.
