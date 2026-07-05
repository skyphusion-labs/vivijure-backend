# Weights base image + runner snapshots (#537)

How the backend bake stopped re-downloading ~87 GB of model weights on every release, and how a
custom runner snapshot makes a bake assemble-and-push only. ICD-grade: the contract is reproducible
from this doc alone.

> **Status (2026-07-05, S19):** the weights-base image + its build workflow (PR 1) land first; the
> backend `deploy/Dockerfile` + `release.yml` refactor to CONSUME the base (PR 2) and the runner
> snapshot workflow (PR 3) follow. The runner snapshot is an ACCELERATOR; the data path is correct
> and provable without it (a release just does a slow `docker pull` of the base). See the ownership
> split at the end -- infra owns onlining the runners; we own everything in this repo.

## The problem it solves

`release.yml` used to, on EVERY `backend-v*` tag, `rclone` the curated ~87 GB (bf16) / ~90 GB (fp8)
weight seed from R2, reconstruct the HF-cache symlinks, bin-pack it into <10 GB layers, THEN build.
That stage + bin-pack is the dominant release cost and it repeats unchanged every release even though
the weights change rarely. The fix: do it ONCE per weights bump, publish the result as a versioned
image, and have the release bake consume that image.

## The three pieces

```
  weights bump (rare, one button)                 backend release (frequent, per backend-v* tag)
  ------------------------------                  ------------------------------------------------
  weights-base.yml (dispatch)                     release.yml (tag / dispatch)
     stage R2 seed -> symlinks                       FROM weights-backend:<ver>@sha256:<digest>
     -> bin-pack -> sha256 manifest        =====>    24x COPY --from=weights bin-NN -> /opt/models
     -> build weights.Dockerfile                     sha256sum -c weights-manifest.sha256  (LOUD gate)
     -> push                                         assemble app + push   (NO R2 stage, NO bin-pack)
     ghcr.io/skyphusion-labs/                        |
       vivijure-weights-backend:<ver>                v
                     ^                            runner-snapshot.yml (dispatch, image-gen runner)
                     |                               docker pull weights-backend:<ver>
                     +---- pre-pulled into --------- install buildx + HF CLI
                           the runner's                snapshot: vivijure-bake-snapshot
                           docker cache                (a stale snapshot degrades to a slow pull,
                                                        NEVER to a wrong image)
```

### 1. Weights base image (`ghcr.io/skyphusion-labs/vivijure-weights-backend:<model_version>-<precision>`)

The versioned, immutable, source-of-truth carrier for the curated model seed. Built by
`.github/workflows/weights-base.yml` (workflow_dispatch only; never fork-reachable; runs on the
existing `vivijure-bake` 1200 GB larger runner). The build:

1. Stages the precision-selected seed (`bake-seed-<precision>/`) from R2 with the same `rclone`
   invocation `release.yml` used, and reconstructs the HF-cache symlinks from the `.rclonelink`
   markers.
2. `assert-weights` (host-side): a non-hollow gate (byte floor + at least one real shard) so a hollow
   base can never be published (the #4 empty-bake defense, restated at this new boundary).
3. Bin-packs into `deploy/seed-bins/bin-00..23` (`deploy/bake_layers.py bin`), each bin < 9 GB (under
   the 10 GB GHCR per-layer ceiling), each mirroring its path relative to `VJ_MODELS_ROOT`.
4. Writes `deploy/seed-bins/weights-manifest.sha256` (`deploy/bake_layers.py manifest`): a
   `sha256sum -c`-compatible, UNION-KEYED SHA-256 of every weight file (the `bin-NN/` prefix stripped,
   so a key is the runtime path under `VJ_MODELS_ROOT`). This is the byte-identity contract the
   consumer verifies.
5. Builds `deploy/weights.Dockerfile` (a tiny `debian:bookworm-slim` carrier that COPYs the 24 bins,
   the manifest, and the third-party model licenses into `/seed-bins`), per-layer-gates it, pushes
   `:<ver>` + an immutable `:<ver>-<run>` tag, and prints the digest to the job summary.

The carrier OS contributes NOTHING to the consumer image -- the consumer `COPY --from`s only
`/seed-bins`. This image is data + a checksum manifest, not a runtime.

### 2. Consumer refactor (`deploy/Dockerfile` + `release.yml`) -- PR 2

`deploy/Dockerfile` pins the base by TAG AND DIGEST and copies the bins out, preserving the <10 GB
layer structure and decoupling the weights (rarely bumped) from the CUDA/torch/app toolchain
(frequently bumped -- a torch bump never re-stages 87 GB):

```dockerfile
ARG WEIGHTS_REF=ghcr.io/skyphusion-labs/vivijure-weights-backend:1-bf16@sha256:<digest>
FROM ${WEIGHTS_REF} AS weights
# ... CUDA + conda + torch + deps + rife + hf-configs (unchanged) ...
COPY --from=weights /seed-bins/bin-00/ /opt/models/
# ... bin-01 .. bin-23 ...
COPY --from=weights /seed-bins/weights-manifest.sha256 /opt/models/weights-manifest.sha256
RUN cd /opt/models && sha256sum -c weights-manifest.sha256 --quiet   # LOUD byte-identity gate
# ... assert-weights + assert-finish-shas + .vj-baked stamp (unchanged) ...
```

`release.yml` loses the R2-stage + bin-pack steps entirely: it becomes assemble + push. It runs on
the snapshot runner when one is online (the base is pre-cached), and degrades to `vivijure-bake` (a
one-time `docker pull` of the base) otherwise -- SLOW, never WRONG.

**Why `COPY --from`, not `FROM weights-base`:** `COPY --from` keeps the weights out of the toolchain's
layer chain, so a CUDA/torch bump rebuilds only the toolchain, never re-stages the seed. Upload per
release is unchanged from today (~87 GB, already the case). `FROM`-inheritance would dedup the weight
layers on push (near-zero upload) but couples the CUDA/apt base into the "weights" image, so a CUDA
bump would force an 87 GB re-stage. We optimize for the frequent case (releases) and clean semantics.

### 3. Runner snapshot (`runner-snapshot.yml`) -- PR 3

GitHub custom images for larger runners (GA 2026-03) use a job-level `snapshot:` keyword: a job runs
on an IMAGE-GENERATION runner, installs tools, and on success GitHub captures the runner state into a
reusable, auto-versioned image. Our snapshot job (`workflow_dispatch` only) `docker pull`s the weights
base and installs buildx + the HF CLI, so those land in the runner image's docker cache. The backend
release then targets a runner built from that image and skips the base pull. Rebuild the snapshot
(one dispatch) after a weights bump.

**A stale snapshot is safe by construction:** the consumer pins the base by DIGEST and verifies the
checksum manifest. If the snapshot's cached base is older than the pinned digest, docker pulls the
newer one (slow); if a cached layer were ever wrong, `sha256sum -c` fails the build LOUD. The snapshot
can only ever make a build FASTER, never change WHAT it builds.

## Ownership split (infra <-> us)

| Owned by INFRA (org admin, UI-only, no REST API) | Owned by US (this repo) |
|---|---|
| Create the IMAGE-GENERATION larger runner (class `ubuntu-latest-32c-128gb-1200-gen`), enable "generate custom images" | `weights.Dockerfile` + `weights-base.yml` (the weights base) |
| Run/authorize the snapshot workflow; online the resulting custom-image runner and set its label | `deploy/Dockerfile` + `release.yml` refactor (consume the base) |
| Keep the `vivijure-bake` build runner online | `runner-snapshot.yml` (the snapshot definition) |
|  | `deploy/bake_layers.py manifest` (the checksum tool) + this doc |

Runner labels are the interface: infra sets the image-generation runner's label to match
`runner-snapshot.yml`'s `runs-on`, and the snapshot-image runner's label to match `release.yml`'s
`runs-on`. The exact strings are tracked in the infra handoff (fleet-chezmoi issue) referenced from
vivijure#537.

## Acceptance

- Weights base builds + pushes on `vivijure-bake`; digest captured. (dispatch)
- Refactored `release.yml` consumes the pinned base, the `sha256sum -c` gate passes on a good base and
  FAILS LOUD on a tampered one, and the built image is byte-identical to the pre-refactor bake.
- Timed: a release bake on the snapshot runner vs the current stage-from-R2 bake.
- Snapshot-runner timing is PENDING-RUNNER until infra onlines the image-generation runner; the data
  path (base builds, consumer consumes, gate proven) is provable now on `vivijure-bake`.
