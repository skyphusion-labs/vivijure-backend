# Stills weights base

Issue #564. The keyframe endpoint carries the stills set in its image, below the code layer.

## Why

A thin image has no weights. A cold worker pulled the stills set from R2 inside the handler:
26.22 GB, measured at 104 to 2618 s per worker, and about 20 minutes per job when 15 workers pulled at
once. That time is on the GPU clock, and every release makes every worker cold.

RunPod pulls the image before a worker starts, and that pull is not billed (RunPod docs:
`serverless/pricing`, "Compute cost breakdown"; `serverless/workers/overview`, "Worker lifecycle"). So
the same bytes cost nothing as image layers. The comparison with a global volume, a network volume, a
smaller pull and FlashBoot is on issue #564.

## Shape

```
toolchain-t1                     CUDA, torch, deps, HF configs (unchanged)
  + stills weights base          the stills set and the finish legs, 4 bin layers  (this page)
    + code image                 deploy/Dockerfile.thin, TOOLCHAIN_REF = the base   (every release)
```

A code-only release rebuilds only the top layer. A host that ran the previous release keeps the weight
layers. Rebuild the base only when the stills set, the finish legs, the toolchain or `VJ_MODEL_VERSION`
changes.

The base has **no `.vj-baked`**. That marker ends every pull for every action, and this image holds only
the stills set and the finish legs. A stills job (`preview`, `regen_shot`, `train_lora`) finds
`.vj-stills-complete` and pulls nothing. The worker checks that sentinel against the files before it
trusts it (`stills_manifest.verify`, a stat of 11 files) and pulls again if a file is missing. A finish
job finds its leg sentinels. Any other action still pulls what it needs from R2.

## Build

> The current stills base (`stills-t1-m1`, the `toolchain_ref` for 1.1.19) was built on the bake
> box, which is retired. **The next one is built by Actions:** `.github/workflows/stills-base.yml`
> (fleet-chezmoi#2620) runs steps 1 to 4 below on a GitHub-hosted runner.

### From Actions (the build path)

Dispatch `stills-base.yml` from `main` with `toolchain_ref` (a full digest of `toolchain-t<N>`), `tag` and
`model_version`. The job runs in the `stills-bake` environment (a required reviewer, main only), which holds three
secrets: `STILLS_R2_ENDPOINT`, `STILLS_R2_ACCESS_KEY_ID` and
`STILLS_R2_SECRET_ACCESS_KEY`, a read-only R2 token with Object Read on bucket `vivijure` only. The
repo-level `R2_*` secrets are not used.

- **Candidate tags only.** `tag` must be `stills-t<T>-m<M>-rc<N>`, with `t` the toolchain's and `m` the
  model version. A final tag such as `stills-t1-m1` is refused, and so is any tag already on GHCR
  (`deploy/stills-base-resolve.sh`, `tests/test_stills_base_resolve.py`). Promoting a proven candidate
  to a final name is a registry-side copy by the lead (`docker buildx imagetools create`), never a rebuild.
- **Disk.** The job first removes preinstalled SDKs it never uses, and refuses below 100 GiB free. The
  measured start is 88 GB on `ubuntu-latest` (`/mnt` is the same disk), and the worst-case peak is about
  93 GB. If a real run shows it does not fit, the fallback is reproducible bin tarballs appended with
  `crane`, not a larger runner.
- **Stage** with the worker's own pull, on the runner's Python, through the mirror's own `MODELS_R2_*`
  names (handed to rclone as `RCLONE_CONFIG_R2_*` environment variables, never a file). No throwaway
  image is needed, because the code IS the checked-out commit.
- **Reproducible layers.** `bin_pack` sorts by (size, path), so the same set always packs the same way,
  and the build sets `SOURCE_DATE_EPOCH` to the commit time with `rewrite-timestamp=true`. So an unchanged
  bin rebuilds to the same layer digest.
- **Push only.** The toolchain is referenced, never pulled or unpacked. The digest the push returned is
  read back from GHCR by tag, and every layer is checked against the 10 GiB limit from the registry
  manifest (`deploy/image_bytes.py` with `MODE=max-layer`; `verify-image` needs a local image).
- **Then** dispatch `thin-image.yml` with `toolchain_ref` = the printed `ref@digest` and the next free
  `X.Y.Z`. The live worker and its tag are never touched. The GPU proof below runs on a scratch endpoint,
  never the live one.

### On a host (the bake box record)

It was run on the bake box ([bake-box.md](bake-box.md)). It needs two credentials, both for this run only:

- An R2 temporary credential, read-only, scoped to the `vivijure` bucket prefix `models/`. Mint it on
  the laptop as [bake-box.md](bake-box.md) describes. It reaches the container as environment, never a
  file and never a build argument.
- The GHCR push token, as for any bake.

### 1. Stage, with the worker's own pull

Use an image whose code has `--stage-stills` (#591 or later), at the commit that will ship on this base. That code reads
the tree. An older image does not know the flag. On the first bake, 1.1.14 printed the usage line and exited 0 with
nothing staged. So on a first bake, build a local thin image at the target commit on the bare toolchain, and stage
with it. Do not push that image.

Pass the R2 credential on the SSH stdin, and source it in a shell that runs in the FOREGROUND. Background only
the `docker run`. A backgrounded `bash -c` has its stdin set to `/dev/null`, so the credential never arrives,
and the stage reports `no_creds_empty_cache`.

```
mkdir -p ~/stills/stage
docker run --rm \
  -e R2_ENDPOINT -e R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY -e RCLONE_CONFIG_R2_SESSION_TOKEN \
  -e R2_BUCKET=vivijure -e VJ_MODEL_VERSION=1 \
  -e VJ_MODELS_ROOT=/stage -e HF_HOME=/stage/hf-cache \
  -v ~/stills/stage:/stage \
  ghcr.io/skyphusion-labs/vivijure-backend:<current tag>@<digest> \
  conda run --no-capture-output -n vivijure \
  python -m vivijure_backend.harness.models_mirror --stage-stills
```

It runs `ensure_models` for a stills action and for every finish leg, which is the pull a cold worker
runs, and then `--check-stills`. It must end with `stills tree at /stage is complete` and exit 0.
`VJ_MODEL_VERSION` must match the endpoint's value, or every worker pulls again.

### 2. Gate the staged tree

```
python3 deploy/bake_layers.py assert-no-tree-cache --root ~/stills/stage/hf-cache
python3 deploy/bake_layers.py assert-finish-shas --root ~/stills/stage
```

### 3. Pack under the layer limit

The largest file is the RealVisXL UNet shard 1, 9,988,139,160 bytes (9.30 GiB). The GHCR layer limit
that `bake_layers.py` enforces is 10 GiB. So the ceiling is 9.5 GiB, and the set fits 3 bins of the 4
the workflow appends (`BINS` in `stills-base.yml`).

```
python3 deploy/bake_layers.py bin --src ~/stills/stage --out ~/stills/bins \
  --bins 4 --ceiling-gb 9.5 --min-gb 23
```

### 4. Tar the bins and append them to the toolchain

As `stills-base.yml` runs it, with `EPOCH` the commit time (`git log -1 --format=%ct`) and `crane` the
pinned release:

```
for b in bin-00 bin-01 bin-02 bin-03; do
  python3 deploy/bake_layers.py layer-tar --bin ~/stills/bins/$b --out ~/stills/layers/$b.tar \
    --epoch "$EPOCH" >> ~/stills/layers/diff-ids
done
crane mutate ghcr.io/skyphusion-labs/vivijure-backend@sha256:<toolchain digest> --platform linux/amd64 \
  --append ~/stills/layers/bin-00.tar --append ~/stills/layers/bin-01.tar \
  --append ~/stills/layers/bin-02.tar --append ~/stills/layers/bin-03.tar \
  --label org.opencontainers.image.revision=<commit> ... \
  -t ghcr.io/skyphusion-labs/vivijure-backend:stills-t1-m1-rc<N>
```

The tag names the toolchain (`t1`) and the model version (`m1`); the workflow writes only an `-rc<N>`
candidate. Read the digest back from GHCR and pin that digest. The first bake (below) was built instead
with `docker buildx build -f deploy/stills-weights.Dockerfile` (four `COPY bin-NN/ /opt/models/`), on
the bake box; that Dockerfile was removed when the build moved to `crane`, and is in git history.

### 5. Build the code image on it

As for any thin release, with the base as the toolchain:

```
docker buildx build -f deploy/Dockerfile.thin --provenance=false \
  --build-arg TOOLCHAIN_REF=ghcr.io/skyphusion-labs/vivijure-backend:stills-t1-m1@sha256:<digest> \
  -t ghcr.io/skyphusion-labs/vivijure-backend:<version> --load .
```

### 6. Gate the image you will push

CPU only. This checks the tree the worker will read, in the image that ships:

```
docker run --rm ghcr.io/skyphusion-labs/vivijure-backend:<version> \
  conda run --no-capture-output -n vivijure \
  python -m vivijure_backend.harness.models_mirror --check-stills
```

It must print `complete` and exit 0. Then push, read the digest back, and repin as
[runpod-endpoint-config.md](runpod-endpoint-config.md) says.

## Proof on the endpoint

The first job on a new worker must show, in its log and progress:

- `mirror_skipped {"reason": "warm_stills"}`, not `mirror_complete`;
- `mirror_done` with `pulled: false`;
- an `executionTime` near a warm job's, plus the model load.

**First bake, 2026-11-06 (`stills-t1-m1@sha256:6d0e19bd18bdba3318e510fcc0055d9d75c1a1353df918795dc92be0d70740b6`, on
`toolchain-t1@sha256:456b417b...`).**
- **Stage:** 26,180,415,017 bytes of stills set in 72.5 s (361 MB/s), plus 555,648,645 bytes of finish legs in 8.4 s.
  It was run in a local thin image at `9de17db`, for the reason above.
- **Gates:** `assert-no-tree-cache` OK; `assert-finish-shas` OK (2 facexlib weights).
- **Bins:** `bin` packed 62 files and 33 symlinks into 4 bins, 24.9 GB in total, with the largest at 9.49 GB. The fourth bin is empty.
- **Build:** 12:44:34Z to 12:57:39Z. `verify-image`: OK, all layers under the 10 GB ceiling.
- **Layers added over the toolchain, compressed:**
  - `76c9fde8193f`, 4,601,336,414 bytes (4.29 GiB);
  - `f953853e0b64`, 7,802,787,740 bytes (7.27 GiB);
  - `89cf0b497764`, 5,400,004,930 bytes (5.03 GiB);
  - the empty bin, which is the shared empty layer `4f4fb700ef54` (32 bytes).
  - In total 16.58 GiB.
- **Push:** 12:57:39Z to 12:58:36Z (57 s). The 23 toolchain layers already existed on GHCR.
- **Credential:** read-only, scoped to `models/`, 4 h TTL. It was minted from a dedicated read-only parent token, and the parent was deleted after the stage.

Record for the first bake: each layer's compressed size, the push time, how long a new worker stays
`INITIALIZING`, and the cold job's `executionTime`. Those are the after numbers for #564.

## Rollback

Repin the previous thin image. It pulls the stills set from R2 as before. Nothing in the code depends on
the base: on an image without it, the stills sentinel is absent and the worker pulls.
