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

> `stills-t1-m1`, the `toolchain_ref` for 1.1.19, was built on the bake box, which is retired.
> `stills-t1-m1-rc1`, the `toolchain_ref` for 1.1.20, is the first built by Actions: `.github/workflows/stills-base.yml`
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
  to a final name is a registry-side tag by the lead, by digest, never a rebuild ("Promote a proven candidate", below).
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
- **The CPU import smoke leaves the weight layers out (#645).** A hosted runner cannot hold the full
  worker: run 43312194021 refused 1.1.20 (31.8 GB compressed) with 43.8 GB free. `deploy/smoke_image.py`
  streams the worker into `docker load` without the base's layers above its toolchain: the toolchain
  layers plus the worker's own thin layers, by digest, each checked against its descriptor, with the
  worker's config minus those layers. The smoke then checks the loaded rootfs is exactly that and runs
  `smoke_imports.py`. That is sound because the weight layers add files under `/opt/models` only
  (`assert-appended`) and the smoke reads no weight on CPU. It is NOT a check of the weights: the GPU proof
  is. To smoke an existing worker again, dispatch with `smoke_ref` = the worker and `smoke_base` = this
  base; that run builds and pushes nothing, so it may run from a branch.

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

## Promote a proven candidate (registry-side, by digest)

A candidate gets a final name only after the GPU proof below passed on a worker built FROM it. The
promotion adds a second tag to the SAME manifest. It is never a rebuild, and the digest does not change,
so every worker already built FROM the candidate digest is built from the promoted base. Keep the `-rc<N>`
tag as well; a pinned digest with no tag is GC-bait (RELEASES.md).

**Final names.** `stills-t<T>-m<M>` is the first build of a set; `stills-t1-m1` is the bake-box build and
stays as it is (a GHCR tag is never overwritten). A later build of the same toolchain and model version,
whose weight layers have new digests, is `stills-t<T>-m<M>-b<N>`, with `stills-t<T>-m<M>` counted as `b1`.
So the first Actions build is `stills-t1-m1-b2`. The name says the layers differ from `stills-t1-m1`,
which is the fact a reader needs: a worker moving between them pulls the weight layers again.

The lead runs it, with a GHCR login that can write the package (`docker login ghcr.io`). It refuses
when the final tag already exists or the registry cannot say, and it proves the digest after:

```
set -euo pipefail
R=ghcr.io/skyphusion-labs/vivijure-backend
D=sha256:9380cc424f8c76eeb1ac6d8ebfb15d98673b26fd0effdced528ddf76d4afd533   # stills-t1-m1-rc1
T=stills-t1-m1-b2
test "$(docker buildx imagetools inspect --format '{{json .Manifest.Digest}}' "$R:stills-t1-m1-rc1")" = "\"$D\""
if out=$(docker buildx imagetools inspect "$R:$T" 2>&1); then echo "REFUSE: $R:$T exists"; exit 1; fi
printf '%s' "$out" | grep -Eqi 'not found|manifest unknown' || { echo "REFUSE: cannot tell: $out"; exit 1; }
docker buildx imagetools create -t "$R:$T" "$R@$D"
test "$(docker buildx imagetools inspect --format '{{json .Manifest.Digest}}' "$R:$T")" = "\"$D\"" && echo "PROMOTED $R:$T@$D"
```

`imagetools create` from one source manifest with no annotations pushes that manifest as it is. The last
line is what proves it: if the digest read back is not `$D`, the tag points at a new manifest, so delete
that tag and use `crane tag "$R@$D" "$T"` (the pinned crane of `stills-base.yml`), which tags the manifest
bytes as they are. Then record the promotion here, with the digest read back.

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

**First Actions build, 2026-11-12 (`stills-t1-m1-rc1@sha256:9380cc424f8c76eeb1ac6d8ebfb15d98673b26fd0effdced528ddf76d4afd533`,
on `toolchain-t1@sha256:456b417b...`).** `stills-base.yml` run 43299859408 at main `9f2505d` (#644), `crane mutate
--append` of 4 bin tars. Final name after promotion: `stills-t1-m1-b2` (the section above); not promoted as of this record.
- **Layers added over the toolchain, compressed (GHCR manifest, read 2026-11-12):**
  - `beda3952af2d`, 5,020,773,145 bytes;
  - `638d992d951a`, 7,971,655,393 bytes;
  - `3575d8216c63`, 5,436,649,064 bytes;
  - `ed5b4e370251`, 126 bytes (the empty fourth bin, a tar of its own, not the shared empty layer).
  - In total 18,429,077,728 bytes (17.16 GiB). The first 22 layers are the same as `stills-t1-m1`'s; the 4 weight
    layers are new digests, so a worker that ran a 1.1.19 or older image pulls them once.
- **Worker:** `vivijure-backend:1.1.20@sha256:961ac67c96e0e971eb3ce90b9f5f9772a3592cf8c7fbdf7a6fd31cbdfa7c985d`, built FROM this
  digest (thin-image run 43312194021). CPU import smoke without the weight layers: main run 43319368393 (`smoke_ref` 1.1.20,
  `smoke_base` rc1), success.
- **GPU proof (lead, scratch endpoint `vj-stills-proof-1113`, H200 `HOPPER_141`, workersMax 1, made from the live endpoint's
  template with only the image changed, deleted after; the live endpoint was untouched, release v46 before and after).**
  Job `bd2ffcd4-cd1b-4e11-8726-6f8936be0e5c-u2`, a `preview` replay of studio film `film-63bbaae3` (project
  `cf1281_hand_mug`, seed 130605). The worker log has `@event mirror_skipped {"reason": "warm_stills"}`, then
  `@event mirror_done {"pulled": false, ...}`, and no `mirror_complete`. COMPLETED: keyframe
  `renders/cf1281_hand_mug/keyframes/bd2ffcd4-cd1b-4e11-8726-6f8936be0e5c-u2/shot_01.png`, no `reused`, plate count clean
  (expected 2, chosen 2), LoRAs A and B trained in the job. `executionTime` 2,271,623 ms, which is two LoRA trainings
  plus the draw, not a warm-draw figure. `delayTime` 961,769 ms, mostly the image pull; H200 capacity read LOW and 2 extra
  placements were THROTTLED.

Record for the first bake: each layer's compressed size, the push time, how long a new worker stays
`INITIALIZING`, and the cold job's `executionTime`. Those are the after numbers for #564.

## Rollback

Repin the previous thin image. It pulls the stills set from R2 as before. Nothing in the code depends on
the base: on an image without it, the stills sentinel is absent and the worker pulls.
