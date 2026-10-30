# Global volume for stills

The keyframe worker skips the bake.
It skips the cold pull.
It reads stills from a global volume.

This page is the process.
Issue #530.

## Why

The baked image is about 87 GB.
A cold pull is about 100 GB.
That pull pinned us to one datacenter.

A global volume mounts on GPU serverless at `/runpod-volume`.
A worker can land in any datacenter.
Stage the stills once.
Every worker reads the same files.

## Image

Build the toolchain only.
GitHub Actions does this on `ubuntu-latest`.
See `.github/workflows/thin-image.yml`.
A public runner has about 90 GB free.
The toolchain is CUDA, torch, and deps.
It is not the 87 GB bake.

```
docker build -f deploy/runtime.Dockerfile --target toolchain -t vivijure-backend-toolchain .
docker build -f deploy/Dockerfile.thin --build-arg TOOLCHAIN_REF=vivijure-backend-toolchain .
```

The thin image has no seed COPY.
It does not stamp `.vj-baked`.
Do not ship the baked runtime for this path.

## Volume

Create a new global volume for stills.
Do not reuse another product volume.

Mount it on the GPU serverless endpoint at `/runpod-volume`.
A global volume and a network volume cannot share one endpoint.

Set these env vars.

| Var | Value |
|---|---|
| `VJ_VOLUME_ROOT` | `/runpod-volume` |
| `VJ_VOLUME_KIND` | `global` |
| `VJ_VOLUME_SELF_PRELOAD` | `0` |

The worker also uses `/runpod-volume` when `VJ_VOLUME_ROOT` is unset and that path is a directory.

## Stage

Fill the volume once, offline.
Use a writer that is not a GPU serverless worker.
A first-worker lock is not safe here.
A global volume has no file lock and no hard link.

```
VJ_MODELS_ROOT=/runpod-volume python -m vivijure_backend.harness.models_mirror --preload --stills --flatten
```

Or run `scripts/stage_global_volume.py`.

`--stills` skips Wan.
Preview and keyframe need the base set only.
`--flatten` copies each HF-cache symlink to a real file.
`from_pretrained` then reads real files.

The writer writes `.vj-mirror-complete`.
It does not write the i2v sentinel.
The worker names that state `READY_NO_I2V`.

Do not set `VJ_VOLUME_ROOT` on the writer.
That key is the worker read path.

## Worker

A thin worker sees no `.vj-baked`.
It then checks the volume.
A ready stills volume repoints `HF_HOME` and `VJ_MODELS_ROOT`.
It skips R2.

If the volume is empty, the worker does not fill it.
It falls back to R2 on local disk.
That fallback is the miss path.
It is not the plan.

## Hydrate

Reading weights straight off a global volume is slow.
Measured on film-a9939630: the 9.99 GiB UNet shard took 404 s, about 26 MB/s.
One job reads that shard twice: LoRA training, then the keyframe pipeline.

So a stills job first copies what it reads to container disk.
See `harness/volume_hydrate.py`.
The copy happens once per worker.
The worker then loads from local disk.

The local tree is always complete.
It starts as symlinks into the volume.
A copied file replaces its symlink.
A time cap, a full disk or a read error leaves a slower job, not a missing file.

| Var | Default | Meaning |
|---|---|---|
| `VJ_HYDRATE` | on when `VJ_VOLUME_KIND=global` | `0` turns it off. `1` turns it on for any volume. |
| `VJ_HYDRATE_ROOT` | `/tmp/vj-hydrate` | Local root. |
| `VJ_HYDRATE_THREADS` | `8` | Parallel 64 MiB reads per file. |
| `VJ_HYDRATE_MAX_SECONDS` | `600` | After this, the rest stays on the volume. |

Read `hydrate_file` for the MB/s of each file.
Read `load_done` for the seconds of each model load.
Both appear as `@event` lines and in the progress snapshot.

## After the thin image is live

Unpin the datacenter allow list.
The Iceland pin was a cold-pull dodge.
It is not a product pin.

Do not raise `workersMin` for a volume fill.
Stage offline first.

## Cost

RunPod bills storage per GB-month.
Reads and writes have their own class fees.
Stage once.
Workers then read.

Do not attach or fill a volume from a docs page.
Create the volume when you cut the thin image.
