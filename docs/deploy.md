# Deploy this backend

This page puts the **GPU render engine** online. It is the short path: supply your keys once, run
one script, and you have a RunPod Serverless endpoint that can train a character face model, draw
keyframes, turn them into video, and hand the finished film back to the Studio.

New here? The one-page picture of how the parts fit together is in
[constellation.md](constellation.md). You are standing up the **vivijure-backend** box on that
map: the cloud GPU the Studio hands its heavy work to.

## What you are deploying, in one breath

A single container image that runs on a rented GPU. You do **not** build it and you do **not**
download any models. The image is public and already has every model weight baked inside it, so
the only thing the deploy needs from you is a few keys.

## Before you start

You need two accounts and two small tools on your computer:

- A **RunPod** account (rents the GPU by the second). Sign up at runpod.io.
- A **Cloudflare** account with an **R2 bucket** (this is the shared drop box where the Studio
  puts a job and the GPU puts the finished video back). Use the **same bucket your Studio uses**.
- **`curl`** and **`python3`**, which almost every Mac and Linux computer already has. Check with
  `curl --version` and `python3 --version`.

You do not need Docker, and you do not need a GPU on your own computer. The GPU is rented.

## The keys you will paste in

The deploy asks for a small handful of values. Each one is for **your own** account, and you pay
your own bills. Every one is explained in plain words further down in
[Every setting explained](#every-setting-explained).

- Your RunPod **API key**.
- Two **R2 storage keys** (an access key and a secret) for the shared bucket.
- Your **R2 address** (or your Cloudflare **account id**, and the script builds the address for
  you).

## The three steps

```bash
# 1. Make your key file from the example, then open it and fill in your keys.
cp deploy.env.example deploy.env

# 2. Deploy. This is safe to re-run.
./deploy.sh

# 3. The script prints a line like:  RUNPOD_ENDPOINT_ID=abc123...
#    Copy that line into the Studio's deploy.env, then run the Studio's deploy.
```

That is it.

> **Keep `deploy.env` private.** It holds your keys. It is already set to be ignored by git, so it
> will not be committed. Never share it or paste it anywhere.

## What the script does for you

You do not have to click through the RunPod dashboard. The script:

1. Reads your keys from `deploy.env` and checks that nothing needed is blank.
2. Creates a RunPod **template** (the recipe: which image to run, and your R2 keys as the
   endpoint's settings). If a template with this name already exists, it updates it instead.
3. Creates a RunPod **Serverless endpoint** from that template, on the GPU you chose, set to
   **scale to zero** so an idle endpoint costs you nothing. If the endpoint already exists, it
   updates it in place.
4. Prints the **endpoint id** you paste into the Studio.

If anything is missing or wrong, it **stops right there** and tells you, so you never end up with
a half-built endpoint. Re-running after a fix is safe: it changes the same template and endpoint,
it does not pile up new ones.

## Picking a GPU

This render engine needs a **big, modern card**. It is not a preference; the work (drawing
keyframes, animating them, training a face model) needs the memory and the compute, and the image
is built for the current NVIDIA "Blackwell / Hopper" line. A small card either will not fit the
models in memory or will not run at all.

The standard for this backend is **H200 or B200** (both listed in `GPU_TYPE_IDS`, and the
scheduler uses whichever has a free card). Here is the part that surprises people: **the big card
is usually the cheaper choice.** RunPod Serverless bills **per second the job runs**, not per
hour. A faster card finishes the job in fewer seconds, and the shorter time more than pays for the
higher per-second rate. On top of that, an idle endpoint set to scale to zero costs **nothing** no
matter how premium the card is, because you only pay while a render is actually running.

So: do not "save money" by choosing a small card. For this GPU-heavy job, the small card is the
more expensive one, and it may not run at all.

To see the exact GPU id strings your account can use, open the RunPod dashboard, start adding a
Serverless endpoint, and look at the GPU picker; the names there are the strings you put in
`GPU_TYPE_IDS`.

## Every setting explained

Nobody should have to read the code to learn what a setting does. Here is every knob, what it is,
why it exists, and an example.

### Settings you put in `deploy.env` (the deploy inputs)

| Setting | What it is | Why it exists | Example |
|---|---|---|---|
| `RUNPOD_API_KEY` | Your RunPod read/write API key. | It is how the script creates and updates your endpoint. | `rpa_ABC123...` |
| `R2_ACCESS_KEY_ID` | The public half of your R2 storage key. | The GPU reads the job and writes the video back to R2 with it. | `a1b2c3d4...` |
| `R2_SECRET_ACCESS_KEY` | The secret half of your R2 storage key. | Same as above; this is the password part. Keep it secret. | `9f8e7d...` |
| `R2_BUCKET` | The name of your R2 bucket. | The shared drop box. Must match the Studio's bucket. | `vivijure` |
| `R2_ENDPOINT` | The web address of your R2 storage. | Tells the GPU where your bucket lives. Leave blank to build it from your account id. | `https://<accountid>.r2.cloudflarestorage.com` |
| `CLOUDFLARE_ACCOUNT_ID` | Your Cloudflare account id. | Only used to build `R2_ENDPOINT` for you when you leave that blank. Not a secret. | `0123456789abcdef...` |
| `BACKEND_IMAGE` | The published render image to run. | This is the ready-made image with models baked in. Leave as-is unless you forked the code. | `ghcr.io/skyphusion-labs/vivijure-backend:0.4.9` |
| `GPU_TYPE_IDS` | The GPU card(s) allowed, comma-separated. | Lets the scheduler pick whichever big card has stock. See [Picking a GPU](#picking-a-gpu). | `NVIDIA H200,NVIDIA B200` |
| `ENDPOINT_NAME` | The name shown in your RunPod dashboard. | Lets the script find and update the same endpoint on a re-run. | `vivijure-backend` |
| `WORKERS_MAX` | The most workers that can run at once. | Each render holds one worker until it finishes; this caps how many renders run in parallel. | `3` |
| `WORKERS_MIN` | The fewest workers kept always-on. | Keep it `0` so an idle endpoint costs nothing. | `0` |
| `CONTAINER_DISK_GB` | Scratch disk per worker, in GB. | The image is large, so the worker needs room. `100` is a safe floor. | `100` |
| `IDLE_TIMEOUT` | Seconds a warm worker waits for the next job before shutting down. | A small buffer so back-to-back renders reuse a warm worker. | `5` |
| `EXECUTION_TIMEOUT_MS` | Optional hard stop for a single job, in milliseconds. | A stuck job would otherwise hold a paid worker forever; this kills it. Leave blank for no limit. | `1800000` (30 min) |

### Settings the GPU worker reads at runtime

These are set for you: `deploy.sh` puts the R2 four onto the endpoint, and the rest are baked into
the image. You normally never touch them. They are listed so the contract is complete.

| Variable | What it is | Why it exists | Set by |
|---|---|---|---|
| `R2_ENDPOINT` | The R2 storage address. | The one place the worker reads the job from and writes the film to. | `deploy.sh` (from your `deploy.env`) |
| `R2_ACCESS_KEY_ID` | Public half of the R2 key. | Lets the worker sign its storage requests. | `deploy.sh` |
| `R2_SECRET_ACCESS_KEY` | Secret half of the R2 key. | The one secret the worker holds. It never holds any other credential. | `deploy.sh` |
| `R2_BUCKET` | The bucket name. | Which bucket to read and write. | `deploy.sh` |
| `HF_HOME` | Folder for the model cache. | Points the model loader at the baked-in weights. | baked into the image (`/opt/models/hf-cache`) |
| `VJ_MODELS_ROOT` | The root folder for all baked models. | Where the worker looks for every model file. | baked into the image (`/opt/models`) |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_DATASETS_OFFLINE` | "Never phone home" switches, all set to `1`. | Force the worker to read weights from the baked image only, never the internet. This is why it runs with no extra downloads and no surprise data leaving your box. | baked into the image |
| `PYTHONPATH` | Where the Python code lives. | So `import vivijure_backend` works. | baked into the image (`/opt/vivijure`) |
| `PYTORCH_CUDA_ALLOC_CONF` | A GPU-memory tuning flag. | Gives the GPU memory allocator more headroom so big renders do not fragment memory. | baked into the image |

### Advanced runtime toggles (leave alone unless you know why)

The published image is self-contained, so you should never need these. They exist for
people who build a custom image or want to tune the model path.

| Variable | Default | What it does |
|---|---|---|
| `VJ_MODEL_VERSION` | `1` | A cache-stamp number. Bump it only if you rebuild with a changed model set, so warm workers refresh instead of using an old cache. |
| `VJ_I2V_DISTILL` | `1` | Turns the fast 4-step video path on. Set to `0` to force the slower full-quality video path for every job. |
| `VJ_I2V_FP8` | `1` | Turns on a memory-saving number format for the video model. Set to `0` to use the larger, higher-precision weights. |
| `HF_TOKEN` | unset | A HuggingFace token. It is used only when **building** an image; at runtime the offline switches make it inert. You do not set this to run the published image. |

## Advanced: build your own image (most people skip this)

You do **not** need to build anything to run this backend; the published image is ready. Building
your own image is only for people who forked the render code and want to ship their change.

Be honest with yourself about the cost first: this is a **model-baked image**. The build stages a
large, curated set of model weights, packs them into layers, and produces an image around
**87 GB**. That needs a build machine with a very large disk (hundreds of GB free) and the curated
model "seed" staged in storage. Because of that, the image is built by the project's automated
build (a GitHub Actions release on a version tag), not on a laptop. The mechanics, the disk needs,
and the exact steps are in [deploy/README.md](../deploy/README.md) and
[operations.md](operations.md). Once you have pushed your own image to your own registry, point
`BACKEND_IMAGE` in `deploy.env` at it and run `./deploy.sh` as usual. If your registry is private,
you also add a RunPod container-registry credential; that path is covered in
[operations.md](operations.md).

## If something goes wrong

- The script prints a clear error and stops. Read the last line; it names what is missing or what
  RunPod rejected.
- Re-running is safe. Fix the value in `deploy.env` and run `./deploy.sh` again.
- A wrong GPU id is the most common trip-up. Open the RunPod dashboard's GPU picker and copy the
  exact name into `GPU_TYPE_IDS`.
- For the deeper operator view (the object-key map, the progress channel, failure modes), see
  [operations.md](operations.md); for GPU sizing and the account worker cap, see
  [runpod-endpoint-config.md](runpod-endpoint-config.md).
