**Fix: deploy, promote and image-bake move to RunPod REST v2 (fleet-chezmoi#2085).**

RunPod REST v1 retired on 2026-11-15. v2 lives on its own host, `https://api.runpod.io/v2`, and every list there is cursor-paginated.

- **`deploy.sh` creates or updates one endpoint, with no template.** On v2 an endpoint carries its own `image`, `env` and `disk`; a v2 template is applied once at create and keeps no link to the endpoint. The script finds the endpoint by name across every page, then PATCHes or POSTs `/v2/serverless`.
  - GPUs are chosen by `gpu.pools`. `GPU_TYPE_IDS` still works, mapped to pools from RunPod's GPU catalog; the new `GPU_POOLS` names pools directly.
  - `EXECUTION_TIMEOUT_MS` blank now means one hour. v2 has no "no limit", and its own default is far shorter than a render.
  - `flashboot` is set to `FLASHBOOT`. A leftover `<name>-template` from an older deploy is no longer used.
- **`deploy/runpod_verify.py`:** promote re-pins the ENDPOINT image and reads it back. Flush and restore PATCH `workers.max` and `workers.min` on `/v2/serverless/{id}`.
- **`scripts/runpod_image_bake.py`:** the CPU pod body uses v2 names (`cpu`, `image`, `disk`, `entrypoint`, `cmd`). A pod's state is read from `status`. Pods-by-name lists every page and matches the name locally, because v2's filters are tag-based.
- **`deploy.env.example`:** `GPU_TYPE_IDS` is quoted. Unquoted, the file stopped `deploy.sh` with "command not found" when it was sourced.
- `tests/test_no_rest_v1_2085.py` fails if a shipped file names the v1 host again.
