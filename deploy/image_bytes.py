"""Print the compressed size in bytes of an image's linux/amd64 manifest (#636).

    IMG=ghcr.io/skyphusion-labs/vivijure-backend@sha256:<64 hex> python3 deploy/image_bytes.py

MODE=max-layer prints the largest compressed layer instead (stills-base.yml, the GHCR layer limit).

stills-base.yml uses MODE=max-layer. (thin-image.yml used the default mode to size its smoke pull until
#645; it now sizes the smoke with deploy/smoke_image.py, which leaves the weight layers out.) It reads the
registry manifest only (the config blob plus every layer), never a layer.

INSPECT is the raw-manifest probe, default "docker buildx imagetools inspect --raw" (needs the
GHCR login first); tests/test_image_bytes.py replaces it. An index picks its linux/amd64 entry.
Fails (exit 1) on a manifest with no layers or no linux/amd64 entry: an unknown size is never 0.
"""
import json
import os
import shlex
import subprocess
import sys

INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}


def raw(ref: str) -> dict:
    cmd = shlex.split(os.environ.get("INSPECT", "docker buildx imagetools inspect --raw")) + [ref]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout
    return json.loads(out)


def image_manifest(ref: str) -> dict:
    """The linux/amd64 image manifest of `ref` (an index resolves to its linux/amd64 entry)."""
    m = raw(ref)
    if m.get("mediaType") in INDEX_TYPES or "manifests" in m:
        pick = [
            d for d in m.get("manifests", [])
            if (d.get("platform") or {}).get("os") == "linux" and (d.get("platform") or {}).get("architecture") == "amd64"
        ]
        if not pick:
            raise SystemExit(f"{ref}: the index has no linux/amd64 manifest")
        repo = ref.split("@", 1)[0]
        m = raw(f"{repo}@{pick[0]['digest']}")
    if not (m.get("layers") or []):
        raise SystemExit(f"{ref}: the manifest lists no layers")
    return m


def manifest_bytes(ref: str) -> int:
    m = image_manifest(ref)
    return int((m.get("config") or {}).get("size", 0)) + sum(int(x["size"]) for x in m["layers"])


def max_layer_bytes(ref: str) -> int:
    """The largest compressed layer. stills-base.yml checks it against the GHCR layer limit from the
    registry, because bake_layers.verify-image reads `docker history` of a LOCAL image and the
    push-only build never loads one (#564, fc#2620)."""
    return max(int(x["size"]) for x in image_manifest(ref)["layers"])


if __name__ == "__main__":
    ref = os.environ.get("IMG", "")
    if "@sha256:" not in ref:
        sys.exit(f"IMG must be a digest ref (got {ref!r})")
    # MODE=max-layer prints the largest layer instead of the total (stills-base.yml).
    print(max_layer_bytes(ref) if os.environ.get("MODE") == "max-layer" else manifest_bytes(ref))
