#!/bin/bash
# Remove every bake input and output from the box, between two bakes. This is
# housekeeping, not erasure: blocks stay on the NVMe until they are overwritten.
# Destroy the box when you want it gone. Run as bake.
set -eu
rm -rf /run/bake-creds/* "$HOME/build"
docker buildx prune -af >/dev/null
docker system prune -af --volumes >/dev/null
echo "docker_images=$(docker images -q | wc -l) creds=$(ls -A /run/bake-creds | wc -l)"
df -h / | tail -1
