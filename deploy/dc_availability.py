#!/usr/bin/env python3
"""List the RunPod datacenters that can host the preloaded-volume strategy (issue #55 Phase C):
network-volume-capable AND currently advertising a GPU at/above our floor.

The floor is **H200 (141 GB)**: Wan2.2-I2V-A14B is a ~28B two-expert MoE whose full-step path
OOMs even an H100-80GB; a 96 GB RTX 6000 PRO only survives via CPU-offload (draft-only). So the
volume coverage targets H200-class DCs. Run this before provisioning (and periodically) so the
DC allow-list never drifts silently from real availability.

Read-only: a single GraphQL query, no mutations, no spend.

    RUNPOD_API_KEY=... python deploy/dc_availability.py            # H200+ floor (default)
    RUNPOD_API_KEY=... python deploy/dc_availability.py --include-rtx6000   # also show 96 GB DCs
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

GRAPHQL = "https://api.runpod.io/graphql"

# GPU type ids (RunPod `gpuTypeId`) by tier. H200+ is the render floor; RTX 6000 PRO (96 GB) is
# draft-only via CPU-offload and shown only with --include-rtx6000. AMD MI300X is excluded (our
# image is CUDA/cu128).
H200_PLUS = {
    "NVIDIA B300 SXM6 AC",
    "NVIDIA B200",
    "NVIDIA H200 NVL",
    "NVIDIA H200",
}
RTX_6000 = {
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
}


def _query(api_key: str) -> list[dict]:
    body = json.dumps({
        "query": "{ dataCenters { id name location storageSupport "
                 "gpuAvailability { gpuTypeId available } } }"
    }).encode()
    # RunPod's API sits behind Cloudflare, which 403s (error 1010) urllib's default User-Agent;
    # send an explicit one so the request isn't bot-blocked.
    req = urllib.request.Request(
        f"{GRAPHQL}?api_key={api_key}", data=body,
        headers={"Content-Type": "application/json", "User-Agent": "vivijure-deploy/1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read())
    if "errors" in out:
        sys.exit("RunPod API error: " + json.dumps(out["errors"])[:300])
    return out["data"]["dataCenters"]


def _short(g: str) -> str:
    return (g.replace("NVIDIA ", "").replace(" SXM6 AC", "")
             .replace(" Blackwell", "").replace(" Workstation Edition", " WK")
             .replace(" Max-Q WK", " MaxQ").replace(" Server Edition", "").strip())


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("set RUNPOD_API_KEY")
    include_rtx = "--include-rtx6000" in sys.argv
    wanted = H200_PLUS | (RTX_6000 if include_rtx else set())

    covered, skipped_no_storage = [], []
    for c in _query(api_key):
        avail = {g["gpuTypeId"] for g in (c.get("gpuAvailability") or []) if g.get("available")}
        hit = sorted(_short(g) for g in (avail & wanted))
        if not hit:
            continue
        (covered if c.get("storageSupport") else skipped_no_storage).append((c["id"], hit))

    floor = "H200+ or RTX 6000" if include_rtx else "H200+ (141 GB)"
    print(f"# Network-volume-capable DCs with {floor} (provisioning candidates):")
    for dc, gpus in sorted(covered):
        print(f"  {dc:<11} {', '.join(gpus)}")
    print(f"\nVOLUME_DCS=\"{' '.join(dc for dc, _ in sorted(covered))}\"")
    if skipped_no_storage:
        print(f"\n# Has the GPU but NO network-volume support (cannot host a volume):")
        for dc, gpus in sorted(skipped_no_storage):
            print(f"  {dc:<11} {', '.join(gpus)}")
    print("\n# NOTE: live snapshot; availability fluctuates. Re-run before provisioning.")


if __name__ == "__main__":
    main()
