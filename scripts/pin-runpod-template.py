#!/usr/bin/env python3
"""Pin the RunPod endpoint template to a built GHCR image (the "deploy" step).

Decoupled from the image build (which happens on a `backend-vX.Y.Z` git tag, build + push only,
see ../.github/workflows/release.yml). Run this deliberately when you want the live RunPod endpoint to start
pulling a new image on its next cold start.

  RUNPOD_API_KEY=...  RUNPOD_TEMPLATE_ID=...  \\
    python3 scripts/pin-runpod-template.py ghcr.io/skyphusion-labs/vivijure-backend:0.1.0

On mindcrime RUNPOD_API_KEY is available after `source ~/.bashrc`; set RUNPOD_TEMPLATE_ID to the
vivijure-backend endpoint's template id.

RunPod's saveTemplate mutation requires the FULL template object (containerDiskInGb, dockerArgs,
env, name, volumeInGb, ... are non-null SaveTemplateInput fields), so we fetch the template,
splice imageName onto it, strip __typename, and POST it back. The default urllib User-Agent trips
Cloudflare error 1010 (403), so we send a named UA. The splice + strip is a pure helper so it
unit-tests without the network.
"""
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UA = "vivijure-backend-deploy/1.0 (+https://github.com/SkyPhusion/vivijure-backend)"


def strip_typename(o):
    """Recursively drop GraphQL `__typename` keys so the fetched object is a valid input."""
    if isinstance(o, dict):
        o.pop("__typename", None)
        for v in o.values():
            strip_typename(v)
    elif isinstance(o, list):
        for v in o:
            strip_typename(v)
    return o


def prepare_template(tpl: dict, new_img: str) -> dict:
    """The SaveTemplateInput for re-pinning: the fetched template with imageName spliced to
    `new_img` and `__typename` stripped. Pure (no network), so it is unit-testable.

    CRITICAL: a pin must change ONLY imageName and carry every other field through UNCHANGED --
    especially `containerRegistryAuthId` (the GHCR pull credential, which lives on the template and
    is already correct). The fetch query above requests it so it round-trips here untouched. Do NOT
    omit it (saveTemplate would clear it -> a present-but-empty cred makes RunPod attempt an empty
    login and FAIL the pull, even for public images) and do NOT hardcode it (clobbers the real cred).
    Just preserve whatever was fetched."""
    tpl = json.loads(json.dumps(tpl))  # copy; never mutate the caller's dict
    tpl["imageName"] = new_img
    return strip_typename(tpl)


def gql(api_key, payload):
    req = Request(
        "https://api.runpod.io/graphql",
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                 "User-Agent": UA},
        data=json.dumps(payload).encode("utf-8"),
    )
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")
    except URLError as e:
        sys.exit(f"network error: {e.reason}")


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: pin-runpod-template.py <registry/owner/image:tag>")
    new_img = sys.argv[1]
    api_key = os.environ.get("RUNPOD_API_KEY") or sys.exit("RUNPOD_API_KEY not set")
    tpl_id = os.environ.get("RUNPOD_TEMPLATE_ID") or sys.exit("RUNPOD_TEMPLATE_ID not set")

    fetch = gql(api_key, {
        "query": "{ myself { podTemplates { id name imageName containerDiskInGb "
                 "dockerArgs volumeInGb volumeMountPath ports env { key value } "
                 "isServerless category readme containerRegistryAuthId } } }"
    })
    if "errors" in fetch:
        sys.exit(f"fetch errors: {json.dumps(fetch['errors'])}")

    templates = ((fetch.get("data") or {}).get("myself") or {}).get("podTemplates") or []
    tpl = next((t for t in templates if t.get("id") == tpl_id), None)
    if tpl is None:
        sys.exit(f"template {tpl_id!r} not in podTemplates; known: {[t.get('id') for t in templates]}")

    print(f"Pinning RunPod template {tpl_id} ({tpl.get('name')}): "
          f"{tpl.get('imageName')} -> {new_img}")
    save = gql(api_key, {
        "query": "mutation Save($input: SaveTemplateInput!) { saveTemplate(input: $input) { id imageName } }",
        "variables": {"input": prepare_template(tpl, new_img)},
    })
    if "errors" in save:
        sys.exit(f"save errors: {json.dumps(save['errors'])}")
    print(f"OK: template {tpl_id} now pinned to {new_img}.")
    print("New endpoint workers pull this on their next cold start.")


if __name__ == "__main__":
    main()
