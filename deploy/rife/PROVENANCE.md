# RIFE inference code -- provenance

This file records exactly where the vendored RIFE inference code came from, so
the vendor directory is reproducible and auditable from this document alone.

## Why this code is vendored (the problem being fixed)

The baked image previously fetched the RIFE `rife/` package at Docker build time
from the Hugging Face Space `svjack/CogVideoX-5B-Space`, which carries **no
license at all**. Redistributing unlicensed third-party code inside a public
image is the compliance defect this vendor directory removes. We replace that
build-time `curl` with code committed to this repo from a canonical,
properly-licensed source at a pinned revision.

## Source

- Repository: `THUDM/CogVideo` (now `zai-org/CogVideo` on GitHub; same repo).
- Pinned commit: `7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2`
- Path in source: `inference/gradio_composite_demo/rife/`
- Source repository license: Apache License 2.0
  (Copyright 2024 CogVideo Model Team @ Zhipu AI). See `./LICENSE`.
- Algorithm / reference-code lineage: MIT, hzwer (Practical-RIFE /
  ECCV2022-RIFE) and Megvii Inc. See `./LICENSE.RIFE-MIT` and `./NOTICE`.

## Files vendored (the loader import closure)

The loader entry point is `from rife.RIFE_HDv3 import Model`
(`src/vivijure_backend/models.py`). Only the files in that import closure are
vendored; the unused training/variant files and the `pytorch_msssim/` subpackage
present in the upstream directory are intentionally NOT vendored (they are not
imported at inference time and would add unrelated third-party surface).

Import closure:
- `RIFE_HDv3.py`  imports `.warplayer`, `.IFNet_HDv3`, `.loss`
- `IFNet_HDv3.py` imports `.warplayer`
- `warplayer.py`  (THUDM variant; backward-warp helper)
- `loss.py`       (imported by `RIFE_HDv3`; pulls torchvision at import time)
- `__init__.py`   (empty package marker)

Each file is vendored byte-for-byte from the source revision above. Their git
blob SHA-1s at that revision (verify with `git hash-object <file>`):

| File          | git blob SHA-1                             |
|---------------|--------------------------------------------|
| `RIFE_HDv3.py`  | `182c78eb01c8e548c1aa44529307c859da870162` |
| `IFNet_HDv3.py` | `ad4a72751ff4edf7067ca0603b642defcb0aa557` |
| `warplayer.py`  | `ff796e897564961845ffb654f5006ae00c09f362` |
| `loss.py`       | `8ed7564ba1e3d98e2d3e4f6a461aa4cddf4150bc` |
| `__init__.py`   | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |

(`e69de29...` is gits canonical empty-blob hash; `__init__.py` is a 0-byte file.)

## Modifications

**None by us.** The `.py` files are vendored verbatim, unmodified by
skyphusion-labs, at the pinned revision. (Note: the upstream CogVideo copy is
itself a modified redistribution of the original hzwer/Megvii RIFE code; those
are CogVideo's Apache-2.0 changes, distinct from our verbatim vendoring. See
NOTICE.) No source edits are made, so Apache-2.0 Section 4(b) ("carry prominent
notices stating that You changed the files") does not apply. If any vendored file
is ever edited, add a prominent change notice to that file and update this
section in the same change.

## Compatibility note (weights)

This RIFE_HDv3 / IFNet (c=90, single-flownet) variant matches the seeded
`flownet.pkl` weights (the ECCV2022-RIFE HDv3 lineage; loaded with `rank=-1` to
strip the `module.` prefix), so the state_dict loads without a key mismatch. The
weights are a separate artifact documented in `THIRD_PARTY_MODELS.md` and
`deploy/licenses/imaginairy/rife/LICENSE`.

## How to re-fetch / re-verify the source

    REPO=https://raw.githubusercontent.com/zai-org/CogVideo
    SHA=7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2
    DIR=inference/gradio_composite_demo/rife
    for f in __init__.py RIFE_HDv3.py IFNet_HDv3.py warplayer.py loss.py; do
      curl -fsSL "$REPO/$SHA/$DIR/$f" -o "/tmp/$f"
      git hash-object "/tmp/$f"   # compare to the table above
    done
