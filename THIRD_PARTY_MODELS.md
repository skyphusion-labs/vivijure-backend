# Third-party model licenses (baked image)

The `ghcr.io/skyphusion-labs/vivijure-backend` image bakes model weights into the
layer so a worker loads them offline. That image is a PUBLIC package, so shipping
it redistributes every embedded model. This document records each baked model, the
exact upstream source and pinned revision, and its license. The full license text
for every model lives under `deploy/licenses/<repo_id>/LICENSE` and is copied into
the image at `/opt/models/licenses` at build time (the Dockerfile COPY is tracked
separately, issue #127).

The authoritative list of what gets baked is `DEFAULT_SPECS` in
`src/vivijure_backend/models.py`. The nine entries below are that list. Each license
was verified at the pinned revision shown (the Hugging Face commit SHA, or, for the
two repos that are access-restricted on the Hub, the upstream source the weights
originate from).

## Models

| Role | repo_id | Pinned revision | License | Source |
|---|---|---|---|---|
| Wan2.2 i2v engine | Wan-AI/Wan2.2-I2V-A14B-Diffusers | `596658fd9ca6b7b71d5057529bbf319ecbc61d74` | Apache-2.0 | https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers |
| SDXL keyframe base | SG161222/RealVisXL_V5.0 | `ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc` | CreativeML Open RAIL++-M | https://huggingface.co/SG161222/RealVisXL_V5.0 |
| Few-step keyframe LoRA | ByteDance/Hyper-SD | `bc08d970a87c74c71209491d64e3525845698863` | CreativeML Open RAIL++-M (SDXL variant) | https://huggingface.co/ByteDance/Hyper-SD |
| i2v distill LoRA | lightx2v/Wan2.2-Lightning | `18bccf8884ec0a078eed79785eb4ef13ea16ce1e` | Apache-2.0 | https://huggingface.co/lightx2v/Wan2.2-Lightning |
| Face identity | InstantX/InstantID | `57b32dfee076092ad2930c71fd6d439c2c3b1820` | Apache-2.0 | https://huggingface.co/InstantX/InstantID |
| IP-Adapter | h94/IP-Adapter | `018e402774aeeddd60609b4ecdb7e298259dc729` | Apache-2.0 | https://huggingface.co/h94/IP-Adapter |
| OpenPose ControlNet | xinsir/controlnet-openpose-sdxl-1.0 | `23f966cd5cfdd3f7729c903e243d87152162d2b7` | Apache-2.0 | https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0 |
| RIFE frame interp | imaginairy/rife | upstream (see note) | MIT | https://huggingface.co/imaginairy/rife |
| Face restore | TencentARC/GFPGANv1.4 | upstream (see note) | Apache-2.0 | https://huggingface.co/TencentARC/GFPGANv1.4 |

### Notes on the two access-restricted repos

`imaginairy/rife` and `TencentARC/GFPGANv1.4` require Hub authentication, so a pinned
Hub commit SHA cannot be recorded anonymously. Their licenses were verified against
the canonical upstream sources the weights come from, and the upstream license text
is what we redistribute:

- **RIFE** (`flownet.pkl`, the ECCV2022-RIFE HDv3 weights vendored as the `rife`
  module): MIT, Copyright (c) Megvii Inc. Upstream:
  https://github.com/hzwer/ECCV2022-RIFE (`LICENSE`). The model card and the loader
  comment both record these weights as MIT and redistribution-clean. The RIFE
  inference *code* (the importable `rife/` package) is a SEPARATE artifact,
  vendored at `deploy/rife/` from THUDM/CogVideo at a pinned revision
  (Apache-2.0, with hzwer/Megvii MIT lineage); see the "Third-party inference
  code" section below. It is no longer fetched at build time from the
  unlicensed `svjack/CogVideoX-5B-Space` Space.
- **GFPGAN** (`GFPGANv1.4.pth`): Apache-2.0, Copyright (C) 2021 THL A29 Limited (a
  Tencent company). Upstream: https://github.com/TencentARC/GFPGAN (`LICENSE`). The
  shipped LICENSE is the full upstream file, which is Apache-2.0 plus the upstream
  third-party-component notices.

### Models NOT bundled

`sczhou/CodeFormer` (S-Lab Non-Commercial) is a deploy-time opt-in face restorer,
never baked into the default image. It is referenced by the loader but is not part
of `DEFAULT_SPECS` and so is not redistributed here. Anyone enabling it at deploy
time is responsible for its non-commercial terms.

## Wan2.2 modifications statement (Apache-2.0 Section 4(b))

Apache-2.0 Section 4(b) requires modified files to carry prominent notices stating
that they were changed. The Wan2.2 i2v weights baked into this image have been
modified by skyphusion-labs:

1. Numeric recast from fp32 to bf16.
2. Numeric recast from bf16 to fp8 (`float8_e4m3fn`), redistributed as the
   `Wan-AI/Wan2.2-I2V-A14B-Diffusers-fp8` variant.

These are precision/quantization recasts of the tensor values only. There is **no**
change to the model architecture, layer topology, configuration, or any accompanying
source code, and the original diffusers layout is preserved. All upstream copyright,
patent, trademark, and attribution notices are retained. The same notice ships with
the weights at `deploy/licenses/Wan-AI/Wan2.2-I2V-A14B-Diffusers/MODIFICATIONS.txt`.

## CreativeML Open RAIL++-M use restrictions (RealVisXL and Hyper-SD)

Two baked models are licensed under CreativeML Open RAIL++-M, not a permissive
license:

- `SG161222/RealVisXL_V5.0` (the SDXL keyframe base), a derivative of SDXL.
- `ByteDance/Hyper-SD` (the few-step keyframe LoRA). Its `LICENSE` is segmented by
  base model; the SDXL variant we bake (`Hyper-SDXL-8steps-lora.safetensors`) falls
  under the "other SD-related models" section, which is CreativeML Open RAIL++-M,
  Copyright (c) 2024 Bytedance Inc.

CreativeML Open RAIL++-M is permissive on IP rights but carries **use-based
restrictions** (its Attachment A). The license requires (paragraph 5 and the
Distribution conditions) that these restrictions be passed through as an enforceable
provision to every downstream user, and that downstream users be given notice that
the model and its derivatives are subject to them.

### Flow-down to image users

By using `ghcr.io/skyphusion-labs/vivijure-backend`, or any output produced with the
RealVisXL or Hyper-SD weights it bakes, **you are bound by the Attachment A use
restrictions reproduced below.** These restrictions flow down to you as a downstream
user and you must in turn pass them through to anyone you distribute the model or its
derivatives to. This is in addition to, not instead of, the full license text in
`deploy/licenses/SG161222/RealVisXL_V5.0/LICENSE` and
`deploy/licenses/ByteDance/Hyper-SD/LICENSE`.

### Attachment A: Use Restrictions

You agree not to use the Model or Derivatives of the Model:

1. In any way that violates any applicable national, federal, state, local or
   international law or regulation;
2. For the purpose of exploiting, harming or attempting to exploit or harm minors in
   any way;
3. To generate or disseminate verifiably false information and/or content with the
   purpose of harming others;
4. To generate or disseminate personal identifiable information that can be used to
   harm an individual;
5. To defame, disparage or otherwise harass others;
6. For fully automated decision making that adversely impacts an individual's legal
   rights or otherwise creates or modifies a binding, enforceable obligation;
7. For any use intended to or which has the effect of discriminating against or
   harming individuals or groups based on online or offline social behavior or known
   or predicted personal or personality characteristics;
8. To exploit any of the vulnerabilities of a specific group of persons based on
   their age, social, physical or mental characteristics, in order to materially
   distort the behavior of a person pertaining to that group in a manner that causes
   or is likely to cause that person or another person physical or psychological
   harm;
9. For any use intended to or which has the effect of discriminating against
   individuals or groups based on legally protected characteristics or categories;
10. To provide medical advice and medical results interpretation;
11. To generate or disseminate information for the purpose to be used for
    administration of justice, law enforcement, immigration or asylum processes, such
    as predicting an individual will commit fraud/crime commitment (e.g. by text
    profiling, drawing causal relationships between assertions made in documents,
    indiscriminate and arbitrarily-targeted use).

## Third-party inference code (not a model weight)

One third-party CODE component is vendored into the image alongside the model
weights: the RIFE frame-interpolation inference code (the importable `rife/`
package). It is code, not weights, so it is tracked here separately from the
model table above.

| Component | Vendored source | Pinned revision | License | Path |
|---|---|---|---|---|
| RIFE HDv3 inference code | THUDM/CogVideo (`zai-org/CogVideo`), path `inference/gradio_composite_demo/rife/` | `7a1af7154511e0ce4e4be8d62faa8c5e5a3532d2` | Apache-2.0 (code), with hzwer/Megvii MIT lineage | `deploy/rife/` |

Only the loader import closure (`RIFE_HDv3.py`, `IFNet_HDv3.py`, `warplayer.py`,
`loss.py`, `__init__.py`) is vendored, byte-for-byte and unmodified. Full
attribution and reproducibility detail live in `deploy/rife/`: the Apache-2.0
`LICENSE` (verbatim from the source repo), the restored Megvii/hzwer
`LICENSE.RIFE-MIT`, the `NOTICE`, and `PROVENANCE.md` (exact source revision,
the vendored file list with per-file git blob SHA-1s, and the unmodified
statement). This replaces the previous build-time fetch of the same code from
the unlicensed `svjack/CogVideoX-5B-Space` Space.

## Maintenance

When `DEFAULT_SPECS` changes (a model added, removed, or its pinned source changed),
update this file and `deploy/licenses/` in the same change so the baked image stays
compliant by construction. Verify each license at the new pinned revision; do not
assume a model card tag without checking the actual license file at that revision.
