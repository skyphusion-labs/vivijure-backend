"""Capability-aware model loading.

Turns `device.py`'s card facts into actual model loads in the right precision, and keeps
them resident: this is a persistent server, each model role loads once into VRAM and is
reused across renders, so only the first render on a fresh worker pays load time.

Clean-room: every loader targets the model's own public API (diffusers, torchao, the model
hubs) built from those docs, not from any prior pipeline.

Quant reality (verified June 2026):
  - SDXL is a UNet. The 4-bit engines (Nunchaku/SVDQuant) only cover DiTs (FLUX, Qwen-Image,
    SANA, Z-Image), so SDXL tops out at fp8 (MXFP8 on Blackwell, plain fp8 on Hopper) via
    torchao. NVFP4 is NOT available for SDXL.
  - A DiT keyframe model (FLUX/Qwen-Image) WOULD unlock NVFP4 (~1.7x on B200), at the cost of
    re-homing the identity stack (LoRA training, InstantID, IP-Adapter) off the SDXL ecosystem.
    That trade is `ModelFamily.DIT`; left as a deliberate future option.
  - Wan i2v is a video DiT: fp8 on both archs (4-bit-for-video is still young).

Heavy imports (torch / diffusers / torchao) are deferred into the load methods so this module
imports and unit-tests on a CPU box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .device import Device, Quant, current


class ModelRole(str, Enum):
    """A role a model plays in the pipeline; the key into the spec table and the load cache."""
    KEYFRAME_BASE = "keyframe_base"      # SDXL checkpoint that draws the still
    KEYFRAME_FEWSTEP = "keyframe_fewstep"  # distill LoRA/unet (Hyper-SD / DMD2) for 4-8 step
    I2V = "i2v"                          # Wan 2.2 image-to-video
    I2V_DISTILL = "i2v_distill"          # Wan2.2-Lightning distill LoRA (few-step i2v)
    INSTANTID = "instantid"              # face identity (insightface + ControlNet)
    IP_ADAPTER = "ip_adapter"            # per-slot identity conditioning
    CONTROLNET_POSE = "controlnet_pose"  # OpenPose: separates bodies in multi-char shots
    FRAME_INTERP = "frame_interp"        # RIFE: finishing-stage frame interpolation (16fps -> smooth)
    FACE_RESTORE = "face_restore"        # blind face restorer: relock identity over the i2v frames


class ModelFamily(str, Enum):
    """The model's architecture class, which decides the precision it can be loaded at."""
    SDXL_UNET = "sdxl_unet"  # fp8 ceiling (no 4-bit engine)
    DIT = "dit"              # FLUX/Qwen/SANA: NVFP4-capable on Blackwell
    VIDEO_DIT = "video_dit"  # Wan: fp8 on both archs
    AUX = "aux"              # adapters / controlnets / detectors: load at base dtype

    @property
    def fp4_capable(self) -> bool:
        return self is ModelFamily.DIT


@dataclass(frozen=True)
class ModelSpec:
    """Where a model role's weights live (HF repo, optional subfolder / weight file) and its family."""
    role: ModelRole
    repo_id: str
    family: ModelFamily
    subfolder: str | None = None
    note: str = ""
    weight_name: str | None = None   # specific file within repo_id (e.g. a per-step LoRA variant)


# Default model set. These are public Hugging Face repo ids; swap per project via overrides.
DEFAULT_SPECS: dict[ModelRole, ModelSpec] = {
    ModelRole.KEYFRAME_BASE: ModelSpec(
        ModelRole.KEYFRAME_BASE, "SG161222/RealVisXL_V5.0", ModelFamily.SDXL_UNET,
        note="photoreal-leaning SDXL default; anime alt cagliostrolab/animagine-xl-4.0",
    ),
    ModelRole.KEYFRAME_FEWSTEP: ModelSpec(
        ModelRole.KEYFRAME_FEWSTEP, "ByteDance/Hyper-SD", ModelFamily.AUX,
        weight_name="Hyper-SDXL-8steps-lora.safetensors",
        note="few-step distill LoRA (Hyper-SDXL); the 8-step variant is forgiving and still renders "
             "well at 4-6 steps, so one warm adapter serves both draft (4) and standard (8). DMD2 is "
             "the alt. Keyframes get animated, so 4-8 steps is plenty",
    ),
    ModelRole.I2V: ModelSpec(
        ModelRole.I2V, "Wan-AI/Wan2.2-I2V-A14B-Diffusers", ModelFamily.VIDEO_DIT,
    ),
    ModelRole.I2V_DISTILL: ModelSpec(
        ModelRole.I2V_DISTILL, "lightx2v/Wan2.2-Lightning", ModelFamily.AUX,
        weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        note="4-step distill LoRA; no root-level .safetensors (repo uses subdirs); I2V path confirmed "
             "from R2 listing. Watch diffusers LoRA-load compat (issue #12535), LightX2V/DiffSynth is the fallback.",
    ),
    ModelRole.INSTANTID: ModelSpec(ModelRole.INSTANTID, "InstantX/InstantID", ModelFamily.AUX),
    ModelRole.IP_ADAPTER: ModelSpec(ModelRole.IP_ADAPTER, "h94/IP-Adapter", ModelFamily.AUX),
    ModelRole.CONTROLNET_POSE: ModelSpec(
        ModelRole.CONTROLNET_POSE, "xinsir/controlnet-openpose-sdxl-1.0", ModelFamily.AUX,
    ),
    ModelRole.FRAME_INTERP: ModelSpec(
        ModelRole.FRAME_INTERP, "imaginairy/rife", ModelFamily.AUX,
        weight_name="flownet.pkl",
        note="RIFE recursive-2x frame interpolation for the finishing stage (Practical-RIFE / "
             "ECCV2022-RIFE weights, MIT -- redistribution-clean). Light next to i2v.",
    ),
    ModelRole.FACE_RESTORE: ModelSpec(
        ModelRole.FACE_RESTORE, "TencentARC/GFPGANv1.4", ModelFamily.AUX,
        weight_name="GFPGANv1.4.pth",
        note="blind face restorer for the finishing stage (relock identity over the i2v frames). "
             "GFPGAN is the default; CodeFormer (sczhou/CodeFormer) is higher quality but S-Lab "
             "NON-COMMERCIAL, so it is a deploy-time spec swap, never the bundled default. "
             "VERIFY the chosen model's license before seeding its weights to R2.",
    ),
}


def quant_for(family: ModelFamily, device: Device) -> Quant:
    """The actual precision to load `family` at on `device`. The card sets the ceiling; the
    model family narrows it (SDXL has no 4-bit engine, so it never gets NVFP4 even on a card
    that supports it)."""
    if family is ModelFamily.AUX:
        return Quant.BF16  # adapters / LoRAs / detectors attach at the base dtype, not quantized
    if family.fp4_capable and device.supports_fp4:
        return Quant.NVFP4
    if device.supports_fp8:
        return Quant.FP8  # MXFP8 on Blackwell, plain fp8 on Hopper (torchao picks the variant)
    return Quant.BF16


class ModelServer:
    """Lazy, persistent model registry. Each role loads once and is cached for the process
    lifetime (the warm worker reuses it). `device` defaults to the live card."""

    def __init__(self, device: Device | None = None, specs: dict[ModelRole, ModelSpec] | None = None):
        self.device = device or current()
        self.specs = {**DEFAULT_SPECS, **(specs or {})}
        self._cache: dict[str, Any] = {}

    def plan(self) -> dict[str, str]:
        """The precision each role WILL load at on this card, no GPU touched. Drives logging
        and lets tests assert the matrix."""
        return {
            role.value: quant_for(spec.family, self.device).value
            for role, spec in self.specs.items()
        }

    # ---- loaders (deferred heavy imports; bodies need a GPU, verified on the pod) ----

    def keyframe_pipeline(self):
        """SDXL pipeline for keyframes: load at bf16, set the attention backend. Adapters and the
        per-scene character LoRAs (plus InstantID / IP-Adapter / ControlNet) are attached by
        keyframe.py per scene.

        NOT fp8-quantized: keyframe.py loads DYNAMIC per-scene character LoRAs onto the UNet, and
        peft cannot attach a LoRA to a torchao-quantized linear (TorchaoLoraLinear init mismatch,
        the #12535 family). SDXL is small (~6.5GB UNet), so bf16 is essentially free and keeps LoRA
        loading working on every card. (i2v CAN use fp8 because it FUSES its single distill LoRA
        before quantizing -- see i2v_pipeline.)"""
        if "keyframe" in self._cache:
            return self._cache["keyframe"]
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        spec = self.specs[ModelRole.KEYFRAME_BASE]
        cn_spec = self.specs[ModelRole.CONTROLNET_POSE]
        # The keyframe pipe is a ControlNet pipeline so the regional multi-character path can plant
        # two distinct bodies with an OpenPose skeleton (keyframe._render_regional); the single path
        # hands it a blank control image at conditioning_scale 0.0, which makes the ControlNet inert
        # (zero residual), so single-subject renders behave exactly like plain SDXL. Sharing one pipe
        # keeps the dynamic per-scene LoRA + IP-Adapter attach points identical across both paths.
        controlnet = ControlNetModel.from_pretrained(
            cn_spec.repo_id, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            spec.repo_id, controlnet=controlnet, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.to("cuda")
        _set_attention(pipe, self.device)

        # Stash the pristine (full-step) scheduler so keyframe._apply_scheduler can restore it for the
        # final tier after a draft/standard render swapped in DDIM-trailing on this warm shared pipe.
        pipe._vj_default_scheduler = pipe.scheduler
        # Load the Hyper-SD few-step distill LoRA as a persistent BASE adapter (named "distill", NOT
        # fused -- unlike i2v, the keyframe pipe stays bf16 so dynamic per-scene character LoRAs can
        # still attach). keyframe._bind_loras keeps it active at weight 1.0 on draft/standard and 0.0
        # on final, so one warm pipe serves every tier.
        self._attach_keyframe_distill(pipe)

        self._cache["keyframe"] = pipe
        return pipe

    def face_analyzer(self):
        """insightface FaceAnalysis on the antelopev2 pack (mirrored to <VJ_MODELS_ROOT>/antelopev2
        on cold start) for InstantID: yields a face embedding + 5 keypoints from a reference face.
        insightface resolves a pack at `<root>/models/<name>`, so its root is the PARENT of the
        mirror's models_root (antelopev2 sits one level up from `<root>/models/`). Cached; deferred
        imports keep this module CPU-importable; validated on a pod (insightface path + onnxruntime
        provider selection are environment-finicky)."""
        if "face_analyzer" in self._cache:
            return self._cache["face_analyzer"]
        import glob
        import os
        import shutil
        from insightface.app import FaceAnalysis

        models_root = os.environ.get("VJ_MODELS_ROOT", "/opt/models")
        root = os.path.dirname(models_root.rstrip("/")) or "/"

        # antelopev2.zip extracts one level too deep (`<dir>/antelopev2/*.onnx`), but insightface
        # scans `<dir>/*.onnx` -> "assert 'detection' in self.models". Flatten the nested pack so both
        # the auto-download and a flat R2-mirror layout work. (Runs before AND would-be after the
        # download, so a fresh pull that nests is recovered on the retry below.)
        def _flatten_antelope():
            mdir = os.path.join(root, "models", "antelopev2")
            nested = os.path.join(mdir, "antelopev2")
            if os.path.isdir(nested) and not glob.glob(os.path.join(mdir, "*.onnx")):
                for f in glob.glob(os.path.join(nested, "*")):
                    shutil.move(f, mdir)

        _flatten_antelope()
        try:
            app = FaceAnalysis(name="antelopev2", root=root,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except AssertionError:
            _flatten_antelope()  # the just-finished download nested it; flatten and retry once
            app = FaceAnalysis(name="antelopev2", root=root,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        self._cache["face_analyzer"] = app
        return app

    def instantid_pipeline(self):
        """InstantID single-character pipeline: a plain SDXL base whose UNet cross-attention is
        augmented with InstantID's face-embedding IP-Adapter (the insightface embedding -> identity
        tokens via the Resampler image-projection, injected through the IP attn side channel). Its own
        components so the per-scene character-LoRA + IP-attn state never tangles with the shared
        keyframe pipe; shares the few-step distill attach so single-char drafts stay cheap.
        keyframe._render_instantid drives it.

        NOTE: this uses InstantID's identity IP-Adapter ONLY, not the IdentityNet (face-keypoints)
        ControlNet. The IdentityNet must receive the face embedding as its encoder_hidden_states (a
        custom unet+controlnet denoise loop the stock pipeline does not expose); fed the text embeds it
        produces noise, and for a scene-posed character keyframe the face-pose lock it adds is usually
        undesirable. The IdentityNet is a documented future enhancement (draw_kps is kept for it)."""
        if "instantid" in self._cache:
            return self._cache["instantid"]
        import torch
        from diffusers import StableDiffusionXLPipeline
        from huggingface_hub import hf_hub_download
        from . import instantid as _iid

        base = self.specs[ModelRole.KEYFRAME_BASE]
        id_spec = self.specs[ModelRole.INSTANTID]
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base.repo_id, torch_dtype=torch.bfloat16, local_files_only=True)
        pipe.to("cuda")
        _set_attention(pipe, self.device)
        pipe._vj_default_scheduler = pipe.scheduler
        self._attach_keyframe_distill(pipe)

        # Identity injection: project the face embedding (Resampler) and wire the IP cross-attention.
        ckpt = torch.load(hf_hub_download(id_spec.repo_id, "ip-adapter.bin"), map_location="cpu")
        pipe._vj_image_proj = _iid.build_image_proj(ckpt["image_proj"]).to("cuda", torch.bfloat16)
        pipe._vj_id_attn = _iid.set_instantid_ip_attn(pipe.unet, ckpt["ip_adapter"])
        self._cache["instantid"] = pipe
        return pipe

    def _attach_keyframe_distill(self, pipe):
        """Load the Hyper-SD few-step distill LoRA as a persistent base adapter "distill" on a
        keyframe-style SDXL pipe (shared by keyframe_pipeline and instantid_pipeline). A finicky
        load degrades to full-step rather than crashing the worker."""
        pipe._vj_distill = None
        few = self.specs.get(ModelRole.KEYFRAME_FEWSTEP)
        if few is None:
            return
        try:
            pipe.load_lora_weights(few.repo_id, weight_name=few.weight_name, adapter_name="distill")
            pipe._vj_distill = "distill"
        except Exception as e:  # noqa: BLE001 -- never let a finicky distill load down the worker
            print(f"keyframe distill LoRA load failed ({few.repo_id} / {few.weight_name}): {e}; "
                  "full-step keyframes only.")

    def i2v_pipeline(self):
        """Wan 2.2 image-to-video: load bf16, quantize the MoE transformers to fp8 with torchao on
        the cards that support it, attach the Lightning distill LoRA for few-step sampling, set the
        attention backend. Final-tier rendering toggles the distill LoRA off (handled by i2v.py);
        here we load the warm baseline."""
        if "i2v" in self._cache:
            return self._cache["i2v"]
        # Lazy mirror: the heavy Wan i2v weights (~120GB) are kept out of the cold-start pull and
        # fetched from R2 here, on first i2v use, so keyframe/preview workers never pay for them.
        # No-op when already present or when there are no R2 creds (weights pre-provisioned / HF).
        from .harness.models_mirror import ensure_i2v_models
        ensure_i2v_models()
        import os
        import torch
        from diffusers import WanImageToVideoPipeline

        spec = self.specs[ModelRole.I2V]
        # The Wan repo ships only bf16 weights (no fp8 variant), so load bf16 and quantize to fp8
        # in place with torchao -- the same path as the SDXL keyframe, not a from_pretrained
        # variant. Wan 2.2 A14B is a ~28B two-expert MoE, too large to hold resident on the tighter
        # cards (and an 80GB H100, where it OOMs), so CPU-offload the inactive expert below the
        # big-VRAM tiers; the H200/B200 keep it resident and quantize to fp8 for full speed.
        pipe = WanImageToVideoPipeline.from_pretrained(
            spec.repo_id, torch_dtype=torch.bfloat16, local_files_only=True)
        offload = bool(self.device.vram_gb) and self.device.vram_gb < 120
        if not offload:
            pipe.to("cuda")

        # The distill LoRA must be FUSED before fp8 quant: a LoRA cannot load onto torchao-quantized
        # linears (#12535 -> TorchaoLoraLinear), so bake it into the base weights here and then
        # quantize the plain fused model. The few-step distill is the draft/standard speed path; a
        # final-tier worker sets VJ_I2V_DISTILL=0 to keep full steps.
        if os.environ.get("VJ_I2V_DISTILL", "1") != "0":
            _load_i2v_distill(pipe, self.specs[ModelRole.I2V_DISTILL])
        else:
            pipe._vj_i2v_distill_loaded = False
            pipe._vj_i2v_distill_fused = False

        use_fp8 = (self.device.supports_fp8 and not offload
                   and os.environ.get("VJ_I2V_FP8", "1") != "0")
        if use_fp8:
            for name in ("transformer", "transformer_2"):  # both MoE experts, after the distill fuse
                module = getattr(pipe, name, None)
                if module is not None:
                    try:
                        _quantize_fp8(module)
                    except Exception as e:  # noqa: BLE001
                        print(f"i2v fp8 quantize of {name} failed ({e}); leaving it bf16.")
        _set_attention(pipe, self.device)
        if offload:
            pipe.enable_model_cpu_offload()  # keep only the active expert on the GPU
        self._cache["i2v"] = pipe
        return pipe

    def frame_interpolator(self):
        """RIFE recursive-2x frame interpolator for the finishing stage. Loads the flownet weights
        named by the FRAME_INTERP spec once and caches them; the returned object exposes
        `interpolate(frame_a, frame_b) -> midpoint_frame` over HxWx3 uint8 arrays, which
        `finish._interpolate_once` calls to insert a frame between every adjacent pair.

        Deferred heavy imports (torch + the RIFE inference module) keep this module CPU-importable;
        the body is validated on a pod. A load failure RAISES (it does not return None): the finish
        stage decides whether a configured-but-unloadable pass is fatal, so the failure must not be
        swallowed here into a silent no-op. The render that asked for smooth motion gets a loud
        error, not a clip that quietly shipped at 16 fps."""
        if "frame_interp" in self._cache:
            return self._cache["frame_interp"]
        import os

        spec = self.specs[ModelRole.FRAME_INTERP]
        weight_path = os.path.join(
            os.environ.get("VJ_MODELS_ROOT", "/opt/models"),
            spec.repo_id.split("/")[-1], spec.weight_name or "flownet.pkl")
        interp = _RifeInterpolator(weight_path)
        self._cache["frame_interp"] = interp
        return interp

    def face_restorer(self, backend=None):
        """Blind face restorer for the finishing stage (relock identity over the i2v frames). The
        returned object exposes `restore(frame, fidelity=, only_faces=)` over an HxWx3 uint8 frame;
        the per-backend argument mapping (GFPGAN `weight` vs CodeFormer `w`) and the paste-back wiring
        live in the wrapper so `finish` stays backend-agnostic.

        `backend` is a `config.FaceRestore` member (or its string value); when None it falls back to
        the FACE_RESTORE spec's default (GFPGAN). `FaceRestore.NONE` is a programming error here (the
        finish stage only calls this when face restore is ON), so it RAISES rather than returning a
        no-op. As with `frame_interpolator`, a load failure RAISES so a configured pass that cannot
        load fails loud rather than silently skipping the restore.

        Deferred heavy imports (torch + the restorer + facelib detection) keep this module
        CPU-importable; the body is validated on a pod."""
        from .config import FaceRestore  # deferred: config imports models, so break the cycle here

        choice = FaceRestore(str(backend.value if isinstance(backend, FaceRestore) else backend).lower()) \
            if backend is not None else FaceRestore.GFPGAN
        if choice is FaceRestore.NONE:
            raise ValueError("face_restorer() called with FaceRestore.NONE; finish should not restore")

        cache_key = f"face_restore:{choice.value}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        import os

        spec = self.specs[ModelRole.FACE_RESTORE]
        models_root = os.environ.get("VJ_MODELS_ROOT", "/opt/models")
        if choice is FaceRestore.CODEFORMER:
            restorer = _CodeFormerRestorer(models_root)
        else:
            weight_path = os.path.join(
                models_root, spec.repo_id.split("/")[-1], spec.weight_name or "GFPGANv1.4.pth")
            restorer = _GfpganRestorer(weight_path)
        self._cache[cache_key] = restorer
        return restorer

    def unload(self) -> None:
        self._cache.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _quantize_fp8(module) -> None:
    """Quantize a diffusers module to fp8 in place via torchao. Blackwell gets MXFP8, Hopper
    plain fp8; torchao selects the kernel from the device. Verified against the diffusers
    torchao quantization guide."""
    from torchao.quantization import quantize_  # deferred
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

    quantize_(module, Float8DynamicActivationFloat8WeightConfig())


def _set_attention(pipe, device: Device) -> None:
    """Select the attention backend the card supports (FlashAttention-3 on Hopper/Blackwell)."""
    from .device import Attention
    if device.attention() is Attention.FLASH3:
        try:
            pipe.set_attention_backend("flash_attention_3")  # diffusers attention dispatch
        except Exception:
            pass  # fall through to the pipeline default (SDPA)


# --------------------------------------------------------------------------- finish-stage models
#
# These three wrappers are the GPU bodies of the finishing stage's two passes. They are constructed
# by `ModelServer.frame_interpolator` / `face_restorer` (so the load runs once and is cached) and
# expose a uniform, backend-agnostic interface that `finish.py` drives: the RIFE interpolator yields
# a midpoint frame for a pair, and either face restorer takes one frame and returns the restored
# frame. The per-backend argument names (GFPGAN `weight` vs CodeFormer `w`) and the only-faces
# paste-back wiring live HERE, so `finish` never has to know which restorer it got. All heavy imports
# (torch / the RIFE module / GFPGAN / CodeFormer / facelib) are deferred into the constructors, so
# this module stays CPU-importable; the bodies are validated on a pod.


# --------------------------------------------------------------------------- i2v distill loaders


def _load_i2v_distill(pipe, distill_spec) -> None:
    """Load the Wan2.2-Lightning distill LoRA onto `pipe`, trying the diffusers path first
    (fuse-before-fp8) then the LightX2V/manual path (direct weight delta, survives fp8 quant).

    Records the outcome on the pipe:
    - `_vj_i2v_distill_loaded` (bool): True when ANY loader succeeded.
    - `_vj_i2v_distill_fused` (bool): True when the diffusers fuse path baked the LoRA into
      the base weights (cannot be toggled off; the model IS distilled for its lifetime).

    Never raises: a load failure leaves both flags False and i2v.py runs full-step instead
    (and will refuse a 4-step-no-distill request via _set_distill)."""
    pipe._vj_i2v_distill_loaded = False
    pipe._vj_i2v_distill_fused = False

    if _try_diffusers_distill(pipe, distill_spec):
        return
    if _try_lightx2v_distill(pipe, distill_spec):
        return
    print(f"i2v: all distill loaders failed for {distill_spec.repo_id}; full-step only. "
          "Set VJ_I2V_DISTILL=0 to suppress this warning.", flush=True)


def _try_diffusers_distill(pipe, distill_spec) -> bool:
    """Load and fuse the distill LoRA via diffusers before fp8 quantization.

    Fusing bakes the LoRA delta permanently into the base weights so the quantizer sees plain
    weight matrices (TorchaoLoraLinear cannot be quantized after a LoRA is attached, #12535).
    Returns True on success and marks the pipe as fused."""
    try:
        # weight_name is required under HF_HUB_OFFLINE=1: diffusers scans the repo to pick a file
        # when weight_name is omitted, and that scan raises OfflineModeIsEnabled (probe 6).
        pipe.load_lora_weights(distill_spec.repo_id, weight_name=distill_spec.weight_name,
                               adapter_name="distill")
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        pipe._vj_i2v_distill_loaded = True
        pipe._vj_i2v_distill_fused = True
        return True
    except Exception as e:  # noqa: BLE001
        print(f"i2v distill (diffusers): {distill_spec.repo_id} failed ({e!r}); "
              "trying LightX2V fallback.", flush=True)
        return False


def _try_lightx2v_distill(pipe, distill_spec) -> bool:
    """LightX2V fallback: apply the LoRA delta directly to the transformer weight matrices.

    Bypasses diffusers' adapter injection entirely -- works on fp8-quantized
    TorchaoLoraLinear layers where diffusers load_lora_weights raises (#12535).

    Pod-validate items (marked POD-TODO):
    - Actual filename in the lightx2v/Wan2.2-Lightning HF repo.
    - Whether `lightx2v` package is installed in the image and the correct import path.
    - LoRA key prefix format (transformer.* vs diffusion_model.*).
    - Alpha key naming convention.

    Returns True on success (sets distill_loaded=True, distill_fused=True -- the weight
    delta is baked, effectively fused; no adapter registered, _set_distill early-returns),
    False on any error."""
    try:
        return _apply_lora_delta_to_wan(pipe, distill_spec)
    except Exception as e:  # noqa: BLE001
        print(f"i2v distill (LightX2V/manual): failed ({e!r})", flush=True)
        return False


def _apply_lora_delta_to_wan(pipe, distill_spec) -> bool:
    """Apply the Wan2.2-Lightning LoRA as a weight delta (B @ A * alpha/rank) directly onto each
    transformer expert's weight matrices. No diffusers LoRA injection -- works on quantized models.

    The safetensors keys use either "transformer.<block>.<proj>.lora_A.weight" or
    "diffusion_model.<...>.lora_A.weight" prefixes; both are handled by the prefix-strip loop.
    Alpha key naming and key format still to be confirmed on a pod (POD-TODO in the loop body)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    # distill_spec.weight_name is the subdirectory path within the repo confirmed from R2:
    # "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
    lora_path = hf_hub_download(distill_spec.repo_id, filename=distill_spec.weight_name)
    sd = load_file(lora_path)

    applied = 0
    for key, val in sd.items():
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in sd:
            continue
        lora_a, lora_b = val, sd[b_key]   # A: (rank, in_feat), B: (out_feat, rank)
        rank = lora_a.shape[0]
        alpha_key = key.rsplit(".lora_A.", 1)[0] + ".alpha"
        alpha = float(sd[alpha_key]) if alpha_key in sd else float(rank)
        scale = alpha / rank

        for expert_attr in ("transformer", "transformer_2"):
            expert = getattr(pipe, expert_attr, None)
            if expert is None:
                continue
            # Strip the expert prefix + optional "diffusion_model." wrapper -- POD-TODO: confirm
            mod_key = key.split(".lora_A.")[0]
            for pfx in (f"{expert_attr}.", "diffusion_model."):
                if mod_key.startswith(pfx):
                    mod_key = mod_key[len(pfx):]
                    break
            try:
                mod = _nested_module(expert, mod_key)
                delta = (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scale
                mod.weight.data += delta.to(mod.weight.dtype)
                applied += 1
            except (AttributeError, RuntimeError):
                continue

    if applied == 0:
        first3 = list(sd.keys())[:3]
        raise RuntimeError(
            f"No LoRA weights applied from {distill_spec.repo_id} -- key format mismatch. "
            f"First 3 sd keys: {first3}. POD-TODO: verify key prefix.")
    print(f"i2v distill (manual delta): applied {applied} LoRA pairs from {distill_spec.repo_id}",
          flush=True)
    pipe._vj_i2v_distill_loaded = True
    # The delta is baked directly into the base weight matrices -- effectively fused.
    # No adapter is registered, so _set_distill must NOT call set_adapters (it would KeyError).
    # Mark fused=True so _set_distill takes the early-return path on every shot.
    pipe._vj_i2v_distill_fused = True
    return True


def _nested_module(root, dotted_key: str):
    """Traverse a dotted module path (e.g. 'blocks.3.attn.to_q') on `root`."""
    mod = root
    for part in dotted_key.split("."):
        mod = getattr(mod, part)
    return mod


def _pad_to_multiple(n: int, multiple: int = 64) -> int:
    """Pixels to add to a spatial dimension `n` so it becomes a multiple of `multiple` (0 if already
    aligned). RIFE's flownet downsamples then concatenates skip features, so each of H and W must be a
    multiple of 64; a non-aligned i2v resolution otherwise throws inside inference, e.g. Wan 2.6's
    1270x726 -> 'tensor a (1270) must match tensor b (1280) at non-singleton dimension 3' (#245). This
    rescues every non-64 backend at any aspect ratio (seedance/veo/vidu 1280x720, hailuo 1364x768)."""
    return (multiple - n % multiple) % multiple


class _RifeInterpolator:
    """RIFE recursive-2x interpolator. `interpolate(a, b)` returns the synthesized midpoint frame
    for the adjacent pair `(a, b)` (HxWx3 uint8 in, same out). The model loads once in __init__.

    RIFE requires 64-divisible H/W, so `interpolate` pads the pair up to the next multiple of 64,
    runs the flownet, and crops the result back to the original dims -- the pad never reaches the
    encoded clip. Without it any non-64 i2v output (Wan 2.6 1270x726, etc.) crashes the finish chain
    at step 0, silently shipping the raw clip (#245)."""

    def __init__(self, weight_path: str):
        import os  # deferred
        import torch  # deferred
        from rife.RIFE_HDv3 import Model  # vendored ECCV2022-RIFE HDv3 inference module (MIT weights)

        self._torch = torch
        self._model = Model()
        # RIFE_HDv3.Model.load_model takes the DIRECTORY that holds flownet.pkl (it appends
        # `/flownet.pkl` itself), not the file path. weight_path is .../rife/flownet.pkl, so pass its
        # parent. rank=-1 strips the `module.` prefix the DDP-saved flownet.pkl checkpoint carries.
        self._model.load_model(os.path.dirname(weight_path), -1)
        self._model.eval()
        self._model.device()

    def interpolate(self, frame_a, frame_b):
        torch = self._torch
        import torch.nn.functional as F  # deferred: keep this module CPU-importable

        h, w = int(frame_a.shape[0]), int(frame_a.shape[1])
        ph, pw = _pad_to_multiple(h), _pad_to_multiple(w)
        a = torch.from_numpy(frame_a).permute(2, 0, 1).float().div(255.0).unsqueeze(0).cuda()
        b = torch.from_numpy(frame_b).permute(2, 0, 1).float().div(255.0).unsqueeze(0).cuda()
        if ph or pw:
            # Pad bottom/right up to the next multiple of 64 so RIFE accepts any i2v resolution
            # (#245). Replicate the edge rather than zero-pad: a hard black seam would pull the
            # optical flow toward it; replicate keeps the synthesized motion stable. F.pad's 4D
            # order is (W_left, W_right, H_top, H_bottom), so we extend only right and bottom.
            a = F.pad(a, (0, pw, 0, ph), mode="replicate")
            b = F.pad(b, (0, pw, 0, ph), mode="replicate")
        with torch.no_grad():
            mid = self._model.inference(a, b)
        out = (mid[0].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        return out[:h, :w]  # crop the pad back off -> exactly the original dims


class _GfpganRestorer:
    """GFPGAN blind face restorer. `restore(frame, fidelity, only_faces)` runs the restorer over the
    detected faces in one frame. GFPGAN's fidelity knob is `weight` (its restoration/fidelity
    balance), and `paste_back` controls whether the restored faces are composited back into the full
    frame: with `only_faces` False we paste the restored faces back over the whole frame; with
    `only_faces` True we still paste them back (the restored faces are what we want), but leave the
    rest of the frame untouched, which is GFPGAN's default paste behavior. The model loads once."""

    def __init__(self, weight_path: str):
        from gfpgan import GFPGANer  # deferred

        self._restorer = GFPGANer(model_path=weight_path, upscale=1, arch="clean",
                                  channel_multiplier=2, bg_upsampler=None)

    def restore(self, frame, *, fidelity: float = 0.7, only_faces: bool = True):
        # paste_back composites the restored faces back into the frame. We always want the restored
        # faces, so paste_back is True; `only_faces` means "do not also touch the background", and
        # GFPGAN with bg_upsampler=None already leaves the non-face background as-is, so the flag is
        # honored by NOT enabling a background upsampler rather than by suppressing the paste.
        _cropped, _restored, restored_img = self._restorer.enhance(
            frame, has_aligned=False, only_center_face=False,
            paste_back=True, weight=float(fidelity))
        return restored_img if restored_img is not None else frame


class _CodeFormerRestorer:
    """CodeFormer blind face restorer (S-Lab NON-COMMERCIAL: a deploy-time opt-in, never bundled).
    `restore(frame, fidelity, only_faces)` mirrors `_GfpganRestorer` but maps fidelity to
    CodeFormer's own argument name `w` (its fidelity weight: higher keeps more of the input
    identity). Loads the net + a FaceRestoreHelper once."""

    def __init__(self, models_root: str):
        import os

        import torch  # deferred
        from basicsr.utils.registry import ARCH_REGISTRY
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"]).to(self._device)
        ckpt = os.path.join(models_root, "CodeFormer", "codeformer.pth")
        net.load_state_dict(torch.load(ckpt, map_location=self._device)["params_ema"])
        net.eval()
        self._net = net
        self._helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, device=self._device)

    def restore(self, frame, *, fidelity: float = 0.7, only_faces: bool = True):
        torch = self._torch
        from torchvision.transforms.functional import normalize
        from basicsr.utils import img2tensor, tensor2img

        helper = self._helper
        helper.clean_all()
        helper.read_image(frame)
        helper.get_face_landmarks_5(only_center_face=False)
        helper.align_warp_face()
        for cropped in helper.cropped_faces:
            t = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
            normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            t = t.unsqueeze(0).to(self._device)
            with torch.no_grad():
                out = self._net(t, w=float(fidelity), adain=True)[0]  # CodeFormer fidelity arg is `w`
            helper.add_restored_face(tensor2img(out, rgb2bgr=True, min_max=(-1, 1)).astype("uint8"))
        helper.get_inverse_affine(None)
        # paste_back True: composite the restored faces back over the frame. only_faces leaves the
        # rest of the frame as-is (no background upsampler is passed), so the flag is honored.
        restored = helper.paste_faces_to_input_image(upsample_img=None)
        return restored if restored is not None else frame
