"""Per-character SDXL LoRA training, straight from the bundle's reference images.

A render that uses characters needs a small identity adapter per slot so the keyframe
model can draw the same person across shots. This module trains that adapter: a DreamBooth
style LoRA on the SDXL UNet, fit to the handful of reference images the control plane packed
into `characters/refs/<SLOT>/`. There is no cast manager and no shared identity service; each
slot trains fresh from its own refs and saves one `.safetensors` the keyframe stage loads.

Scope is deliberately narrow. We adapt the UNet only (the cross/self-attention projections),
which is the cheapest place to bind a face and avoids the text-encoder drift that makes a LoRA
forget how to render anything but the training set. fp8 belongs to inference, not here: the
trainable LoRA path runs in bf16 so the optimizer sees real gradients.

Clean-room: the training mechanics here are built from the diffusers + PEFT public training
guides (the LoRA guide and the SDXL DreamBooth-LoRA reference script), not from any prior
pipeline. The SDXL-specific bookkeeping (micro-conditioning `time_ids`, the two-text-encoder
pooled embeds, the `save_lora_weights` adapter format) follows those docs.

Heavy imports (torch / diffusers / transformers / peft) are deferred into `train_slot` so the
module imports and unit-tests on a CPU box, the same convention `models.py` uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .contract import Character
from .device import current
from .models import DEFAULT_SPECS, ModelRole

# The UNet attention projections a character LoRA needs to bind identity. Matches the
# diffusers LoRA guide's SDXL target set; the text encoders are deliberately left frozen.
UNET_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


@dataclass
class LoraTrainConfig:
    """Knobs for one slot's training run. Defaults are tuned for a few-reference character
    fit (5-20 images): a small rank that captures a face without memorizing backgrounds, a
    short run that converges before it overfits, and gradient checkpointing so a 1024 SDXL
    UNet trains inside a single mid-tier card's VRAM."""
    rank: int = 16
    resolution: int = 1024
    learning_rate: float = 1e-4
    max_steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 0
    random_flip: bool = True
    gradient_checkpointing: bool = True
    # The caption every reference trains under. `{name}` and `{prompt}` are filled from the
    # slot's registry entry; the name is the trigger token the keyframe stage will prompt with.
    caption_template: str = "{name}, {prompt}"
    save_every: int = 0  # 0 = only the final adapter; >0 writes intermediate checkpoints


@dataclass
class TrainedLora:
    """Where one slot's adapter landed, plus the facts the planner records."""
    slot: str
    path: Path                      # the saved pytorch_lora_weights.safetensors
    trigger: str                    # the token the keyframe prompt uses to summon this identity
    steps: int
    rank: int
    ref_count: int
    base_repo: str
    meta: dict = field(default_factory=dict)


def default_base_repo() -> str:
    """The SDXL checkpoint LoRAs train against: the same keyframe base the renderer draws with,
    so the adapter is fit to the model that will use it."""
    return DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id


def caption_for(char: Character, template: str) -> str:
    """The training caption for a slot.

    Uses simple str.replace substitution (not str.format) so the template cannot carry
    attribute-access or format_spec payloads like {name:{prompt.__class__.__mro__}}. A
    template with unrecognised placeholders has them left literal rather than raising (safe
    default). Falls back to the name alone when the registry has no prompt, and collapses
    stray punctuation so an empty field never leaks a dangling comma."""
    if "{" in template or "}" in template:
        # Reject any remaining braces after substitution to prevent future .format() reuse
        # and catch templates with unknown placeholders that signal a mis-authored config.
        filled = template.replace("{name}", char.name or char.slot).replace(
            "{prompt}", char.prompt or "")
        remaining_braces = [c for c in filled if c in "{}"]
        if remaining_braces:
            raise ValueError(
                f"caption_template contains unsupported placeholders: {template!r}; "
                "only {name} and {prompt} are allowed")
        text = filled
    else:
        text = template
    return ", ".join(part.strip() for part in text.split(",") if part.strip())


def train_slot(
    char: Character,
    out_dir: Path,
    *,
    config: LoraTrainConfig | None = None,
    base_repo: str | None = None,
    progress_cb=None,
) -> TrainedLora:
    """Train one character's SDXL LoRA from its reference images and save the adapter.

    `char.ref_paths` is the list of bundle reference images (already resolved by
    `Bundle.extract`). Writes `pytorch_lora_weights.safetensors` under `out_dir` and returns
    a `TrainedLora` describing it. Needs a CUDA device; raises if there are no references.
    """
    cfg = config or LoraTrainConfig()
    base = base_repo or default_base_repo()
    refs = list(char.ref_paths)
    if not refs:
        raise ValueError(f"slot {char.slot} ({char.name}) has no reference images to train on")

    # Deferred heavy imports: the module must load on a CPU box for unit tests.
    import torch
    import torch.nn.functional as F
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    weight_dtype = torch.bfloat16
    torch.manual_seed(cfg.seed)

    # --- frozen backbone: VAE (fp32 for stable latents), both text encoders, the UNet ---
    # The VAE stays fp32 because SDXL's VAE produces NaNs under fp16/bf16 encode; everything
    # else lives in bf16. Only the UNet gets a trainable LoRA; the rest is inference-only.
    vae = AutoencoderKL.from_pretrained(
        base, subfolder="vae", torch_dtype=torch.float32, local_files_only=True).to(device)
    # CLIPTokenizer, not AutoTokenizer: Auto* probes AutoConfig for a config.json the CLIP
    # tokenizer subfolders do not have, which is a graceful 404 online but a FATAL
    # LocalEntryNotFoundError under HF_HUB_OFFLINE=1 on the deployed worker. CLIPTokenizer loads
    # the tokenizer files directly (the diffusers SDXL path), so it works offline from the cache.
    tokenizer_one = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    # local_files_only=True: transformers probes the repo tree for additional_chat_templates on
    # from_pretrained; that tree-listing call raises under HF_HUB_OFFLINE=1. Safe here because
    # the R2 mirror has run before lora_train is called, so the weights are in the local cache.
    text_encoder_one = CLIPTextModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=weight_dtype,
        local_files_only=True).to(device)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        base, subfolder="text_encoder_2", torch_dtype=weight_dtype,
        local_files_only=True).to(device)
    # local_files_only=True: diffusers calls model_info() to check for a newer revision before
    # loading; that API call raises OfflineModeIsEnabled under HF_HUB_OFFLINE=1. The R2 mirror
    # has already run, so the weights are in the local cache -- it is safe to skip the check.
    unet = UNet2DConditionModel.from_pretrained(
        base, subfolder="unet", torch_dtype=weight_dtype, local_files_only=True).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler",
                                                    local_files_only=True)

    for module in (vae, text_encoder_one, text_encoder_two, unet):
        module.requires_grad_(False)

    # --- attach the trainable LoRA to the UNet only ---
    unet.add_adapter(LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=UNET_TARGET_MODULES,
    ))
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=cfg.learning_rate)

    # --- caption embeddings: one fixed instance prompt, so encode it once ---
    # SDXL needs the concatenated penultimate hidden states of both encoders plus encoder-2's
    # pooled output (the "text_embeds" micro-condition). The caption is constant across refs,
    # so the embeds are computed a single time and reused every step.
    caption = caption_for(char, cfg.caption_template)
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(
        caption, [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two], device, weight_dtype)

    # --- references -> latents (precomputed once; the set is tiny) ---
    target_size = (cfg.resolution, cfg.resolution)
    latents: list[torch.Tensor] = []
    time_ids: list[torch.Tensor] = []
    for ref in refs:
        pixels, original_size, crop_top_left = _load_image(ref, cfg.resolution)
        pixels = pixels.to(device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        latents.append(latent.to(weight_dtype).squeeze(0))
        time_ids.append(_time_ids(original_size, crop_top_left, target_size, device, weight_dtype))

    # --- training loop ---
    unet.train()
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    n = len(latents)
    add_text_embeds = pooled_prompt_embeds  # encoder-2 pooled output, the SDXL "text_embeds"
    last_loss = 0.0
    for step in range(cfg.max_steps):
        idx = int(torch.randint(0, n, (1,), generator=generator, device=device).item())
        latent = latents[idx].unsqueeze(0)
        add_time = time_ids[idx]
        if cfg.random_flip and torch.rand(1, generator=generator, device=device).item() < 0.5:
            latent = torch.flip(latent, dims=[-1])  # horizontal flip in latent space

        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=weight_dtype)
        timestep = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,), generator=generator, device=device).long()
        noisy = noise_scheduler.add_noise(latent, noise, timestep)

        model_pred = unet(
            noisy, timestep, prompt_embeds,
            added_cond_kwargs={"text_embeds": add_text_embeds, "time_ids": add_time},
            return_dict=False,
        )[0]
        target = _loss_target(noise_scheduler, latent, noise, timestep)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss = loss / cfg.gradient_accumulation_steps
        loss.backward()
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
        last_loss = float(loss.item()) * cfg.gradient_accumulation_steps

        if cfg.save_every and (step + 1) % cfg.save_every == 0 and (step + 1) < cfg.max_steps:
            _save_adapter(unet, out_dir / f"checkpoint-{step + 1}",
                          StableDiffusionXLPipeline, get_peft_model_state_dict, convert_state_dict_to_diffusers)
        if (step + 1) % 50 == 0 or step == 0:
            print(f"[lora {char.slot}] step {step + 1}/{cfg.max_steps} loss={last_loss:.4f}", flush=True)
            if progress_cb is not None:
                try:
                    progress_cb(step + 1, cfg.max_steps, last_loss)  # throttled: only at the log cadence
                except Exception:
                    pass  # best-effort: a progress callback must never break training

    # --- save the final adapter in diffusers' SDXL LoRA format ---
    _save_adapter(unet, out_dir, StableDiffusionXLPipeline,
                  get_peft_model_state_dict, convert_state_dict_to_diffusers)
    saved = out_dir / "pytorch_lora_weights.safetensors"

    return TrainedLora(
        slot=char.slot,
        path=saved,
        trigger=char.name or char.slot,
        steps=cfg.max_steps,
        rank=cfg.rank,
        ref_count=n,
        base_repo=base,
        meta={"caption": caption, "final_loss": round(last_loss, 4),
              "device": current().tier.value},
    )


# --------------------------------------------------------------------------- internals

def _encode_prompt(prompt, tokenizers, text_encoders, device, dtype):
    """SDXL's two-encoder prompt embedding: concatenate the penultimate hidden states of both
    CLIP encoders along the feature axis, and take encoder-2's pooled projection as the pooled
    embed. Follows the diffusers SDXL training `encode_prompt` helper."""
    import torch

    embeds_list = []
    pooled = None
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        ids = tokenizer(
            prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt").input_ids.to(device)
        out = text_encoder(ids, output_hidden_states=True, return_dict=False)
        pooled = out[0]                 # encoder-2's pooled output is the one we keep
        embeds_list.append(out[-1][-2])  # penultimate hidden state
    prompt_embeds = torch.concat(embeds_list, dim=-1).to(dtype)
    return prompt_embeds, pooled.to(dtype)


def _time_ids(original_size, crop_top_left, target_size, device, dtype):
    """SDXL micro-conditioning vector: original size + crop top-left + target size."""
    import torch
    return torch.tensor([list(original_size) + list(crop_top_left) + list(target_size)],
                        device=device, dtype=dtype)


def _loss_target(noise_scheduler, latent, noise, timestep):
    """The regression target for this scheduler's prediction type: the noise itself for
    epsilon-prediction, the velocity for v-prediction."""
    if noise_scheduler.config.prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latent, noise, timestep)
    return noise


def _load_image(path: Path, resolution: int):
    """Load a reference image and center-crop it to a square at `resolution`, returning the
    pixel tensor in [-1, 1] plus the (original_size, crop_top_left) SDXL needs for time_ids."""
    import torch
    from PIL import Image

    img = Image.open(path).convert("RGB")
    original_size = (img.height, img.width)
    # Resize the short side to `resolution`, then center-crop a square.
    scale = resolution / min(img.width, img.height)
    new_w, new_h = round(img.width * scale), round(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))

    import numpy as np
    arr = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0)  # HWC in [0,1]
    pixels = arr.permute(2, 0, 1) * 2.0 - 1.0                          # CHW in [-1,1]
    return pixels, original_size, (top, left)


def _save_adapter(unet, out_dir, pipeline_cls, get_peft_model_state_dict, convert_state_dict_to_diffusers):
    """Write the UNet LoRA as diffusers' `pytorch_lora_weights.safetensors` (UNet-only;
    no text-encoder layers, since the encoders were frozen)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    pipeline_cls.save_lora_weights(save_directory=str(out_dir), unet_lora_layers=unet_lora_layers)
