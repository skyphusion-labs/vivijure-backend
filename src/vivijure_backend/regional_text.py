"""Per-region text for a two-person plate (vivijure-cf#1143 option a).

WHY. SDXL reads one prompt for the whole frame, so in a two-person plate it does not bind "grey zip hoodie" to Dev
or "dark shoulder-length curls" to Mara. Measured at seed 110501: Mara's hair went up, and with the plate's portrait
reference (option b) her jacket became Dev's hoodie. Binding traits to names in the text made it worse.

HOW. Each cross-attention layer of the UNet runs once on the SHARED prompt, which the background keeps, and once per
person on that person's own prompt (the shared text plus their description only). Each person's result replaces the
shared one inside their region box, scaled down to the layer's own resolution. The shared prompt carries the scene,
the action, the style, the setting and the adult clause, and is part of every person's prompt, so none of it is ever
masked away (keyframe.plate_region_prompts). Only the conditioned half of a guidance batch is regional; the negative
half is untouched. An IP-Adapter's image embeddings ride through unchanged.

The geometry below is pure and tested on CPU. The processor needs torch and is checked against diffusers' own
Attention on a tiny module (scripts/check_regional_text.py), and validated on a pod.
"""
from __future__ import annotations

from contextlib import contextmanager


def layer_hw(q_len: int, latent_h: int, latent_w: int) -> tuple[int, int]:
    """The (h, w) grid of a UNet layer whose attention sees `q_len` query tokens, for a latent of
    latent_h x latent_w. SDXL halves both sides per level; a side may round up on an odd size."""
    h, w = latent_h, latent_w
    for _ in range(8):
        if h * w == q_len:
            return h, w
        h, w = (h + 1) // 2, (w + 1) // 2
    raise ValueError(f"no UNet level of a {latent_h}x{latent_w} latent has {q_len} query tokens")


def box_to_grid(box: tuple[int, int, int, int], width: int, height: int, h: int, w: int) -> list[list[bool]]:
    """A pixel-space region box `(left, top, right, bottom)` as a boolean h x w grid: a cell is in when its
    centre falls inside the box."""
    left, top, right, bottom = box
    grid = []
    for y in range(h):
        cy = (y + 0.5) * height / h
        grid.append([left <= (x + 0.5) * width / w < right and top <= cy < bottom for x in range(w)])
    return grid


class RegionalTextProcessor:
    """Wraps one cross-attention processor. `regions` is a list of (prompt_embeds, box); the shared prompt is what the
    pipe passes in. Set `regions` before the plate call and clear it after (the InstantID side-channel pattern)."""

    def __init__(self, inner, *, width: int, height: int, cfg_batch: bool, vae_scale_factor: int = 8):
        self.inner = inner
        self.width = width
        self.height = height
        self.vae_scale_factor = vae_scale_factor  # SDXL is 8; read from the pipe at install, never assumed
        self.cfg_batch = cfg_batch
        self.regions: list = []
        # Option (c): what the cells outside every region read instead of the shared prompt. None = the shared one.
        self.background = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, *args, **kwargs):
        out = self.inner(attn, hidden_states, encoder_hidden_states, *args, **kwargs)
        if encoder_hidden_states is None or not self.regions:
            return out  # self-attention, or regional text off
        import torch

        ip = None
        text = encoder_hidden_states
        if isinstance(encoder_hidden_states, tuple):  # IP-Adapter: (text, image embeds) rides through unchanged
            text, ip = encoder_hidden_states
        b = hidden_states.shape[0]
        start = b // 2 if self.cfg_batch and b > 1 else 0  # the conditioned half of [uncond; cond]
        hs = hidden_states[start:]
        f = self.vae_scale_factor
        h, w = layer_hw(hs.shape[1], self.height // f, self.width // f)
        result = out.clone()
        if self.background is not None:
            # The band and the background: every cell in no region reads the people-free prompt (option c).
            bg = self.background.to(device=text.device, dtype=text.dtype).expand(hs.shape[0], -1, -1)
            bg_out = self.inner(attn, hs, (bg, [t[start:] for t in ip]) if ip is not None else bg, *args, **kwargs)
            inside = None
            for _, box in self.regions:
                g = torch.tensor(box_to_grid(box, self.width, self.height, h, w), device=out.device).reshape(1, h * w, 1)
                inside = g if inside is None else (inside | g)
            outside = ~inside if inside is not None else torch.ones(1, h * w, 1, dtype=torch.bool, device=out.device)
            result[start:] = torch.where(outside, bg_out, result[start:])
        for embeds, box in self.regions:
            emb = embeds.to(device=text.device, dtype=text.dtype).expand(hs.shape[0], -1, -1)
            ehs = (emb, [t[start:] for t in ip]) if ip is not None else emb
            region_out = self.inner(attn, hs, ehs, *args, **kwargs)
            grid = box_to_grid(box, self.width, self.height, h, w)
            mask = torch.tensor(grid, device=out.device).reshape(1, h * w, 1)
            result[start:] = torch.where(mask, region_out, result[start:])
        return result


def install(pipe, regions, *, width: int, height: int, cfg_batch: bool, background=None) -> dict:
    """Wrap every cross-attention processor of `pipe.unet` with the regions set. Returns the original processors for
    `restore`. Self-attention ("attn1") processors are left as they are."""
    originals = dict(pipe.unet.attn_processors)
    wrapped = {}
    for name, proc in originals.items():
        if name.endswith("attn2.processor"):
            p = RegionalTextProcessor(proc, width=width, height=height, cfg_batch=cfg_batch,
                                      vae_scale_factor=int(getattr(pipe, "vae_scale_factor", 8)))
            p.regions = regions
            p.background = background
            wrapped[name] = p
        else:
            wrapped[name] = proc
    pipe.unet.set_attn_processor(wrapped)
    return originals


def restore(pipe, originals: dict) -> None:
    pipe.unet.set_attn_processor(originals)


@contextmanager
def applied(pipe, regions, *, width: int, height: int, cfg_batch: bool, background=None):
    """Install the regions for one pipe call and ALWAYS put the original processors back, even if the call raises.
    The pipe is shared across shots, so a processor left installed would make every later shot regional. With no
    regions this does nothing."""
    if not regions:
        yield
        return
    originals = install(pipe, regions, width=width, height=height, cfg_batch=cfg_batch, background=background)
    try:
        yield
    finally:
        restore(pipe, originals)
