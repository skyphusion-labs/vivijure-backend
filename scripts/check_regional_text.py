# Run: python scripts/check_regional_text.py  (needs torch, transformers, diffusers; network for the tiny test pipe)
"""Proof that regional_text.RegionalTextProcessor gives each region the RIGHT tokens (vivijure-cf#1143 option a).

On diffusers' own Attention, with distinct random text embeddings for the shared prompt, person A and person B:
  1. every query cell inside A's box equals attention over A's prompt alone, inside B's box over B's alone, and
     every cell outside both (the gutter, the background) over the shared prompt;
  2. the unconditioned half of a guidance batch is untouched;
  3. the same holds under an IP-Adapter processor, whose image embeddings pass through;
  4. a CONTROL: with the region masks swapped, check 1 fails (so the check can see a wrong assignment);
  5. end to end on the tiny SDXL test pipe: installed, a plate is drawn; restored, the processors are the originals.
Exit 0 only if every check passes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0, IPAdapterAttnProcessor2_0

from vivijure_backend.keyframe import region_boxes
from vivijure_backend.regional_text import RegionalTextProcessor, box_to_grid, install, restore

torch.manual_seed(0)
fails = []


def check(name, ok, got=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {got}")
    if not ok:
        fails.append(name)


W, H = 128, 64                   # pixels -> an 8 x 16 latent
lh, lw = H // 8, W // 8
boxes = region_boxes(W, H, 2, orientation="vertical", gutter=32)
D, L = 32, 7
attn = Attention(query_dim=D, cross_attention_dim=D, heads=2, dim_head=16)
inner = AttnProcessor2_0()
shared = torch.randn(1, L, D)
person_a, person_b, neg = torch.randn(1, L, D), torch.randn(1, L + 3, D), torch.randn(1, L, D)
hs = torch.randn(2, lh * lw, D)  # [uncond; cond]
ehs = torch.cat([neg, shared])


def cells(box):
    g = box_to_grid(box, W, H, lh, lw)
    return torch.tensor(g).reshape(-1)


def compare(out, regions, label):
    ref_shared = inner(attn, hs[1:], shared)
    ref = {0: inner(attn, hs[1:], person_a), 1: inner(attn, hs[1:], person_b)}
    a, b = cells(boxes[0]), cells(boxes[1])
    bg = ~(a | b)
    check(f"{label}: region A reads only A's prompt", torch.allclose(out[1:, a], ref[0][:, a], atol=1e-5))
    check(f"{label}: region B reads only B's prompt", torch.allclose(out[1:, b], ref[1][:, b], atol=1e-5))
    check(f"{label}: the gutter and background read the shared prompt", bool(bg.any()) and torch.allclose(out[1:, bg], ref_shared[:, bg], atol=1e-5))
    check(f"{label}: the unconditioned half is untouched", torch.allclose(out[:1], inner(attn, hs[:1], neg), atol=1e-5))


with torch.no_grad():
    p = RegionalTextProcessor(inner, width=W, height=H, cfg_batch=True)
    p.regions = [(person_a, boxes[0]), (person_b, boxes[1])]
    compare(p(attn, hs, ehs), p.regions, "plain")

    # 4. CONTROL: swapped masks must fail the same comparison.
    p.regions = [(person_a, boxes[1]), (person_b, boxes[0])]
    swapped = p(attn, hs, ehs)
    a = cells(boxes[0])
    check("CONTROL: swapping the regions is seen as wrong",
          not torch.allclose(swapped[1:, a], inner(attn, hs[1:], person_a)[:, a], atol=1e-5))

    # 3. IP-Adapter: (text, [image embeds]) rides through; each region still reads its own text.
    ipp = IPAdapterAttnProcessor2_0(hidden_size=D, cross_attention_dim=D, num_tokens=(4,), scale=0.5)
    ip = [torch.randn(2, 4, D)]
    p2 = RegionalTextProcessor(ipp, width=W, height=H, cfg_batch=True)
    p2.regions = [(person_a, boxes[0]), (person_b, boxes[1])]
    out = p2(attn, hs, (ehs, ip))
    ref_a = ipp(attn, hs[1:], (person_a, [ip[0][1:]]))
    check("IP-Adapter: region A reads A's prompt with the image embeds kept", torch.allclose(out[1:, a], ref_a[:, a], atol=1e-5))
    p2.regions = []
    check("IP-Adapter: with no regions the processor is the inner one", torch.allclose(p2(attn, hs, (ehs, ip)), ipp(attn, hs, (ehs, ip)), atol=1e-6))

    # 5. End to end on the tiny SDXL pipe.
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe").to("cpu")
    before = dict(pipe.unet.attn_processors)
    tok = pipe.tokenizer
    def emb(text):
        from vivijure_backend.prompt_chunks import encode_long_prompt
        return encode_long_prompt(pipe, text, "", shot_id="t")
    base = emb("a workshop, two adults")
    regs = [(emb("a workshop, two adults, dark curls")["prompt_embeds"], b) for b in region_boxes(64, 64, 2, gutter=8)]
    saved = install(pipe, regs, width=64, height=64, cfg_batch=True)
    wrapped = sum(isinstance(v, RegionalTextProcessor) for v in pipe.unet.attn_processors.values())
    factors = {v.vae_scale_factor for v in pipe.unet.attn_processors.values() if isinstance(v, RegionalTextProcessor)}
    img_on = pipe(**base, num_inference_steps=2, guidance_scale=5.0, height=64, width=64, output_type="np",
                  generator=torch.Generator().manual_seed(1)).images
    restore(pipe, saved)
    after = dict(pipe.unet.attn_processors)
    img_off = pipe(**base, num_inference_steps=2, guidance_scale=5.0, height=64, width=64, output_type="np",
                   generator=torch.Generator().manual_seed(1)).images
    attn2 = sum(1 for k in before if k.endswith("attn2.processor"))
    check("tiny pipe: the latent size comes from the pipe's own VAE factor", factors == {pipe.vae_scale_factor}, f"{factors} vs {pipe.vae_scale_factor}")
    check("tiny pipe: every cross-attention processor was wrapped, and no self-attention one", wrapped == attn2 and attn2 > 0, f"{wrapped}/{attn2}")
    check("tiny pipe: restore puts back the exact originals", all(after[k] is before[k] for k in before))
    check("tiny pipe: a plate is drawn with regions on, and differs from off", img_on.shape == img_off.shape and not (img_on == img_off).all())

print("FAIL" if fails else "PASS", fails)
sys.exit(1 if fails else 0)
