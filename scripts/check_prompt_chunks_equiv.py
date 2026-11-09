# Run: python scripts/check_prompt_chunks_equiv.py  (needs torch, transformers, diffusers and network for the tiny test pipe)
"""Equivalence check for prompt_chunks.encode_long_prompt against diffusers' own SDXL encode_prompt.

The tiny SDXL test pipe has random weights, which is fine: the claim is that our chunk framing and pooling
reproduce what the pipeline itself computes, for a prompt that fits one chunk, and that a long prompt grows
the sequence by whole 77-token chunks with the negative padded to match."""
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
from diffusers import StableDiffusionXLPipeline
from vivijure_backend.prompt_chunks import encode_long_prompt

pipe = StableDiffusionXLPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
pipe.to("cpu")
print("tokenizers", type(pipe.tokenizer).__name__, type(pipe.tokenizer_2).__name__,
      "pad", pipe.tokenizer.pad_token_id, pipe.tokenizer_2.pad_token_id)

short, neg = "photoreal, Mara and Dev, both adults, grey zip hoodie", "lowres, bad anatomy"
ours = encode_long_prompt(pipe, short, neg, shot_id="eq")
with torch.no_grad():
    pe, npe, ppe, nppe = pipe.encode_prompt(short, device="cpu", num_images_per_prompt=1,
                                           do_classifier_free_guidance=True, negative_prompt=neg)
fails = []
for name, a, b in [("prompt_embeds", ours["prompt_embeds"], pe), ("negative_prompt_embeds", ours["negative_prompt_embeds"], npe),
                   ("pooled", ours["pooled_prompt_embeds"], ppe), ("negative_pooled", ours["negative_pooled_prompt_embeds"], nppe)]:
    same_shape = tuple(a.shape) == tuple(b.shape)
    close = same_shape and torch.allclose(a.float(), b.float(), atol=1e-5)
    print(f"{name}: ours {tuple(a.shape)} diffusers {tuple(b.shape)} equal={close}")
    if not close:
        fails.append(name)

long = ", ".join(["dark shoulder-length curls"] * 30)  # well past 75 tokens
lo = encode_long_prompt(pipe, long, neg, shot_id="long")
print("long:", tuple(lo["prompt_embeds"].shape), "neg", tuple(lo["negative_prompt_embeds"].shape),
      "pooled", tuple(lo["pooled_prompt_embeds"].shape))
if lo["prompt_embeds"].shape[1] != 154 or lo["negative_prompt_embeds"].shape != lo["prompt_embeds"].shape:
    fails.append("long shape")

# The pipe must accept the embeds end to end (one tiny step).
img = pipe(**lo, num_inference_steps=1, guidance_scale=5.0, height=64, width=64, output_type="np").images
print("pipe ran with chunked embeds:", img.shape)
print("FAIL" if fails else "PASS", fails)
sys.exit(1 if fails else 0)
