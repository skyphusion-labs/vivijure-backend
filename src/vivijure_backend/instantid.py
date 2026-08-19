"""InstantID single-character face identity for the keyframe stage.

InstantID raises single-subject face fidelity above the plain IP-Adapter path by injecting a face
EMBEDDING from insightface (antelopev2) through an image-projection (a Resampler) as extra
cross-attention tokens -- like an IP-Adapter but keyed on identity rather than a CLIP image embed.

This module is a clean-room reimplementation built from the published InstantID architecture and
the diffusers IP-Adapter attention-processor interface. It shares NOTHING with any prior render
pipeline. The Resampler dims are InstantID's documented public spec.

The face-KEYPOINTS ControlNet (IdentityNet) is a planned future enhancement; the IP-Adapter face
embedding path here is the live implementation.
"""
from __future__ import annotations

# InstantID's published image-projection (Resampler) shape for SDXL: face embedding (512) -> 16
# identity tokens at the UNet cross-attention dim (2048). These are the model's own documented
# hyperparameters, not tunable knobs.
FACE_EMBED_DIM = 512
NUM_ID_TOKENS = 16
CROSS_ATTENTION_DIM = 2048

def largest_face(faces):
    """Pick the dominant face from an insightface detection list: the one with the largest bounding
    box (so a background bystander never steals the identity). `faces` items expose `.bbox` (x1, y1,
    x2, y2). Returns the chosen face, or None for an empty list. Pure: no model imports."""
    best = None
    best_area = -1.0
    for f in faces or []:
        x1, y1, x2, y2 = (float(v) for v in f.bbox)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best, best_area = f, area
    return best


def analyze_face(analyzer, image):
    """Run insightface on a reference and return (normed_embedding, kps, (width, height)) for its
    largest face, or None if no face is found. `image` may be a PIL Image (what keyframe._ref_images
    hands us) or a path/str. The embedding feeds the image-projection; kps (in the reference's pixel
    coords, hence the size) from the reference face.
    GPU/onnxruntime path: deferred imports, validated on a pod."""
    import numpy as np
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # insightface wants BGR
    face = largest_face(analyzer.get(arr))
    if face is None:
        return None
    return face.normed_embedding, face.kps, img.size


def crop_from_bbox(image, bbox, *, pad_ratio: float = 0.35):
    """PIL RGB crop of an insightface bbox `(x1, y1, x2, y2)`, padded and clipped to the image.

    Pure except PIL. A degenerate or empty bbox returns the original image so a caller that
    already decided a face exists never gets a zero-size crop."""
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("RGB")
    w, h = img.size
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return img
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return img
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(w, int(x2 + pad_x))
    bottom = min(h, int(y2 + pad_y))
    if right <= left or bottom <= top:
        return img
    return img.crop((left, top, right, bottom))


def crop_face(analyzer, image, *, pad_ratio: float = 0.35):
    """Largest-face crop via the insightface bbox, or None if no face.

    Pass B feeds this crop as `ip_adapter_image`, never the full studio portrait. GPU/onnx
    path: deferred imports, validated on a pod."""
    import numpy as np
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.convert("RGB")
    arr = np.array(img)[:, :, ::-1]  # insightface wants BGR
    face = largest_face(analyzer.get(arr))
    if face is None:
        return None
    return crop_from_bbox(img, face.bbox, pad_ratio=pad_ratio)


def build_image_proj(state_dict):
    """Construct InstantID's image-projection (a Resampler / perceiver) and load its weights from the
    `image_proj` sub-dict of `ip-adapter.bin`. The Resampler maps one 512-d face embedding to
    NUM_ID_TOKENS identity tokens at CROSS_ATTENTION_DIM. Architecture is InstantID's public spec;
    deferred torch import keeps the module CPU-importable. Validated on a pod."""
    import torch
    from torch import nn

    class PerceiverAttention(nn.Module):
        def __init__(self, dim, dim_head=64, heads=20):
            super().__init__()
            self.scale = dim_head ** -0.5
            self.dim_head = dim_head
            self.heads = heads
            inner = dim_head * heads
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, inner, bias=False)
            self.to_kv = nn.Linear(dim, inner * 2, bias=False)
            self.to_out = nn.Linear(inner, dim, bias=False)

        def forward(self, x, latents):
            x = self.norm1(x)
            latents = self.norm2(latents)
            b, n, _ = latents.shape
            q = self.to_q(latents)
            kv = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)
            k, v = kv

            def split(t):
                return t.reshape(b, t.shape[1], self.heads, self.dim_head).transpose(1, 2)

            q, k, v = map(split, (q, k, v))
            attn = (q * self.scale) @ (k * self.scale).transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            out = attn @ v
            out = out.transpose(1, 2).reshape(b, n, -1)
            return self.to_out(out)

    def feed_forward(dim, mult=4):
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * mult, bias=False),
                             nn.GELU(), nn.Linear(dim * mult, dim, bias=False))

    class Resampler(nn.Module):
        def __init__(self, dim=1280, depth=4, dim_head=64, heads=20, num_queries=NUM_ID_TOKENS,
                     embedding_dim=FACE_EMBED_DIM, output_dim=CROSS_ATTENTION_DIM, ff_mult=4):
            super().__init__()
            self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
            self.proj_in = nn.Linear(embedding_dim, dim)
            self.proj_out = nn.Linear(dim, output_dim)
            self.norm_out = nn.LayerNorm(output_dim)
            self.layers = nn.ModuleList(
                [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), feed_forward(dim, ff_mult)])
                 for _ in range(depth)])

        def forward(self, x):
            latents = self.latents.repeat(x.size(0), 1, 1)
            x = self.proj_in(x)
            for attn, ff in self.layers:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents
            return self.norm_out(self.proj_out(latents))

    model = Resampler()
    r = model.load_state_dict(state_dict, strict=False)
    if r.missing_keys or r.unexpected_keys:  # surfaces a future checkpoint-shape drift, not silence
        print(f"[instantid] image_proj load mismatch: missing={len(r.missing_keys)} "
              f"unexpected={len(r.unexpected_keys)} (identity may be degraded)", flush=True)
    return model


def faceid_tokens(image_proj, face_embedding):
    """Project a single insightface face embedding into the InstantID identity tokens the UNet's
    IP-Adapter cross-attention consumes. Deferred torch; validated on a pod."""
    import torch

    emb = torch.as_tensor(face_embedding, dtype=image_proj.proj_in.weight.dtype,
                          device=image_proj.proj_in.weight.device).reshape(1, 1, FACE_EMBED_DIM)
    with torch.no_grad():
        return image_proj(emb)


def set_instantid_ip_attn(unet, ip_state_dict, *, num_tokens: int = NUM_ID_TOKENS, scale: float = 0.8):
    """Wire InstantID's identity tokens into the UNet cross-attention. Replaces each cross-attention
    processor with an IP-Adapter processor that adds a second attention over the `num_tokens`
    identity tokens (the projected face embedding) and sums it into the text attention at `scale`;
    self-attention layers keep the default processor. Weights come from the `ip_adapter` sub-dict of
    InstantID's `ip-adapter.bin`. Built from the documented IP-Adapter attention-processor interface
    in diffusers; deferred imports, validated on a pod.

    Returns the dict of installed identity processors so the caller can retune `scale` per render
    without rebuilding them.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    class IPAttnProcessor(nn.Module):
        """Cross-attention with an added IP-Adapter branch. The text cross-attention is computed
        normally over `encoder_hidden_states` (the 77 prompt tokens, kept clean so the ControlNet is
        unaffected); the identity tokens arrive via the SIDE channel `self.id_embeds` (set per render),
        get their own attention, and are summed in at `self.scale`. NOT concatenated onto the prompt
        embeds (that would corrupt the ControlNet, which never saw appended tokens)."""
        def __init__(self, hidden_size, cross_attention_dim, num_tokens, scale):
            super().__init__()
            self.num_tokens = num_tokens
            self.scale = scale
            self.id_embeds = None  # (batch, num_tokens, cross_attention_dim); set just before pipe()
            self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
            self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states  # self-attn fallback (not hit on attn2)
            q = attn.to_q(hidden_states)
            k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)
            heads, dim = attn.heads, q.shape[-1] // attn.heads

            def shape(t):
                return t.view(t.shape[0], t.shape[1], heads, dim).transpose(1, 2)

            out = F.scaled_dot_product_attention(shape(q), shape(k), shape(v))
            if self.id_embeds is not None and self.scale:
                ie = self.id_embeds.to(q.dtype)
                if ie.shape[0] != q.shape[0]:  # match the (CFG or not) batch
                    ie = ie[-q.shape[0]:] if ie.shape[0] > q.shape[0] else ie.repeat(q.shape[0], 1, 1)
                ip_out = F.scaled_dot_product_attention(
                    shape(q), shape(self.to_k_ip(ie)), shape(self.to_v_ip(ie)))
                out = out + self.scale * ip_out
            out = out.transpose(1, 2).reshape(out.shape[0], -1, heads * dim)
            return attn.to_out[1](attn.to_out[0](out))

    # InstantID's ip_adapter weights are keyed by each layer's index over ALL attn processors (self +
    # cross) in unet.attn_processors order; self-attn slots carry no params, so the checkpoint simply
    # has no keys for them. Enumerate the full processor list and load each cross-attention layer's
    # to_k_ip / to_v_ip from the checkpoint by its OVERALL index (a ModuleList over the full set is
    # not possible: diffusers' default self-attn processor is not an nn.Module).
    procs = {}
    installed = {}
    for idx, name in enumerate(unet.attn_processors.keys()):
        if not name.endswith("attn2.processor"):  # cross-attention only
            procs[name] = unet.attn_processors[name]
            continue
        if name.startswith("mid_block"):
            hidden = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            i = int(name[len("up_blocks.")])
            hidden = list(reversed(unet.config.block_out_channels))[i]
        else:
            i = int(name[len("down_blocks.")])
            hidden = unet.config.block_out_channels[i]
        p = IPAttnProcessor(hidden, CROSS_ATTENTION_DIM, num_tokens, scale).to(
            device=unet.device, dtype=unet.dtype)
        kw = ip_state_dict.get(f"{idx}.to_k_ip.weight")
        vw = ip_state_dict.get(f"{idx}.to_v_ip.weight")
        if kw is not None:
            p.to_k_ip.weight.data.copy_(kw.to(device=unet.device, dtype=unet.dtype))
        if vw is not None:
            p.to_v_ip.weight.data.copy_(vw.to(device=unet.device, dtype=unet.dtype))
        procs[name] = p
        installed[name] = p
    unet.set_attn_processor(procs)
    return installed
