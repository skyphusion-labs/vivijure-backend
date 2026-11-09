"""Long SDXL prompts, encoded in 75-token chunks (vivijure-cf#1143).

SDXL's two CLIP text encoders each read 77 tokens: a start token, 75 of ours, an end token. The stock
pipelines truncate anything longer, silently. A two-person identity prompt that carries both people's hair
and clothes does not fit in 75, and the cut fell on Mara's hair and Dev's hoodie (vivijure-cf#1143, read
from the prompt this code builds for film-076828c5 shot_03).

So the prompt is split into chunks of at most 75 tokens. Each chunk is wrapped in its own start and end
tokens, padded to 77, and encoded on its own, and the hidden states are joined along the sequence axis.
The pooled embedding SDXL also needs comes from the first chunk. The negative prompt is encoded the same
way, padded with empty chunks to the same length, because the UNet's cross-attention takes one length for
both halves of the guidance batch. This is the long-prompt method used by the common SDXL front ends.

A split prefers the last comma in the final `backtrack` tokens of a chunk, so a clause such as "grey zip
hoodie" is not cut in two across chunks.

The split is pure and tested on CPU. The encoder needs torch and the pipe's own tokenizers and encoders,
and is validated on a pod.
"""
from __future__ import annotations

import json

CLIP_CHUNK_TOKENS = 75   # 77 less the start and end tokens
PROMPT_CHUNKS = 2        # 150 tokens of prompt. Past this the rest is dropped, and reported.
COMMA_BACKTRACK = 20


def chunk_token_ids(ids: list[int], comma_id: int | None, *, size: int = CLIP_CHUNK_TOKENS,
                    backtrack: int = COMMA_BACKTRACK, max_chunks: int = PROMPT_CHUNKS) -> tuple[list[list[int]], int]:
    """Split prompt token ids (no start or end tokens) into chunks of at most `size`.

    A chunk that would be cut mid-run ends instead just after the last comma in its final `backtrack`
    tokens, when there is one. Returns `(chunks, dropped)`: at most `max_chunks` chunks, and how many
    tokens did not fit in them. An empty prompt is one empty chunk, so the encoder still has something
    to encode."""
    if size <= 0 or max_chunks <= 0:
        raise ValueError("size and max_chunks must be positive")
    chunks: list[list[int]] = []
    i = 0
    n = len(ids)
    while i < n and len(chunks) < max_chunks:
        end = min(i + size, n)
        if end < n and comma_id is not None:
            window_start = max(i + 1, end - backtrack)
            for j in range(end - 1, window_start - 1, -1):
                if ids[j] == comma_id:
                    end = j + 1
                    break
        chunks.append(list(ids[i:end]))
        i = end
    if not chunks:
        chunks = [[]]
    return chunks, n - i


def frame_chunk(chunk: list[int], bos: int, eos: int, pad: int, size: int = CLIP_CHUNK_TOKENS) -> list[int]:
    """One chunk as the encoder reads it: start, the chunk, end, then padding to `size + 2`."""
    if len(chunk) > size:
        raise ValueError(f"chunk of {len(chunk)} tokens exceeds {size}")
    return [bos, *chunk, eos] + [pad] * (size - len(chunk))


def _comma_id(tokenizer) -> int | None:
    ids = tokenizer(",", add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def _encode_one(tokenizer, encoder, chunks: list[list[int]], device):
    import torch

    rows = [frame_chunk(c, tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) for c in chunks]
    ids = torch.tensor(rows, dtype=torch.long, device=device)
    out = encoder(ids, output_hidden_states=True, return_dict=False)
    hidden = out[-1][-2]                    # penultimate layer, as SDXL reads it: (chunks, 77, dim)
    seq = hidden.reshape(1, -1, hidden.shape[-1])
    return seq, out[0][:1]                  # out[0] is encoder-2's pooled projection; first chunk only


def encode_long_prompt(pipe, prompt: str, negative: str, *, shot_id: str = "shot",
                       max_chunks: int = PROMPT_CHUNKS) -> dict:
    """The keyword arguments an SDXL pipe takes IN PLACE OF `prompt` and `negative_prompt`:
    `prompt_embeds`, `negative_prompt_embeds`, `pooled_prompt_embeds`, `negative_pooled_prompt_embeds`.

    Both tokenizers share SDXL's vocabulary, so the split is computed once and used for both encoders.
    If the two ever tokenize differently, this refuses rather than pair mismatched chunks."""
    import torch

    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    enc1, enc2 = pipe.text_encoder, pipe.text_encoder_2
    device = enc2.device

    def split(text: str):
        ids1 = tok1(text, add_special_tokens=False, truncation=False).input_ids
        ids2 = tok2(text, add_special_tokens=False, truncation=False).input_ids
        if ids1 != ids2:
            raise RuntimeError("SDXL tokenizers disagree on the prompt; refusing to pair mismatched chunks")
        return chunk_token_ids(ids1, _comma_id(tok1), max_chunks=max_chunks), len(ids1)

    (pos, pos_dropped), pos_tokens = split(prompt)
    (neg, neg_dropped), _ = split(negative or "")
    n = max(len(pos), len(neg))
    pos = pos + [[]] * (n - len(pos))
    neg = neg + [[]] * (n - len(neg))
    if pos_dropped or neg_dropped:
        # Never silent: a cut prompt is reported with its size, never its text.
        print("@event keyframe.prompt_cut " + json.dumps(
            {"shot_id": shot_id, "tokens": pos_tokens, "dropped": pos_dropped, "negative_dropped": neg_dropped,
             "limit": max_chunks * CLIP_CHUNK_TOKENS}, sort_keys=True), flush=True)

    with torch.no_grad():
        p1, _ = _encode_one(tok1, enc1, pos, device)
        p2, pooled = _encode_one(tok2, enc2, pos, device)
        n1, _ = _encode_one(tok1, enc1, neg, device)
        n2, npooled = _encode_one(tok2, enc2, neg, device)
    dtype = enc2.dtype
    return {
        "prompt_embeds": torch.cat([p1, p2], dim=-1).to(dtype),
        "negative_prompt_embeds": torch.cat([n1, n2], dim=-1).to(dtype),
        "pooled_prompt_embeds": pooled.to(dtype),
        "negative_pooled_prompt_embeds": npooled.to(dtype),
    }
