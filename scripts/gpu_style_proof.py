#!/usr/bin/env python3
"""GPU proof: photoreal RealVisXL and anime Animagine each draw one still.

Run on a CUDA box. Downloads public HF checkpoints if they are not cached.
Does not train a LoRA. Does not call i2v. Writes two PNGs and a json report.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vivijure_backend.models import (  # noqa: E402
    DEFAULT_SPECS,
    KEYFRAME_ANIME_REPO,
    ModelRole,
    validate_repo_id,
)

PHOTOREAL = DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id
ANIME = KEYFRAME_ANIME_REPO
PROMPT = "two people standing in a rainy neon alley, night, film still"
NEGATIVE = "blurry, extra limbs, watermark, text"
STEPS = 6
SIZE = 768


def _draw(repo_id: str, out_path: Path) -> dict:
    import torch
    from diffusers import StableDiffusionXLPipeline

    rid = validate_repo_id(repo_id)
    t0 = time.time()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(rid, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe.to("cuda")
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
        width=SIZE,
        height=SIZE,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "repo": rid,
        "path": str(out_path),
        "bytes": out_path.stat().st_size,
        "seconds": round(time.time() - t0, 1),
        "exists": out_path.is_file(),
    }


def main() -> int:
    import torch

    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/style-proof")
    report = {
        "cuda": bool(torch.cuda.is_available()),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "photoreal_repo": validate_repo_id(PHOTOREAL),
        "anime_repo": validate_repo_id(ANIME),
        "steps": STEPS,
        "size": SIZE,
    }
    if not report["cuda"]:
        print(json.dumps({"ok": False, "error": "no CUDA", **report}, indent=2))
        return 1
    print("STYLE_PROOF_START " + json.dumps(report), flush=True)
    photo = _draw(PHOTOREAL, out_dir / "photoreal.png")
    print("STYLE_PROOF_PHOTOREAL " + json.dumps(photo), flush=True)
    anime = _draw(ANIME, out_dir / "anime.png")
    print("STYLE_PROOF_ANIME " + json.dumps(anime), flush=True)
    ok = bool(photo["exists"] and anime["exists"] and photo["bytes"] > 10_000 and anime["bytes"] > 10_000)
    final = {"ok": ok, "photoreal": photo, "anime": anime, **report}
    print("STYLE_PROOF_DONE " + json.dumps(final), flush=True)
    (out_dir / "report.json").write_text(json.dumps(final, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
