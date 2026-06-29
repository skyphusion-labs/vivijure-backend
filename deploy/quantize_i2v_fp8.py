#!/usr/bin/env python3
"""One-time prep: quantize the Wan2.2 I2V A14B bf16 diffusers repo to fp8_e4m3fn, IN PLACE in the
diffusers directory layout, then push to R2 as the bake seed (r2:vivijure/models-fp8/...).

Method (ii), ratified: cast each weight tensor to torch.float8_e4m3fn and re-save -- plain fp8
storage (NOT a torchao tensor-subclass save, which does not round-trip through from_pretrained).
The runtime loader (models.ModelServer.i2v_pipeline, baked path) reads these fp8 shards and applies
the dynamic-activation wrapper for compute, so the bake matches the validated runtime-quant numerics.

CPU-ONLY + STREAMING: the fp8 cast runs on CPU (verified on jello), one shard at a time
(load ~4.9GB bf16 -> cast -> save ~2.5GB -> free), so peak RAM stays ~one shard. FREE (no GPU).
Only the two MoE experts (transformer/, transformer_2/) are quantized; vae/ text_encoder/ tokenizer/
scheduler/ + index/config json are copied byte-for-byte. Each fp8 shard is ~half its bf16 source =>
well under GHCR 10GB/layer.

Usage:
  quantize_i2v_fp8.py --src <bf16 diffusers dir> --dst <fp8 out dir> [--push r2:vivijure/models-fp8/...]
"""
import argparse, shutil, subprocess
from pathlib import Path

QUANTIZED_SUBDIRS = ("transformer", "transformer_2")


def _fp8_cast_file(src_file, dst_file, log):
    import torch
    from safetensors.torch import load_file, save_file
    sd = load_file(str(src_file))
    out = {}
    for k, v in sd.items():
        if v.dtype in (torch.bfloat16, torch.float16, torch.float32):
            out[k] = v.to(torch.float8_e4m3fn)
        else:
            out[k] = v
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst_file))
    del sd, out
    return src_file.stat().st_size, dst_file.stat().st_size


def quantize_repo(src, dst, log=print):
    dst.mkdir(parents=True, exist_ok=True)
    over = 10 * 1024**3
    biggest = 0
    for entry in sorted(src.rglob("*")):
        if entry.is_dir():
            continue
        rel = entry.relative_to(src)
        out = dst / rel
        in_expert = rel.parts and rel.parts[0] in QUANTIZED_SUBDIRS
        if in_expert and entry.suffix == ".safetensors":
            sb, db = _fp8_cast_file(entry, out, log)
            biggest = max(biggest, db)
            log("  fp8 %s: %.2fGB bf16 -> %.2fGB fp8" % (rel, sb/1024**3, db/1024**3))
            if db > over:
                raise SystemExit("FATAL: fp8 shard %s is %.2fGB > 10GB GHCR limit -- reshard needed" % (rel, db/1024**3))
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, out)
    log("DONE. largest emitted shard = %.2fGB (limit 10GB)." % (biggest/1024**3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--push", default=None)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit("src not a dir: %s" % src)
    print("quantize_i2v_fp8: %s -> %s (fp8_e4m3fn, CPU streaming)" % (src, dst))
    quantize_repo(src, dst)
    if args.push:
        print("pushing %s -> %s" % (dst, args.push))
        subprocess.run(["rclone", "copy", str(dst), args.push, "--progress"], check=True)
        print("pushed.")


if __name__ == "__main__":
    main()
