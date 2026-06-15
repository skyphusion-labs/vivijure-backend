"""Build-time script: bake HF repo config files (no weights) into the image.

Runs at image BUILD TIME with HF_HUB_OFFLINE overridden to 0 so network calls succeed.
At RENDER TIME the worker runs with HF_HUB_OFFLINE=1 from the Dockerfile ENV, finding the
baked configs in the image layer and the .no_exist stubs for the known-absent probe paths
rather than raising LocalEntryNotFoundError and aborting.

Standalone: no imports from vivijure_backend so this runs BEFORE `COPY src/` in the
Dockerfile (avoids busting this layer on every code change). Keep STUBS in sync with
models_mirror.HF_OFFLINE_STUBS which is the testable copy used by tests.
"""
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HF_HOME = os.environ.get("HF_HOME", "/opt/models/hf-cache")

# Exclude all weight tensors; only config JSONs, tokenizers, and metadata land in the image.
_WEIGHT_GLOBS = [
    "*.safetensors", "*.bin", "*.pt", "*.gguf", "*.pkl", "*.onnx",
    "*.pth", "flax_model.msgpack", "tf_model.h5", "rust_model.ot", "*.h5",
]

REPOS = [
    "SG161222/RealVisXL_V5.0",            # SDXL base (keyframe + LoRA train)
    "xinsir/controlnet-openpose-sdxl-1.0", # ControlNet pose
    "h94/IP-Adapter",                      # IP-Adapter (image_encoder config)
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",   # Wan I2V (configs only; weights arrive from R2)
]
for repo in REPOS:
    print(f"bake_hf_configs: downloading configs for {repo} ...", flush=True)
    snapshot_download(repo, ignore_patterns=_WEIGHT_GLOBS)
    print(f"bake_hf_configs: done {repo}", flush=True)

# Write .no_exist stubs for 3 known-absent files that diffusers probes for offline.
# Format: hub/<cache-dir>/.no_exist/<revision>/<file-path> (empty file = negative-cache entry).
# Probe #4 (additional_chat_templates tree listing) is handled in lora_train.py via
# local_files_only=True -- tree-listing responses are not cached as .no_exist stubs.
# Keep in sync with models_mirror.HF_OFFLINE_STUBS (the testable copy).
STUBS = [
    # probe 1: shard-index check for the VAE; single-file VAE has no index.json
    ("models--SG161222--RealVisXL_V5.0",             "vae/diffusion_pytorch_model.safetensors.index.json"),
    # probe 2: same shard-index check for the xinsir ControlNet weights
    ("models--xinsir--controlnet-openpose-sdxl-1.0", "diffusion_pytorch_model.safetensors.index.json"),
    # probe 3: PEFT adapter_config probe for IP-Adapter image_encoder (not a PEFT model)
    ("models--h94--IP-Adapter",                      "sdxl_models/image_encoder/adapter_config.json"),
]

hub = Path(HF_HOME) / "hub"
written = 0
for cache_dir, absent_path in STUBS:
    ref_file = hub / cache_dir / "refs" / "main"
    if not ref_file.exists():
        print(f"bake_hf_configs: WARNING no refs/main for {cache_dir}; skipping stub", flush=True)
        continue
    rev = ref_file.read_text().strip()
    stub = hub / cache_dir / ".no_exist" / rev / absent_path
    stub.parent.mkdir(parents=True, exist_ok=True)
    stub.write_text("")
    print(f"bake_hf_configs: stub {cache_dir}/.no_exist/{rev[:12]}/{absent_path}", flush=True)
    written += 1

print(f"bake_hf_configs: complete ({written} stub(s) written).", flush=True)
