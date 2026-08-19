"""Build-time script: bake HF repo config files (no weights) into the image.

Runs at image BUILD TIME with HF_HUB_OFFLINE overridden to 0 so network calls succeed.
At RENDER TIME the worker runs with HF_HUB_OFFLINE=1 from the Dockerfile ENV, finding the
baked configs in the image layer and the .no_exist stubs for the known-absent probe paths
rather than raising LocalEntryNotFoundError and aborting.

Standalone: no imports from vivijure_backend so this runs BEFORE `COPY src/` in the
Dockerfile (avoids busting this layer on every code change). Keep STUBS in sync with
models_mirror.HF_OFFLINE_STUBS which is the testable copy used by tests.

TREE-CACHE SCRUB (#206). huggingface_hub 1.x writes a per-repo tree listing at
`hub/<repo>/trees/<commit>.json` on every ONLINE snapshot_download -- the full sibling list of
the revision. At RENDER TIME (offline) hf_hub `_raise_if_incomplete_snapshot` reads that listing
and hard-fails (`IncompleteSnapshotError`) unless EVERY sibling it names is present on disk. Our
baked image ships a deliberately CURATED subset (diffusers folder layout only; the redundant root
single-file checkpoints are dropped, and one of them -- RealVisXL_V5.0 fp32, 13.8 GB -- physically
cannot be baked, being over the 10 GB GHCR per-layer ceiling). So the online config bake here
poisons the image with a tree listing that claims files we intentionally, correctly omit, and the
first prod keyframe job dies offline. The check is a documented no-op when no tree listing is cached
("If the tree listing is not cached we cannot tell, we do nothing"), so we SCRUB every `trees/` dir
the config bake wrote: offline load then returns the folder as-is (the pre-1.x behaviour) using
exactly the files the pipeline loads. The rclone-staged R2 mirror never carried a tree cache, which
is why only the baked path hit this. deploy/bake_layers.py `assert-no-tree-cache` re-asserts, at
BAKE, that no tree listing survived (fail-at-bake, not at the first prod job).
"""
import os
from pathlib import Path

HF_HOME = os.environ.get("HF_HOME", "/opt/models/hf-cache")

# Exclude all weight tensors; only config JSONs, tokenizers, and metadata land in the image.
_WEIGHT_GLOBS = [
    "*.safetensors", "*.bin", "*.pt", "*.gguf", "*.pkl", "*.onnx",
    "*.pth", "flax_model.msgpack", "tf_model.h5", "rust_model.ot", "*.h5",
]

REPOS = [
    "SG161222/RealVisXL_V5.0",            # SDXL base (keyframe + LoRA train)
    "xinsir/controlnet-openpose-sdxl-1.0", # ControlNet pose
    "xinsir/controlnet-canny-sdxl-1.0",    # ControlNet canny (scene-lock Pass B)
    "h94/IP-Adapter",                      # IP-Adapter (image_encoder config)
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",   # Wan I2V (configs only; weights arrive from R2)
]

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
    ("models--xinsir--controlnet-canny-sdxl-1.0",    "diffusion_pytorch_model.safetensors.index.json"),
    # probe 3: PEFT adapter_config probe for IP-Adapter image_encoder (not a PEFT model)
    ("models--h94--IP-Adapter",                      "sdxl_models/image_encoder/adapter_config.json"),
]


def scrub_tree_cache(hub: Path, log=print) -> int:
    """Remove every huggingface_hub tree-cache listing under `hub` (the `<repo>/trees/` dirs).

    Stdlib-only + import-free so it is unit-testable without huggingface_hub (the hf_hub import in
    bake_configs is deferred for the same reason). See the module docstring for WHY: a baked tree
    listing turns the offline completeness check into a hard failure against our deliberately-curated
    model subset (#206). Idempotent; returns the number of listing files removed."""
    removed = 0
    for trees in hub.glob("*/trees"):
        if not trees.is_dir():
            continue
        for entry in sorted(trees.glob("*.json")):
            entry.unlink()
            removed += 1
            log(f"bake_hf_configs: scrubbed tree cache {entry.relative_to(hub)}")
        try:
            trees.rmdir()  # drop the now-empty dir so the baked layer carries no `trees/` at all
        except OSError:
            pass
    log(f"bake_hf_configs: tree-cache scrub removed {removed} listing(s) under {hub}.")
    return removed


def bake_configs(hub: Path, log=print) -> None:
    """Download each repo config/tokenizer/metadata (no weights) into the HF cache, then scrub the
    tree-cache listings the online snapshot_download wrote (#206). hf_hub is imported HERE, not at
    module top, so scrub_tree_cache stays importable in a test env without huggingface_hub."""
    from huggingface_hub import snapshot_download

    for repo in REPOS:
        log(f"bake_hf_configs: downloading configs for {repo} ...")
        snapshot_download(repo, ignore_patterns=_WEIGHT_GLOBS)
        log(f"bake_hf_configs: done {repo}")
    scrub_tree_cache(hub)


def write_stubs(hub: Path, log=print) -> int:
    """Write the .no_exist negative-cache stubs for the known-absent offline probe paths."""
    written = 0
    for cache_dir, absent_path in STUBS:
        ref_file = hub / cache_dir / "refs" / "main"
        if not ref_file.exists():
            log(f"bake_hf_configs: WARNING no refs/main for {cache_dir}; skipping stub")
            continue
        rev = ref_file.read_text().strip()
        stub = hub / cache_dir / ".no_exist" / rev / absent_path
        stub.parent.mkdir(parents=True, exist_ok=True)
        stub.write_text("")
        log(f"bake_hf_configs: stub {cache_dir}/.no_exist/{rev[:12]}/{absent_path}")
        written += 1
    return written


def main() -> None:
    hub = Path(HF_HOME) / "hub"
    bake_configs(hub)
    written = write_stubs(hub)
    print(f"bake_hf_configs: complete ({written} stub(s) written).", flush=True)


if __name__ == "__main__":
    main()
