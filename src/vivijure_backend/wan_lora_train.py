"""Per-character Wan 2.2 A14B LoRA training, wrapping the ai-toolkit (Ostris, MIT) trainer.

The sibling `lora_train.py` fits an SDXL identity adapter in-process. This module fits the
Wan 2.2 A14B *video* character LoRA the cloud-i2v cost door needs (vivijure-cf #29). The
difference is architectural, so it is a separate module rather than an overload of `train_slot`:

  - Base model: the Wan 2.2 A14B mixture-of-experts DiT, not SDXL. Wan A14B carries TWO experts
    (a high-noise and a low-noise pass), so a Wan character LoRA is TWO `.safetensors` files, not
    one. The `alibaba-wan-lora` render module consumes them as `high_noise_loras` /
    `low_noise_loras`.
  - Trainer: ai-toolkit's `run.py`, driven by a generated config, NOT a hand-rolled diffusers
    loop. We do not vendor a 14B training loop; we wrap the maintained MIT trainer as a
    subprocess and own only the config we hand it and the two files we harvest back.

The recipe is the one the Phase-0 spike proved on an 80GB A100 (vivijure-cf #29): bf16 with BOTH
experts resident on the GPU (`quantize:false`, `low_vram:false`) so training is GPU-bound rather
than stalling on CPU-side uint4 quantization; identity carried by a per-slot trigger token in the
captions. `low_vram` is exposed for a smaller-card deploy, but the proven, default path is 80GB
bf16.

Clean-room: the config schema follows ai-toolkit's public `train_lora_wan22_14b` example; nothing
here is torch code. The module imports and unit-tests on a CPU box -- the only GPU touch is the
`run.py` subprocess, which is the single injectable seam (`runner=`) so config generation, dataset
prep, and expert harvesting are all testable without a GPU or ai-toolkit installed.

QUEUED (spike follow-up): training on the Wan2.2-**I2V**-A14B base directly (vs the T2V base the
spike used) for cleaner low-scale binding on the i2v endpoint. Change `WanLoraTrainConfig.base_repo`
to try it; the T2V base is the default because it is what the spike validated end to end.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

from .contract import Character

# The ai-toolkit checkout the subprocess runs from. The GPU/training image provides it; overridable
# for a non-standard install. Kept as an env var (not hardcoded) so the deploy owns the location.
AITOOLKIT_DIR_ENV = "VIVIJURE_AITOOLKIT_DIR"
DEFAULT_AITOOLKIT_DIR = "/opt/ai-toolkit"

# The interpreter the ai-toolkit `run.py` subprocess runs under. ai-toolkit's dependency set
# (a diffusers git pin, plus transformers/torchao/av at versions that conflict irreconcilably with
# the worker "vivijure" env's validated render pins) cannot coexist in the worker's own env, so the
# training image isolates ai-toolkit in a SEPARATE conda env and points this at that env's python
# (cf#29 D1). DEFAULT is sys.executable -- the worker's own interpreter -- so a single-env install
# (and every existing CPU test) keeps the pre-isolation behavior unchanged. The deploy sets this to
# the aitoolkit env's python; nothing in this module hardcodes an interpreter location.
AITOOLKIT_PYTHON_ENV = "VIVIJURE_AITOOLKIT_PYTHON"

# The Wan 2.2 A14B base ai-toolkit trains against. The T2V-A14B bf16 diffusers re-host is what the
# Phase-0 spike validated; the adapters it emits load on the i2v `wan-2-2-t2v-720-lora` endpoint
# (shared DiT attention layers). See the module docstring for the queued I2V-base experiment.
#
# Hub id is the LOGICAL name (tests / docs). The train IMAGE sets VIVIJURE_WAN_BASE_PATH to the
# bake-time stable dir (/opt/models/aitoolkit/wan-base) so config never ships a hub id at runtime
# (Conrad: bake it, don't just remap). default_wan_base_path() prefers that env / dir.
DEFAULT_WAN_BASE_REPO = "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16"
WAN_BASE_PATH_ENV = "VIVIJURE_WAN_BASE_PATH"
DEFAULT_WAN_BASE_PATH = "/opt/models/aitoolkit/wan-base"

# Hardcoded HF hub IDs inside upstream ai-toolkit (not our config). The train Dockerfile patches
# these at bake time to absolute local dirs. Runtime rewrite below is belt-and-suspenders only
# (idempotent no-op when the image already scrubbed the strings).
AITOOLKIT_UMT5_REPO = "ai-toolkit/umt5_xxl_encoder"
AITOOLKIT_VAE_REPO = "ai-toolkit/wan2.1-vae"
_AITOOLKIT_HUB_ID_FILES = (
    # (relative path under the ai-toolkit checkout, hub id string as it appears in source)
    ("toolkit/models/wan21/wan21.py", AITOOLKIT_UMT5_REPO),
    ("extensions_built_in/diffusion_models/wan22/wan22_14b_model.py", AITOOLKIT_VAE_REPO),
)

# The two MoE experts a Wan A14B LoRA is split across. Order is fixed; the harvest + the upload
# keys key off these exact names.
EXPERTS = ("high", "low")

# How many trailing stdout lines to keep for a non-zero ai-toolkit exit (RunPod stream often empty).
_AITOOLKIT_TAIL_LINES = 40

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def default_wan_base_path() -> str:
    """The base path/id written into ai-toolkit config for a train run.

    Prefer (1) VIVIJURE_WAN_BASE_PATH env (train image bake), (2) the stable bake dir if it
    exists on disk, else (3) the hub id -- which resolve_local_hf_snapshot must still turn into
    a local snapshot before run.py (belt-and-suspenders; Conrad: bake is the real fix).
    """
    env = (os.environ.get(WAN_BASE_PATH_ENV) or "").strip()
    if env:
        return env
    if Path(DEFAULT_WAN_BASE_PATH).is_dir():
        return DEFAULT_WAN_BASE_PATH
    return DEFAULT_WAN_BASE_REPO


def resolve_local_hf_snapshot(repo_or_path: str) -> str:
    """Resolve a hub id or filesystem path to an absolute local snapshot directory.

    Hub ids under HF_HUB_OFFLINE=1 cannot be passed to diffusers from_pretrained: the loader still
    calls huggingface model_info() for sharded checkpoints and raises OfflineModeIsEnabled even when
    the weights are fully baked. Prefer bake-time stable dirs; snapshot_download(local_files_only=True)
    is the fallback resolver the train image asserts at bake time.
    """
    p = Path(repo_or_path)
    if p.is_dir():
        return str(p.resolve())
    # Hub id that matches the baked Wan base: prefer the stable symlink the image materializes.
    if repo_or_path == DEFAULT_WAN_BASE_REPO and Path(DEFAULT_WAN_BASE_PATH).is_dir():
        return str(Path(DEFAULT_WAN_BASE_PATH).resolve())
    from huggingface_hub import snapshot_download  # local: keep module import light for CPU tests

    try:
        return snapshot_download(repo_or_path, local_files_only=True)
    except Exception as e:
        raise FileNotFoundError(
            f"HF snapshot {repo_or_path!r} not in local cache "
            f"(HF_HOME={os.environ.get('HF_HOME', '')!r}); train image must bake it. "
            f"underlying={e}"
        ) from e


def patch_aitoolkit_hub_ids_for_offline(cwd: Path) -> dict[str, str]:
    """Rewrite ai-toolkit's hardcoded UMT5/VAE hub ids to absolute local snapshot paths.

    Idempotent: skips a file whose hub-id string is already gone. Returns the resolved paths
    (keys: umt5, vae) for logging/tests. If NONE of the target files exist (a stub checkout used
    by CPU unit tests), returns {} without resolving. A partial checkout (some files present,
    some missing) raises -- that is a broken train image, not a test stub.
    """
    cwd = Path(cwd)
    present = [(rel, hub_id) for rel, hub_id in _AITOOLKIT_HUB_ID_FILES if (cwd / rel).is_file()]
    if not present:
        return {}
    if len(present) != len(_AITOOLKIT_HUB_ID_FILES):
        missing = [rel for rel, _ in _AITOOLKIT_HUB_ID_FILES if not (cwd / rel).is_file()]
        raise FileNotFoundError(
            f"ai-toolkit offline hub-id patch: incomplete checkout under {cwd}; "
            f"missing {missing} (set {AITOOLKIT_DIR_ENV})")
    resolved = {
        "umt5": resolve_local_hf_snapshot(AITOOLKIT_UMT5_REPO),
        "vae": resolve_local_hf_snapshot(AITOOLKIT_VAE_REPO),
    }
    by_repo = {AITOOLKIT_UMT5_REPO: resolved["umt5"], AITOOLKIT_VAE_REPO: resolved["vae"]}
    for rel, hub_id in present:
        path = cwd / rel
        text = path.read_text(encoding="utf-8")
        needle = f'"{hub_id}"'
        if needle not in text:
            continue  # already patched (or upstream renamed the string)
        path.write_text(text.replace(needle, repr(by_repo[hub_id])), encoding="utf-8")
    return resolved


@dataclass
class WanLoraTrainConfig:
    """Knobs for one slot's Wan 2.2 A14B LoRA run. Defaults are the spike-proven recipe: rank 32,
    a run that binds identity without over-baking (the spike harvested a strong bind by ~1000
    steps), bf16 with both experts resident on an 80GB card. `low_vram=True` trades speed for a
    smaller card (CPU expert-swapping); the default False is the proven, GPU-bound path."""
    rank: int = 32
    resolution: tuple[int, ...] = (512, 768, 1024)
    learning_rate: float = 1e-4
    steps: int = 2000               # 500-4000 is a sane range; identity binds well by ~1000
    save_every: int = 250
    max_step_saves_to_keep: int = 8
    switch_boundary_every: int = 10  # Wan MoE: alternate high/low expert training every N steps
    caption_dropout_rate: float = 0.05
    weight_decay: float = 1e-4
    seed: int = 0
    low_vram: bool = False          # False = both experts GPU-resident (80GB, proven); True = swap
    # Default is bake-time local dir when present (see default_wan_base_path); hub id otherwise.
    base_repo: str = field(default_factory=default_wan_base_path)
    # The caption every reference trains under. `{name}` fills from the slot; the name is the
    # trigger token the render prompt uses to summon the identity. Same substitution rules as the
    # SDXL trainer (str.replace, {name}/{prompt} only).
    caption_template: str = "{name}"


@dataclass
class TrainedWanLora:
    """Where one slot's two Wan experts landed, plus the facts the harness records + uploads."""
    slot: str
    high_path: Path                 # the high-noise expert .safetensors
    low_path: Path                  # the low-noise expert .safetensors
    trigger: str                    # the token a render prompt uses to summon this identity
    steps: int
    rank: int
    ref_count: int
    base_repo: str
    meta: dict = field(default_factory=dict)


def aitoolkit_dir() -> Path:
    """The ai-toolkit checkout to run `run.py` from (env override, else the image default)."""
    return Path(os.environ.get(AITOOLKIT_DIR_ENV) or DEFAULT_AITOOLKIT_DIR)


def aitoolkit_python() -> str:
    """The interpreter to launch ai-toolkit's `run.py` with. Env override (the deploy points it at
    the isolated aitoolkit conda env), else `sys.executable` -- the worker's own interpreter, the
    pre-isolation default that keeps single-env installs and the CPU tests unchanged."""
    return os.environ.get(AITOOLKIT_PYTHON_ENV) or sys.executable


def wan_train_runtime_ready() -> bool:
    """True when this worker image can run the ai-toolkit Wan trainer (cf#29 train image).

    The render image lacks the isolated aitoolkit env and baked Wan base weights; a train_lora job
    that resolves to the Wan family on that image must fail loud (see pipeline._train_slot_wan).
    CPU tests keep the pre-train-image behavior (False unless env/deps are stubbed in).
    """
    env_py = (os.environ.get(AITOOLKIT_PYTHON_ENV) or "").strip()
    if env_py:
        p = Path(env_py)
        if p.is_file() and os.access(p, os.X_OK):
            return True
    if Path(DEFAULT_WAN_BASE_PATH).is_dir():
        return True
    run_py = Path(DEFAULT_AITOOLKIT_DIR) / "run.py"
    return run_py.is_file()


def caption_for(char: Character, template: str) -> str:
    """The training caption for a slot. Same safe substitution as `lora_train.caption_for`:
    str.replace (never str.format, so no attribute-access payloads), only `{name}`/`{prompt}`
    allowed, unknown placeholders raise, and stray punctuation from an empty field is collapsed."""
    if "{" in template or "}" in template:
        filled = template.replace("{name}", char.name or char.slot).replace("{prompt}", char.prompt or "")
        if any(c in "{}" for c in filled):
            raise ValueError(
                f"caption_template contains unsupported placeholders: {template!r}; "
                "only {name} and {prompt} are allowed")
        text = filled
    else:
        text = template
    return ", ".join(part.strip() for part in text.split(",") if part.strip())


def _run_name(slot: str) -> str:
    """ai-toolkit uses the config `name` for BOTH the output subdir and the filename stem; reduce
    the slot to a filesystem-safe stem so a slot id can never smuggle a slash or space into it."""
    return "_".join(str(slot).strip().split()).replace("/", "_") or "wan_lora"


def build_aitoolkit_config(
    name: str, dataset_dir: Path, training_folder: Path, cfg: WanLoraTrainConfig,
) -> dict:
    """Pure: the ai-toolkit job config for one Wan 2.2 A14B character LoRA run, as a dict ready to
    dump to YAML. Encodes the spike-proven bf16 both-experts recipe. No I/O, so it unit-tests
    without a GPU or ai-toolkit -- the assertions this pipeline depends on (dual-expert output,
    bf16/no-quant, dataset + steps wiring) are all checkable here."""
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(training_folder),
                    "device": "cuda:0",
                    "network": {"type": "lora", "linear": cfg.rank, "linear_alpha": cfg.rank},
                    "save": {
                        "dtype": "float16",
                        "save_every": cfg.save_every,
                        "max_step_saves_to_keep": cfg.max_step_saves_to_keep,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_dir),
                            "caption_ext": "txt",
                            "caption_dropout_rate": cfg.caption_dropout_rate,
                            "num_frames": 1,               # image dataset: one frame per still
                            "resolution": list(cfg.resolution),
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": cfg.steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,       # Wan: TE stays frozen
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "timestep_type": "linear",
                        "optimizer": "adamw8bit",
                        "lr": cfg.learning_rate,
                        "optimizer_params": {"weight_decay": cfg.weight_decay},
                        "dtype": "bf16",
                        "switch_boundary_every": cfg.switch_boundary_every,  # MoE expert alternation
                        "cache_text_embeddings": True,     # captions carry the trigger; cache + free the TE
                        "skip_first_sample": True,         # production run: no baseline sample overhead
                        "disable_sampling": True,          # evidence comes from the render A/B, not samples
                    },
                    "model": {
                        "name_or_path": cfg.base_repo,
                        "arch": "wan22_14b",
                        # bf16, both experts resident -- the spike-proven, GPU-bound path (no
                        # CPU-side uint4 quantization stall). low_vram flips to CPU expert-swapping.
                        "quantize": False,
                        "quantize_te": False,
                        "low_vram": cfg.low_vram,
                        "model_kwargs": {"train_high_noise": True, "train_low_noise": True},
                    },
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def prepare_dataset(char: Character, dataset_dir: Path, cfg: WanLoraTrainConfig) -> int:
    """Stage the slot's reference images + per-image caption `.txt` files into `dataset_dir` (the
    layout ai-toolkit expects: image `X.ext` beside caption `X.txt`). The caption carries the
    trigger token so identity binds to it. Returns the reference count. Raises if the slot has no
    references (the same honest failure as the SDXL trainer)."""
    refs = list(char.ref_paths)
    if not refs:
        raise ValueError(f"slot {char.slot} ({char.name}) has no reference images to train on")
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    caption = caption_for(char, cfg.caption_template)
    for i, ref in enumerate(refs):
        ref = Path(ref)
        stem = f"ref_{i:03d}"
        dest_img = dataset_dir / (stem + ref.suffix.lower())
        shutil.copyfile(ref, dest_img)
        (dataset_dir / (stem + ".txt")).write_text(caption + "\n", encoding="utf-8")
    return len(refs)


def harvest_experts(run_dir: Path, name: str) -> tuple[Path, Path]:
    """Find the trained high-noise + low-noise expert `.safetensors` under ai-toolkit's output
    dir (`<training_folder>/<name>/`), taking the HIGHEST-step checkpoint of each. Pure filesystem
    read, no torch. Raises if either expert is missing -- a half-trained run must fail loud, never
    ship one expert and silently skip the other."""
    run_dir = Path(run_dir)
    out: dict[str, Path] = {}
    for expert in EXPERTS:
        cands = sorted(run_dir.glob(f"{name}_*_{expert}_noise.safetensors"))
        # Fallback for a no-step-suffix final save, if ai-toolkit ever writes one.
        cands += sorted(run_dir.glob(f"{name}_{expert}_noise.safetensors"))
        if not cands:
            raise FileNotFoundError(
                f"wan_lora harvest: no {expert}-noise expert .safetensors under {run_dir} "
                f"(looked for {name}_*_{expert}_noise.safetensors); training did not complete")
        out[expert] = cands[-1]  # sorted lexicographically -> zero-padded step -> highest step last
    return out["high"], out["low"]


def _run_aitoolkit(config_path: Path, *, cwd: Path, progress_cb=None) -> None:
    """The single un-stubbable seam: run ai-toolkit's `run.py <config>` as a subprocess and raise
    on a non-zero exit. The interpreter comes from `aitoolkit_python()` (the isolated aitoolkit env
    in the training image, else `sys.executable`), NOT a hardcoded interpreter, so the worker env
    and ai-toolkit's conflicting deps stay in separate conda envs (cf#29 D1). Streams the child's
    output to this worker's stdout (so RunPod captures the training log) and forwards coarse
    progress lines to `progress_cb` when one is wired. Injectable (`runner=` on `train_slot_wan`)
    so everything else in this module tests without a GPU.

    Before launch: rewrite ai-toolkit's hardcoded UMT5/VAE hub ids to local snapshot paths so
    HF_HUB_OFFLINE=1 does not crash on model_info() (cf#29 D2c). On non-zero exit, the exception
    carries the last N stdout lines (RunPod's status stream is often empty).
    """
    run_py = Path(cwd) / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(
            f"ai-toolkit run.py not found at {run_py} (set {AITOOLKIT_DIR_ENV} to the checkout; "
            "the training image must provide ai-toolkit)")
    patch_aitoolkit_hub_ids_for_offline(cwd)
    proc = subprocess.Popen(
        [aitoolkit_python(), "run.py", str(config_path)],
        cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    tail: deque[str] = deque(maxlen=_AITOOLKIT_TAIL_LINES)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail.append(line.rstrip("\n"))
        if progress_cb is not None:
            try:
                progress_cb(line.rstrip("\n"))
            except Exception:
                pass  # best-effort: a progress hook must never break training
    rc = proc.wait()
    if rc != 0:
        excerpt = "\n".join(tail) if tail else "(no stdout captured)"
        raise RuntimeError(
            f"ai-toolkit run.py exited {rc} for config {config_path}\n"
            f"--- last {len(tail)} lines ---\n{excerpt}"
        )


def train_slot_wan(
    char: Character,
    out_dir: Path,
    *,
    config: WanLoraTrainConfig | None = None,
    progress_cb=None,
    runner=None,
) -> TrainedWanLora:
    """Train one character's Wan 2.2 A14B LoRA (both experts) from its reference images via
    ai-toolkit, and return where the two `.safetensors` landed.

    `char.ref_paths` is the bundle's reference set (already resolved by `Bundle.extract`), the
    SAME plumbing the SDXL trainer uses. Writes the dataset, the ai-toolkit config, and the two
    experts under `out_dir`. `runner(config_path, cwd=, progress_cb=)` is the subprocess seam
    (default: the real ai-toolkit run); inject it to test the wiring without a GPU.
    """
    import yaml  # local: keep the module import CPU-only and dependency-light at import time

    cfg = config or WanLoraTrainConfig()
    # Hub id -> local snapshot so model.name_or_path never trips OfflineModeIsEnabled (cf#29 D2c).
    # Injected runners / unit tests that pass an absolute path are unchanged (is_dir short-circuit).
    cfg = replace(cfg, base_repo=resolve_local_hf_snapshot(cfg.base_repo))
    run = runner or _run_aitoolkit
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = _run_name(char.slot)
    dataset_dir = out_dir / "dataset"
    training_folder = out_dir / "output"
    ref_count = prepare_dataset(char, dataset_dir, cfg)

    config_dict = build_aitoolkit_config(name, dataset_dir, training_folder, cfg)
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")

    run(config_path, cwd=aitoolkit_dir(), progress_cb=progress_cb)
    high_path, low_path = harvest_experts(training_folder / name, name)
    return TrainedWanLora(
        slot=char.slot,
        high_path=high_path,
        low_path=low_path,
        trigger=char.name or char.slot,
        steps=cfg.steps,
        rank=cfg.rank,
        ref_count=ref_count,
        base_repo=cfg.base_repo,
        meta={"caption": caption_for(char, cfg.caption_template), "low_vram": cfg.low_vram},
    )
