#!/usr/bin/env python3
"""Build-time offline smoke: prove Wan train loads never need HuggingFace network.

Conrad ruling (cf#29): bake the assets; do not ship a "fix" that still depends on HF at
runtime. This smoke runs in the train image AFTER bake_aitoolkit_offline_paths.py:

  1. Stable dirs + BAKE_OFFLINE.json exist
  2. ai-toolkit source no longer carries the UMT5/VAE hub IDs
  3. Hub-id from_pretrained under HF_HUB_OFFLINE raises OfflineModeIsEnabled (the D2c failure
     class) -- proves hub IDs are still toxic offline even when the bake is complete
  4. Local-path config load with a network-forbid hook succeeds -- proves baked dirs are enough

Does NOT load the 14B DiT into RAM (build runners cannot). Config / model_index reads only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

STABLE_ROOT = Path(os.environ.get("VIVIJURE_AITOOLKIT_WEIGHTS", "/opt/models/aitoolkit"))
AITOOLKIT_DIR = Path(os.environ.get("VIVIJURE_AITOOLKIT_DIR", "/opt/ai-toolkit"))
WAN_BASE_HUB = "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16"
HUB_IDS = (
    "ai-toolkit/umt5_xxl_encoder",
    "ai-toolkit/wan2.1-vae",
)
PATCH_FILES = (
    "toolkit/models/wan21/wan21.py",
    "extensions_built_in/diffusion_models/wan22/wan22_14b_model.py",
)


class _NetworkForbidden(RuntimeError):
    pass


def _forbid_hf_http():
    """Raise if huggingface_hub attempts any HTTP (proves local paths need zero network)."""
    import huggingface_hub.utils._http as http_mod

    class _BlockedSession:
        def request(self, *args, **kwargs):
            raise _NetworkForbidden(f"HF HTTP forbidden in offline smoke: args={args[:1]!r}")

        def get(self, *args, **kwargs):
            return self.request("GET", *args, **kwargs)

        def head(self, *args, **kwargs):
            return self.request("HEAD", *args, **kwargs)

        def post(self, *args, **kwargs):
            return self.request("POST", *args, **kwargs)

    http_mod.get_session = lambda *a, **k: _BlockedSession()  # type: ignore[assignment]


def check_marker() -> dict:
    marker = STABLE_ROOT / "BAKE_OFFLINE.json"
    if not marker.is_file():
        raise FileNotFoundError(f"missing {marker}; bake_aitoolkit_offline_paths.py did not run")
    data = json.loads(marker.read_text(encoding="utf-8"))
    for key in ("wan-base", "umt5", "vae"):
        p = Path(data["stable"][key])
        if not p.is_dir():
            raise FileNotFoundError(f"stable dir missing: {key} -> {p}")
        # non-hollow: at least one config / model_index
        if not any(p.glob("*.json")) and not (p / "config.json").exists():
            # wan-base has model_index.json; umt5/vae have config.json
            found = list(p.rglob("config.json"))[:1] or list(p.rglob("model_index.json"))[:1]
            if not found:
                raise FileNotFoundError(f"stable dir looks hollow (no config): {p}")
    print("marker OK:", marker)
    return data


def check_aitoolkit_patched() -> None:
    for rel in PATCH_FILES:
        text = (AITOOLKIT_DIR / rel).read_text(encoding="utf-8")
        for hub_id in HUB_IDS:
            if f'"{hub_id}"' in text:
                raise RuntimeError(f"{rel} still contains hub id {hub_id!r}")
    print("ai-toolkit hub-id strings scrubbed OK")


def check_hub_id_still_toxic_offline() -> None:
    """Reproduce the D2c failure class: hub id + HF_HUB_OFFLINE -> OfflineModeIsEnabled."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from diffusers import WanTransformer3DModel
    from huggingface_hub.errors import OfflineModeIsEnabled

    try:
        WanTransformer3DModel.from_pretrained(WAN_BASE_HUB, subfolder="transformer")
    except OfflineModeIsEnabled as e:
        print("hub-id toxic offline OK:", type(e).__name__)
        return
    except Exception as e:
        # Some hub versions wrap OfflineModeIsEnabled; accept any offline/network refusal
        name = type(e).__name__
        if "Offline" in name or "offline" in str(e).lower():
            print("hub-id toxic offline OK (wrapped):", name)
            return
        raise RuntimeError(f"expected OfflineModeIsEnabled for hub id, got {name}: {e}") from e
    raise RuntimeError("hub-id from_pretrained unexpectedly succeeded under HF_HUB_OFFLINE=1")


def check_local_path_config_no_network(data: dict) -> None:
    """Local snapshot must load config with HF HTTP fully forbidden."""
    _forbid_hf_http()
    from diffusers import WanTransformer3DModel

    base = data["stable"]["wan-base"]
    # config-only path: from_pretrained still reads local json/shards metadata; we only need it
    # not to phone home. Loading full weights is too heavy for the build runner -- use
    # local_files_only + a tiny probe via the config file existence + Diffusers config load.
    cfg_path = Path(base) / "transformer" / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing transformer config under baked wan-base: {cfg_path}")
    # Diffusers load_config is enough to prove local resolution without allocating the DiT.
    _ = WanTransformer3DModel.load_config(str(Path(base) / "transformer"))
    print("local wan-base transformer config load OK (network forbidden)")

    for key in ("umt5", "vae"):
        root = Path(data["stable"][key])
        cfg = root / "config.json"
        if not cfg.is_file():
            # some snapshots nest config
            found = list(root.rglob("config.json"))
            if not found:
                raise FileNotFoundError(f"no config.json under {root}")
            cfg = found[0]
        json.loads(cfg.read_text(encoding="utf-8"))
        print(f"local {key} config OK:", cfg)


def main() -> int:
    data = check_marker()
    check_aitoolkit_patched()
    check_hub_id_still_toxic_offline()
    check_local_path_config_no_network(data)
    wan_env = os.environ.get("VIVIJURE_WAN_BASE_PATH", "")
    expected = data["stable"]["wan-base"]
    if wan_env and wan_env != expected:
        raise RuntimeError(f"VIVIJURE_WAN_BASE_PATH={wan_env!r} != baked {expected!r}")
    print("offline smoke OK (bake-it, no HF at train time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
