**Fix: scene-lock Canny loads from a local snapshot and moves to CUDA (#528).**

`bake_hf_configs` writes a Canny config snapshot and can point `refs/main` at it.
`from_pretrained(..., local_files_only=True)` then raises `OSError` even when the
weight snapshot is on disk, or when only the stub is present.
The loader now picks a snapshot that holds tensors.
It loads from that path, not the hub id.
It then moves the ControlNet to CUDA.
If no tensors exist, it pulls `xinsir/controlnet-canny-sdxl-1.0` from the model
mirror, then from Hub.
`scene_lock` stays on. Off remains a debug hatch.
