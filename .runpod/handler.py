"""RunPod Hub entrypoint (real file).

Root ``handler.py`` is a git symlink to ``src/vivijure_backend/worker.py``. Hub's listing
probe does not always treat symlinks as a present ``handler.py``, so this real file under
``.runpod/`` (Hub precedence) re-exports the same serverless start path.
"""
from __future__ import annotations

from vivijure_backend.worker import handler, main

__all__ = ["handler", "main"]

if __name__ == "__main__":
    main()
