### feat(bake): a stills weights base, so a cold keyframe worker pulls nothing

A cold keyframe worker pulled the stills set, 26.22 GB, from R2 inside the handler: 104 to 2618 s on the GPU clock per worker, about 20 minutes per job when 15 pulled at once, and again after every release. RunPod pulls image layers before a worker starts, and does not bill that. So the stills set and the finish legs now go into a weights base image below the code layer (`deploy/stills-weights.Dockerfile`, runbook `docs/stills-weights-base.md`). A code-only release changes only the code layer.

`python -m vivijure_backend.harness.models_mirror --stage-stills` fills the tree with the worker's own pull and checks it; `--check-stills` checks a built image. The base carries no `.vj-baked`, so any action other than a stills job or a finish step still pulls what it needs. The warm stills path now checks the files behind its sentinel before it skips, and pulls again if one is missing.

Nothing changes on the endpoint until a base is built and a code image on it is pinned.
