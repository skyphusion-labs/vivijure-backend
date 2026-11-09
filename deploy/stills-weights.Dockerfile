# Stills weights base (#564): the toolchain plus the stills set and the finish legs, as weight layers.
#
# A cold keyframe worker pulled about 26 GB from R2 inside the handler, on the GPU clock, after every
# release. RunPod pulls image layers before a worker starts, and that is not billed. So the stills set
# lives here, BELOW the code layer: deploy/Dockerfile.thin builds on this image through TOOLCHAIN_REF,
# a code-only release changes only the small app layer, and a host that held the previous version
# reuses every weight layer.
#
# The build context is the BIN directory, not the repo. The bins come from a tree that the worker's own
# pull staged and checked (`models_mirror --stage-stills`), packed by deploy/bake_layers.py under the
# GHCR layer limit. The full procedure, with every gate, is docs/stills-weights-base.md.
#
# No .vj-baked, on purpose. That marker ends every pull for every action, and this image holds only the
# stills set and the finish legs. The sentinels the stage wrote are inside the bins: a stills job finds
# `.vj-stills-complete` (checked against the files at runtime) and pulls nothing, and any other action
# still pulls what it needs. tests/test_stills_weights_base.py holds this file to that.
#
# Pin TOOLCHAIN_REF by tag and digest. There is no default: a base built on an unnamed toolchain is not
# reproducible.
ARG TOOLCHAIN_REF
FROM ${TOOLCHAIN_REF}

COPY bin-00/ /opt/models/
COPY bin-01/ /opt/models/
COPY bin-02/ /opt/models/
COPY bin-03/ /opt/models/
