**Feat: thin image reads stills from a global volume (#530).**

The keyframe worker skips the bake and the cold pull.
A mounted `/runpod-volume` is the default weight root.
`VJ_VOLUME_KIND=global` refuses first-worker lock.
`--preload --stills --flatten` stages the base set as real files.
`deploy/Dockerfile.thin` FROMs the toolchain stage only.
This PR does not create, fill, or attach a volume.
