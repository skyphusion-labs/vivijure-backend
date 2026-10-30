**Feat: a stills job pulls only what it loads.**

`preview`, `regen_shot` and `train_lora` now pull 25.75 GB from R2 instead of 52.29 GB, plus `antelopev2`.
The list is `harness/stills_manifest.json`, by exact blob at the pinned revisions.
The volume hydrate reads the same list.
A missing file fails the job at the mirror, naming the file.
Other actions keep the full mirror. `VJ_STILLS_MIRROR=0` restores it for all.
