**Docs: the backend endpoint runs 1.1.13, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.13 thin image built from main `a3d48a6` (#587, `plate_count_retries`, default 0, which is OFF);
- that its face count uses the antelopev2 analyser already mirrored for stills jobs;
- the rollback to 1.1.12.
