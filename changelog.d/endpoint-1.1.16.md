**Docs: the backend endpoint runs 1.1.16, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.16 code-only image built from main `6edad52` (#597) on the existing `stills-t1-m1` base, which reused all three weight layers;
- that two-person scene-lock stills redraw once;
- the rollback to 1.1.15.
