**Docs: the backend endpoint runs 1.1.17, with its rollback; the disk history.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.17 code-only image built from main `eed4b57` (#601 per-slot ip scale, #602 disk guards) on the existing `stills-t1-m1` base;
- the rollback to 1.1.16;
- the container disk raise of 2026-11-06 from the release history: 80 to 200 GB at 15:01:44Z (v41), then 200 to 300 GB at 15:20:17Z (v42), kept at 300 through the repin.
