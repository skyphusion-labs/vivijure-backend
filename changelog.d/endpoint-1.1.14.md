**Docs: the backend endpoint runs 1.1.14, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.14 thin image built from main `a2f3b18` (#589, `plate_count_retries` default 3 for two-person-plus shots);
- that the first re-render of each stored two-person still redraws it once;
- the rollback to 1.1.13.
