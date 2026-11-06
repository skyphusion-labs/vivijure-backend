**Docs: the backend endpoint runs 1.1.8, with its rollback.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.8 thin image built from main `957e4ce` (#578, `plate_ip_adapter_scale`, default 0, which is OFF);
- the 60 s wait after a repin;
- the fresh-draw check;
- the rollback to 1.1.7.
