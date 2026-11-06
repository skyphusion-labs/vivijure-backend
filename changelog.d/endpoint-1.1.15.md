**Docs: the backend endpoint runs 1.1.15 on the stills weights base; first-bake numbers.**

`docs/runpod-endpoint-config.md` records:
- the 1.1.15 pin, built from main `82b2324` on `stills-t1-m1@6d0e19bd`, with its in-image checks;
- the rollback to 1.1.14.

`docs/stills-weights-base.md`:
- records the first bake (stage size and time, the gates, the bins, the three layer sizes, the push time);
- corrects step 1: stage with code that has `--stage-stills`, and keep the shell that reads the credential in the foreground.
