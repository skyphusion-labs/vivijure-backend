# AGENTS.md

## Cursor Cloud specific instructions

Standard test config is in `pytest.ini` (`pythonpath=src`, `testpaths=tests`) and
`CLAUDE.md`. Cloud notes:

- Use a per-repo venv (`.venv` is git-ignored; `python3.12-venv` is installed by the
  environment update script, which also creates the venv and installs dev deps):
  `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`.
- Run the CPU test suite: `.venv/bin/python -m pytest`.

Verified in this environment: `pytest` -> 651 passed, 3 skipped. The heavy GPU/ML
deps in `requirements.txt` (Torch etc.) import only on the card and are not needed
for this CPU suite.
