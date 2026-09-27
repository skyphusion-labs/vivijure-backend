#!/usr/bin/env python3
"""
A PR touching src/ needs a changelog entry OF ITS OWN: a `changelog.d/` fragment, a direct
CHANGELOG.md edit, or the `no-changelog` label as a recorded, deliberate skip (backend#471).

Ported from vivijure-cf's scripts/changelog-entry-required.py (cf#510/cf#539, itself ported from
vivijure-control-plane cp#147), so the estate has one convention, not several dialects.

WHY FRAGMENTS. Every entry used to land under the one shared `## Unreleased` heading, so the moment
ANY PR merged that hunk moved and re-conflicted every other open PR touching it. Several fixes here
(R2 `exists()` error handling, `finish_clip` deadline, stored LoRA staging, `face_restore` boolean,
`face_restored` accuracy) shipped with no note for exactly that reason. One fragment file per PR
under `changelog.d/`, assembled into CHANGELOG.md at release by scripts/changelog-assemble.py,
means two PRs never touch the same file.

WHY THREE-DOT. `changed_files()` diffs `BASE...HEAD`, not `BASE..HEAD`: three-dot compares from the
MERGE BASE, so a moving `main` cannot lend this PR somebody else's changelog edit. Two-dot lets a
PR that touched src/ with no entry pass whenever a DIFFERENT merged PR touched CHANGELOG.md in
between. tests/test_changelog_entry_required.py reproduces that false pass as a fixture and proves
three-dot refuses it.

SCOPE, deliberately narrow: `src/` only, the code the image COPYs (`src/vivijure_backend`, plus the
`handler.py` symlink into it). Tests, docs, deploy/, scripts/ and workflow-only changes do not need
an entry; a check that fires on everything gets bypassed on reflex, which is worse than no check.

EITHER FORM ACCEPTED during the migration window: a fragment or a direct CHANGELOG.md edit, so PRs
already open when the convention landed are not refused. Tightening to fragment-only once the queue
drains is a deliberate follow-up, not this change.

Logic lives here rather than in the workflow so a test can drive it against a synthetic repository.
A guard whose failure cannot be watched is a guard nobody has verified.
"""
import json
import subprocess
import sys

GATED_PREFIXES = ("src/",)


def changed_files(root, base, head, two_dot=False):
    """Files this PR changed. THREE-dot by default: from the merge base, so a moving main cannot
    lend this PR somebody else's edits. two_dot exists ONLY so the test can reproduce the old bug."""
    spec = [base, head] if two_dot else [base + "..." + head]
    out = subprocess.run(
        ["git", "-C", root, "diff", "--name-only", *spec],
        capture_output=True, text=True, check=True,
    )
    return [l for l in out.stdout.split("\n") if l.strip()]


def has_fragment(files):
    """A changelog.d/ touch that is an actual fragment, not the tracked .gitkeep placeholder or
    the README -- otherwise the guard degenerates to "did you touch this directory", which an
    unrelated touch would satisfy without adding any entry at all."""
    return any(
        f.startswith("changelog.d/") and f.endswith(".md") and f != "changelog.d/README.md"
        for f in files
    )


def verdict(files, labels=()):
    """(ok, message). Pure, so the decision is testable without a repository at all."""
    if any(l == "no-changelog" for l in labels):
        return True, "no-changelog label present: deliberate skip, recorded on the PR."
    touches_code = any(f.startswith(p) for f in files for p in GATED_PREFIXES)
    if not touches_code:
        return True, "no src/ changes."
    if "CHANGELOG.md" in files:
        return True, "src/ changed and CHANGELOG.md was updated."
    if has_fragment(files):
        return True, "src/ changed and a changelog.d/ fragment was added."
    return False, (
        "This PR touches src/ but neither CHANGELOG.md nor a changelog.d/ fragment changed. "
        "Preferred: add a fragment file under changelog.d/ (see changelog.d/README.md) named "
        "<issue>-<slug>.md, containing the block that would have gone under `## Unreleased`. "
        "CHANGELOG.md itself is still accepted during the migration window. Or apply the "
        "`no-changelog` label if this is a deliberate skip."
    )


def main(argv):
    root, base, head = argv[1], argv[2], argv[3]
    labels = []
    if len(argv) > 4 and argv[4].strip():
        try:
            labels = json.loads(argv[4])
        except json.JSONDecodeError:
            labels = []
    files = changed_files(root, base, head)
    print("Changed files (three-dot, from the merge base):")
    for f in files:
        print("  " + f)
    ok, message = verdict(files, labels)
    if not ok:
        print("::error::" + message)
        return 1
    print("OK: " + message)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
