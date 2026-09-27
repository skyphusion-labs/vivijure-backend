#!/usr/bin/env python3
"""
Assembles changelog.d/ fragments (plus any legacy `## Unreleased` body) into a new released
section at RELEASE time (backend#471). Ported from vivijure-cf's scripts/changelog-assemble.py
(cf#539, itself from vivijure-control-plane cp#358) so the estate has one convention.

WHAT THIS SCRIPT DOES, in order:
  1. reads every changelog.d/*.md fragment, sorted by FILENAME (deterministic; README.md and
     `.gitkeep` excluded)
  2. ALSO reads whatever is still sitting under `## Unreleased` -- the migration window means the
     guard accepts EITHER a fragment OR a direct edit, so both sources can be populated at once
  3. emits a `## [X.Y.Z] -- YYYY-MM-DD` section, the heading format this CHANGELOG already uses,
     placed directly below the (now empty) `## Unreleased` heading
  4. deletes the consumed fragment files and leaves `## Unreleased` empty

<version> IS AN EXPLICIT ARGUMENT, ALWAYS THE VERSION BEING CUT -- never derived from "the topmost
heading" in CHANGELOG.md, so an assemble run cannot silently target the wrong release. Accepts
`1.0.13`, `v1.0.13` or `backend-v1.0.13` (the tag spelling); the heading always spells `[1.0.13]`.

ORDERING: the legacy `## Unreleased` body (if any) comes FIRST, in whatever order it was already in,
because it predates fragments existing at all; the fragments follow, in filename-sorted order,
which is issue-number order by the naming convention in changelog.d/README.md.

IDEMPOTENT-SAFE. Refuses loudly, and writes nothing, when the version already appears as a heading
anywhere in CHANGELOG.md, released or not.

Run from the repo root: python3 scripts/changelog-assemble.py <version> <date>
"""
import pathlib
import sys

FRAGMENT_DIR = "changelog.d"
NOT_FRAGMENTS = (".gitkeep", "README.md")


def normalize_version(v):
    """'backend-v1.0.13', 'v1.0.13' and '1.0.13' all become '1.0.13'."""
    for prefix in ("backend-v", "v"):
        if v.startswith(prefix):
            return v[len(prefix):]
    return v


def section_bounds(lines, heading_text):
    """(start, end) line indices for the FIRST `## <heading_text>` section, end exclusive at the
    next `## ` heading or EOF. None if the heading is not present at all."""
    start = None
    for i, line in enumerate(lines):
        if line == "## " + heading_text:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    return start, end


def version_heading_present(lines, version):
    """Is there already a heading for <version>, in either spelling this file has used
    ('## [1.0.12] -- 2026-07-31' or '## backend-v0.1.17')? The bracket closes the bracketed form,
    and the backend-v form is boundary-checked, so '1.0.1' cannot false-positive against '1.0.12'."""
    bracketed = "## [" + version + "]"
    tagged = "## backend-v" + version
    for line in lines:
        if line == bracketed or line.startswith(bracketed + " "):
            return True
        if line == tagged or line.startswith(tagged + " "):
            return True
    return False


def read_fragments(root):
    """Fragment BODIES, sorted by filename, README/.gitkeep and non-.md files excluded. Each
    fragment's content is exactly the block an author would have added under `## Unreleased` --
    no new syntax, so this is a plain read-and-strip, never a parse."""
    d = root / FRAGMENT_DIR
    if not d.is_dir():
        return [], []
    names = sorted(
        p.name for p in d.iterdir()
        if p.is_file() and p.name.endswith(".md") and p.name not in NOT_FRAGMENTS
    )
    bodies = [(d / n).read_text().strip() for n in names]
    return names, bodies


def assemble(text, fragments_names, fragments_bodies, version, date):
    """Pure: returns (ok, new_text_or_message). Takes CHANGELOG.md's current text and the fragment
    bodies already read from disk, and returns the fully-updated text, so it is testable without
    touching a filesystem."""
    lines = text.split("\n")
    version = normalize_version(version)

    if version_heading_present(lines, version):
        return False, (
            "refusing: a heading for '" + version + "' already appears in CHANGELOG.md. "
            "Re-running this script for a version already promoted would duplicate the heading. "
            "Nothing was written."
        )

    bounds = section_bounds(lines, "Unreleased")
    if bounds is None:
        return False, (
            "refusing: CHANGELOG.md has no '## Unreleased' heading to promote from. Add one before "
            "running this script."
        )
    start, end = bounds
    legacy_body = "\n".join(lines[start + 1 : end]).strip()

    parts = []
    if legacy_body:
        parts.append(legacy_body)
    parts.extend(fragments_bodies)
    content = "\n\n".join(parts).strip()

    new_section = ["## [" + version + "] -- " + date, ""]
    if content:
        new_section.append(content)
    new_section.append("")  # exactly one blank line before whatever heading follows

    new_lines = (
        lines[: start + 1]  # up to and including "## Unreleased"
        + [""]              # Unreleased left empty
        + new_section
        + lines[end:]        # the rest of the file, untouched
    )
    return True, "\n".join(new_lines)


def main(argv):
    if len(argv) != 3:
        print("usage: changelog-assemble.py <version> <date>", file=sys.stderr)
        return 2
    version, date = argv[1], argv[2]
    root = pathlib.Path(".")
    changelog_path = root / "CHANGELOG.md"
    text = changelog_path.read_text()

    names, bodies = read_fragments(root)
    ok, result = assemble(text, names, bodies, version, date)
    if not ok:
        print("changelog-assemble: " + result, file=sys.stderr)
        return 1

    changelog_path.write_text(result)
    for name in names:
        (root / FRAGMENT_DIR / name).unlink()

    summary = "consumed " + str(len(names)) + " fragment(s)"
    if names:
        summary += ": " + ", ".join(names)
    print("changelog-assemble: wrote '## [" + normalize_version(version) + "] -- " + date + "', " + summary)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
