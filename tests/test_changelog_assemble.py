"""Drives scripts/changelog-assemble.py against synthetic fixtures (backend#471).

Output is asserted BYTE-FOR-BYTE against a written expectation, including the migration case where
a fragment and a hand-edited `## Unreleased` body are BOTH populated at once (the entry guard
accepts either form, so a release can genuinely see both). The version is data, never inferred
from CHANGELOG.md's existing structure.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "changelog-assemble.py"


def _load():
    spec = importlib.util.spec_from_file_location("changelog_assemble", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load()

BASE = (
    "# Changelog\n\n"
    "All notable changes.\n\n"
    "## Unreleased\n\n"
    "## [1.0.0] -- 2026-01-01\n\n"
    "old release\n"
)


def test_fragment_only_assembly_is_byte_correct():
    ok, out = ca.assemble(
        BASE, ["100-a.md", "200-b.md"], ["**Fix: a.**\n\nbody a.", "**Fix: b.**\n\nbody b."],
        "1.1.0", "2026-08-14",
    )
    assert ok
    assert out == (
        "# Changelog\n\n"
        "All notable changes.\n\n"
        "## Unreleased\n\n"
        "## [1.1.0] -- 2026-08-14\n\n"
        "**Fix: a.**\n\nbody a.\n\n**Fix: b.**\n\nbody b.\n\n"
        "## [1.0.0] -- 2026-01-01\n\n"
        "old release\n"
    )


LEGACY = (
    "# Changelog\n\n"
    "## Unreleased\n\n"
    "**Fix: direct edit, no fragment.**\n\nprose.\n\n"
    "## [1.0.0] -- 2026-01-01\n\nold\n"
)


def test_legacy_unreleased_only_assembly_is_byte_correct():
    ok, out = ca.assemble(LEGACY, [], [], "1.1.0", "2026-08-14")
    assert ok
    assert out == (
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "## [1.1.0] -- 2026-08-14\n\n"
        "**Fix: direct edit, no fragment.**\n\nprose.\n\n"
        "## [1.0.0] -- 2026-01-01\n\nold\n"
    )


def test_migration_case_legacy_body_first_then_fragments():
    ok, out = ca.assemble(LEGACY, ["050-e.md"], ["**Fix: fragment entry.**\n\nfragment prose."], "1.1.0", "2026-08-14")
    assert ok
    assert out == (
        "# Changelog\n\n"
        "## Unreleased\n\n"
        "## [1.1.0] -- 2026-08-14\n\n"
        "**Fix: direct edit, no fragment.**\n\nprose.\n\n"
        "**Fix: fragment entry.**\n\nfragment prose.\n\n"
        "## [1.0.0] -- 2026-01-01\n\nold\n"
    )


def test_empty_release_still_writes_a_bare_heading():
    ok, out = ca.assemble("# Changelog\n\n## Unreleased\n\n## [1.0.0] -- 2026-01-01\n\nold\n", [], [], "1.1.0", "2026-08-14")
    assert ok and "## [1.1.0] -- 2026-08-14" in out and "## [1.0.0] -- 2026-01-01" in out


def test_version_is_the_explicit_argument_not_the_topmost_heading():
    base = "# Changelog\n\n## Unreleased\n\n## [1.26.0] -- 2026-08-14\n\nexisting top\n"
    ok, out = ca.assemble(base, ["1-a.md"], ["**Fix: new.**"], "9.9.9", "2099-01-01")
    assert ok
    assert out.split("## Unreleased\n\n", 1)[1].startswith("## [9.9.9] -- 2099-01-01")
    assert "## [1.26.0] -- 2026-08-14\n\nexisting top" in out


def test_version_spellings_normalize_to_one_bracketed_heading():
    for spelling in ("1.1.0", "v1.1.0", "backend-v1.1.0"):
        ok, out = ca.assemble(BASE, [], [], spelling, "2026-08-14")
        assert ok and "## [1.1.0] -- 2026-08-14" in out


def test_refuses_a_version_that_already_has_a_heading_in_either_spelling():
    released = "# Changelog\n\n## Unreleased\n\n## [1.1.0] -- 2026-08-01\n\nthere\n\n## backend-v0.1.17\n\nold\n"
    for spelling in ("1.1.0", "v1.1.0", "backend-v1.1.0"):
        ok, msg = ca.assemble(released, ["100-a.md"], ["new"], spelling, "2026-08-14")
        assert not ok and "already appears" in msg
    ok, msg = ca.assemble(released, [], [], "0.1.17", "2026-08-14")
    assert not ok and "already appears" in msg


def test_a_prefix_of_an_existing_version_is_not_a_duplicate():
    """CONTROL: 1.1 must not be refused because 1.1.0 exists, nor 1.0.1 because 1.0.12 does."""
    base = "# Changelog\n\n## Unreleased\n\n## [1.0.12] -- 2026-07-31\n\nx\n"
    ok, _ = ca.assemble(base, [], [], "1.0.1", "2026-08-14")
    assert ok


def test_refuses_without_an_unreleased_heading():
    ok, msg = ca.assemble("# Changelog\n\n## [1.0.0] -- 2026-01-01\n\nold\n", [], [], "1.1.0", "2026-08-14")
    assert not ok and "Unreleased" in msg


def test_main_reads_sorted_deletes_consumed_and_is_idempotent_safe(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(BASE)
    d = tmp_path / "changelog.d"
    d.mkdir()
    (d / ".gitkeep").write_text("")
    (d / "README.md").write_text("naming rules, not an entry")
    (d / "200-later.md").write_text("**Fix: later.**\n\nlater body.")
    (d / "050-earlier.md").write_text("**Fix: earlier.**\n\nearlier body.")

    run = lambda: subprocess.run(
        [sys.executable, str(SCRIPT), "backend-v1.1.0", "2026-08-14"], cwd=tmp_path, capture_output=True, text=True
    )
    proc = run()
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in d.iterdir()) == [".gitkeep", "README.md"]
    written = (tmp_path / "CHANGELOG.md").read_text()
    assert written.index("earlier body") < written.index("later body")
    assert "not an entry" not in written

    again = run()
    assert again.returncode == 1 and "already appears" in again.stderr
