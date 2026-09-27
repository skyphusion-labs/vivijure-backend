"""Drives scripts/changelog-entry-required.py against a SYNTHETIC repository (backend#471).

THE FIXTURE IS THE POINT. The guard diffs THREE-dot from the merge base; two-dot against a base
that has MOVED lets a src-only PR pass vacuously because a DIFFERENT merged PR touched
CHANGELOG.md. The fixture reproduces that false pass (CONTROL) and then proves three-dot refuses
it, on the same repository. A guard whose failure cannot be watched is a guard nobody has verified.
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "changelog-entry-required.py"
    spec = importlib.util.spec_from_file_location("changelog_entry_required", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cer = _load()
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "changelog-entry-required.py"


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)


def _build_repo(root):
    """main advances with SOMEBODY ELSE'S PR touching CHANGELOG.md; our branch touches src/ only."""
    root = Path(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n")
    (root / "src").mkdir()
    (root / "src" / "a.py").write_text("a = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")

    _git(root, "checkout", "-qb", "pr")
    (root / "src" / "a.py").write_text("a = 2\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "pr: src only, no changelog entry")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()

    _git(root, "checkout", "-q", "main")
    (root / "CHANGELOG.md").write_text("# Changelog\n\n## Unreleased\n\nsomeone else\n")
    (root / "src" / "b.py").write_text("b = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "other PR: touches CHANGELOG.md")
    moved_base = _git(root, "rev-parse", "HEAD").stdout.strip()
    return head, moved_base


def test_fixture_reproduces_the_two_dot_false_pass_and_three_dot_refuses_it(tmp_path):
    head, moved_base = _build_repo(tmp_path)

    # CONTROL: two-dot against the MOVED base passes vacuously, or the fix below proves nothing.
    two_dot = cer.changed_files(str(tmp_path), moved_base, head, two_dot=True)
    assert "CHANGELOG.md" in two_dot and "src/a.py" in two_dot
    assert cer.verdict(two_dot)[0]

    # THE FIX: three-dot sees only what this PR did, so the missing entry is caught.
    three_dot = cer.changed_files(str(tmp_path), moved_base, head)
    ok, msg = cer.verdict(three_dot)
    assert not ok
    assert "CHANGELOG.md" not in three_dot and "src/a.py" in three_dot
    assert "neither CHANGELOG.md nor a changelog.d/ fragment" in msg


def test_a_pr_with_its_own_fragment_passes_end_to_end(tmp_path):
    head, moved_base = _build_repo(tmp_path)
    _git(tmp_path, "checkout", "-q", "pr")
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d" / "1-x.md").write_text("**Fix: x.**\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "pr: add fragment")
    head2 = _git(tmp_path, "rev-parse", "HEAD").stdout.strip()
    assert cer.verdict(cer.changed_files(str(tmp_path), moved_base, head2))[0]


def test_main_exit_codes_red_then_green(tmp_path):
    """The workflow calls main(); pin the exit code the job actually turns on."""
    head, moved_base = _build_repo(tmp_path)
    red = subprocess.run(
        ["python3", str(SCRIPT), str(tmp_path), moved_base, head, "[]"], capture_output=True, text=True
    )
    assert red.returncode == 1 and "::error::" in red.stdout
    labelled = subprocess.run(
        ["python3", str(SCRIPT), str(tmp_path), moved_base, head, '["no-changelog"]'],
        capture_output=True, text=True,
    )
    assert labelled.returncode == 0


def test_docs_tests_and_workflow_only_prs_need_no_entry():
    assert cer.verdict(["docs/x.md", "tests/y.py", ".github/workflows/z.yml", "deploy/Dockerfile"])[0]


def test_no_changelog_label_is_the_loud_escape_hatch_and_control_without_it_refuses():
    assert cer.verdict(["src/vivijure_backend/a.py"], ["no-changelog"])[0]
    assert not cer.verdict(["src/vivijure_backend/a.py"])[0]


def test_a_fragment_alone_and_a_changelog_edit_alone_are_each_accepted():
    assert cer.verdict(["src/vivijure_backend/a.py", "changelog.d/471-x.md"])[0]
    assert cer.verdict(["src/vivijure_backend/a.py", "CHANGELOG.md"])[0]


@pytest.mark.parametrize("stray", ["changelog.d/.gitkeep", "changelog.d/README.md", "changelog.d/x.txt"])
def test_touching_changelog_d_without_adding_a_fragment_does_not_satisfy_the_guard(stray):
    """CONTROL: otherwise the check degenerates to "did you touch this directory"."""
    assert not cer.verdict(["src/vivijure_backend/a.py", stray])[0]
