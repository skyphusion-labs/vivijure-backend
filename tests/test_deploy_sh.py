"""deploy.sh first-time create must read the created template/endpoint ids from the create response.

The real deploy.sh runs against a stub `curl` on PATH, so no RunPod call is made. The stub answers
GET with an empty list (nothing exists yet) and POST /templates and POST /endpoints with an id."""
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_CURL = r'''#!/usr/bin/env bash
# stub curl: emit "<body>\n<http code>" like `curl -w '\n%{http_code}'`; log "<METHOD> <url>".
method=GET; url=""
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift 2 ;;
    -H|-w|--data) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
cat >/dev/null 2>&1 || true
echo "$method $url" >> "$STUB_LOG"
case "$method $url" in
  "GET "*) printf '[]\n200' ;;
  "POST "*/templates) printf '{"id":"tpl-created"}\n200' ;;
  "POST "*/endpoints) printf '{"id":"ep-created"}\n200' ;;
  *) printf '{}\n200' ;;
esac
'''

DEPLOY_ENV = """RUNPOD_API_KEY=stub
R2_ACCESS_KEY_ID=stub
R2_SECRET_ACCESS_KEY=stub
R2_ENDPOINT=https://stub.invalid
"""


def test_first_time_create_reads_ids_from_the_create_responses(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "deploy.sh", work / "deploy.sh")
    (work / "deploy.env").write_text(DEPLOY_ENV)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.log"
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "STUB_LOG": str(log),
           "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(work / "deploy.sh")], cwd=work, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "RUNPOD_ENDPOINT_ID=ep-created" in r.stdout
    calls = log.read_text().splitlines()
    assert any(c.startswith("POST ") and c.endswith("/templates") for c in calls)
    assert any(c.startswith("POST ") and c.endswith("/endpoints") for c in calls)
