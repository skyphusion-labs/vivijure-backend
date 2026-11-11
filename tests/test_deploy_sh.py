"""deploy.sh on RunPod REST v2 (fleet-chezmoi#2085).

The real deploy.sh runs against a stub `curl` on PATH, so no RunPod call is made. The stub logs
"<METHOD> <url>" and the request body, and answers by STUB_MODE. On v2 an endpoint carries its own
image, env and disk, so deploy.sh makes no template: it finds the endpoint by name across every list
page, then PATCHes it or POSTs a new one at /v2/serverless."""
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

FAKE_CURL = r'''#!/usr/bin/env bash
# stub curl: emit "<body>\n<http code>" like `curl -w '\n%{http_code}'`; log "<METHOD> <url>" and the body.
method=GET; url=""; data=0
while [ $# -gt 0 ]; do
  case "$1" in
    -X) method="$2"; shift 2 ;;
    --data) data=1; shift 2 ;;
    -H|-w) shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
body=""
if [ "$data" = 1 ]; then body="$(cat)"; else cat >/dev/null 2>&1 || true; fi
echo "$method $url" >> "$STUB_LOG"
[ -n "$body" ] && printf "%s\n" "$body" >> "$STUB_BODIES"
last='"pagination":{"nextCursor":null,"hasNextPage":false}'
case "$STUB_MODE:$method $url" in
  "fresh:GET "*/serverless*) printf '{"endpoints":[],%s}\n200' "$last" ;;
  "page2:GET "*cursor=p2*) printf '{"endpoints":[{"id":"ep-2","name":"vivijure-backend"}],%s}\n200' "$last" ;;
  "page2:GET "*/serverless*) printf '{"endpoints":[{"id":"ep-1","name":"other"}],"pagination":{"nextCursor":"p2","hasNextPage":true}}\n200' ;;
  "nocursor:GET "*/serverless*) printf '{"endpoints":[],"pagination":{"nextCursor":null,"hasNextPage":true}}\n200' ;;
  *":POST "*/serverless) printf '{"id":"ep-created"}\n200' ;;
  *":PATCH "*/serverless/*) printf '{"id":"ep-2"}\n200' ;;
  *) printf '{}\n404' ;;
esac
'''

DEPLOY_ENV = """RUNPOD_API_KEY=stub
R2_ACCESS_KEY_ID=stub-id
R2_SECRET_ACCESS_KEY=stub-secret
R2_ENDPOINT=https://stub.invalid
"""


def _run(tmp_path, mode, extra_env=""):
    work = tmp_path / "work"
    work.mkdir()
    shutil.copy(ROOT / "deploy.sh", work / "deploy.sh")
    (work / "deploy.env").write_text(DEPLOY_ENV + extra_env)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    log, bodies = tmp_path / "calls.log", tmp_path / "bodies.log"
    log.write_text("")
    bodies.write_text("")
    env = {"PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}", "STUB_LOG": str(log),
           "STUB_BODIES": str(bodies), "STUB_MODE": mode, "HOME": str(tmp_path)}
    r = subprocess.run(["bash", str(work / "deploy.sh")], cwd=work, env=env,
                       capture_output=True, text=True, timeout=60)
    sent = [json.loads(b) for b in bodies.read_text().splitlines() if b.strip()]
    return r, log.read_text().splitlines(), sent


def test_first_time_create_posts_a_v2_endpoint_with_everything_on_it(tmp_path):
    r, calls, sent = _run(tmp_path, "fresh")
    assert r.returncode == 0, r.stderr
    assert "RUNPOD_ENDPOINT_ID=ep-created" in r.stdout
    assert calls[0] == "GET https://api.runpod.io/v2/serverless?limit=1000"
    assert calls[1] == "POST https://api.runpod.io/v2/serverless"
    assert not any("/templates" in c for c in calls), "v2 deploy makes no template"
    assert {urlsplit(c.split(" ", 1)[1]).hostname for c in calls} == {"api.runpod.io"}
    body = sent[0]
    assert body["type"] == "QUEUE"
    assert body["name"] == "vivijure-backend"
    assert body["image"].startswith("ghcr.io/skyphusion-labs/vivijure-backend:")
    assert body["env"] == {"R2_ENDPOINT": "https://stub.invalid", "R2_ACCESS_KEY_ID": "stub-id",
                           "R2_SECRET_ACCESS_KEY": "stub-secret", "R2_BUCKET": "vivijure"}
    assert body["gpu"] == {"pools": ["HOPPER_141", "BLACKWELL_180"], "count": 1}
    assert body["workers"] == {"min": 0, "max": 3, "idleTimeout": 5}
    assert body["scaling"] == {"type": "QUEUE_DELAY", "queueDelay": 4}
    assert body["flashboot"] == "FLASHBOOT"
    # v2 has no "no limit"; a blank EXECUTION_TIMEOUT_MS is one hour, never RunPod's short default.
    assert body["timeout"] == 3600000
    assert body["disk"] == 100


def test_an_existing_endpoint_on_a_later_page_is_patched_in_place(tmp_path):
    r, calls, sent = _run(tmp_path, "page2")
    assert r.returncode == 0, r.stderr
    assert "RUNPOD_ENDPOINT_ID=ep-2" in r.stdout
    assert calls == [
        "GET https://api.runpod.io/v2/serverless?limit=1000",
        "GET https://api.runpod.io/v2/serverless?limit=1000&cursor=p2",
        "PATCH https://api.runpod.io/v2/serverless/ep-2",
    ]
    assert sent[0]["image"].startswith("ghcr.io/skyphusion-labs/vivijure-backend:")
    assert "type" not in sent[0], "type is create-only on v2"


def test_a_list_that_claims_more_without_a_cursor_stops_the_run(tmp_path):
    r, calls, _ = _run(tmp_path, "nocursor")
    assert r.returncode != 0
    assert "gave no cursor" in r.stderr
    assert not any(c.startswith("POST ") for c in calls), "must not create a possible duplicate"


def test_an_unmapped_gpu_type_stops_before_any_call(tmp_path):
    r, calls, _ = _run(tmp_path, "fresh", "GPU_TYPE_IDS=\"NVIDIA Z9\"\n")
    assert r.returncode != 0
    assert "GPU_POOLS" in r.stderr
    assert calls == []


def test_gpu_pools_and_timeout_can_be_set_directly(tmp_path):
    r, _, sent = _run(tmp_path, "fresh", "GPU_POOLS=BLACKWELL_96,HOPPER_141\nEXECUTION_TIMEOUT_MS=1800000\n")
    assert r.returncode == 0, r.stderr
    assert sent[0]["gpu"]["pools"] == ["BLACKWELL_96", "HOPPER_141"]
    assert sent[0]["timeout"] == 1800000
