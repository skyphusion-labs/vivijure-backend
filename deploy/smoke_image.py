"""Stream a worker image WITHOUT its weight layers into `docker load`, for the CPU import smoke (#645).

    WORKER=ghcr.io/skyphusion-labs/vivijure-backend:1.1.20@sha256:<64 hex> \
    BASE=ghcr.io/skyphusion-labs/vivijure-backend:stills-t1-m1-rc1@sha256:<64 hex> \
      python3 -I deploy/smoke_image.py plan          # JSON: what is kept, what is dropped, bytes
      python3 -I deploy/smoke_image.py stream | docker load

WHY. A worker built FROM the stills weights base is the toolchain, then the base's weight layers, then
the thin layers. The build pushes it by layer reference and fits a hosted runner. A `docker pull` does
not: 1.1.20 is 31.8 GB compressed, and run 43312194021 refused it with 43.8 GB free. The weight layers
are 18.4 GB of that. They add files under /opt/models only (#644: `assert-appended` proves the base
config is the toolchain config plus exactly the bin diff_ids), and `smoke_imports.py` reads no weight
file on CPU (PROBE_MODEL is used only when a GPU is visible). So the smoke runs on the worker with the
weight layers taken out, and every other byte it runs is the worker's own blob, fetched by digest and
checked. Nothing is rebuilt and nothing is pushed. The GPU proof on a scratch endpoint stays the
end-to-end check of the full image.

WHAT IS DROPPED. BASE is the image the worker was built FROM. The dropped layers are the base's layers
above ITS toolchain, read from the base's `org.skyphusion.vivijure.toolchain` label (set by
stills-base.yml). Every prefix is checked, in the manifest layer digests and in the config diff_ids:
the worker starts with the base, and the base starts with the toolchain. Without BASE, or on a base
with no toolchain label (a plain toolchain), nothing is dropped and the smoke image is the worker.

THE IMAGE LOADED. The worker's config, with the dropped layers' diff_ids and history entries removed,
and one label added (org.skyphusion.vivijure.smoke-of = the worker digest), so it can never be taken
for the shipped config. The archive carries an OCI layout (index.json) and a Docker manifest.json, the
same shape `docker save` writes, so both Docker image stores load it. Layers stream from the registry
straight into the tar: no archive is written to disk. A blob whose size or sha256 differs from its
descriptor stops the stream with exit 1.

GHCR_TOKEN (optional) is a GitHub token used for the registry token exchange; without it the exchange
is anonymous, which a public package allows. It is never printed.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request

OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
INDEX_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
ACCEPT = ",".join(sorted(INDEX_TYPES | {OCI_MANIFEST, "application/vnd.docker.distribution.manifest.v2+json"}))
TOOLCHAIN_LABEL = "org.skyphusion.vivijure.toolchain"
SMOKE_OF_LABEL = "org.skyphusion.vivijure.smoke-of"
REF_RE = re.compile(r"^(?P<host>ghcr\.io)/(?P<repo>skyphusion-labs/[a-z0-9._-]+)(?::[A-Za-z0-9._-]+)?@(?P<digest>sha256:[0-9a-f]{64})$")
CHUNK = 1 << 20


class Refused(Exception):
    """The inputs do not describe a worker the smoke may load. The message says which check."""


def parse_ref(ref: str, what: str) -> tuple[str, str]:
    """(repo, digest) of a full digest ref on GHCR, or Refused. A tag alone is never accepted."""
    m = REF_RE.match(ref or "")
    if not m:
        raise Refused(f"{what} must be ghcr.io/skyphusion-labs/<image>[:tag]@sha256:<64 hex> (got {ref!r})")
    return m.group("repo"), m.group("digest")


class Registry:
    """The few GHCR reads this needs: a manifest (an index resolves to linux/amd64), a JSON blob, and
    a blob as a stream. Tests replace it with a fake that has the same three methods."""

    def __init__(self, token: str | None = None):
        self._gh_token = token
        self._bearer: dict[str, str] = {}

    def _auth(self, repo: str) -> str:
        if repo not in self._bearer:
            url = "https://ghcr.io/token?" + urllib.parse.urlencode({"scope": f"repository:{repo}:pull"})
            req = urllib.request.Request(url)
            if self._gh_token:
                cred = base64.b64encode(b"x:" + self._gh_token.encode()).decode()
                req.add_header("Authorization", "Basic " + cred)
            with urllib.request.urlopen(req, timeout=60) as r:
                self._bearer[repo] = json.load(r)["token"]
        return self._bearer[repo]

    def _open(self, repo: str, path: str, accept: str):
        # Redirects are followed by hand, WITHOUT the bearer: a blob GET redirects to a signed CDN URL
        # on another host, which must not be sent the registry credential.
        url = f"https://ghcr.io/v2/{repo}/{path}"
        headers = {"Authorization": "Bearer " + self._auth(repo), "Accept": accept}
        opener = urllib.request.build_opener(_NoRedirect)
        for _ in range(5):
            try:
                return opener.open(urllib.request.Request(url, headers=headers), timeout=300)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                    url = urllib.parse.urljoin(url, e.headers["Location"])
                    headers = {}
                    continue
                raise
        raise RuntimeError(f"too many redirects for {repo}/{path}")

    def manifest(self, repo: str, digest: str) -> dict:
        with self._open(repo, "manifests/" + digest, ACCEPT) as r:
            m = json.load(r)
        if m.get("mediaType") in INDEX_TYPES or "manifests" in m:
            pick = [d for d in m.get("manifests", [])
                    if (d.get("platform") or {}).get("os") == "linux"
                    and (d.get("platform") or {}).get("architecture") == "amd64"]
            if not pick:
                raise Refused(f"{repo}@{digest}: the index has no linux/amd64 manifest")
            with self._open(repo, "manifests/" + pick[0]["digest"], ACCEPT) as r:
                m = json.load(r)
        return m

    def blob_json(self, repo: str, digest: str) -> dict:
        with self._open(repo, "blobs/" + digest, "*/*") as r:
            data = r.read()
        if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
            raise Refused(f"{repo}: blob {digest} does not hash to its digest")
        return json.loads(data)

    def blob_stream(self, repo: str, digest: str):
        return self._open(repo, "blobs/" + digest, "*/*")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


def _layers(m: dict) -> list[dict]:
    layers = m.get("layers") or []
    if not layers:
        raise Refused("a manifest lists no layers")
    return layers


def _digests(layers: list[dict]) -> list[str]:
    return [x["digest"] for x in layers]


def plan(worker: str, base: str | None, reg) -> dict:
    """What the smoke image is: the worker's manifest layers and config with the base's weight layers
    taken out. Every relation is checked; any doubt is Refused, never guessed."""
    wrepo, wdig = parse_ref(worker, "WORKER")
    wm = reg.manifest(wrepo, wdig)
    wlayers = _layers(wm)
    wcfg = reg.blob_json(wrepo, wm["config"]["digest"])
    wdiff = list((wcfg.get("rootfs") or {}).get("diff_ids") or [])
    if len(wdiff) != len(wlayers):
        raise Refused(f"the worker config lists {len(wdiff)} diff_ids for {len(wlayers)} layers")

    drop_from = drop_to = len(wlayers)  # an empty slice: nothing dropped
    toolchain = None
    if base:
        brepo, bdig = parse_ref(base, "BASE")
        bm = reg.manifest(brepo, bdig)
        blayers = _layers(bm)
        bcfg = reg.blob_json(brepo, bm["config"]["digest"])
        bdiff = list((bcfg.get("rootfs") or {}).get("diff_ids") or [])
        if _digests(wlayers[:len(blayers)]) != _digests(blayers) or wdiff[:len(bdiff)] != bdiff \
                or len(blayers) != len(bdiff):
            raise Refused("the worker does not start with the base's layers: it was not built FROM this BASE")
        toolchain = ((bcfg.get("config") or {}).get("Labels") or {}).get(TOOLCHAIN_LABEL)
        if toolchain:
            wlabel = ((wcfg.get("config") or {}).get("Labels") or {}).get(TOOLCHAIN_LABEL)
            if wlabel != toolchain:
                raise Refused(f"the worker's {TOOLCHAIN_LABEL} label ({wlabel!r}) is not the base's ({toolchain!r})")
            trepo, tdig = parse_ref(toolchain, f"the base's {TOOLCHAIN_LABEL} label")
            tm = reg.manifest(trepo, tdig)
            tlayers = _layers(tm)
            tcfg = reg.blob_json(trepo, tm["config"]["digest"])
            tdiff = list((tcfg.get("rootfs") or {}).get("diff_ids") or [])
            if _digests(blayers[:len(tlayers)]) != _digests(tlayers) or bdiff[:len(tdiff)] != tdiff \
                    or len(tlayers) != len(tdiff):
                raise Refused("the base does not start with its toolchain's layers")
            if len(tlayers) >= len(blayers):
                raise Refused("the base adds no layers over its toolchain, yet carries a toolchain label")
            drop_from, drop_to = len(tlayers), len(blayers)

    keep = list(range(drop_from)) + list(range(drop_to, len(wlayers)))
    dropped = list(range(drop_from, drop_to))

    history = list(wcfg.get("history") or [])
    if history:
        layer_entries = [i for i, h in enumerate(history) if not h.get("empty_layer")]
        if len(layer_entries) != len(wlayers):
            raise Refused(f"the worker history has {len(layer_entries)} layer entries for {len(wlayers)} layers")
        gone = {layer_entries[i] for i in dropped}
        history = [h for i, h in enumerate(history) if i not in gone]

    cfg = json.loads(json.dumps(wcfg))
    cfg["rootfs"]["diff_ids"] = [wdiff[i] for i in keep]
    if "history" in cfg:
        cfg["history"] = history
    cfg.setdefault("config", {})
    labels = dict(cfg["config"].get("Labels") or {})
    labels[SMOKE_OF_LABEL] = f"ghcr.io/{wrepo}@{wdig}"
    cfg["config"]["Labels"] = labels
    cfg_bytes = json.dumps(cfg, separators=(",", ":"), sort_keys=True).encode()
    cfg_desc = {"mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": "sha256:" + hashlib.sha256(cfg_bytes).hexdigest(), "size": len(cfg_bytes)}
    kept_layers = [dict(wlayers[i]) for i in keep]
    manifest = {"schemaVersion": 2, "mediaType": OCI_MANIFEST, "config": cfg_desc, "layers": kept_layers}
    man_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    return {
        "worker": f"ghcr.io/{wrepo}@{wdig}",
        "repo": wrepo,
        "base": base or None,
        "toolchain": toolchain,
        "kept": len(keep),
        "dropped": [{"digest": wlayers[i]["digest"], "diff_id": wdiff[i], "size": int(wlayers[i]["size"])} for i in dropped],
        "dropped_bytes": sum(int(wlayers[i]["size"]) for i in dropped),
        "bytes": len(cfg_bytes) + sum(int(x["size"]) for x in kept_layers),
        "diff_ids": cfg["rootfs"]["diff_ids"],
        "config": cfg_bytes.decode(),
        "manifest": man_bytes.decode(),
    }


class _Checked:
    """A read-through view of a blob stream that hashes what passes and refuses a wrong size or digest."""

    def __init__(self, raw, digest: str, size: int):
        self._raw, self._digest, self._size = raw, digest, size
        self._h, self._n = hashlib.sha256(), 0

    def read(self, n: int = -1) -> bytes:
        data = self._raw.read(n if n is not None and n >= 0 else CHUNK)
        self._h.update(data)
        self._n += len(data)
        if self._n > self._size:
            raise Refused(f"blob {self._digest} is longer than its descriptor ({self._size} bytes)")
        return data

    def verify(self) -> None:
        # tarfile reads exactly `size` bytes, so a blob LONGER than its descriptor would pass on its
        # first `size` bytes: one more read must find the end.
        if self._raw.read(1):
            raise Refused(f"blob {self._digest} is longer than its descriptor ({self._size} bytes)")
        if self._n != self._size or "sha256:" + self._h.hexdigest() != self._digest:
            raise Refused(f"blob {self._digest}: got {self._n} bytes hashing to sha256:{self._h.hexdigest()}, "
                          f"want {self._size} bytes")


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size, ti.mode, ti.mtime = len(data), 0o644, 0
    tf.addfile(ti, io.BytesIO(data))


def stream(p: dict, reg, out, tag: str) -> None:
    """Write the smoke image to `out` as a `docker load` archive. Layers go registry -> tar, checked."""
    cfg = p["config"].encode()
    man = p["manifest"].encode()
    manifest = json.loads(man)
    blob = lambda d: "blobs/sha256/" + d.split(":", 1)[1]  # noqa: E731
    man_digest = "sha256:" + hashlib.sha256(man).hexdigest()
    with tarfile.open(fileobj=out, mode="w|", format=tarfile.PAX_FORMAT) as tf:
        _add_bytes(tf, "oci-layout", b'{"imageLayoutVersion":"1.0.0"}')
        _add_bytes(tf, blob(manifest["config"]["digest"]), cfg)
        _add_bytes(tf, blob(man_digest), man)
        for layer in manifest["layers"]:
            raw = reg.blob_stream(p["repo"], layer["digest"])
            try:
                chk = _Checked(raw, layer["digest"], int(layer["size"]))
                ti = tarfile.TarInfo(blob(layer["digest"]))
                ti.size, ti.mode, ti.mtime = int(layer["size"]), 0o644, 0
                tf.addfile(ti, chk)
                chk.verify()
            finally:
                close = getattr(raw, "close", None)
                if close:
                    close()
        name = tag if "/" in tag.split(":", 1)[0] or "." in tag.split("/", 1)[0] else "docker.io/library/" + tag
        index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
                 "manifests": [{"mediaType": OCI_MANIFEST, "digest": man_digest, "size": len(man),
                                "annotations": {"io.containerd.image.name": name,
                                                "org.opencontainers.image.ref.name": tag.rsplit(":", 1)[-1]}}]}
        _add_bytes(tf, "index.json", json.dumps(index, separators=(",", ":")).encode())
        legacy = [{"Config": blob(manifest["config"]["digest"]), "RepoTags": [tag],
                   "Layers": [blob(x["digest"]) for x in manifest["layers"]]}]
        _add_bytes(tf, "manifest.json", json.dumps(legacy, separators=(",", ":")).encode())


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("plan", "stream"):
        print("usage: WORKER=<ref@digest> [BASE=<ref@digest>] smoke_image.py plan|stream", file=sys.stderr)
        return 2
    reg = Registry(os.environ.get("GHCR_TOKEN") or None)
    try:
        p = plan(os.environ.get("WORKER", ""), os.environ.get("BASE") or None, reg)
        if argv[1] == "plan":
            print(json.dumps({k: v for k, v in p.items() if k not in ("config", "manifest")}, indent=1))
        else:
            stream(p, reg, sys.stdout.buffer, os.environ.get("TAG", "vj-smoke:local"))
            sys.stdout.buffer.flush()
    except Refused as e:
        print(f"smoke_image: REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
