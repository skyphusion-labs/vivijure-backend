#!/bin/bash
# Check the inputs of .github/workflows/thin-image.yml before anything is built
# or pushed, and say what to build from.
#
#   IMAGE=ghcr.io/skyphusion-labs/vivijure-backend VER=1.1.19 \
#     REF=ghcr.io/skyphusion-labs/vivijure-backend:stills-t1-m1@sha256:<64 hex> \
#     deploy/thin-image-resolve.sh
#
# REF (the workflow's toolchain_ref) is a full digest of an existing base image.
# With it, nothing is built for the toolchain and no toolchain tag is pushed:
# the thin worker is built FROM that digest, so its lower layers are the base's
# and RunPod pulls only the thin layers. Without REF, TOOL (t<N>) names the
# toolchain tag that the workflow builds and pushes, as before.
#
# Fail closed, before any build:
# - VER is not X.Y.Z;
# - REF is not ghcr.io/skyphusion-labs/<image>[:tag]@sha256:<64 hex>, or it
#   cannot be read on GHCR;
# - without REF, TOOL is not t<N>;
# - IMAGE:VER already exists on GHCR. A GHCR tag is never overwritten
#   (RELEASES.md), so a re-dispatch of a published version is refused;
# - the registry cannot tell whether IMAGE:VER exists (an error other than
#   "not found"): that is never read as "free".
#
# INSPECT is the registry probe, default "docker buildx imagetools inspect"
# (needs the GHCR login first). tests/test_thin_image_resolve.py replaces it.
# Output: version=, mode= (prebuilt or build) and toolchain_ref= lines, to
# $GITHUB_OUTPUT when it is set, else to stdout.
set -euo pipefail
: "${IMAGE:?set IMAGE, for example ghcr.io/skyphusion-labs/vivijure-backend}"
read -r -a INSPECT_CMD <<< "${INSPECT:-docker buildx imagetools inspect}"
out() { if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "$1" >> "$GITHUB_OUTPUT"; else echo "$1"; fi; }
fail() { echo "::error::$1"; exit 1; }

# probe <ref>: 0 the ref exists, 1 the registry says it is not there, 2 cannot tell.
probe() {
  local err rc=0
  err=$("${INSPECT_CMD[@]}" "$1" 2>&1 >/dev/null) || rc=$?
  [ "$rc" = 0 ] && return 0
  if printf '%s' "$err" | grep -Eqi 'not found|manifest unknown'; then return 1; fi
  echo "probe of $1 failed (exit $rc): $(printf '%s' "$err" | head -c 300)" >&2
  return 2
}

printf '%s' "${VER:-}" | grep -Eqx '[0-9]+\.[0-9]+\.[0-9]+' || fail "version must be X.Y.Z (got '${VER:-}')"

if [ -n "${REF:-}" ]; then
  printf '%s' "$REF" | grep -Eqx 'ghcr\.io/skyphusion-labs/[a-z0-9._-]+(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}' \
    || fail "toolchain_ref must be a full digest: ghcr.io/skyphusion-labs/<image>[:tag]@sha256:<64 hex> (got '$REF')"
  rc=0; probe "$REF" || rc=$?
  [ "$rc" = 0 ] || fail "toolchain_ref $REF cannot be read on GHCR (probe $rc); nothing was built"
  mode=prebuilt
  tref=$REF
else
  printf '%s' "${TOOL:-}" | grep -Eqx 't[0-9]+' || fail "toolchain_tag must be t<N> (got '${TOOL:-}')"
  mode=build
  tref="$IMAGE:toolchain-$TOOL"
fi

rc=0; probe "$IMAGE:$VER" || rc=$?
case "$rc" in
  0) fail "$IMAGE:$VER already exists on GHCR; refusing to overwrite it (pick the next free semver, RELEASES.md)" ;;
  1) ;;
  *) fail "cannot tell whether $IMAGE:$VER exists on GHCR; refusing (an unknown is never read as free)" ;;
esac

out "version=$VER"
out "mode=$mode"
out "toolchain_ref=$tref"
echo "Building $IMAGE:$VER from $tref ($mode)." >&2
