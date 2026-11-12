#!/bin/bash
# Check the inputs of .github/workflows/stills-base.yml before anything is staged, built or pushed
# (#564, fleet-chezmoi#2620), and say what to build.
#
#   IMAGE=ghcr.io/skyphusion-labs/vivijure-backend TAG=stills-t1-m1-rc1 MODEL_VERSION=1 \
#     REF=ghcr.io/skyphusion-labs/vivijure-backend:toolchain-t1@sha256:<64 hex> \
#     deploy/stills-base-resolve.sh
#
# The workflow pushes ONLY a candidate tag. A final name (stills-t<T>-m<M>, the shape of the live
# stills-t1-m1 under 1.1.19) is never written here; promoting a proven candidate to one is a
# registry-side copy by the lead, after the proof in docs/stills-weights-base.md.
#
# Fail closed, before any work:
# - TAG is not stills-t<T>-m<M>-rc<N>. That refuses stills-t1-m1 and every other final-shaped tag;
# - REF is not a full digest of IMAGE:toolchain-t<N>, or it cannot be read on GHCR;
# - the tag's t<T> is not the toolchain's t<N>, or its m<M> is not MODEL_VERSION (the tag would
#   then name a base it does not hold, and VJ_MODEL_VERSION must match the endpoint's or every
#   worker pulls again);
# - IMAGE:TAG already exists on GHCR. A GHCR tag is never overwritten (RELEASES.md);
# - the registry cannot tell whether IMAGE:TAG exists: that is never read as "free".
#
# INSPECT is the registry probe, default "docker buildx imagetools inspect" (needs the GHCR login
# first). tests/test_stills_base_resolve.py replaces it. Output: tag=, toolchain_ref= and
# model_version= lines, to $GITHUB_OUTPUT when it is set, else to stdout.
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

TAG=${TAG:-}
printf '%s' "$TAG" | grep -Eqx 'stills-t[0-9]+-m[0-9]+-rc[0-9]+' \
  || fail "tag must be a candidate, stills-t<T>-m<M>-rc<N> (got '$TAG'); a final tag such as stills-t1-m1 is never pushed here"

printf '%s' "${MODEL_VERSION:-}" | grep -Eqx '[0-9]+' || fail "model_version must be a whole number (got '${MODEL_VERSION:-}')"

img_re=$(printf '%s' "$IMAGE" | sed 's/[.]/\\./g')
printf '%s' "${REF:-}" | grep -Eqx "${img_re}:toolchain-t[0-9]+@sha256:[0-9a-f]{64}" \
  || fail "toolchain_ref must be a full digest of ${IMAGE}:toolchain-t<N>@sha256:<64 hex> (got '${REF:-}')"

tag_t=$(printf '%s' "$TAG" | sed -E 's/^stills-(t[0-9]+)-m.*/\1/')
tag_m=$(printf '%s' "$TAG" | sed -E 's/^stills-t[0-9]+-m([0-9]+)-rc.*/\1/')
ref_t=$(printf '%s' "$REF" | sed -E 's/.*:toolchain-(t[0-9]+)@.*/\1/')
[ "$tag_t" = "$ref_t" ] || fail "tag names toolchain $tag_t but toolchain_ref is $ref_t"
[ "$tag_m" = "$MODEL_VERSION" ] || fail "tag names model version m$tag_m but model_version is $MODEL_VERSION"

rc=0; probe "$REF" || rc=$?
[ "$rc" = 0 ] || fail "toolchain_ref $REF cannot be read on GHCR (probe $rc); nothing was built"

rc=0; probe "$IMAGE:$TAG" || rc=$?
case "$rc" in
  0) fail "$IMAGE:$TAG already exists on GHCR; refusing to overwrite it (pick the next -rc<N>)" ;;
  1) ;;
  *) fail "cannot tell whether $IMAGE:$TAG exists on GHCR; refusing (an unknown is never read as free)" ;;
esac

out "tag=$TAG"
out "toolchain_ref=$REF"
out "model_version=$MODEL_VERSION"
echo "Building $IMAGE:$TAG from $REF (VJ_MODEL_VERSION=$MODEL_VERSION)." >&2
