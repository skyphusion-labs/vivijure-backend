# De-risk offline-correctness proof (#17): no-egress verification

Status: DESIGN (gated on `:0.3.3` for the runs). Owner: Strummer (harness + fire wiring); the inner
egress guard lives in `vj_derisk.py` (Rollins, PR #161) and is change-controlled -- coordinate with
Rollins + lead before merge. CAP_NET_ADMIN is SETTLED (see below): the primary Layer-2 mechanism is the
userspace socket-guard, NOT iptables.

## Why

The render-leg no-pull proof is currently **rclone-scoped**: the inner shadows `rclone` with a fake
earlier in PATH and asserts the sentinel never fires (`@event rclone_tripwire fired=false`). That proves
"no `rclone` pull from R2", NOT "no network egress". The facexlib finish-leg fetch
(`detection_Resnet50_Final.pth` + `parsing_parsenet.pth` from github releases, via torchvision/facexlib)
proved the hole concretely on the sm_120 `:0.3.2` pass: it sails past the rclone tripwire because it is a
github/torch.hub fetch, not `rclone`. **Baked != proven-no-phone-home.** Before any "fully baked / offline"
claim or the #5 serverless promote, the 3-arch fan-out must run with egress BLOCKED so a clean
`render_done` is a POSITIVE proof of self-containment, not just an absence of one egress mechanism.

## Two complementary layers

### Layer 1 -- `docker --network=none` verify (local, GPU-free, $0)

Run the image with ZERO network on a docker host (jello / the bake host). Proves every model the runtime
RESOLVES is baked + offline-loadable; no R2 needed (read docker stdout locally). Covers the CPU legs only
(the GPU render is layer 2).

```
docker run --rm --network=none ghcr.io/skyphusion-labs/vivijure-backend:<ver> \
  conda run --no-capture-output -n vivijure python /opt/vivijure/vj_derisk.py probe
```

PASS = `@event baked_probe vj_baked:true` + `render_repos_active` populated + NO
`@event derisk_fail stage=render_repo_not_cached`, all with the network namespace empty. This layer alone
would have caught facexlib instantly: the finish-stage download errors with no network. (A full render is
GPU-bound, so layer 1 proves offline-RESOLVABILITY of the repos, not the GPU render itself -- that is
layer 2.)

### Layer 2 -- egress-blocked 3-arch fan-out (RunPod, GPU)

The render process is fully baked + credential-free, so under Design B (FULL-BLOCK) it needs NO network
at all. The pod is not `--network=none` only because it still runs the SEPARATE R2 telemetry uploader (a
different process, OUTSIDE the in-process guard). Run the full render on each of the 3 pooled arches
(RTX PRO 6000 sm_120 / H200 sm_90 / B200 sm_100). A clean `render_portrait` + `render_landscape` +
`derisk_pass` UNDER the block proves the GPU render does no phone-home AT ALL.

**Mechanism: a userspace socket-guard in `vj_derisk.py` (PR #161), NOT iptables.** CAP_NET_ADMIN is not
available on RunPod (see the next section), so a kernel-level OUTPUT-DROP is off the table. The guard
wraps `socket.getaddrinfo` + `socket.socket.connect` in the render process and, in Design B (FULL-BLOCK),
allows ONLY `AF_UNIX` + loopback and DROPs everything else INCLUDING R2; any other `connect()` raises. It installs when `DERISK_EGRESS_LOCK=1` is present in the env, which
the fire injects pod-level via `runpodctl --env` so the inner $RUN render children inherit it (no inner
code change beyond the flag; baseline runs omit it, so zero behavior change).

Controls (Design B -- there is NO R2 allowlist; R2 is dropped to the render too):
- NEGATIVE: a huggingface connect AND a github connect are attempted and MUST BOTH be blocked
  (`egress_guard_proven {hf_blocked: true, github_blocked: true}`).
- POSITIVE (sanity): a loopback connect MUST still succeed (`egress_guard_sane {loopback_ok: true}`),
  proving the guard blocks egress without bricking local IPC.
- No R2 positive control: the render makes no R2 connection, so the guard does not allow R2. `R2_S3_*`
  stays pod-level for the SEPARATE uploader, NOT for the render.

**Finding (record this):** the render process makes NO R2 connection of its own -- it is baked + offline +
credential-free (the backend `R2_*` model-pull names are deliberately NOT injected). The R2 `@event`
uploader is a SEPARATE process (the read-path wrapper), untouched by the in-process guard. So Design B
(FULL-BLOCK) SUPERSEDES the earlier allow-R2 design: the render is credential-free AND R2-free, the guard
is a FULL egress block on it (loopback + `AF_UNIX` only), and a clean render under the block is positive
proof of self-containment.

@event sequence the watcher asserts under the lock:
`egress_guard_installed {mode: full_block}` ->
`egress_guard_proven {hf_blocked: true, github_blocked: true}` (NEGATIVE) ->
`egress_guard_sane {loopback_ok: true}` (POSITIVE) -> `render_done` (portrait) -> `render_done` (landscape)
-> `derisk_pass`. FAIL = `derisk_fail stage=egress_guard_inactive` on ANY control miss (the lock was
requested but the guard did not install, or a negative control reached the network) -- a hard do-not-trust.

On the block, an un-baked dep (facexlib pre-`:0.3.3`) FAILS -> `derisk_fail` at finish -> exactly the proof
the bake is incomplete. On `:0.3.3` (facexlib baked) the render succeeds = true-offline proof. The boto3
read-path uploader keeps working (it is a SEPARATE process, outside the guard), so the @event stream still
reaches R2 even though the render itself has zero network.

EXPECT_SHA for the injected driver is set in #161 (`22709023...`); the inner integrity gate uses exactly
these driver bytes.

## CAP_NET_ADMIN -- SETTLED: not available on RunPod (iptables is OUT)

`iptables` / `nft` OUTPUT-DROP require `CAP_NET_ADMIN`. VERDICT: a RunPod SECURE container does NOT grant
it. `runpodctl pod create` exposes no `--cap-add` / `--privileged` surface, the container drops the cap,
and Secure Cloud is sandboxed. The API surface is conclusive, so this was settled WITHOUT burning a pod
($0). Consequence: the kernel-level iptables approach is OUT; the **userspace socket-guard (above) is the
PRIMARY Layer-2 mechanism**, not a fallback.

(Belt, not a substitute: the inner PATH-shadow tripwire can also shadow `curl` / `wget` / `git` so a
github/torch.hub fetch trips a sentinel even outside the Python socket surface.)

## Honest scope of the proof (do NOT overclaim)

The userspace guard catches the **Python egress surface** -- `hf_hub` / `torch.hub` / `requests` / `boto3`,
which are ALL the vectors in play here. A native (non-Python) raw socket would bypass it; we do not claim
otherwise. Frame the proof EXACTLY as:

> Layer 1: kernel-airtight offline-RESOLVABILITY (`--network=none`, the CPU legs).
> Layer 2: userspace FULL-BLOCK GPU render with negative (HF + github) + positive (loopback sanity) controls.

Do NOT imply kernel-airtight for the GPU leg -- it is userspace-blocked, which is the strongest available
on RunPod given no CAP_NET_ADMIN.

## Sign-off + resume sequencing

1. CAP_NET_ADMIN: SETTLED (not available; userspace guard is primary). DONE.
2. Sign-off the inner egress guard (PR #161, `vj_derisk.py`) + the fire-wiring flag with lead + Rollins
   (change-controlled).
3. Gated on `:0.3.3` (facexlib baked, #15): run Layer 1 (`--network=none`, $0) first, then the
   egress-blocked 3-arch fan-out (RTX PRO 6000 + H200 + B200) with `DERISK_EGRESS_LOCK=1`.
4. A clean 3-arch `derisk_pass` under the block = the "fully baked / offline" proof -> #5 serverless
   promote, on Conrad's word.

Tokens are throwaway + revoked per run (see `docs/derisk-3arch.md`); the egress allowlist intentionally
keeps R2 reachable so the read path survives the block. No em-dashes/en-dashes anywhere (lint).
