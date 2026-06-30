# De-risk offline-correctness proof (#17): no-egress verification

Status: DESIGN (gated on `:0.3.3` for the runs). Owner: Strummer (harness); inner egress-block is
change-controlled, coordinate with Rollins (owns `vj_derisk.py`) + lead before merge.

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

The pods need the R2 read path, so this is NOT `--network=none`; it is egress-blocked EXCEPT the R2 exfil
endpoint. Run the full render on each of the 3 pooled arches (RTX PRO 6000 sm_120 / H200 sm_90 / B200
sm_100). A clean `render_portrait` + `render_landscape` + `derisk_pass` UNDER the block proves the GPU
render does no phone-home beyond the R2 read path.

Egress allowlist (the ONLY things permitted out):
- DNS (UDP+TCP 53) to the resolver -- needed to resolve the R2 endpoint host.
- The R2 S3 endpoint host (`<accountid>.r2.cloudflarestorage.com`) on TCP 443 (the boto3 uploader).
Everything else: DROP. No github, no huggingface, no pypi, no torch.hub.

Mechanism (in `deploy/derisk_pod_start.sh`, BEFORE `$RUN render`, gated behind a `DERISK_EGRESS_LOCK`
env flag so normal de-risk runs are unaffected):

```
# Resolve + pin the R2 host, then default-drop egress except loopback, established, DNS, and R2:443.
R2_HOST=$(printf '%s' "$R2_S3_ENDPOINT" | sed -E 's#https?://([^/]+).*#\1#')
iptables -P OUTPUT DROP
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT
for ip in $(getent ahosts "$R2_HOST" | awk '{print $1}' | sort -u); do
  iptables -A OUTPUT -p tcp -d "$ip" --dport 443 -j ACCEPT
done
echo "@event egress_locked {\"allow\": [\"dns\", \"$R2_HOST:443\"], \"default\": \"DROP\"}"
```

On the block, an un-baked dep (facexlib pre-`:0.3.3`) FAILS -> `derisk_fail` at finish -> exactly the proof
the bake is incomplete. On `:0.3.3` (facexlib baked) the render succeeds = true-offline proof. The boto3
read-path uploader keeps working (R2:443 + DNS allowed), so the @event stream still reaches R2.

Complementary cheap measure (not a substitute for the block): broaden the inner's PATH-shadow tripwire to
also shadow `curl` / `wget` / `git` (same trick as the fake `rclone`), so a github/torch.hub fetch trips a
sentinel too. The egress-block is the gold standard; the broadened tripwire is a cheap belt for runs where
the block is unavailable.

## THE feasibility unknown -- CAP_NET_ADMIN

`iptables` / `netns` require `CAP_NET_ADMIN`. Does a RunPod SECURE container grant it? UNKNOWN -- settle
BEFORE committing layer 2, with a $0 check: smallest/CPU pod running `capsh --print` + `iptables -L` (or
RunPod docs).
- If YES -> the iptables approach above, in the inner behind `DERISK_EGRESS_LOCK`.
- If NO -> fallbacks:
  - (a) Python-level socket guard in the driver: monkeypatch `socket.getaddrinfo` / `socket.socket.connect`
    to allowlist ONLY the R2 host; any other `connect()` raises. Pure userspace, no cap needed, asserts at
    the exact syscall the render uses. (Coordinate with Rollins -- it lives in the driver.)
  - (b) Run Layer 1 (`--network=none`) on a self-hosted GPU box if the fleet has one, for the true
    zero-network GPU render, and keep RunPod for the perf/JIT numbers (network-on).

## Sign-off + resume sequencing

1. Settle CAP_NET_ADMIN (the $0 check above).
2. Sign-off the inner egress-block (or the driver socket-guard fallback) with lead + Rollins
   (change-controlled; the inner is the de-risk pod start, the socket guard touches `vj_derisk.py`).
3. Gated on `:0.3.3` (facexlib baked, #15): run Layer 1 (`--network=none`, $0) first, then the
   egress-blocked 3-arch fan-out (RTX PRO 6000 + H200 + B200).
4. A clean 3-arch `derisk_pass` under the block = the "fully baked / offline" proof -> #5 serverless
   promote, on Conrad's word.

Tokens are throwaway + revoked per run (see `docs/derisk-3arch.md`); the egress allowlist intentionally
keeps R2 reachable so the read path survives the block. No em-dashes/en-dashes anywhere (lint).
