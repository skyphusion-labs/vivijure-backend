# Security policy

## Supported versions

This is a rolling, single-`main`-branch project released as `backend-vX.Y.Z` tags. Only
the latest release receives security fixes. If you are running an older revision, upgrade
to the newest tag to pick them up.

## Reporting a vulnerability

Please do not file a public GitHub issue for a security problem. Report it privately to
**security@skyphusion.org**. If you would rather use GitHub, open the repository's **Security** tab and
click **"Report a vulnerability"** to file a private advisory that only you and the maintainers can
see.

Please include:

- A description of the issue and its impact
- Steps to reproduce, including a minimal example if possible
- The affected version (tag or commit SHA if known)
- Any suggestions for remediation

What to expect:

- **Acknowledgment** within a reasonable window (target: 5 business days).
- A **fix** in the latest release once we confirm the issue; time-sensitive reports should say so.
- **Credit** for your report when the fix ships, unless you would rather stay anonymous.

Please give us a chance to ship a fix before any public disclosure (target: up to 90 days for a
coordinated fix).

## Scope

This is the render backend behind a serverless GPU endpoint (RunPod). It is driven by a
trusted control plane: the control plane writes a project bundle to object storage and
submits a render job; this backend pulls the bundle, renders, and writes artifacts back.
The security boundary is:

- The worker holds exactly one credential: an R2 (S3-compatible) API token scoped to a
  single bucket, delivered through the environment. It is the only secret in the runtime.
  ([docs/operations.md](docs/operations.md) details the runtime, the key layout, and the
  trust boundary in full.)
- Render-job input arrives from the control plane; this backend does not authenticate end
  users itself (the control plane does, behind Cloudflare Access).
- The studio is **single-operator** (the anti-SaaS identity strip, vivijure #292): the control
  plane sends no submitter identity, and `/api/artifact` serves objects by key with no per-row
  ownership check. This backend therefore stamps **no identity** onto artifacts; a job body that
  still carries a `user_email` is ignored, so a stripped identity cannot resurface as object
  metadata. There is no per-user ownership control to defeat (there are no users, only the operator).

In-scope vulnerabilities include:

- Escapes from the bundle reader (e.g. path traversal via `tar`/zip entries or crafted
  keys) that read or write outside the intended job workspace or bucket prefix.
- Server-side request forgery or arbitrary object access via attacker-influenced keys.
- Code execution via crafted bundle contents (`storyboard.yaml`, registry, refs).
- Leakage of the R2 credential, or any reintroduction of submitter identity into artifact
  metadata (the identity strip must hold; see Scope above).
- Injection issues in any shell-out (ffmpeg, model tooling) driven by job input.

Out-of-scope:

- Issues that require an already-compromised control plane or already-leaked R2 token.
- Denial of service from intentionally expensive but well-formed render jobs (render cost
  is the operator's concern; submit access is gated by the control plane).
- The security posture of the upstream model weights or third-party libraries themselves
  (report those to their projects), beyond how this backend invokes them.

## Scope of reports

Security reports should concern this code and its runtime. Please do not send code, diffs,
or excerpts you do not have the rights to share.
