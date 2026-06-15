# Security policy

## Supported versions

This is a rolling, single-`main`-branch project released as `backend-vX.Y.Z` tags. Only
the latest release receives security fixes. If you are running an older revision, upgrade
to the newest tag to pick them up.

## Reporting a vulnerability

Please do not file a public GitHub issue for security problems. Instead, report it
privately through GitHub's private vulnerability reporting: open the repository's
**Security** tab and click **"Report a vulnerability"**. This creates a private advisory
visible only to you and the maintainers. (If that option is not visible, open a minimal
public issue asking for a private channel, without disclosing any details.)

Please include:

- A description of the issue
- Steps to reproduce, including a minimal example if possible
- The affected version (tag or commit SHA if known)
- Any suggestions for remediation

Reports will be acknowledged within a reasonable window (target: 5 business days).
Time-sensitive issues should say so. Please allow up to 90 days for a coordinated fix
before public disclosure.

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
- Generated artifacts are stamped with the requesting `user_email` so the control plane's
  `/api/artifact` ownership check can gate them.

In-scope vulnerabilities include:

- Escapes from the bundle reader (e.g. path traversal via `tar`/zip entries or crafted
  keys) that read or write outside the intended job workspace or bucket prefix.
- Server-side request forgery or arbitrary object access via attacker-influenced keys.
- Code execution via crafted bundle contents (`storyboard.yaml`, registry, refs).
- Leakage of the R2 credential, or writing artifacts under another user's ownership stamp.
- Injection issues in any shell-out (ffmpeg, model tooling) driven by job input.

Out-of-scope:

- Issues that require an already-compromised control plane or already-leaked R2 token.
- Denial of service from intentionally expensive but well-formed render jobs (render cost
  is the operator's concern; submit access is gated by the control plane).
- The security posture of the upstream model weights or third-party libraries themselves
  (report those to their projects), beyond how this backend invokes them.

## A note on provenance

This backend is a clean-room reimplementation written from the control-plane API contract
and the underlying models' own documentation. Security reports should concern this code
and its runtime; please do not send code, diffs, or excerpts from any other render
pipeline.
