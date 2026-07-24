# Security audit false positives

Documented dismissals for adversarial-audit (K2.7/K3) findings that are not actionable bugs in this repo's threat model.

## GPU operator trust boundary

The RunPod worker runs with operator-configured environment and filesystem layout. Findings that assume an external attacker can set `VIVIJURE_AITOOLKIT_PYTHON`, place files under `DEFAULT_WAN_BASE_PATH`, or modify the container image are **out of scope**: only the deployer controls those surfaces.

## Subprocess list invocation (wan_lora_train)

`_run_aitoolkit` uses `subprocess.Popen` with a argv list (no shell). Config paths are built from slugged `project` / workdir segments server-side. Shell-metacharacters in paths are passed as literal filename arguments, not interpreted by a shell.

## Cast registry LoRA reads

Cast-banked adapters live under global `loras/cast-{id}/…` and `loras/lora-{slug}-…/…` shapes outside the render project slug. The control plane resolves cast bindings (`resolveCastLoras`) and passes keys as `pretrained_loras`; cross-cast reads within the shared bucket are intentional studio behavior, not a tenancy bypass. `check_scoped_job_key` always runs `check_job_key` first (rejects `..`, absolute paths, wrong prefixes); the cast-registry early return applies only after those checks pass.

## Record

| Date | Audit | Finding | Rationale |
| --- | --- | --- | --- |
| 2026-07-23 | K3 repo | Cross-project R2 keys | **Hardened** -- bundle_key + scoped read keys bound to project slug (#fix/kf3-r2-tenant-binding) |
| 2026-07-23 | K2.7 PR #319 | aitoolkit_python env injection | Operator-controlled GPU env |
| 2026-07-23 | K2.7 PR #319 | config_path subprocess injection | List-form Popen, no shell |
| 2026-07-23 | K3 repo | Cross-project isolation via key-prefix checks | **Architecture** -- single R2 token + `check_scoped_job_key()` at every read path; documented in `harness/keys.py` |
| 2026-07-23 | K3 repo | Model-divergence guard swallows exceptions | **Accepted** -- guard is best-effort telemetry; job proceeds with loaded weights (operator monitors @event) |
| 2026-07-23 | K3 repo | fp8 runtime placeholder digest | **Build-time** -- `release.yml` passes real digest; placeholder only when unset locally |
| 2026-07-23 | K3 repo | RunPod API error bodies echo secrets | **Operator trust** -- `deploy.sh` runs on operator workstation with their creds |
| 2026-07-23 | K3 verify ~18:04 | curl argv exposes RunPod API key | Operator deploy.sh; process-local only |
| 2026-07-23 | K3 verify ~18:04 | bundle_key hyphen-suffix cross-project | Prefix bind via check_scoped_job_key; hyphen is slug canonicalization |
| 2026-07-23 | K3 verify ~18:04 | Job-controlled HF repo in allowlisted namespaces | Operator GPU worker; namespace allowlist is intentional |
| 2026-07-23 | K3 verify ~18:04 | Unauthenticated health GPU inventory | RunPod health probe; no tenant-facing surface |
| 2026-07-23 | K2.7 PR #332 | Path traversal via cast-banked LoRA key | **Not bypassed** -- `check_job_key` rejects `..` before cast early return |
| 2026-07-23 | K2.7 PR #332 | Cross-project cast LoRA reads via slug prefix | **By design** -- cast registry keys; control plane binds cast to render |
| 2026-07-23 | K2.7 PR #330 | runpod-verify workflow_dispatch command injection | Pre-existing operator workflow; not introduced by #330 bundle-key diff |
| 2026-07-23 | K2.7 PR #330 | Hard-coded api_key in promote test | Pre-existing test fixture; fake operator-scoped key, not repo secret |
