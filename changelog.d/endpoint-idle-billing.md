**Docs: separate the idle timeout from standby in the billing note.**

A worker bills through the idle timeout after a job, by RunPod's rule. A scaled-down worker does not. `docs/runpod-endpoint-config.md` cites that rule, and the hourly buckets for 18:00 to 20:00Z on 2026-10-30 as evidence on standby only. The image line reads 1.1.3.
