# GPU Performance Comparison Plan

**Goal:** Measure the verified H1 baseline against the current H2-H4 runtime on the same self-hosted NVIDIA GPU, with identical deterministic smoke data and requested timesteps, and publish a canonical comparison artifact without assuming that the candidate is faster.

## Constraints

- Keep production `NO-GO`.
- Use the verified H1 implementation head `1f597caf85fe5200fe7abc34461236b65ebb8b1d` as the default historical baseline.
- Run baseline and candidate in separate exact-ref checkouts and separate uv environments.
- Preserve the H1-compatible runtime for the baseline.
- Enable `torch.compile`, pinned non-blocking sequence transfer, and compact subprocess sequence environments only for the candidate accelerated profile.
- Require CUDA allocator and throughput evidence from every sample.
- Use medians across repeated samples; do not enforce a positive speedup threshold before target-GPU evidence exists.
- Bind the comparison report with a content digest.

## Tasks

1. Extend the maintained GPU smoke with explicit `compatibility` and `accelerated` runtime profiles and record the selected profile and git commit in schema v7.
2. Add a comparison utility that validates v6/v7 smoke artifacts, enforces matched workload identity, aggregates medians, computes candidate-to-baseline ratios, and writes canonical JSON.
3. Add a self-hosted workflow that checks out the H1 baseline and current main separately, runs repeated samples, compares them, and uploads the complete evidence directory.
4. Add unit and workflow-contract tests, then run focused and repository-wide CI.
5. After a real RTX 4070 Ti SUPER run, add a verification record with observed throughput, phase timings, and memory ratios. Until then, make no numerical speedup claim.
