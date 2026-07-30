## Summary

- define the controlled Gate-versus-direct target-weight actor ablation
- add failing contract tests before production implementation
- preserve target-weight, PPO, reward, risk, execution, and serving boundaries

## Current state

This PR is intentionally RED. The first implementation checkpoint verifies that the new tests fail for the missing direct-head configuration, shared action-stage API, policy identity v4, causal direct BC gate, common telemetry, and paired experiment profiles.

## Safety

Direct exchange routing remains NO-GO and is outside this PR.