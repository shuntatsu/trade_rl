# Incident Response Runbook

## Scope

This runbook covers Shadow, Canary, and Production incidents for `mars_lite`
model serving and evaluation workflows.

## Severity Levels

| Severity | Trigger | Response target |
| --- | --- | --- |
| SEV1 | Production order safety breach or data corruption | Immediate stop and rollback |
| SEV2 | Canary drift, degraded Sharpe, or failed guardrail | Pause promotion within 30 minutes |
| SEV3 | Shadow-only metric regression | Triage before next promotion |

## First Actions

1. Freeze the current deployment stage.
2. Capture active model version, registry history, metrics, and drift report.
3. If the active stage is Production, run model rollback and move execution to flat
   exposure until the deployment gate is cleared again.
4. Open an incident record with owner, start time, affected symbols, and evidence.

## Recovery Criteria

Production can resume only after Shadow and Canary evidence pass again, the model
decision log is updated, and the approval ticket is recorded.
