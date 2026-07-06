# Compliance Runbook

## Required Evidence

Each model promotion must preserve:

- Model registry version and active version.
- Training manifest with git commit, data hash, hyperparameters, and seed.
- Drift monitor report for the promotion window.
- Deployment gate evidence for Shadow, Canary, and Production.
- Human approval ticket for Production.

## Jurisdiction Checklist

| Area | Evidence |
| --- | --- |
| Data provenance | Source, time range, data hash |
| Model reproducibility | Manifest and registry version |
| Risk control | Pre-trade and deployment gate reports |
| Auditability | Model decision log entry |

## Retention

Keep promotion evidence, incident records, and decision logs for at least seven
years unless a stricter venue or jurisdiction policy applies.
