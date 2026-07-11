# Deployment evidence bundle

Canary and Production promotion consume an immutable GitHub Actions artifact named `deployment-evidence`. Boolean self-attestation is not accepted.

## Required files

```text
deployment-evidence/
├── candidate.json
├── model.zip                 # path is declared by candidate.json
├── shadow.json
├── drift.json
├── incident.json
└── canary.json               # Production only
```

`candidate.json` records the exact model version, Git commit and SHA-256 digests of the model and every evidence report. The deployment gate recalculates every digest, rejects path traversal, and verifies that every report identifies the same model artifact and Git commit.

## Identity fields

Every Shadow, Drift, Incident and Canary report must contain:

```json
{
  "model_version": "1.0.0",
  "git_commit": "40-character SHA-1",
  "artifact_sha256": "64-character SHA-256"
}
```

The Canary report must also contain `parent_shadow_run_id`, which must exactly match the verified Shadow `run_id`.

## Promotion workflow

1. A validation workflow uploads the complete bundle as a GitHub Actions artifact.
2. An operator starts `Deployment Gate` and supplies the source workflow run ID.
3. The workflow downloads the immutable artifact with `gh run download`.
4. `mars_lite.server.deployment_gate` verifies file digests, identity linkage, thresholds, incident state and stage ordering.
5. Production additionally requires a `PROD-<digits>` ticket and approval through the GitHub `production` Environment.
