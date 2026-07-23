# Paper Reconciliation CLI Design

## Purpose

Make the verified `paper_reconciliation_evidence_v1` contract operationally producible from normalized external paper-run measurements and make its path explicit at the Serving packaging boundary.

Production remains `NO-GO`. This work does not add a paper broker, exchange connectivity, credentials, order routing, or a claim that supplied measurements came from a real maintained paper run.

## Alternatives considered

### Raw order/fill log ingestion and reconciliation

The CLI could ingest venue-specific raw logs and compute all matches itself. This is premature because no maintained paper venue or canonical raw-log schema is selected. It would couple the repository to an invented adapter contract.

### Serving path argument only

The CLI could expose `--paper-reconciliation` without creating the artifact. This makes packaging clearer but leaves operators without a maintained way to produce the required artifact.

### Strict normalized request plus explicit package path — selected

Add a strict JSON request whose fields exactly cover `PaperReconciliationEvidence.create(...)`. The command validates field closure and types, derives pass/fail through the existing domain contract, writes the immutable artifact, and returns one machine-readable result. A separate explicit Serving option passes the artifact path to the already fail-closed package boundary.

## Command contract

```text
trade-rl reconciliation create \
  --request /secure/paper-reconciliation-request.json \
  --output /secure/paper-reconciliation.json
```

The request schema is `paper_reconciliation_request_v1`. It contains:

- dataset, environment, policy, and training-run digests;
- start, end, and creation timestamps;
- order and fill log digests;
- submitted/terminal order counts;
- observed/matched fill counts;
- unknown-order fill, duplicate-fill, and open-order counts;
- maximum position-notional, cash, and equity difference fractions;
- the three declared tolerance fractions.

The request does not contain `passed`, `sealed`, `schema_version` for the artifact, or an evidence digest. Those values are derived by the artifact implementation.

The command writes immutable `paper_reconciliation_evidence_v1` and returns:

```json
{
  "artifact_path": "...",
  "evidence_digest": "...",
  "passed": true,
  "production_status": "NO-GO",
  "schema": "paper_reconciliation_creation_result_v1",
  "status": "sealed_for_fresh_confirmation_review"
}
```

A failed reconciliation report is still valid evidence and may be written with `passed=false`; later promotion rejects it. Malformed requests fail before artifact publication.

## Serving package contract

`trade-rl serving package` gains required `--paper-reconciliation`. The CLI passes this explicit path into `package_selected_training_run()` rather than relying on sibling-file discovery. The Python API retains its sibling fallback for compatibility and direct library callers.

## Security and trust boundary

The reconciliation artifact is content-addressed but unsigned. Its digest must subsequently be included in Ed25519-signed fresh confirmation. Therefore the required operational order is:

```text
normalized external measurements
  -> reconciliation create
  -> confirmation request references reconciliation digest
  -> confirmation create signs the complete evidence identity
  -> serving package verifies both artifacts and their chronology
```

Private signing keys remain used only by the existing confirmation command. The reconciliation command loads no private keys.

## Error handling

- Reject non-object JSON, missing/extra fields, unsupported request schema, booleans in numeric fields, invalid datetimes, invalid digests, negative counts, invalid fractions, and immutable-output conflicts.
- Emit the standard machine-readable `production_status: NO-GO` error payload.
- Do not write partial artifacts on validation failure.

## Testing

Use TDD to cover a valid passing report, a valid failed report, malformed field closure, immutable output, machine-readable error output, and explicit Serving path forwarding. Then run the complete exact-head CI and PostgreSQL workflows before squash merge.
