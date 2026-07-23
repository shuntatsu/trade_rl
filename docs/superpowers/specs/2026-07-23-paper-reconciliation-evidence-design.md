# Paper Reconciliation Evidence Design

## Purpose

Close the remaining paper-trading reconciliation evidence gap without adding direct exchange connectivity. The repository must be able to verify a normalized external paper-trading reconciliation report before a selected-final run can be packaged for Serving.

## Problem

`FreshConfirmationEvidence` binds `order_log_digest`, `fill_log_digest`, and `reconciliation_digest`, but the referenced reconciliation report has no maintained schema or machine-evaluated promotion contract. A confirmation signer can therefore attest a digest, while repository code cannot establish what the report measured or whether its accounting reconciliation passed.

Production remains `NO-GO`; this change strengthens evidence integrity only.

## Design

Add `trade_rl.evaluation.paper_reconciliation` with a content-addressed `PaperReconciliationEvidence` artifact.

The artifact binds:

- dataset, environment, policy, and selected-final training-run identities;
- confirmation collection start/end times;
- order-log and fill-log digests;
- submitted and terminal order counts;
- observed and matched fill counts;
- unknown-order fills, duplicate fills, and open orders;
- maximum relative position-notional, cash, and equity differences;
- the exact tolerances used to evaluate those differences;
- a derived `passed` value, sealing flag, schema version, and artifact digest.

`passed` is recomputed from the retained observations. It requires complete terminal order coverage, complete fill matching, zero unknown/duplicate/open-order counts, and every observed accounting difference to remain inside its declared tolerance.

Release promotion applies an additional policy boundary. Declared tolerances may not exceed `1e-6` for position-notional, cash, or equity differences. This prevents an externally generated artifact from declaring arbitrarily permissive tolerances and still becoming release evidence.

## Serving-package integration

`package_selected_training_run()` resolves a paper-reconciliation artifact from an explicit optional path or, by default, `paper-reconciliation.json` beside the confirmation file. Packaging rejects missing, malformed, failed, or over-tolerant reconciliation evidence.

The reconciliation artifact must match the confirmation and selected-final training identities exactly:

- reconciliation digest equals `FreshConfirmationEvidence.reconciliation_digest`;
- order/fill log digests equal the confirmation values;
- dataset, environment, policy, and training-run digests match;
- start/end times equal the confirmation interval.

The verified artifact is copied into the immutable serving bundle and participates in bundle file closure. The existing confirmation evidence digest continues to bind its reconciliation digest, so no serving-manifest schema bump is required for this first increment.

## Non-goals

- No exchange websocket or broker API.
- No order submission, cancellation, replacement, or venue reconciliation adapter.
- No claim that paper fills equal live fills.
- No production authorization or profitability claim.
- No CLI or full-research state-machine argument change in this PR; explicit orchestration wiring is a following independent PR.

## Verification

Use TDD:

1. Add failing unit tests for artifact construction, round-trip loading, tamper rejection, and promotion-policy rejection.
2. Implement the minimal evidence module.
3. Add failing Serving-package tests for missing, mismatched, and failed reconciliation evidence.
4. Wire the package boundary and verify focused tests.
5. Run Ruff, formatter, MyPy, the complete Python suite, Studio checks, import-linter, and normal exact-head GitHub Actions before merge.
