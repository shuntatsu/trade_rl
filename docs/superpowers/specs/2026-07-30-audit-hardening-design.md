# Post-audit hardening design

## Scope

This change hardens the contracts identified by the July 30, 2026 repository audit without a broad architectural rewrite. It addresses execution-promotion evidence, unsupported isolated-margin semantics, tail-slippage validation, execution-cost identity, artifact publication failure handling, and checkpoint/replay deserialization boundaries.

## Design decisions

### Execution promotion evidence

Promotion evidence is fail closed. A promotable artifact must describe at least one order event, use complete order evidence, remain dataset-bound, and match the full execution configuration digest. The evidence schema advances to `execution_promotion_evidence_v2`, so legacy artifacts are rejected rather than silently reinterpreted.

### Margin semantics

Multi-asset `margin_mode="isolated"` is rejected until a per-symbol collateral ledger exists. Single-asset isolated execution remains accepted because no cross-symbol collateral transfer is possible in that case.

### Tail slippage

A non-zero tail-event probability requires `tail_slippage_multiplier >= 1.0`; enabling tail stress may not improve fills. Zero-probability configurations may keep a zero multiplier for the canonical zero-cost configuration.

### Execution identity

`execution_policy_digest` advances to the `execution_policy_v2` payload and binds both order mechanics and economic settings: fees, spread, impact, stochastic and tail slippage, financing, margin, lot/tick/notional constraints, multiplier, and random seed. A fee or risk-setting change therefore invalidates previously generated execution evidence.

### Artifact publication

Temporary pointer files use process-unique names and exclusive creation, so concurrent writers do not share a fixed `.tmp` path. If the atomic `latest.json` pointer update fails after the run directory is published, the run is rolled back to staging. Automatically generated run IDs include microseconds to eliminate the former same-second collision class.

The `latest.json` pointer remains intentionally last-writer-wins for distinct successfully published runs; it is an atomic convenience pointer, not a transaction log.

### Trusted checkpoint boundary

Checkpoint manifests, checkpoint policies, replay manifests, and replay buffers must be regular non-symlink files. Hash verification and unsafe SB3/pickle deserialization are separated by copying the opened, verified source into a private temporary directory and loading only that copy. This closes the previous verify-path-then-reopen TOCTOU window.

These checks provide integrity and filesystem-race protection. They do not turn an untrusted pickle or SB3 archive into a safe format: checkpoint and replay artifacts must still originate from the repository's trusted local/attested training workflow.

## Compatibility

These are intentional fail-closed contract changes. Legacy execution-promotion evidence and multi-asset isolated-margin configurations are rejected. Cross margin, single-asset isolated execution, zero-cost tests, normal checkpoint resume, and cross-triplet transfer remain supported.

## Verification

Regression tests first reproduced the audited failures. Targeted simulation, promotion, artifact-store, checkpoint, resume, and transfer suites then passed after the implementation. The pull request must also pass the repository's full Linux/Windows CI, Ruff, Mypy, Import Linter, dead-code, coverage, serving, workflow-security, PostgreSQL, and training-image checks before merge.
