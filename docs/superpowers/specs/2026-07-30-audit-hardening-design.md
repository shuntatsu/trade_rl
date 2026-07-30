# Post-audit hardening design

## Scope

This change hardens the contracts identified by the July 30, 2026 repository audit without broad architectural rewrites. It addresses execution-promotion evidence, unsupported isolated-margin semantics, tail-slippage validation, execution-cost identity, artifact publication concurrency, and trusted checkpoint loading boundaries.

## Design decisions

### Execution promotion evidence

Promotion evidence becomes fail-closed. A promotable artifact must describe at least one order event, bind both execution mechanics and complete economic cost configuration, and remain dataset-bound. `complete_order_evidence` remains for backward-readable diagnostics but is no longer sufficient by itself. The evidence schema is advanced so legacy artifacts are rejected rather than silently reinterpreted.

### Margin semantics

`margin_mode="isolated"` is rejected until a per-symbol collateral ledger exists. The current proportional allocation of account-wide collateral is removed from the supported contract because it is not isolated margin.

### Tail slippage

A non-zero tail-event probability requires `tail_slippage_multiplier >= 1.0`; tail stress may not improve fills. Zero-probability configurations may keep a zero multiplier for the canonical zero-cost configuration.

### Execution identity

The existing policy digest continues to identify order mechanics. A separate economics digest binds fee, spread, impact, slippage, financing, margin, lot/tick/notional, and multiplier settings. Promotion evidence requires both digests.

### Artifact publication

Temporary files use process-unique names and exclusive creation. Run identifiers include microseconds and a random suffix. Publication uses a store-scoped lock and refuses concurrent mutation rather than sharing a fixed temporary filename.

### Trusted checkpoint boundary

Checkpoint manifests and referenced payloads must be regular, non-symlink files inside the declared artifact root. Verification and consumption operate on a private verified copy to avoid path replacement between hash verification and deserialization. Replay-buffer loading follows the same rule.

## Compatibility

These are intentional fail-closed contract changes. Legacy execution evidence and isolated-margin configurations are rejected. Cross-margin, zero-cost tests, normal checkpoint resume, and existing policy mechanics remain supported.

## Verification

Each behavior receives a regression test that fails against the audited main branch. Targeted tests run before the full pytest, Ruff, Mypy, Import Linter, and security-oriented workflow checks.