# Multi-Timeframe Full Research

The maintained full-research workflow is phase-separated and fail-closed. It does not train a selected-final policy in the same process that chooses or approves it.

## Phases

1. `develop` builds the dataset twice, verifies deterministic identity, runs sealed walk-forward research, validates the complete walk-forward artifact directory, evaluates the research and execution-sensitivity gates, normalizes the selected recipe, and writes an immutable `selection-proposal.json`. A successful result is `awaiting_selection_authorization` with exit code 0.
2. `train-selected` requires the exact proposal, an externally signed Ed25519 authorization, and a read-only public-key store. Authorization is verified before normalizer fitting or model construction. Selected-final training forbids resume checkpoints and writes `training_run_v3` with the complete selection chain. A successful result is `awaiting_fresh_confirmation` with exit code 0.
3. `finalize` requires Ed25519-signed fresh confirmation whose declared boundary exactly equals the selected-final completion time, begins no earlier than that boundary, covers at least 30 days, and does not extend beyond trusted current time. It writes final gate state without mutating earlier evidence.

Research rejection returns exit code 2. Infrastructure, integrity, or identity failure returns exit code 3. Waiting for an external approval is not an error.

## Commands

```bash
uv run python examples/binance-multitimeframe/run_full_research_hardened.py \
  --phase develop \
  --work-root var/binance-multitimeframe-full

uv run python examples/binance-multitimeframe/run_full_research_hardened.py \
  --phase train-selected \
  --work-root var/binance-multitimeframe-full \
  --selection-authorization /secure/selection-authorization.json \
  --selection-public-keys /secure/selection-public-keys.json

uv run python examples/binance-multitimeframe/run_full_research_hardened.py \
  --phase finalize \
  --work-root var/binance-multitimeframe-full \
  --confirmation /secure/fresh-confirmation.json \
  --confirmation-public-keys /secure/confirmation-public-keys.json \
  --trusted-now 2026-08-18T03:00:00Z
```

`run_full_research.py` is a compatibility launcher to the same state machine; it is not an independent implementation. The maintained code never loads another script through `runpy`.

## Metadata modes

- `frozen_snapshot` preserves one official current Binance payload byte-for-byte and clearly marks it unauthenticated and non-point-in-time.
- `historical_signed` accepts only the v4 rule-history schema with explicit ordered `symbol_order`, exact market and coverage, issue time, source URI, complete effective-dated rules, and an Ed25519 envelope verified against a read-only public-key store. The original signed document is retained as run evidence.
- `conservative_static` accepts only an explicitly versioned static payload and remains approximation evidence.

Current metadata is never silently projected backwards as historical truth.

## Deployment status

All artifacts remain research-only. Direct exchange routing and live capital deployment are `NO-GO`.
