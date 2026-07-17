# Binance Metadata Modes Design

## Understanding summary

- The maintained 226-feature Binance research runner must no longer deadlock when an authoritative historical execution-rule collector does not exist.
- `historical_signed` remains the highest-integrity mode and keeps the current HMAC and effective-dated coverage checks unchanged.
- `frozen_snapshot` captures one current USD-M `exchangeInfo` response, preserves its raw bytes, URI, UTC retrieval time, and SHA-256 digest, and applies the selected rules statically as an explicitly non-point-in-time approximation.
- `conservative_static` evaluates fixed execution rules under predeclared stricter scenarios; it never claims that those rules are historical observations.
- All modes bind mode, evidence digest, source, as-of time, coverage, and policy version into dataset identity and publish the same facts in run evidence and summary output.
- Binance Vision market ZIPs and their existing Docker-volume cache remain the price, volume, trade, and funding source. Exchange metadata remains a separate evidence stream.
- The workflow remains research-only and production `NO-GO`; no mode authorizes live trading.

## Assumptions and non-functional requirements

- The selected symbols remain `BTCUSDT`, `ETHUSDT`, and `BNBUSDT`, with research coverage `2024-12-01T00:00:00Z` through `2026-07-01T00:00:00Z`.
- Frozen snapshot values are positive and all selected contracts must be `TRADING` at capture time.
- Raw snapshot bytes are immutable evidence and must be written atomically before dataset publication.
- Dataset A and B reuse one in-memory snapshot resolution so repeated builds cannot observe different live metadata.
- Metadata evidence changes must change `dataset_id`, even when resolved numeric execution arrays are equal.
- HMAC keys stay runtime-only; frozen snapshots do not need a project HMAC because their raw SHA-256 and immutable artifact bind the received payload without claiming Binance attestation.
- The existing four CUDA rollout environments and approximately 7.1M-parameter sequence policy remain unchanged.
- Conservative sensitivity must replay the closed loop with the selected policy and baseline; it must not post-process returns or influence recipe selection.

## Approaches considered

1. **Recommended: explicit evidence modes with identity-bound provenance.** Preserve strict history, add an immutable raw snapshot, and add versioned conservative replay. This is transparent, reproducible, and unlocks research without fabricating history.
2. Self-sign a synthetic historical payload. Rejected because HMAC proves only internal integrity and would misrepresent current values as historical evidence.
3. Remove execution metadata checks. Rejected because it would discard useful order-grid and minimum-notional realism and weaken artifact reproducibility.

## Architecture and data flow

`BinancePublicTransport` gains a snapshot operation that fetches the official exchange-information URI exactly once and returns raw bytes, parsed payload, URI, retrieval time, and raw SHA-256. The existing compatibility loader delegates to this operation.

The full runner resolves one `MetadataEvidence` before either dataset build:

- `historical_signed`: verify the existing envelope, parse effective-dated histories, and report authenticated point-in-time evidence.
- `frozen_snapshot`: capture raw current exchange information, extract the selected symbols, create static contracts without histories, and report a non-point-in-time limitation.
- `conservative_static`: load explicit static evidence or derive a declared stress scenario from a frozen snapshot, record the policy version and factors, and never label it authenticated or point-in-time.

The builder receives a canonical metadata-evidence payload. It places this payload in `identity_payload_json` before computing `dataset_id`; no `MarketDataset` storage field or artifact migration is required. Resolved tick-size, lot-size, and minimum-notional arrays remain independently identity-bound.

The runner writes `exchange-info.raw.json` byte-for-byte for snapshots and writes `exchange-info.json` for canonical evidence. Summary output repeats mode, digest, source, as-of, coverage, authentication, point-in-time status, policy version, and limitations.

## Conservative sensitivity

The first required scenario pack is nominal 1x, individual tick/lot/minimum-notional 2x, joint 2x, and joint 5x. Joint 2x is the required robustness gate; joint 5x is report-only until calibrated. Replays use the selected policy and baseline deterministically on each OOS fold with the same normalizers. Execution effects feed subsequent observations, so no return-only adjustment is allowed. Scenario access is declared before the sealed test and recorded in the access ledger. Sensitivity never participates in model selection.

## Error handling and security

- Unknown modes, missing symbols, non-trading contracts, missing filters, invalid JSON, naive capture times, nonpositive rules, digest mismatches, and partial static metadata fail closed.
- `historical_signed` continues to reject unknown HMAC key IDs, altered payloads, and incomplete time coverage.
- `frozen_snapshot` never reads `TRADE_RL_METADATA_KEYS` and never emits `authenticated=true` or `point_in_time=true`.
- An existing run generation or snapshot output is never overwritten.

## Decision log

| Decision | Alternatives | Reason |
|---|---|---|
| Keep three explicit modes | One permissive fallback | Prevents silent integrity downgrade. |
| Default this Docker research run to `frozen_snapshot` | Keep strict default | The user explicitly selected the practical approximation path; the report remains `NO-GO`. |
| Bind evidence in identity metadata | Add fields to `MarketDataset` | Avoids artifact-schema migration while making provenance affect `dataset_id`. |
| Preserve raw response bytes | Store reserialized JSON only | Exact bytes and digest are stronger reproducibility evidence. |
| Replay sensitivity closed-loop | Adjust returns after evaluation | Execution outcomes affect later observations and positions. |
| Require joint 2x, report joint 5x | Require every tail scenario | Provides a meaningful initial gate without an uncalibrated extreme veto. |

