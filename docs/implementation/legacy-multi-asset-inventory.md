# Legacy multi-asset inventory

Date: 2026-08-07

## Maintained product boundary

The maintained Binance research path has exactly one ordered instrument:

```text
symbols       = ("BTCUSDT",)
action shape  = (1,)
action name   = target_weight:BTCUSDT
```

One model, checkpoint and evidence chain belongs to one instrument. Capital
allocation across independently trained runs is outside `trade_rl`.

## Classification

| Path or concept | Classification | Reason |
|---|---|---|
| `examples/binance-multitimeframe/full_research_pipeline.py` | `REWRITE_SINGLE` | Maintained entrypoint and BTC-only facade. |
| `examples/binance-multitimeframe/run_full_research_state.py` | `REWRITE_SINGLE` | Maintained state runner; derives action width from the dataset. |
| Maintained `training-full` and target-weight profiles | `REWRITE_SINGLE` | New runs use one target-weight action. |
| `full_research_pipeline_legacy.py` | `LEGACY_READER_KEEP` | Preserves prior triplet/disjoint research helpers and historical evidence interpretation. It is not a maintained training entrypoint. |
| Symbol-disjoint and symbol-triplet workflow modules | `LEGACY_READER_KEEP` | Required to inspect prior experiment manifests. No maintained call site activates them. |
| Generic `MarketDataset` multi-symbol arrays | `LEGACY_READER_KEEP` | Needed for historical artifacts and generic library compatibility. The maintained entrypoint enforces one symbol instead of narrowing the data type. |
| Three-symbol PostgreSQL assembly | `LEGACY_READER_KEEP` | Preserved for historical fixture compatibility; one-symbol assembly is now also supported. Two-symbol maintained datasets remain rejected. |
| Cross-asset Transformer implementation | `LEGACY_READER_KEEP` | Still required for historical three-symbol policy identities. The maintained one-symbol encoder bypasses its computation. |
| Three-action checkpoints and serving bundles | `LEGACY_READER_KEEP` | Read-only compatibility and provenance. Resume or transfer into a one-action policy is rejected by architecture and asset-binding identity. |
| Triplet CLI controls in maintained state runner | `DELETE` | Removed from the maintained entrypoint. |
| Fixed `action_size=3` observation construction | `DELETE` | Replaced by `dataset.n_symbols`. |
| Fixed three-symbol maintained data sync | `DELETE` | Maintained sync now requests BTCUSDT only. |

## Invariants

- Historical artifacts are not rewritten into a one-symbol identity.
- Generic data structures remain capable of reading historical multi-symbol data.
- New maintained configs must use `target_weight` with `target_weight_count=1`.
- A three-symbol architecture or asset-binding digest cannot satisfy a one-symbol expected identity.
- The maintained sequence encoder retains four timeframe contexts but does not execute cross-asset fusion.
- Production status remains `NO-GO`.

## Deferred cleanup

Physical deletion of legacy modules is intentionally deferred. A later cleanup
must first prove, from the then-current `main`, that each candidate is not needed
for artifact reading, experiment comparison, provenance, or migration tests.
