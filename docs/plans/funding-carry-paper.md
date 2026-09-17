# Prospective carry implementation plan

Status: Active

- [x] Complete and independently reconcile all twelve fixed development replays.
- [x] Preserve the tested source; isolate this work from completed study artifacts.
- [x] Verify current official public-data endpoints with a read-only probe.
- [x] Implement and test immutable fresh quote/funding evidence capture.
- [x] Verify real capture, failure behavior and permanent repository CI.
- [x] Implement `simulation/depth.py` and canonical fill/valuation separation.
  Test two-level weighted fills, aggregated sub-lot depth, asymmetric capacity,
  one-time fees, clock ordering, rejected rules and unchanged historical fills.
- [x] Independently review depth execution and run the full repository CI.
- [x] Add a verified offline forward-snapshot reader; test raw/summary/timing
  tampering and freshness at consumption, without changing historical artifacts.
- [x] Capture and revalidate current exchange rules; test market lot intersections,
  disabled zero steps, notional flags, unsupported symbols and metadata age.
- [ ] Specify and test restart-safe prospective paper execution on the canonical ledger.
  - [x] Implement a protocol-bound SQLite event chain with compare-and-append,
    duplicate-command idempotency, concurrent-writer and process-interruption tests.
  - [ ] Compose saved decisions, verified later captures, depth fills and funding
    settlement events into replayable canonical account transitions.
- [ ] Freeze forward evaluation duration/gates before starting paper positions.
- [ ] Run prospective paper observation and report the complete declared gate.

No successful probe, past development profit or partial paper record completes
the user's goal of a reliably profit-seeking operational bot.
