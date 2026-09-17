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
- [ ] Independently review depth execution and run the full repository CI.
- [ ] Specify and test restart-safe prospective paper execution on the canonical ledger.
- [ ] Freeze forward evaluation duration/gates before starting paper positions.
- [ ] Run prospective paper observation and report the complete declared gate.

No successful probe, past development profit or partial paper record completes
the user's goal of a reliably profit-seeking operational bot.
