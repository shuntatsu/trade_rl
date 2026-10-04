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
- [x] Specify and test restart-safe prospective paper execution on the canonical ledger.
  - [x] Implement a protocol-bound SQLite event chain with compare-and-append,
    duplicate-command idempotency, concurrent-writer and process-interruption tests.
  - [x] Compose saved decisions, verified later captures, depth fills and funding
    settlement events into replayable canonical account transitions.
  - [x] Independently review account composition and pass the full repository CI.
- [x] Add source/runtime-bound study sealing, supervised public collection and
  operational restart/report commands; verify software against real source data.
  - [x] Bind source/runtime identity before collection, check it before every
    command, and preserve a durable failure if collection or identity fails.
  - [x] Reuse fresh rules, persist each capture before a command, resume pending
    decisions only inside their ten-second window, and supervise one collector.
  - [x] Independently review the collector, verify full CI and exercise real inputs.
  - [x] Expose a cadence CLI and report command with the separately frozen gate.
  - [x] Verify the expanded, versioned 100-level source profile and full cadence
    against real public inputs; preserve the rejected 20-level software probe.
- [x] Freeze forward evaluation duration/gates before starting paper positions.
- [x] Bind paper attempt lineage into each sealed protocol and require an explicit
  lineage file at `seal`; preserve the failed first attempt as the predecessor.
- [x] Restrict predecessor lineage to operational dispositions/reasons so prior
  economic pass/reject labels cannot enter result-blind review.
- [x] Align screen v2 and assessment: declare all four required instrument fills,
  assign funding coverage by settlement time inside `[start, close)`, and reject
  nonzero funding settled at or after close.
- [x] Add red/green regression tests for repeat-attempt lineage, instrument-fill
  and funding-window coverage, and post-close funding qualification.
- [x] Complete fresh result-blind source review of the known predecessor and
  four-fill / funding-window alignments. The predecessor chain is verified
  within the inspected workspace; global attempt uniqueness and a full
  source-to-ledger oracle remain unestablished.
- [ ] Run prospective paper observation and report the complete declared gate.

No successful probe, past development profit or partial paper record completes
the user's goal of a reliably profit-seeking operational bot.
