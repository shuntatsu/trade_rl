# Tier D NONRL48 ignored prototype

This folder is a result-blind, consumed-development preparation. No Task18 market costs, forecast fit, replay or P&L has been executed by this worker. The source-only mode writes the immutable recipe for review; normal technical retries create separate attempt directories and retain failures. Nothing is committed or merged.

Run commands from `C:\Users\shun\.codex\worktrees\nonrl-activity-alias-guard\trade_rl` with the local Python 3.12 environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH=(Join-Path $PWD 'output\issue810-tier-d-nonrl-preparation')
.\.venv\Scripts\python.exe -B output\issue810-tier-d-nonrl-preparation\tier_d_runner.py --mode source
.\.venv\Scripts\python.exe -B output\issue810-tier-d-nonrl-preparation\tier_d_runner.py --mode synthetic
.\.venv\Scripts\python.exe -B -m pytest output\issue810-tier-d-nonrl-preparation\test_runner.py -q -p no:cacheprovider --basetemp output\issue810-tier-d-nonrl-preparation\test-work-new
```

The artificial fixture has 256 hourly rows, two invented symbols and six invented local features. It fits H72/alpha10 once, then compares one NONRL and one CASH native account on actor `[128,160)` with eight reserve advances through168 under actual MARKET latency2. Its manufactured fit-prefix rise followed by an actor/reserve plateau exercises a delayed fill and native closing. Nonzero manufactured funding/borrow separately check that COST2 doubles execution costs only. This fixture proves software wiring; its returns are not market evidence.

The market plan fixes four H24/H72 × Ridge alpha1/10 configurations, three April/July/October vintages, BTC/ETH independent nominal USDT100000 accounts, and BASEL0/COST2L0/BASEL2. One preload, four cost producers and four public preparations precede declaration of all24 matched pairs. Forecasts share no results-dependent tuning; cost/latency treatments rerun complete native execution on separate books. Actor `[2903,9430)` and72 reserve advances end at9502. The loaded Jan1 row9503 stays outside fit labels, prediction decisions, selection and execution. The whole9504-row artifact is already consumed development; it is not claimed unused.

The new declared 24h funding coverage recipe is min2 counted events, max9h counted-bar-close gap, max8h age. Counts rather than nonzero rates define events. Funding uses declared bar-close proxies and current venue snapshots; historical settlement calibration and historical point-in-time rules are not established.

Execution uses the existing public native carrier/declaration/terminal reader/comparison APIs. Full native receipts and the complete pre-execution matrix are gzip JSON. Closed comparison validation runs per matched pair to avoid retaining48 full traces in RAM. The compact complete matrix checks account/scenario coverage and ranks BTC+ETH within one configuration/scenario using denominator200000; it does not claim a single shared portfolio. The read-only cost decomposition reconciles native interval price PnL, actual fill fee/spread/impact, funding, borrow, dividend/interest and closing costs (closing is a subset). Expected allocator costs are never debited again.

The sampled supervisor proposes RSS2GiB, disk4GiB, wall3600s and a10s monitoring interval before any actual market fit. Windows RSS includes the owned redirector and its descendants from one bounded CIM snapshot per sample. Creation-time identities retain orphaned owned descendants and exclude reused PIDs. Cleanup stops deepest owned descendants, then waits/reaps the worker. Nonzero child exit marks the attempt FAILED even without another error. These are unbenchmarked caps, not speed or memory facts. Samples can overshoot; one native arm's full capture remains in RAM because the public API is not streaming. Completion requires the supervisor result as well as the child result. A hard supervisor interruption may preserve logs/state without a valid returned native receipt. Technical failure leaves all remaining cells UNATTEMPTED, with no fabricated cash-zero observations.

ROOT's preserved R1 `market-20261014T1305Z` completed four cost producers and four public preparations/twelve Ridge vintages, then was stopped for the redirector-only RSS defect before matrix/native execution. R1 is FAILED with all48 native slots UNATTEMPTED and no replay/P&L. Its forecasts, logs, prior snapshots and ROOT's operator finalization remain historical evidence. This resource-control repair has not rerun Task18; the next attempt awaits a fresh review of the changed control source.

Actual market execution requires ROOT's different-model result-blind review of the final source hashes and protocol digest. `--review-record` is an operator reminder/provenance record, not authentication of an AI provider or data eligibility. Its JSON must contain these exact bindings:

```json
{
  "scope": "CONSUMED_2025_DEVELOPMENT_NONRL48_ONLY",
  "status": "APPROVED_FOR_TIER_D_DEVELOPMENT_EXECUTION",
  "reviewer_model": "gpt-6-astra",
  "protocol_digest": "use implementation-harness-provenance.json",
  "source_sha256": {"use every entry": "from implementation-harness-provenance.json"}
}
```

ROOT supplies an actual reviewed record after reviewing; the worker has not created an approval. The post-review command is `tier_d_runner.py --mode market --review-record <ROOT-review.json>` with the same Python/workdir. No PID, DLL, one-shot permanent retirement or old R4 SourceGate participates. Source/runtime versions and input/output hashes supply normal development reproducibility. The configured implementation model/effort record is a harness configuration, not a claim about unobservable provider weights.

Reports remain development-only with NO adoption and NO winner. Both symbols and all three stress scenarios stay visible. DD20%, terminal flatness, no pending orders and full receipt coverage remain necessary; minimum operational improvement and fixed infrastructure costs are unknown, so account results are pre-tax/before unknown operating costs. The six-fit PPO learning result remains FAILED, including fee-only seed17; Phase1 profits and sealed/unused/final/live boundaries remain unchanged.
