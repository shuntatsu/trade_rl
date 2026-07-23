# Live Training Status Sync Design

## Purpose

Synchronize the maintained research-status document with the Live Training environment and episode-isolation implementation already merged through PR #85 and PR #103.

## Problem

`docs/RESEARCH_STATUS.md` still reports `TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY_WITH_STREAM_ISOLATION_GAP` and describes the missing environment/episode selector as a current limitation. The maintained implementation and the 2026-07-23 architecture closeout establish that one selected vector environment and its current episode drive the replay, and that producer-issued nullable `episode_id` values close the remaining episode-boundary ambiguity while preserving historical telemetry compatibility.

This is a current-state documentation regression. Historical audit files remain correct records of what was observed at their audited commits and must not be rewritten.

## Design

1. Update the capability date and Trade RL Studio capability token in `docs/RESEARCH_STATUS.md`.
2. Replace the obsolete limitation paragraph with the current contract:
   - one selected vector environment drives the displayed replay;
   - the browser displays the selected environment's current episode;
   - producer-issued nullable `episode_id` is preferred;
   - historical null-ID records retain terminal/counter-rollback fallback;
   - telemetry remains diagnostic and excluded from fitting, selection, sealed evaluation, promotion, release, Serving activation, and order routing.
3. Extend `tests/test_current_documentation_contract.py` with an explicit regression contract that rejects the stale status token and limitation phrases and requires the current episode-isolation markers.
4. Keep production status `NO-GO`, direct exchange routing `NOT_IMPLEMENTED`, and profitability claim `NONE` unchanged.

## Non-goals

- No Python, Studio, training, evaluation, Serving, release, or execution behavior changes.
- No alteration of historical verification or audit records.
- No claim of production readiness, profitability, or exchange-equivalent fills.

## Verification

- Run the new focused documentation contract first and confirm it fails before the documentation update.
- Run the focused documentation contract after the update.
- Run the complete `tests/test_current_documentation_contract.py` file.
- Require normal exact-head GitHub Actions CI before merge.
