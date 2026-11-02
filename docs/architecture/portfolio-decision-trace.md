# Shared portfolio decision and execution trace

`SharedPortfolioGymEnv.step()` emits a JSON-compatible `decision_trace` in its
Gym `info`. `PortfolioEpisodeJournalWrapper` copies that value through canonical
JSON and persists it in each successful step event. The trace links the action
codes to the proposed target, the risk-approved target, and the native execution
events for each interval. It records the native result; it does not change
portfolio accounting, risk constraints, order behavior, rewards, or legacy
defaults.

## Versions and fields

New episode journals use `shared_portfolio_episode_journal_v2` in their manifest
and event records. The trace itself uses
`shared_portfolio_decision_trace_v1`. Existing v1 journal files are not loaded,
rewritten, or relabeled by this writer; readers should branch on the journal
schema.

The trace contains:

- `raw_action_codes` and `proposal_weights`, preserving the submitted action and
  target proposal. The existing top-level `target_weights` field retains its
  proposal meaning.
- `transitions`, one per native interval advanced by the Gym step. An actor
  interval is marked `actor_action`; reserved close intervals are marked
  `terminal_settlement`. Each transition records its start and next index, the
  proposal and pre-trade weights, the approved weights, risk reasons, and
  constraint facts.
- Each transition's `execution_observation`, copied from the same native
  execution interval. It includes canonical order events, active remainder
  quantities, and terminal order reasons appended during that interval. The
  native archive remains cumulative; interval suffixes preserve archive order
  and a new session starts from an empty archive. Native order IDs, event
  sequence, processing index, and timestamp remain attached to each event.
- `account_delta`, containing cash and equity before and after the Gym step,
  the equity change, and exact quantity strings before and after. Terminal
  settlement intervals are included in this step-level delta.

The trace uses plain dictionaries, lists, strings, booleans, integers, finite
numbers, and null values so it remains compatible with Gym info copying. The
journal canonicalizes it before persistence, so later changes to the returned
info cannot change the recorded event. It stores compact event and account
facts rather than duplicating observations or the full portfolio book.

## Observer and failure behavior

The environment composes internal capture with the existing optional observer.
For each native interval it calls the caller's observer once with the unchanged
cumulative observation, when configured. Trace capture builds interval facts
from the same native result and the suffix appended since the prior archive
snapshot. The default trace path walks only new terminal-order nodes, while the
public observer retains its cumulative archive contract. A step that includes
terminal settlement can therefore contain multiple transitions and execution
observations. The capture checks that each native interval produced exactly one
observation at the expected next index.

Observer or trace-capture errors propagate through the existing fail-closed
path. After possible native mutation, the environment and session are poisoned
and cannot be retried. The journal records a failed call without a successful
`step_result`; it does not turn a missing or partial trace into success.

In joint PPO training, the existing actor/native join continues to bind the
policy sample to the persisted step by the exact submitted action codes and
event reference. The trace repeats those action codes and adds the risk and
execution path. It is software lineage evidence; it does not establish
profitability, an approved research result, or live-trading readiness.
