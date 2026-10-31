# Issue #810 P3: synthetic PPO rollout-boundary protocol

Status: Frozen synthetic software/learnability protocol. This document defines the fixed diagnostic; runtime state and results belong in its generated run artifacts.

## Question and evidence boundary

Issue #810 P3 asks whether a delayed payoff beyond an active PPO rollout boundary is credited through the critic bootstrap, and separately whether the trained actor exhibits the predeclared behavior on held-out synthetic windows. The 48-transition arm is the in-rollout positive control; the 72-transition arm crosses the 64-step boundary by eight transitions. P5's continuous-account handover is separate and is not tested here.

The historical evidence remains part of interpretation. In the 2026-10-06 report (Issue #810 comment [6010140068](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6010140068)), delay 36 with rollout 32 and 64 and a one-bar cue had zero held-out cue alignment. A separate delay-20/rollout-16 persistent-cue case moved critic values but left the deterministic actor flat with zero held-out reward and alignment. Critic movement alone is not actor success. Closed, unmerged [PR #833](https://github.com/shuntatsu/trade_rl/pull/833) used an eight-bar payoff shorter than the compared rollout horizons and explicitly did not establish cross-rollout credit.

This protocol is result-blind and synthetic-only. It is not a market hypothesis test, profitability test, or authorization for real-data research. No full PPO fit or outcome is implied by the protocol text.

## Fixture and fixed task

The closed fixture is `tools/issue810_p3_rollout_boundary_fixture.py`; its schema is `issue810_p3_rollout_boundary_synthetic_v1`. It builds one in-memory `MarketDataset` with four windows on the same Dataset identity. The diagnostic is `tools/issue810_p3_rollout_boundary_diagnostic.py`, schema `issue810_p3_rollout_boundary_diagnostic_v1`. The runner records the canonical protocol hash and exact source identity in `protocol.json` and `run.json`.

The task is one symbol with `$1,000` initial capital, target absolute weight `0.5`, maximum gross exposure `1.0`, maximum absolute weight `0.5`, maximum drawdown `0.20`, and no turnover cap. Execution is fixed at 5 bp fee, 2 bp spread, 1 bp impact, and 5% maximum participation. Dataset fee, spread, funding, and borrow arrays are zero; execution costs are charged by the native executor. The bound clock uses one-hour decision/execution/reward intervals, a 64-step rollout, `gamma=1.0`, `gae_lambda=0.95`, and equity-delta rewards.

There are eight 128-transition blocks per window and 1,024 actor transitions per window. Window starts are `(0, 1025, 2050, 3075)` and each window's reserved settlement row is `window_start + 1025`. Actor decisions span `window_start` through `window_start + 1023`. Each window is a separate fixed-capital native episode; this protocol does not carry an account between training and held-out windows or between held-out windows. Each non-neutral block has one 1.5% price event at `block_start + payoff_transition`, measured against the preceding close; `payoff_transition` is only 48 or 72. The price change compounds through later blocks. Liquidity exists only at `block_start + 1` and the reserved settlement row. Neutral blocks have no price event. No other rows are made liquid to permit a late entry.

Training cues are `(+, -, 0, +, -, 0, +, -)`. Held-out windows 1–3 use `(-, 0, +, -, +, 0, -, +)`, `(+, 0, -, +, -, 0, +, -)`, and `(-, +, 0, -, +, 0, -, +)`. The cue stays observable for the entire block. This persistence prevents the delayed arm from changing the information schedule while the sparse opportunity row ensures that waiting until after the boundary cannot capture the payoff.

## Fit, held-out scope, and controls

The fixed matrix is the Cartesian product of payoff transitions `(48, 72)` and seeds `(0, 7, 17)`, six cells total. For each cell, PPO seed is the listed seed, reset seed is seed+1, and execution seed is seed+2. Use CPU Stable-Baselines3 PPO with 4,096 transitions, `n_steps=64`, `batch_size=32`, `n_epochs=10`, `gamma=1.0`, and `gae_lambda=0.95`. The current canonical path is `fit_shared_portfolio_ppo`; do not substitute the retired allocation environment. There is no seed, budget, network, reward, cue, cost, threshold, or evaluation-schedule search after outcomes.

Fit the observation transform and PPO only on window 0. The three held-out windows have later, non-overlapping actor ranges on the same Dataset ID and use a runtime binding to the training transform. They are never passed to `learn` and never enter a transform fit or optimizer join. The report must show training sample/join scope and held-out ranges so leakage is detectable. The initialized checkpoint, trained checkpoint, literal fixed-long action, and cash/HOLD control use the same held-out capital, risk, execution, costs, and endpoint. Initialized and fixed-long are the two actor acceptance comparators; cash is a diagnostic control.

## Frozen measures and pass rule

For each held-out policy and each of its eight blocks, the report at `heldout[].{initialized,trained,fixed_long,cash}.block_results` includes `block_index`, `cue`, `actor_decision_index`, `native_processing_index`, `elapsed_intervals`, `action_code`, `post_action_weight`, `aligned` (boolean for nonzero cues and null for neutral), `flat_at_snapshot`, and `pending_order_at_snapshot`. The snapshot is after the block-start action has processed the only liquid row at `block_start + 1`. For a nonzero cue, alignment is exactly `cue * post_action_weight > 0`; at least five of six nonzero blocks must align. At the one snapshot in each neutral block, require `abs(post_action_weight) <= 1e-12` and no pending order. Throughout-neutral counters may be reported diagnostically but are not an additional pass gate.

Each evaluation must complete exactly 1,024 actor actions with no truncation or early termination, then perform exactly one valid native terminal settlement. The terminal endpoint must be flat and have no active or pending order. Compute terminal log wealth after all execution costs as `L = log(W_terminal / W_initial)`.

Acceptance is assessed independently for every seed × held-out-window cell; do not average across cells. For the trained policy in each window, require:

- at least 5 of 6 nonzero cue snapshots aligned;
- both neutral snapshots flat within `1e-12` with no pending order;
- valid one-time terminal settlement, flat position, and no active order;
- `L_trained > 0.03`;
- `L_trained - L_initialized > 0.02`; and
- `L_trained - L_fixed_long > 0.02`.

The uplift is a difference of log wealth, not a subtraction of simple returns. A seed cell passes only if all three held-out windows meet every criterion and the software/provenance gates pass. The 48-transition positive control must also pass. If the positive control or 72-transition arm fails, report the exact failed cell and criterion; do not tune the protocol or label P3 established.

## Required software oracles and stop conditions

Before any fit, the deterministic native guard must demonstrate both event placements through the real portfolio environment and executor, including the fixed late-entry negative control. The late action must not collect a fill at the event or profit from the price move, and its cancellation/native order facts must be retained. Entry fees may make pre-event reward negative. The independent arithmetic oracle computes notional from `abs(filled_quantity * execution_price)` and checks fee, spread, impact, signed cash flow, cash, equity, and the net terminal result against native accounting.

The actual SB3 `DictRolloutBuffer` oracle must show that at transition 64 the episode is still active and no liquidation/reset occurred. For the 72-transition event, state `s64` bootstraps the after-cost future terminal equity change; with `gamma=1`, the value and transition-64 target equal that future net change. A true-terminal mask must suppress the bootstrap. This is critic/buffer conformance only. It does not count as actor learnability.

For every cell, require exact requested and realized training steps, complete actor/native joins, fit-only training observations, successful checkpoint save/load and matching digest, deterministic held-out inference, finite rewards and account state, exact source imports under a clean expected HEAD, and the terminal requirements above. Missing evidence, leakage, source mismatch, invalid endpoint, early reset, truncation, non-finite values, or an exception fails closed. Keep the run disposition `NOT_ESTABLISHED` unless the fixed matrix and all acceptance rules pass.

## Interpretation

A passing run would support only this narrow statement: under this closed synthetic fixture and frozen configuration, the actual SB3 path showed the specified critic-bootstrap behavior and the trained actor met the frozen synthetic held-out criteria. It would not establish market profitability, a trading edge, generalization to real data, G4, P5 continuous-account economics, unused/final evaluation, or live readiness. It does not change the historic PPO six-fit `FAILED`, P3 conditional diagnostic `NOT_ATTEMPTED`, NONRL48 interrupted, or M34 `FAILED_SCREEN` classifications. Only a new, separately authorized and reviewed evidence path may change those states.
