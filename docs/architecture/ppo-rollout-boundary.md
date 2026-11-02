# PPO rollout-boundary diagnostic contract

This document defines the software boundary for the Issue #810 synthetic PPO rollout-boundary diagnostic. The implementation lives in `tools/issue810_p3_rollout_boundary_fixture.py` and `tools/issue810_p3_rollout_boundary_diagnostic.py`; it is an opt-in research tool, not a production strategy or a change to the public PPO API.

## Closed synthetic source

`build_synthetic_dataset(payoff_transition)` constructs the complete `MarketDataset` in memory. It accepts only the preregistered values 48 and 72. There is no data-path argument, data loader, or option to substitute market data. The dataset contains one symbol, one persistent signed cue feature, four fixed windows, deterministic price events, and liquidity only at each block's first opportunity and its reserved terminal settlement row. All other nonterminal rows have zero volume.

The two values of `payoff_transition` change only where each non-neutral block's 1.5% one-step price event occurs relative to the block start. The event is relative to the immediately preceding close. The 48 case is inside the first 64-step PPO rollout; the 72 case is eight transitions beyond that rollout boundary. The opportunity occurs at the block start, before either event. This arrangement and the native late-entry negative control prevent a policy from earning the event by waiting until after the rollout boundary to enter.

The fixture pins initial capital, one-symbol target weight, drawdown limits, execution rates, a bound financial clock, and the episode horizon. `SharedPortfolioGymEnv` continues the same account through rollout boundaries. A rollout boundary is neither an episode end nor a settlement point. At the actual episode endpoint the native environment settles once and must report a valid endpoint, zero quantity, and no active order.

## Fit and held-out boundary

The diagnostic fits one transform and one PPO model on window 0 only. It uses the existing `fit_shared_portfolio_ppo` path and actual Stable-Baselines3 components. The three later windows share the same immutable synthetic Dataset identity and are bound through `PortfolioObservationRuntimeBinding` and `PortfolioGymObservationWrapper`; their observations are used only after fitting. Optimizer joins must be fully contained in the training actor range. The reserved settlement row may be used for native close accounting but is not an optimizer observation.

Each window contains eight 128-transition cue blocks (1,024 actor transitions) and one reserved terminal settlement row. Actor decisions run from `window_start` through `window_start + 1,023`; the settlement reserve is at `window_start + 1,025`. Window starts are 0, 1,025, 2,050, and 3,075. Each window starts its own fixed-capital native episode; no account is carried from training into held-out windows or between held-out windows. A held-out run uses deterministic policy actions and must complete all 1,024 actor transitions without truncation or an early termination.

The held-out schedules are fixed in the implementation. Training uses `(+, -, 0, +, -, 0, +, -)`. Held-out windows 1–3 use `(-, 0, +, -, +, 0, -, +)`, `(+, 0, -, +, -, 0, +, -)`, and `(-, +, 0, -, +, 0, -, +)`. A cue persists through its 128-transition block. Each non-neutral block has exactly one liquid opportunity at its first transition; neutral blocks contain no price event.

## Evidence and joins

For every evaluated policy and held-out window, `heldout[].{initialized,trained,fixed_long,cash}.block_results` contains eight records. Each record has `block_index`, `cue`, `actor_decision_index`, `native_processing_index`, `elapsed_intervals`, `action_code`, `post_action_weight`, `aligned`, `flat_at_snapshot`, and `pending_order_at_snapshot`. The weight is read from `observation["account"][0, 1]` immediately after the sole liquid opportunity; the observation transform preserves it. `aligned` is a boolean for nonzero cues and null for a neutral cue. A non-neutral block is aligned only when `cue * post_action_weight > 0`. Each held-out window must align at least five of its six nonzero cues. At the corresponding snapshot in each of its two neutral blocks, the weight must be within `1e-12` of zero and no order may be pending. Throughout-block neutral exposure/order counters may be retained as diagnostics, but they do not replace or strengthen this frozen snapshot acceptance rule.

The native reward remains the equity-delta reward emitted by the portfolio environment. Entry and terminal execution costs are included by the canonical session and executor. The fixture tests independently calculate filled notional from quantity and execution price, then calculate fees, spread, participation impact, cash, and equity. They do not derive the expected value from the native total-cost field.

The SB3 boundary check uses the installed `DictRolloutBuffer`, not a hand-written GAE substitute. At transition 64, the account remains open and the environment is not terminated or truncated. The value target at that boundary bootstraps to the remaining after-cost terminal equity change; with `gamma=1`, the numeric target equals that future net change. A terminal-mask control must suppress the bootstrap. This establishes a buffer/critic-target plumbing check only; it does not establish that PPO learned the actor behavior.

## Provenance and fail-closed outcomes

The runner requires a full expected commit SHA, a clean source worktree, and imported modules resolved inside that exact checkout. It constructs a new output directory outside the source tree and writes the frozen protocol identity before running cells. Results retain each fixed cell, training completion and join facts, checkpoint digests, held-out controls, settlement facts, and the final disposition. Any missing join, source mismatch, invalid/non-finite result, unexpected truncation or termination, incomplete training, invalid endpoint, non-flat terminal state, pending order, or failed acceptance criterion remains a failure or `NOT_ESTABLISHED`; it is never converted to a successful result by fallback.

The tool records only software-mechanism evidence. It does not read real data, establish profitability or generalization, close P3 by itself, or satisfy P5's separate continuous-account handover requirement. It does not authorize real-data work, unused/final evaluation, or live trading. A software pass cannot change prior negative or incomplete economic classifications.
