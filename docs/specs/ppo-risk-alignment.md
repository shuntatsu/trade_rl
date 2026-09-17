# PPO training risk alignment

Status: Active

The user prioritizes PPO and improvements to data/training. The completed first
PPO seed in the separately frozen directional study loses 17.11% with 2252.96
USDT execution costs. This is a diagnostic observation, not final family evidence.

Source audit identified a specific train/evaluation mismatch: training defaults
allow per-symbol/gross exposure 1.0 and only scale at drawdown 1.0, whereas the
shared-account evaluation uses per-symbol 0.1, account gross 0.5, and drawdown
scaling from 0.1 to 0.2. Risk-driven fills must not be mislabeled policy reversals.
Reward already includes execution costs and funding through log net return.

Add an optional immutable `PreTradeRiskConfig` to the PPO environment and fitter.
Omission preserves the maintained defaults exactly, including sequential and
interleaved behavior. An explicit configuration is used at construction and every
reset; each environment owns its risk object. It must survive fitter assembly and
cannot silently revert on an episode boundary. Existing reward, observation,
action, feature, execution, time window, and network contracts remain unchanged.

The next economic comparison changes only training risk configuration to match
the directional evaluation. Keep five seeds 0–4, 262144 steps, sequential layout,
the same dataset and 12 inputs, before-2023 fit and 2023–2024 development replay.
Freeze a separate protocol bound to completed baseline artifacts before any
candidate fit. Compare paired seed net return/cost/turnover/drawdown, retain all
seeds, and keep absolute profitability and operational gates separate from
relative improvement. No final/unused data or live orders are authorized here.

Relative improvement requires at least four paired net-return wins, a positive
median return delta across all five seeds, complete 17544-bar replay for both
sides, all candidate ledger drawdowns at most 20%, and no increase in termination
count per seed. Costs and turnover are diagnostics, not independent vetoes: net
profit is the objective. The original absolute five-seed and stress gate is
unchanged. Lower losses alone yield `RELATIVE_IMPROVEMENT_ONLY` and never profit
or deployment eligibility. Freeze baseline result/model hashes, selection,
dataset/config, current source, and the identical runtime before candidate fit.

This aligns the configurable risk limits, not the account topology. Training
still has one active symbol per account, evaluation shares cash across five, and
the policy still lacks shared-account drawdown observations. Terminal execution
and residual-holding rejection also remain unchanged. These limitations prevent
interpreting this one factor as a complete solution to the training mismatch.

Feature normalization, extra observations (including shared drawdown), training
layout, reward changes, hyperparameters, and larger budgets are separate factors.
In particular this does not alter or rerun the sealed #629/#632/#635 experiment.

Tests must establish default compatibility, explicit risk enforcement on real
synthetic price paths, reset persistence, and propagation through both fitter
layouts. Run full permanent CI before calling the capability complete.
