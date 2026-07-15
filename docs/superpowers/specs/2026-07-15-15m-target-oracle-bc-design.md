# 15-Minute Target-Weight Oracle BC Design

## Understanding summary

- The maintained research policy observes the market and recomputes desired
  portfolio weights every 15 minutes.
- A policy action is a BTCUSDT, ETHUSDT, and BNBUSDT target-weight vector, not
  an order command. Execution remains a separate cost- and liquidity-aware
  concern.
- Small target changes do not trade. Entry/hold/exit hysteresis prevents
  threshold chatter, while causal emergency risk can bypass ordinary trading
  friction only to reduce risk.
- The dataset uses a native 15-minute clock plus causal 1-hour, 4-hour, and
  1-day features. The 96 raw indicators remain four clocks times 24 indicators;
  the policy observation remains much wider because it also includes masks,
  staleness, execution state, book state, factors, and global context.
- Pure PPO and DP-oracle behavior cloning followed by PPO are predeclared
  candidates. Both use the same action, risk, execution, and network contracts.
- Oracle labels may use future returns only inside a fold's train range. The
  deployed student sees causal observations only, and checkpoint, selection,
  and outer ranges never contribute labels.
- Completion still requires two distinct selected RL policy digests, positive
  cost-adjusted mean outer return, non-negative uplift over Trend, and maximum
  independently reset fold drawdown no greater than 20%.

## Assumptions and non-goals

- The single RTX 4050 Laptop GPU has 6 GiB VRAM. Seeds and A/B candidates run
  sequentially; each seed collects rollouts in four CPU subprocess environments.
- `[256, 256]` policy/value heads and 128-dimensional asset/global embeddings
  remain the selected network. The observed VRAM margin is sufficient and the
  input width does not increase when the bar clock changes.
- Public Binance Vision data, immutable artifacts, policies, checkpoints, and
  logs remain inside the `trade-rl-training-data` Docker named volume.
- The pipeline is research-only. It does not route live orders, use exchange
  credentials, run the oracle during inference, or claim future profitability.
- The frozen June 2026 outer windows are regenerated on a 15-minute clock and
  are opened only after this design, configuration, and experiment-plan digest
  are committed.

## Architecture

### Data and timing

The dataset base timeframe changes from `1h` to `15m`. Feature specs explicitly
request `1h`, `4h`, and `1d`; the builder supplies native `15m` features, keeping
all four clocks and 96 causal feature columns. The period from 2024-12-01 to
2026-07-01 contains 55,392 regular 15-minute bars. A decision advances one bar
(`decision_hours=0.25`). The 168-hour discount half-life resolves to gamma from
the maintained `gamma_from_half_life` function rather than a rounded constant.

The two outer tests cover 2026-06-01 through 2026-06-16 and 2026-06-16 through
2026-07-01. In 15-minute bars the fold contract is: train 43,936, checkpoint
4,000, selection 4,000, purge 192, test 1,440, step 1,440, expanding train,
maximum two folds.

### Direct target action and rebalance policy

`ActionSpec` gains an explicit direct-target mode with three canonical action
names bound to the dataset symbols. Values remain finite and bounded in
`[-1, 1]`; the composer applies per-asset and gross limits before execution.
Residual mode remains readable for old artifacts but is not a candidate in the
new experiment.

A pure `TargetRebalancePolicy` owns train/evaluation-identical post-processing:

- flat positions enter only when absolute requested weight is at least 10%;
- an existing same-direction position is held while requested magnitude is
  between the 3% exit threshold and 10% entry threshold;
- reversals require the 10% entry threshold in the opposite direction;
- per-asset changes below 5% remain at the current weight;
- the existing 2% portfolio turnover limit then bounds ordinary rebalances.

The environment records requested and executed target weights, suppressed
turnover, hysteresis state, and bypass reasons in diagnostics and observation
execution state.

### Emergency risk

A causal `EmergencyRiskMonitor` runs at every 15-minute decision before the
ordinary no-trade band. It can only preserve or reduce absolute exposure.

- portfolio drawdown uses the existing 10%-to-20% deleveraging curve;
- signed one-hour asset loss of at least 3% flattens the affected position;
- a 4% 15-minute gap flattens an adversely exposed position;
- 24-hour realized volatility at least 2.5 times its trailing 30-day reference
  halves gross exposure;
- non-tradable or directionally disallowed assets receive a zero target;
- portfolio exposure and market-notional limits retain their existing hard
  enforcement.

Emergency reductions bypass hysteresis, the no-trade band, and ordinary
turnover limits. They never authorize increasing exposure or trading an
untradable asset. All inputs are timestamp-causal and digest-bound.

### Oracle labels and behavior cloning

`OracleTargetTeacher` ports the historical cost-aware Viterbi/DP teacher into
the maintained package. Each asset has discrete states `-0.45`, `0`, and
`+0.45`; transition reward includes the configured fee, spread, and impact
costs. Concurrent asset targets are projected to the same gross and per-asset
limits as the environment.

For each fold and candidate, teacher paths are built strictly from the fold's
train view. A deterministic teacher rollout executes those targets through the
same rebalance, emergency, risk, and execution pipeline and records causal
observations paired with raw target actions. Dataset range, configuration,
observation digest, action digest, label digest, and sample count form an
immutable teacher artifact.

The SB3 backend optionally performs actor-only MSE behavior cloning before PPO.
The pure candidate skips this phase. BC configuration, epoch losses, final MSE,
teacher artifact digest, and random seed enter training and policy manifests.
PPO then performs the same number of fine-tuning timesteps for both candidates.

### Selection and evidence

The walk-forward config predeclares two candidates: `ppo-target-15m` and
`oracle-bc-ppo-target-15m`. Each trains three seeds. Checkpoint-validation keeps
the top three checkpoints per seed, configuration-selection chooses one policy
across all finalists and both candidates, and only then does the sealed ledger
authorize the outer test. All teacher and BC identities are included in the
full experiment-plan digest. The final gate continues to require exactly two
different canonical RL policy digests; a baseline fallback cannot pass.

## Error handling and verification

- Reject direct-target action dimensions that do not match dataset symbols.
- Reject teacher ranges outside train, labels with non-finite values, or any
  teacher artifact whose observation/action/environment identities differ.
- Reject BC for non-PPO algorithms in this initial implementation.
- Reject emergency calculations without sufficient causal history by returning
  a non-emergency assessment; never backfill from future observations.
- Keep dataset determinism, prefix causality, sealed-access counts, complete
  plan identity, Docker provenance, and research-gate tests fail-closed.
- Run targeted RED/GREEN tests per component, then Ruff formatting/checking,
  MyPy, import contracts, full Pytest with coverage, CUDA smoke, and a complete
  volume-backed Docker training generation.

## Decision log

1. Chose direct target weights over residual-action inversion because oracle
   labels and the user's execution semantics map one-to-one to portfolio intent.
2. Chose one shared post-processing pipeline for both A/B candidates so BC is
   the only experimental difference and turnover comparisons are meaningful.
3. Chose top-three checkpoint finalists per seed because the previous global
   checkpoint winner was unstable across checkpoint and selection ranges.
4. Chose 15-minute base data with hourly higher-timeframe features instead of
   interpolating an hourly environment, preserving emergency reaction latency.
5. Chose discrete cost-aware oracle targets rather than next-bar direction
   labels because the former naturally learns flat/no-trade regions.
6. Chose sequential A/B GPU training with four parallel rollout environments;
   simultaneous GPU models are unsafe on the available 6 GiB device.
7. Kept the frozen June outer dates but changed their bar representation before
   downloading or evaluating the 15-minute experiment; pre-June evidence is
   development validation only.
