# Trade RL

Trade RL is a research-grade baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy does not choose arbitrary portfolio weights directly. It makes bounded residual decisions around a deterministic trend baseline, while an independent shadow book provides the reward and evaluation reference.

> Production status: **NO-GO**. The repository provides research, evaluation, artifact and serving contracts. It does not claim a profitable or production-authorized strategy.

## What changed

The repository was rebuilt around one authoritative `trade_rl` package. The former `mars_lite` stack, direct-action PPO path, legacy scripts, duplicate metric implementations and legacy tests were removed rather than deprecated.

Key boundaries are now explicit:

- immutable domain manifests for datasets, signals, policy ensembles, selections and releases;
- canonical, content-addressed artifacts with staged and atomic publication;
- one implementation of return, risk and paired-comparison metrics;
- pure leak-checked walk-forward fold construction and chronological OOS stitching;
- a two-dimensional residual action schema with exact zero-action baseline identity;
- isolated execution, accounting and pre-trade risk components;
- validated serving bundles, atomic registry activation and fail-closed hot swaps;
- one `trade-rl` CLI and typed workflow boundaries.

## Install

```bash
uv sync --extra dev
```

## Commands

```bash
uv run trade-rl --version
uv run trade-rl train config \
  --timesteps 102400 --decision-hours 4 --discount-half-life-hours 168 \
  --n-steps 2048 --batch-size 64 --n-epochs 10 \
  --learning-rate 0.0003 --device auto \
  --seed 0 --seed 1 --seed 2
uv run trade-rl walk-forward plan \
  --bars 220 --train-bars 80 --checkpoint-bars 10 \
  --selection-bars 10 --test-bars 20 --purge-bars 2 --max-folds 2
```

The `data`, `signal`, `evaluate`, `registry`, and `serve` command groups expose the corresponding architectural boundaries. Additional application adapters should be added behind these boundaries rather than inside domain or evaluation code.

## Concrete nested walk-forward

`ConcreteFoldRunner` executes one nested fold through candidate training, checkpoint validation, configuration selection, baseline fallback, and sealed outer-OOS evaluation. Training and evaluation implementations remain injected adapters. Each adapter receives immutable `IndexRange` requests rather than unrestricted fold data.

`execute_walk_forward` runs all planned folds, validates dataset and fold identity, stitches selected and baseline OOS return series independently, computes both metric sets, and produces a content-addressed final evaluation digest. Gate decisions bind the evaluated dataset, selected policy identity when applicable, and final evaluation digest; release construction rejects mismatched identities.

The orchestration is concrete and tested, but the repository deliberately does not guess the project's storage format, preprocessing graph, checkpoint-selection policy, or data source. A real-data integration must provide adapters that fit preprocessing inside the train range, choose checkpoints only with checkpoint-validation data, and evaluate frozen candidates only on the requested selection or outer-test range.

## AUM and environment identity

`ResidualMarketEnvConfig.initial_capital` has no silent default. Every training or evaluation adapter must provide the intended quote-currency AUM explicitly. This is required because participation limits, market impact, turnover costs, minimum-equity termination, and liquidation feasibility all change with portfolio size.

The environment computes a content digest over the dataset identity, resolved episode and decision cadence, trend horizons, risk limits, execution-cost model, reward settings, action and observation schemas, alpha mode, and initial capital. Every policy ensemble records this environment digest and AUM, and ensemble construction rejects members trained with different environment identities or capital scales.

A model trained with a one-dollar account is therefore not silently interchangeable with a model intended for a 100,000-dollar or 1,000,000-dollar account. Capacity should ultimately be reported across several predeclared AUM scenarios rather than inferred from a scale-free backtest.

## Serving identity and activation

Serving bundle schema v2 carries the action schema, observation schema, expected flattened observation size, environment digest, and initial capital in addition to dataset, signal, selection, policy, release, and artifact-file identities. Every field participates in the bundle digest.

`ServingRuntime` rejects a bundle whose action or observation schema does not match the maintained runtime and rejects inference vectors whose size does not match the active manifest. `ServingRegistry` and `ServingRuntime` require an approved release identity by default. Research-only activation of unreleased bundles is possible only through the explicit `allow_unreleased=True` escape hatch.

This prevents a correctly shaped action model from being deployed with the wrong feature order, wrong observation width, wrong environment configuration, or wrong capital scale.

## GPU utilization

The maintained PPO uses a small MLP and currently trains through one environment. In this shape, environment stepping, NumPy accounting, and rollout collection can dominate wall-clock time, so low GPU utilization is expected and does not by itself indicate that training is broken.

Use `--device cuda` to require CUDA or `--device auto` to let Stable-Baselines3 choose. The training artifact records the resolved device, requested timesteps, rollout-rounded actual timesteps, complete PPO configuration digest, environment digest, and initial capital. Compare wall-clock samples per second and sealed OOS quality before increasing network width, batch size, or environment parallelism merely to make a GPU busier.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

See [Architecture](docs/ARCHITECTURE.md) and [Research Status](docs/RESEARCH_STATUS.md).
