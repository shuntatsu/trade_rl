# Trade RL

Trade RL is a research-grade, baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy adjusts a deterministic causal baseline through a bounded, explicitly identified action specification while an independent shadow book supplies the comparison path.

> Production status: **NO-GO**. The repository provides research, evaluation, artifact and serving contracts. It does not claim profitability or production authorization.

## Start here

For a copy-paste first training run, including deterministic demo-data generation, a maintained PPO configuration, artifact inspection, real-data replacement, GPU settings and troubleshooting, read [START.md](START.md).

## Action and environment v3

The maintained environment now provides:

- a dynamic residual action contract with independent fast, slow and risk controls, optional alpha scale and optional causal factor residuals;
- no dead alpha action when alpha is disabled, with exact action names and an ActionSpec digest bound into training and serving artifacts;
- time-series, cross-sectional, long-only, market-neutral and cash-or-trend deterministic baselines, including a nonzero one-symbol mode;
- continuous 24/7 and irregular session-calendar datasets with feature-level availability, staleness and execution metadata;
- next-open execution with fees, spread, impact, paired stochastic slippage, participation limits, minimum notional, lot/tick rounding, borrow, funding schedules, market/limit orders, latency, margin, corporate actions and delisting settlement;
- hard risk limits that override soft turnover throttles during concentration, leverage or emergency drawdown violations;
- economic terminal states for insolvency, minimum equity, execution-cost exhaustion, margin call and liquidation instead of training-process crashes;
- cash, baseline, random, stressed, partial-fill and restored causal reset states, duration curricula and regime/stress episode sampling;
- observation schema v3 with per-feature masks and staleness, factor loadings, requested and realized execution state, cash/net/gross/margin state, previous action and optional finite-horizon time;
- fold-fitted, frozen, content-addressed observation normalization that preserves categorical masks exactly;
- reward schema v4 prioritizing absolute log-wealth growth, then excess growth, with incremental drawdown and rolling baseline-underperformance progressive hinges;
- PPO exploration/network controls plus sealed-comparison support for SAC, TD3 and TQC;
- AUM capacity curves, action saturation/projection diagnostics and permutation-aware shared per-asset encoding.

## Identity and serving

Environment identity includes dataset, calendar, action specification, content-addressed alpha/factor artifacts, normalizer, episode curriculum, trend, reward, risk, execution and AUM. Signal filesystem paths are source references only and never change experiment identity. Serving bundle v3 binds the exact action size, action names, ActionSpec digest, observation schema and size, environment digest and normalizer digest. Runtime inference rejects non-finite, incorrectly shaped or out-of-range actions rather than clipping them silently.

The framework-independent serving layer accepts a `PolicyLoader`. `trade_rl.integrations.StableBaselines3PolicyLoader` is the maintained concrete adapter for PPO, SAC, TD3 and TQC ensemble bundles. It validates every declared member and averages deterministic actions only after all members pass shape, finite-value and bounds checks.

## Training artifacts

A market dataset artifact is a validated directory containing canonical `manifest.json` and deterministic `arrays.npz`. The maintained API is `write_market_dataset_files` for deterministic staging files, `publish_market_dataset_artifact` for exclusive atomic publication, and `load_market_dataset_artifact` for verified loading; older same-named writers remain warning-only compatibility wrappers for one release. Training and walk-forward runs are staged, validated and then atomically published under `runs/<run-id>`; incomplete runs are isolated under `failed/<run-id>` and never replace `latest.json`.

A published training run contains the source dataset identity, resolved training/environment configuration, ensemble manifest, one authoritative `policy.zip` per seed, content-addressed intermediate checkpoints selected only on checkpoint-validation data, a `policy-loader.json`, optional verified ONNX/TorchScript actors and a content-addressed `run.json`. `policy.zip` remains the authoritative recovery and retraining format. ONNX is an optional required export when requested; TorchScript is best-effort and records an explicit unsupported reason when conversion is unsafe.

Nested walk-forward execution fits normalization on each train range only, evaluates checkpoint and configuration-selection ranges without reading sealed test data, and evaluates each selected policy on the outer test range only after selection.

## Commands

```bash
uv sync --extra dev
uv run trade-rl train config \
  --timesteps 102400 --decision-hours 4 --discount-half-life-hours 168 \
  --n-steps 2048 --batch-size 64 --log-std-init -0.5 --target-kl 0.02 \
  --policy-net-arch 128 --policy-net-arch 128 --seed 0 --seed 1 --seed 2
uv run trade-rl environment config \
  --initial-capital 100000 --calendar-kind continuous_24_7 \
  --episode-hour-choice 168 --episode-hour-choice 720 \
  --initial-state-mode cash --initial-state-mode stress
```

Execute actual training and atomic artifact publication:

```bash
uv run trade-rl train run \
  --config configs/train.json \
  --dataset artifacts/datasets/btc-eth \
  --output artifacts/research \
  --run-id btc-eth-ppo-001
```

Execute real-data nested walk-forward research:

```bash
uv run trade-rl walk-forward run \
  --config configs/walk-forward.json \
  --dataset artifacts/datasets/btc-eth \
  --output artifacts/research \
  --run-id btc-eth-wf-001
```

Both execution commands print one machine-readable JSON result and retain `production_status: "NO-GO"` until a separate approved release exists.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

Install export verification dependencies with `uv sync --extra dev --extra export`.

See [Architecture](docs/ARCHITECTURE.md) and [Research Status](docs/RESEARCH_STATUS.md).
