# Trade RL

Trade RL is a research-grade, baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy adjusts a deterministic causal baseline through a bounded, explicitly identified action specification while an independent shadow book supplies the comparison path.

> Capability status: **research-ready** and **attested paper-serving-ready**. Direct exchange order routing remains **NO-GO**. The repository does not claim profitability or authorize live capital deployment.

## Start here

For a copy-paste first training run, including deterministic demo-data generation, a maintained PPO configuration, artifact inspection, real-data replacement, GPU settings and troubleshooting, read [START.md](START.md). For the maintained public Binance ingestion and live-smoke path, read [Binance Public Data Workflow](docs/BINANCE.md).

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

Dataset identity v6 is recomputed from every observation, eligibility, execution and accounting field, including effective-dated tick/lot/minimum-notional rules, aggregated funding-event counts, fees, spread, participation, borrow/funding schedules, mark/index prices, corporate actions, cash rates, volume-unit semantics, contract multipliers and feature availability/age. Published dataset artifacts reject arbitrary identities, symlinks, root escapes and undeclared files.

Environment identity includes the verified dataset, calendar, action specification, content-addressed fold-local alpha/factor artifacts, semantic normalizer, episode curriculum, trend, reward, portfolio risk, execution and AUM. Signal filesystem paths are diagnostics only and never change experiment identity.

Serving candidate bundle v4 contains no release identifier. A separate HMAC-SHA256-authenticated `ReleaseAttestation` binds the immutable bundle digest to dataset, selection, evaluation, gate evidence, selected policy, source commit, dependency provenance, approver and approval time. Registry and runtime activation require an explicitly trusted key ID, verify the signature, load the shared observation/normalization pipeline and run deterministic probe observations through every ensemble member before live state is swapped. Structured prediction additionally requires a monotonic identity-bound account-state snapshot in released mode. Runtime inference rejects non-finite, incorrectly shaped or out-of-range actions rather than clipping them silently.

The framework-independent serving layer accepts a `PolicyLoader`. `trade_rl.integrations.StableBaselines3PolicyLoader` is the maintained concrete adapter for PPO, SAC, TD3 and TQC ensemble bundles. Stable-Baselines3 and PyTorch are installed only with the `train-sb3` extra.

## Training artifacts

A market dataset artifact is a validated directory containing canonical `manifest.json` and deterministic `arrays.npz`. The maintained API is `write_market_dataset_files` for deterministic staging files, `publish_market_dataset_artifact` for exclusive atomic publication, and `load_market_dataset_artifact` for verified loading; older same-named writers remain warning-only compatibility wrappers for one release. Training and walk-forward runs are staged, validated and then atomically published under `runs/<run-id>`; incomplete runs are isolated under `failed/<run-id>` and never replace `latest.json`.

A published training run contains the source dataset identity, resolved training/environment configuration, ensemble manifest, one authoritative `policy.zip` per seed, content-addressed intermediate checkpoints selected only on checkpoint-validation data, a `policy-loader.json`, optional verified ONNX/TorchScript actors and a content-addressed `run.json`. `policy.zip` remains the authoritative recovery and retraining format. ONNX is an optional required export when requested; TorchScript is best-effort and records an explicit unsupported reason when conversion is unsafe.

Nested walk-forward execution builds fold-local causal signals, fits only exogenous normalization statistics on each train capability, records one-shot sealed-test access, and evaluates the exact deterministic mean seed ensemble used by serving on the outer test range only after selection. Training, behavior cloning, checkpoint selection and sealed evaluation all use liquidation-at-close terminal accounting. Independent folds retain full execution evidence and are reported as a distribution; continuous return and drawdown are produced only with verified contiguous account-state handoff. An optional identity-bound post-selection execution-sensitivity pack replays the selected ensemble and baseline closed-loop under nominal, individual 2x, joint 2x and report-only joint 5x rule stresses without participating in selection.

## Commands

```bash
uv sync --extra dev --extra train-sb3
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

Build a deterministic public Binance USDⓈ-M dataset:

```bash
uv run trade-rl data binance \
  --market usds-m --symbol BTCUSDT --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --transport vision \
  --tick-size 0.1 --lot-size 0.001 --minimum-notional 5 \
  --listed-at 2019-09-08T00:00:00Z \
  --output artifacts/datasets/binance-btcusdt
```

Spot and USDⓈ-M are supported linear products. COIN-M inverse futures fail closed because the current accounting model is linear and must not publish misleading inverse-contract PnL. See [docs/BINANCE.md](docs/BINANCE.md) for the fixed-range end-to-end smoke.

Both execution commands print one machine-readable JSON result. Research runs remain non-production artifacts; a paper-serving activation additionally requires a signed external release attestation from an explicitly configured trusted key. Fresh confirmation evidence is likewise signed and recomputes return/drawdown from its immutable return series rather than trusting summary fields. Direct exchange connectivity is not implemented.

## Docker GPU full research run

The maintained container requires CUDA, keeps runtime data in the
`trade-rl-training-data` Docker volume, and runs the complete Binance
multi-timeframe research workflow. Build and run it from the repository root:

```bash
docker compose -f compose.training.yaml build trainer
docker compose -f compose.training.yaml run --rm trainer
```

The second command exits nonzero when CUDA preflight, training, evaluation, or
the research gate fails. A successful process is research evidence only: it is
not a profitability guarantee and production status remains `NO-GO`.

The Docker workflow defaults to `TRADE_RL_METADATA_MODE=frozen_snapshot`: one official current `exchangeInfo` payload is preserved byte-for-byte, identity-bound and disclosed as unauthenticated, non-point-in-time evidence. `historical_signed` remains the highest-integrity explicit opt-in and requires `TRADE_RL_BINANCE_RULE_HISTORY` plus `TRADE_RL_METADATA_KEYS`; `conservative_static` requires a versioned payload path. No mode silently projects current values backward as historical truth. The full preset also publishes identity-bound closed-loop execution sensitivity and requires the joint-2x research gate while keeping joint 5x report-only.

See [Docker GPU full-training operations](docs/operations/docker-gpu-full-training.md)
for exact detached start, status, logs, volume inspection, artifact extraction,
fresh retry, and cleanup commands. Artifact extraction uses an absolute
PowerShell host path and a running Alpine copy container. The CUDA preflight and
smoke tools live under `examples/binance-multitimeframe/`.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

Install export verification dependencies with `uv sync --extra dev --extra train-sb3 --extra export`.

See [Architecture](docs/ARCHITECTURE.md), [Research Status](docs/RESEARCH_STATUS.md), and [Binance Public Data Workflow](docs/BINANCE.md).
