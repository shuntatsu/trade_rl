# Trade RL

Trade RL is a research-grade, baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy can adjust a deterministic causal baseline through a bounded residual action specification or emit explicitly identified per-symbol target weights. An independent shadow book supplies the comparison path.

> Capability status: **research-ready** and **attested paper-serving-ready**. Direct exchange order routing remains **NO-GO**. The repository does not claim profitability or authorize live capital deployment.

## Current maintained system

The current repository combines six distinct capabilities that must not be conflated:

1. causal market-data and multi-timeframe feature artifacts;
2. exploratory and selected-final reinforcement-learning workflows;
3. nested walk-forward selection with persistent sealed-test access;
4. conservative, stateful OHLCV order simulation with auditable order events;
5. local research visualization through Trade RL Studio;
6. attested, read-only paper-serving bundles.

PostgreSQL is an optional metadata catalog for reusable artifact identities, provenance, dependencies, lifecycle state, and sealed-test reservations. Large arrays, datasets, checkpoints, models, and run evidence remain immutable filesystem artifacts; PostgreSQL is not the numerical source of truth.

Normal environment transitions use persistent market, limit, or stop-market orders. Orders carry latency, time-in-force, remaining quantity, trigger state, replacement linkage, and deterministic evidence across decision boundaries. Fills use the processing bar's capacity under an explicit OHLC path assumption. Selected-final promotion requires conservative path mode, processing-bar capacity, partial-fill carry, complete order evidence, and the expected execution-policy digest. OHLCV simulation is not an order-book reconstruction or a guarantee of exchange-equivalent fills.

## Trade RL Studio

The local GUI lives in `studio/`. Its fixed-layout React + Vite + TypeScript interface operates on validated datasets, training configs, exploratory training jobs, run artifacts, comparisons, evidence, and read-only paper-serving state. Live Training presents sampled exploration as a chart-first market replay with near-live or buffered playback, candle or event-compressed views, position/risk markers, PnL, reward, and drawdown.

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
# In another terminal
npm ci --prefix studio
npm run dev --prefix studio
```

Studio does not submit exchange orders. BUY/SELL markers visualize changes in learned target exposure, not real orders. Training telemetry is diagnostic, seed-scoped, append-only, and excluded from selection, artifact identity, release approval, and execution. See [`studio/README.md`](studio/README.md).

## Start here

For a copy-paste first training run, including deterministic demo-data generation, a maintained PPO configuration, artifact inspection, real-data replacement, GPU settings, and troubleshooting, read [START.md](START.md). For the maintained public Binance ingestion and live-smoke path, read [Binance Public Data Workflow](docs/BINANCE.md).

## Action and environment v3

The maintained environment provides:

- a dynamic residual action contract with independent fast, slow, and risk controls, optional alpha scale, and optional causal factor residuals;
- an explicit direct `target_weight:<symbol>` mode whose symbol order is identity-bound;
- no dead alpha action when alpha is disabled, with exact action names and an ActionSpec digest bound into training and serving artifacts;
- time-series, cross-sectional, long-only, market-neutral, and cash-or-trend deterministic baselines, including a nonzero one-symbol mode;
- continuous 24/7 and irregular session-calendar datasets with feature-level availability, staleness, and execution metadata;
- next-open execution with fees, maker/taker costs, spread, impact, paired stochastic slippage, participation limits, minimum notional, lot/tick rounding, borrow, funding schedules, latency, margin, corporate actions, and delisting settlement;
- persistent market, limit, and stop-market order state with partial-fill carry, cancel-and-replace, deterministic OHLC path assumptions, and shared processing-bar capacity;
- hard risk limits that override soft turnover throttles during concentration, leverage, or emergency drawdown violations;
- economic terminal states for insolvency, minimum equity, execution-cost exhaustion, margin call, liquidation, and drawdown stop instead of training-process crashes;
- cash, baseline, random, stressed, partial-fill, and restored causal reset states, duration curricula, and regime/stress episode sampling;
- observation schema v3 with per-feature masks and staleness, factor loadings, requested and realized execution state, pending-order state, cash/net/gross/margin state, previous action, and optional finite-horizon time;
- fold-fitted, frozen, content-addressed observation normalization that preserves categorical masks exactly;
- reward schema v4 prioritizing absolute log-wealth growth, then excess growth, with incremental drawdown and rolling baseline-underperformance progressive hinges;
- PPO exploration/network controls plus sealed-comparison support for SAC, TD3, and TQC;
- AUM capacity curves, action saturation/projection diagnostics, and permutation-aware shared per-asset encoding.

## Identity and serving

Dataset identity v6 is recomputed from every observation, eligibility, execution, and accounting field, including effective-dated tick/lot/minimum-notional rules, aggregated funding-event counts, fees, spread, participation, borrow/funding schedules, mark/index prices, corporate actions, cash rates, volume-unit semantics, contract multipliers, and feature availability/age. Published dataset artifacts reject arbitrary identities, symlinks, root escapes, and undeclared files.

Environment identity includes the verified dataset, calendar, action specification, content-addressed fold-local alpha/factor artifacts, semantic normalizer, episode curriculum, trend, reward, portfolio risk, execution policy, AUM, and sequence-policy architecture. Signal filesystem paths are diagnostics only and never change experiment identity.

Serving bundle v5 contains the complete selected-final evidence chain but no approval material. A detached Ed25519 `ReleaseAttestation` binds the immutable bundle digest to the training run, selection proposal and authorization, walk-forward and gate evidence, fresh confirmation, selected policy, source commit, dependency provenance, approver, and expiry. Runtime and registry processes receive purpose-bound public keys only; private keys are accepted exclusively by explicit offline CLI commands. Exploratory runs, unsigned bundles, legacy release sidecars, wrong-purpose keys, incomplete evidence chains, and incompatible execution evidence fail closed before activation.

The framework-independent serving layer accepts a `PolicyLoader`. `trade_rl.integrations.StableBaselines3PolicyLoader` is the maintained concrete adapter for PPO, SAC, TD3, and TQC ensemble bundles. Stable-Baselines3 and PyTorch are installed only with the `train-sb3` extra.

## Training artifacts and evidence

A market dataset artifact is a validated directory containing canonical `manifest.json` and deterministic `arrays.npz`. The maintained API is `write_market_dataset_files` for deterministic staging files, `publish_market_dataset_artifact` for exclusive atomic publication, and `load_market_dataset_artifact` for verified loading. Training and walk-forward runs are staged, validated, and atomically published under `runs/<run-id>`; incomplete runs are isolated under `failed/<run-id>` and never replace `latest.json`.

A published training run contains the source dataset identity, resolved training/environment configuration, ensemble manifest, one authoritative `policy.zip` per seed, content-addressed intermediate checkpoints selected only on checkpoint-validation data, a `policy-loader.json`, optional verified ONNX/TorchScript actors, execution evidence, and a content-addressed `run.json`. `policy.zip` remains the authoritative recovery and retraining format.

Exploratory Stable-Baselines3 training emits sampled `training_telemetry_v1` JSON Lines beneath each seed's `telemetry/` directory. Normal rollout transitions are sampled, while material position changes, risk events, and episode termination are retained. Telemetry is diagnostic and does not participate in model selection, artifact identity, serving approval, or order execution.

Nested walk-forward execution builds fold-local causal signals, fits normalization only on each train capability, records one-shot sealed-test access, and evaluates the exact deterministic mean seed ensemble used by serving on the outer test range only after selection. Independent folds retain complete execution evidence and are reported as a distribution; continuous return and drawdown require verified contiguous account-state handoff. An optional identity-bound execution-sensitivity pack replays the selected ensemble and baseline under declared rule stresses without participating in selection.

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

Start the optional PostgreSQL catalog:

```bash
cp .env.example .env
docker compose up -d postgres
uv sync --extra postgres
export TRADE_RL_DATABASE_URL=postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
uv run trade-rl catalog migrate
uv run trade-rl catalog health
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

Spot and USDⓈ-M are supported linear products. COIN-M inverse futures fail closed because the current accounting model is linear. See [docs/BINANCE.md](docs/BINANCE.md).

Both execution commands print one machine-readable JSON result. Exploratory training cannot be packaged or released. A selected-final run requires an externally signed selection authorization before model construction, at least 30 days of Ed25519-signed post-training confirmation, deterministic bundle packaging, matching conservative execution evidence, and a separate offline release approval. Direct exchange connectivity is not implemented.

### Offline evidence and release commands

Private Ed25519 key files use schema `ed25519_private_key_v1`, must be purpose-bound, and on POSIX must have mode `0600`. They are used only on an offline approval host.

```bash
uv run trade-rl selection authorize \
  --proposal var/research/selection-proposal.json \
  --private-key /secure/selection-key.json \
  --approver research-committee \
  --approved-at 2026-07-18T03:00:00Z \
  --expires-at 2026-07-25T03:00:00Z \
  --output /secure/selection-authorization.json

uv run trade-rl confirmation create \
  --request /secure/fresh-confirmation-request.json \
  --private-key /secure/confirmation-key.json \
  --output /secure/fresh-confirmation.json

uv run trade-rl release approve \
  --bundle var/serving/candidate \
  --private-key /secure/release-key.json \
  --git-commit <40-character-commit> \
  --dependency-digest <uv-lock-sha256> \
  --approver release-committee \
  --approved-at 2026-08-18T03:00:00Z \
  --expires-at 2026-09-18T03:00:00Z
```

The authorization, confirmation, and release files are immutable and remain external to the candidate bundle until verification.

## Docker GPU full research run

The maintained container requires CUDA, stores runtime data in the `trade-rl-training-data` Docker volume, and runs the complete Binance multi-timeframe research workflow:

```bash
docker compose -f compose.training.yaml build trainer
docker compose -f compose.training.yaml run --rm trainer
```

The run exits nonzero when CUDA preflight, training, evaluation, or the research gate fails. A successful process is research evidence only; it is not a profitability guarantee and production status remains `NO-GO`.

The Docker workflow defaults to `TRADE_RL_METADATA_MODE=frozen_snapshot`: one official current `exchangeInfo` payload is preserved byte-for-byte, identity-bound, and disclosed as unauthenticated, non-point-in-time evidence. `historical_signed` is the highest-integrity opt-in and requires a signed point-in-time document plus a read-only Ed25519 public-key store. `conservative_static` requires a versioned payload path. No mode silently projects current values backward as historical truth.

See [Docker GPU full-training operations](docs/operations/docker-gpu-full-training.md) for detached start, status, logs, volume inspection, artifact extraction, retry, and cleanup commands.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Install export verification dependencies with `uv sync --extra dev --extra train-sb3 --extra export`.

See [Architecture](docs/ARCHITECTURE.md), [Research Status](docs/RESEARCH_STATUS.md), [Binance Public Data Workflow](docs/BINANCE.md), and the latest [documentation and architecture audit](docs/verification/2026-07-22-documentation-and-architecture-audit.md).
