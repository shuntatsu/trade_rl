# Trade RL Architecture

## Status

`trade_rl` is a research-grade baseline-anchored residual-RL core with attested local/paper-serving contracts. Direct exchange trading remains unavailable. Architecture quality, empirical profitability and operational authorization are separate gates.

## Responsibility map

```text
trade_rl/
  domain/        immutable dataset, policy, selection and release identities
  artifacts/     canonical serialization, hashing and atomic publication
  data/          market calendar, feature and execution-data contracts
  strategies/    deterministic causal baselines
  risk/          pure hard/soft pre-trade constraints
  simulation/    execution, costs, carry, margin and accounting
  evaluation/    metrics, paired tests, folds, gates and AUM capacity
  learning/      framework-light teacher, behavior-cloning and supervised-data contracts
  rl/            actions, observations, normalization, rewards, environment and framework-neutral training protocols
  release/       verification contracts for non-circular external release attestations
  serving/       framework-independent candidate bundles, registry and fail-closed runtime
  integrations/  Stable-Baselines3 training/serving and other framework adapters
  workflows/     typed training, walk-forward and publication orchestration
  cli/           single authoritative configuration and execution entry point
```

The enforced dependency order is `cli -> workflows -> integrations -> serving -> release -> rl -> learning -> risk -> simulation -> strategies -> data -> evaluation -> artifacts -> domain`. Integrations own external model-framework adapters. Release verification cannot import serving, integrations or workflows. Learning contracts cannot depend on Stable-Baselines3 or sb3-contrib. Core training and serving modules do not import Stable-Baselines3 directly.

## Privileged GPU boundary

Self-hosted GPU runners never execute pull-request-controlled workflows. The maintained full-run workflow is restricted to the protected `main` ref, the repository owner and the `gpu-full-training` GitHub Environment. External Actions in that privileged workflow are pinned to immutable commit SHAs. Detached full runs are labeled and supervised through explicit `start`, `status` and `stop` operations; terminal exit status and retained logs are collected without deleting the failed container before evidence is captured.

## Action contract

`portfolio_action_v3` supports either the maintained baseline-residual controls or direct per-symbol `target_weight:<symbol>` controls. The environment derives the exact dimension from `ActionSpec` and binds symbol order into its identity. In target-weight mode, zero action means flat; in residual mode, zero action remains exact baseline identity. Training may clip with diagnostics; evaluation and serving can reject out-of-range actions fail closed.

## Baseline contract

Trend baselines distinguish time-series direction from cross-sectional ranking. `auto` uses time-series trend for a one-symbol universe and cross-sectional trend for multiple symbols. Directional modes preserve signal confidence and cash allocation rather than always normalizing to full gross exposure.

## Market and execution contract

Bar timestamps are close times and decisions first execute at the next open. Continuous datasets require regular cadence; session datasets use wall-clock lookup helpers across overnight, weekend and holiday gaps. Execution uses only information available at the decision and execution timestamps. Hybrid and shadow books share one episode RNG stream while different episodes receive distinct deterministic seeds.

The execution layer models partial fills, fees, spread, impact, random and tail slippage, per-bar constraints, minimum notional, lot/tick rules, borrow, carry, latency, market/limit orders, margin, dividends, splits and delisting recovery. Economic failures become structured terminal transitions; malformed market data remains an exception.

Binance `historical_signed` execution rules carry an authenticated semantic scope: market, ordered symbols, coverage start/end, issue time, source URI, policy version and payload digest. The maintained resolver requires an exact USD-M, symbol-order and research-interval match. Tick size, lot size and minimum notional are strictly positive for every signed metadata entry and effective rule. Frozen and conservative-static modes remain explicitly non-point-in-time and retain their limitations in dataset identity.

## Risk ordering

Turnover is a soft operational constraint. Concentration, gross leverage and emergency drawdown limits are hard constraints applied afterward and validated again. A hard deleveraging requirement may override turnover, and every projection exposes reasons and L1 distance.

## Observation and normalization

Observation schema v3 carries feature values, feature-level availability/staleness/reasons, active/tradable state, all baseline and factor inputs, current and requested portfolios, fill/cost/capacity state, cash/net/gross/margin state and previous action. Remaining time is included only for an explicitly finite-horizon MDP. Normalization statistics are fitted on an explicit train range, frozen elsewhere, content-addressed, and preserve mask/categorical coordinates exactly.

The maintained multi-timeframe policy uses a structured Dict observation rather than flattening history into one vector. Completed native sequences are 15m=96 bars, 1h=168, 4h=120 and 1d=60. Each clock has ordered values, availability and staleness tensors. Timeframe-specific left-padded residual TCNs use per-timestep LayerNorm and dilation schedules whose receptive fields cover the complete declared window, so neither convolution nor normalization can read future timesteps and the oldest declared observation can still affect the final latent. A timestep mask is true when any channel is available; unavailable channels are zeroed and remain individually masked. Train-range-only robust median/IQR statistics normalize sequence values per native feature while availability and staleness remain separate inputs.

Cross-asset attention preserves one contextual token per ordered symbol. Learned symbol embeddings bind token identity, the actor receives ordered per-symbol tokens so each target weight remains attached to its symbol, and only the critic/global context uses pooled portfolio representations. PPO and behavior cloning share this exact feature extractor.

The maintained dataset has 226 ordered point-in-time channels across the four clocks, including explicit BTC-relative return, rolling BTC correlation/beta, cross-sectional momentum rank and dispersion. Derived features preserve native source age through as-of alignment. Dataset and policy identities bind symbols, feature order, window lengths, availability rules and architecture settings, and reject mismatched artifacts.

## Reward contract

Reward schema v4 prioritizes absolute log-wealth growth. Baseline-relative growth is secondary. Drawdown is penalized only on new excess drawdown beyond a dead zone. Baseline underperformance uses a fixed real-time rolling window, tolerance and progressive hinge. Terminal penalties are continuous in equity shortfall rather than fixed jackpots. Every component is returned for audit.

## Dataset and run artifacts

Dataset identity v6 is recomputed from every observation, execution and accounting scalar/array, including effective-dated instrument filters, funding-event counts, quantity semantics and corporate actions. The maintained dataset artifact consists of canonical `manifest.json` and deterministic `arrays.npz`. `write_market_dataset_files`, `publish_market_dataset_artifact`, and `load_market_dataset_artifact` are the authoritative staging, publication, and loading APIs. Loading verifies exact file closure, rejects symlinks/root escapes, recomputes dataset identity, and checks array allow-list, shape, dtype, ordering and `MarketDataset` invariants. `MarketDatasetView` carries an immutable half-open absolute range and rejects subviews outside its parent range. The former duplicate `write_market_dataset_artifact` compatibility export is not part of the maintained API.

Training outputs are first written to `ArtifactStore/.staging/<run-id>`. `run.json` binds every declared file by relative path, byte size and SHA-256 together with dataset, environment, training-config and ensemble identities. Only a fully validated run is atomically moved to `runs/<run-id>` and made current through `latest.json`; partial failures are moved to `failed/<run-id>`.

## Training and nested walk-forward

`trade-rl train run` loads a validated dataset artifact, resolves content-addressed alpha/factor artifacts independently of filesystem location, requires a complete causal reward pre-roll, fits flat and sequence normalizers only on the declared training capability, constructs the real residual market environment, trains one Stable-Baselines3 checkpoint per seed, writes canonical configuration and ensemble manifests, validates the complete run and publishes atomically. Flat and structured policies both emit their matching SB3 loader contract; structured bundles include the native dataset/sequence adapter and never masquerade as flat policies. Maintained training, behavior cloning, checkpoint validation, configuration selection and outer testing share liquidation-at-close terminal accounting.

Direct training without selection evidence is labeled `research_exploratory`. A `research_selected_final` run requires an immutable `SelectionAuthorization` before normalizer fitting or model construction. That authorization binds the walk-forward run digest, gate-evidence digest, dataset ID, selected configuration, canonical candidate-config digest and fixed seed set. The authorization is copied into the staged run and therefore included in `run.json` file closure.

`trade-rl walk-forward run` uses the existing nested fold contract with concrete market adapters. Each fold receives disjoint train, checkpoint-validation, configuration-selection and sealed-test ranges. Training views include the maximum of trend and native-sequence history plus any reward/baseline pre-roll, so the nominal train start is never silently discarded. Flat and sequence normalizers fit only the chronological fold-train capability and are frozen and identity-bound thereafter. Masks, actions, portfolio/risk and bounded execution coordinates are pass-through or fixed-scaled.

Checkpoint validation selects the deterministic winner for each predeclared seed. Configuration selection retains per-seed evidence for median score, worst-seed uplift, seed dispersion, success fraction, daily turnover, cost fraction and drawdown, then separately evaluates the exact deterministic mean ensemble across those fixed seed members. That same ensemble rule is used for the sealed fold evaluation and final serving. The maintained hardened research runner executes walk-forward before final training, freezes the agreed recipe and complete seed set, writes `SelectionAuthorization`, and blocks final RL training when folds disagree, select baseline, or authorization identity differs. Sealed outer-test returns never flow back into training or selection.

Fold results retain complete execution evidence. Independent folds expose fold-distribution summaries rather than synthetic continuous-account metrics; continuous results require verified state handoff. Final `policy.zip` files are authoritative model artifacts. Training emits bounded, atomic intermediate policy checkpoints during each SB3 learning call. The checkpoint selector compares intermediate and final member checkpoints only on the checkpoint-validation range and records the selected content digest before sealed outer-test evaluation.

## Export contract

ONNX and TorchScript export a deterministic actor wrapper rather than the optimizer or replay state. Each export is compared with Stable-Baselines3 deterministic predictions on a fixed finite observation corpus. Shape, finite values and maximum absolute error must satisfy the configured tolerance. Requested ONNX failure rejects the run. TorchScript is best-effort and records an explicit `unsupported` result when conversion cannot be proven safe.

## Serving and release

Candidate bundle v4 binds action size/names/spec digest, observation schema/size, environment, normalizer and alpha/factor artifacts, but no release identity. The immutable candidate digest is approved by a separate HMAC-SHA256-authenticated `ReleaseAttestation` bound to dataset, evaluation, evidence-based gates, selected policy, source/dependency provenance, approver and time. Registry and runtime accept release evidence only through a purpose-bound `VerificationKey` whose purpose is `release-verification`; raw signing material and metadata/confirmation-purpose keys are rejected for released activation. Signing helpers are not re-exported by the runtime-facing release package and offline approval is isolated in `release.offline_approval`.

This purpose boundary prevents accidental key reuse inside the application but HMAC still uses symmetric key material. Public-key approval remains a future schema-versioned migration when an approved cryptography/KMS dependency is selected.

`StableBaselines3PolicyLoader` lives in the integration layer. It verifies the bundle-declared `policy-loader.json`, loads every declared PPO, SAC, TD3 or TQC member and runs deterministic probe observations before the runtime swaps active state. Every member must return a finite dynamic action vector inside `[-1, 1]`; the ensemble is averaged only when all members succeed. The runtime loads the same normalizer contract used in training.

Direct exchange connectivity, order routing, broker reconciliation and operational secrets/alerting are deliberately outside this repository's current capability boundary.

## Final causal sequence hardening

The maintained policy uses a shared per-asset actor over contextual asset tokens. The critic remains portfolio-level. PPO uses an index-backed rollout: persistent rollout state contains decision indices and current state, while overlapping native sequences are reconstructed from the immutable causal dataset only for sampled minibatches.

The BC teacher is an approximate portfolio teacher rather than a globally optimal oracle. It uses a bounded beam and executor-compatible minimum-notional and partial-fill semantics. Model artifacts record the approximation contract.

Structured sequence serving is native SB3 serving. The runtime restores the flat and sequence normalizers, validates symbols, ordered features, global features, cadence and sequence layout, rebuilds the Dict observation from a bounded rolling dataset, and requires a monotonic `ServingStateSnapshot` binding dataset, decision index, portfolio state, pending target and current observation before released inference. Live order routing remains outside the policy artifact.

Production remains NO-GO until the maintained GPU verification, 180 OOS days, a strictly positive paired block-bootstrap lower bound on RL-minus-baseline daily log excess, a signed fresh confirmation interval and paper-trading reconciliation all pass.
