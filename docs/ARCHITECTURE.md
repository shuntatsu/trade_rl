# Trade RL Architecture

## Status

Trade RLは、Portfolio RLを因果データ、状態付き約定、Nested walk-forward、不変Evidence、Fail-closed Servingで検証する研究基盤です。Direct exchange tradingは未実装で、Production statusは`NO-GO`です。

Architecture integrity、Empirical profitability、Operational authorizationは別のGateです。CI通過は利益や運用認可を意味しません。

## Responsibility map

```text
trade_rl/
  domain/        不変Identityと標準ライブラリCanonical primitive
  telemetry/     Append-only診断Event契約
  artifacts/     Hash、File closure、Atomic publish
  release/       Offline署名と外部Attestation
  evaluation/    Metrics、Paired inference、Fold、Gate、AUM capacity
  catalog/       Artifact metadata契約とPostgreSQL adapter
  data/          Market calendar、因果特徴量、Dataset
  strategies/    決定論的な因果Baseline
  simulation/    Order、Liquidity、Fill、Carry、Margin、Accounting
  risk/          Pre-trade制約とEmergency deleverage
  rl/            Action、Observation、Reward、Environment、Training
  learning/      Teacher、Behavior cloning、Supervised data
  serving/       Bundle、Registry、Fail-closed runtime
  integrations/  Stable-Baselines3などのAdapter
  workflows/     Training、Walk-forward、Publication orchestration
  studio/        Local research read modelとJob control
  operations/    Maintained GPU・運用検証entry point
  cli/           trade-rlコマンド
```

Import Linterの強制順序は次のとおりです:

```text
cli
operations
studio
workflows
integrations
serving
learning
rl
risk
simulation
strategies
data
catalog
evaluation
release
artifacts
telemetry
domain
```

上位Layerから下位Layerへの依存だけを許可します。`operations`は維持対象のGPU・運用検証を所有し、Exampleには薄い互換entry pointだけを残します。`domain`は標準ライブラリ専用です。`telemetry`は数値計算やModel frameworkへ依存しません。ServingはTraining workflowやStable-Baselines3へ直接依存しません。

## Data flow

```text
Raw market source
  -> point-in-time metadata resolution
  -> causal feature construction
  -> Dataset identity and immutable publication
  -> fold-scoped normalizer fitting
  -> training / checkpoint validation
  -> configuration selection
  -> one-shot sealed evaluation
  -> release evidence and serving bundle
```

Bar timestampはClose時刻です。行`t`のObservationにはCloseまでに利用可能な情報だけを含め、注文は最短でも`t + 1`のOpen以降で処理します。

Dataset identityは、FeatureだけでなくAvailability、Staleness、Tradability、Fee、Spread、Participation、Tick/Lot、Funding、Borrow、Mark/Index price、Corporate action、Cash rateなどを含みます。Artifact load時にFile closure、Symlink、Shape、Dtype、Ordering、Digestを検証します。

## Action、Observation、Reward

Action schemaは`portfolio_action_v3`です。Residual modeではゼロActionがBaseline identity、Target-weight modeでは銘柄順に対応する直接Weightを出力します。

Flat observationの正本は`baseline_residual_observation_v5`です。Feature、Availability、Staleness、Current/Requested portfolio、Fill/Cost/Capacity、Cash、Exposure、Margin、Previous action、Pending order stateを含みます。

Reward schema v4は絶対対数資産成長を主目的にします。Baseline-relative growthは補助的な非劣後判定で、DrawdownとBaseline劣後は許容幅を超えて悪化した増分だけを罰します。

## Observation encoder architecture

`training_run_config_v3`は`observation_encoder`と階層Actor契約を明示します。

- `flat_mlp`: Flat observationをMLPへ入力
- `asset_set`: 銘柄別構造を使う非系列Encoder
- `hierarchical_sequence_v2`: Clock内、Clock間、銘柄間を階層的に処理

### Hierarchical sequence v2

Native Clockは15m、1h、4h、1dです。各Clockは`values`、`available`、`staleness`を保持します。

```text
15m/1h/4h/1d values
  -> Clock別 left-padded Causal TCN
  -> 最後の利用可能な因果State
  -> Clock latent

Current snapshot + Asset state
  -> Context token

[Context, 15m, 1h, 4h, 1d]
  -> Gated Cross-Timeframe Attention
  -> Asset token
  -> Symbol embedding
  -> Gated Cross-Asset Attention
  -> Actor token / Critic global context
```

TCNの受容野は宣言Window全体を覆います。完全欠損ClockはAttention keyからMaskし、入力変更とGradientが出力へ漏れないことをTestします。Context tokenは常に存在するため、全Clock欠損でもAll-masked rowを作りません。

Gated TransformerはPre-LayerNorm、Attention residual gate、FFN residual gateを使います。Gate biasの既定は`-2.0`で、新BlockをIdentityに近い状態から導入します。

Timeframe AttentionとAsset AttentionはHead、Layer、FFN倍率、Gate biasを別々に設定します。詳細は[CONFIGURATION.md](CONFIGURATION.md)を参照してください。

## Policy identity

`sb3_policy_identity_v1`は、ConfigをそのままHashせず、組み立て済みFeature extractorから構造を読み取ります。

Sequence identityは次を含みます。

- Clock順とWindow
- Input/Latent width
- TCN width、Dilation、受容野
- Timeframe/Asset Attention設定
- Symbol順
- Action名と順序

BC、PPO、CostCriticPPO、LagrangianPPOは同じPolicy architecture identityを使用します。Algorithm固有IdentityはこのIdentityへ合成します。

CheckpointはTraining config digestとArchitecture identityを別々に保存します。Resume時にどちらかが異なればFail closedします。

## Stateful execution

```text
Target proposal
  -> Risk projection
  -> Target-to-order reconciliation
  -> Persistent OrderBookState
  -> Latency / eligibility / trigger / expiry
  -> Shared processing-bar capacity
  -> Fill and cost
  -> BookState accounting
  -> OrderEvent evidence
```

Market、Limit、Stop-market、Partial-fill carry、GTC/IOC/Day、Cancel/Replace、Gap、Funding、Borrow、Margin、Liquidation、Corporate actionを扱います。

OHLCVから真のIntrabar順序、Queue position、Hidden liquidity、Auction、Adverse selection、L2 depthは復元できません。Conservative pathは研究上の保守的近似です。

## Training and evaluation

`trade-rl train run`はDatasetを検証し、Train capabilityだけでNormalizerをFitし、SeedごとにModelを学習し、RunをAtomic publishします。

Nested walk-forwardは次を分離します。

1. Fold-local training
2. Checkpoint validation
3. Configuration selection
4. One-shot sealed outer test
5. Execution sensitivity
6. Release gate

Configuration selection、Sealed評価、Servingは同じ決定論的Mean ensembleを使用します。独立Foldは分布として報告し、Account-state handoffなしに1本の連続Portfolioへ合成しません。

## TensorBoard and Studio boundary

Exploratory trainingはAppend-only `training_telemetry_v1`とTensorBoard scalarを出力できます。

Sequence diagnosticsは、Clock別Attention share、Missing ratio、Attention entropy、Gate飽和、Gradient normを記録します。診断Forwardは低頻度で、通常の学習Hot pathから分離します。

TelemetryとTensorBoardは、Fitting、Checkpoint選択、Sealed評価、Run identity、Release、Order executionへ使用しません。StudioはRead-onlyに表示します。

## Export and serving

Flat actorのONNX/TorchScript Exportと、Sequence actorの構造化Exportを分離します。

`structured_policy_export_v2`はCanonical Dict input order、Shape、Dtype、Parity corpus、Model digest、Policy identity、Architecture digestを保存します。構造化Modelは`policy.structured.torchscript.pt`です。

Serving bundleの正本は`serving_bundle_v6`です。`policy_mode`はBaselineか学習済みPolicyかを表し、`action_mode`はResidualかTarget-weightかを表します。この2つは別IdentityとしてBundle digestとRuntime identityへ固定されます。`CanonicalStructuredPolicyLoader`は、Sequence observation schema、Bundle file closure、Export manifest、Model digest、Architecture digestをPolicy実行前に検証します。

Release approvalはBundle外のEd25519 Attestationとして保持します。RuntimeへPrivate keyを渡しません。

## Artifact store and PostgreSQL

Filesystem artifactが数値データの正本です。Runは`.staging/<run-id>`で作成し、検証後に`runs/<run-id>`へAtomic publishします。

PostgreSQLは任意のmetadata catalogです。Artifact identity、Cache key、Location、Size、Dependency、Lifecycle、Sealed-test reservationを保存します。ModelやDatasetのBLOBを数値計算の正本にはしません。

## Privileged GPU boundary

Self-hosted GPU runnerはPull Requestの任意Codeを実行しません。維持対象の完全実行は、Owner-authored `main`、指定Environment、固定Action SHA、read-only Sourceを要求します。

GPU Evidenceが欠けてもCPU CIを成功扱いに偽装せず、GPU Gateを未取得として別に記録します。

## Maintained contract clarifications

因果データ契約は`data/contracts.py`だけに閉じていません。FeatureとInstrumentの宣言は`contracts.py`、Barと`available_at`は`RawMarketSeries`、市場配列は`MarketDataset`、`values`・`available`・`staleness`は`SequenceObservation`がそれぞれ保持し、frozen型、read-only配列、実行時検証を重ねて将来参照を拒否します。

7種類のconstraint costはscalar rewardから独立したCost Critic／Lagrangian用チャネルです。すべてがhard constraintという意味ではありません。Weight、gross、margin、liquidation、exchange rule等のhard safetyは環境とpre-trade riskが強制し、turnoverやexecution cost等はsoft budgetとして扱います。

`hierarchical_gate_target_v1`という互換名はCheckpoint identityのため維持しますが、sigmoid出力の意味はBernoulli Gateではなく連続的な**change intensity**です。TensorBoardはdeterministic composed actionと探索後のsampled policy actionのL1差を、Environment infoはsampled policy actionから約定後effective filled weightsまでの差を記録します。

Constrained PPOのPR C正本は修正版PR #193です。PR #191は置換前のDraft履歴であり、維持対象実装の根拠には使用しません。

Workflow securityはrunnerの任意の表示名をallowlistするのではなく、GitHub-hosted形式かprivileged runnerかを分類し、privileged runnerについてtrigger、owner、main、Environment、権限、immutable checkoutを検証します。この方針をrunner classificationと呼びます。

構造化配信の正本は`structured_policy_export_v2`と`serving_bundle_v6`です。秘密鍵ファイルのloaderは`offline_keys`、鍵生成と署名は`offline_signing`、承認署名は`offline_approval`等の明示的offline moduleへ限定します。Import Linterはruntime/trainingからこれらへの静的依存を禁止しますが、OS sandboxそのものを主張するものではありません。
