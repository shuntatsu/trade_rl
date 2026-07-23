# Trade RL

Trade RLは、暗号資産を中心としたポートフォリオ配分を、**因果性・再現性・実行現実性・証拠保存**を重視して研究するための強化学習基盤です。

決定論的なベースライン戦略に制約付きResidual行動を加える方式と、銘柄ごとのTarget Weightを直接出力する方式を扱います。学習方策とは独立したShadow baseline bookを維持し、候補方策が本当にベースラインを上回ったかを比較できるようにしています。

> **現在の状態**
>
> - 研究ワークフロー: 利用可能
> - 保守的な状態付きOHLCV約定シミュレーション: 利用可能
> - Trade RL Studioによる学習中の探索観察: 利用可能
> - 適格なSelected-final bundleのRead-only Paper Serving: 利用可能
> - 取引所への直接注文送信: **未実装**
> - Production status: **NO-GO**
> - 収益性の主張: **なし**

CI、学習、Studio表示、Paper Servingが成功しても、利益、本番資金の投入、実取引所と同等の約定、運用認可を保証しません。

## まず動かす

Pythonは`>=3.12,<3.13`です。最短の動作確認では、決定論的なデモ相場を生成し、小規模なPPO学習を実行します。

```bash
python -m pip install uv
uv sync --extra dev --extra train-sb3

uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset

uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-001
```

このQuickstartはPipelineの確認用です。デモデータや短時間学習を収益性評価には使用できません。

詳しい手順、成果物の見方、GPU設定、実データへの置換方法は[START.md](START.md)を参照してください。

## このリポジトリで扱うもの

Trade RLは、次の機能を一つの研究基盤として提供します。

1. 因果的な市場データとMulti-Timeframe特徴量artifact
2. Exploratory学習とSelected-final学習
3. Nested walk-forward、Checkpoint選択、永続Sealed-test ledger
4. Market・Limit・Stop-market注文を扱う状態付き約定シミュレータ
5. 学習中の探索行動を観察するTrade RL Studio
6. 外部署名と証拠Chainで検証するRead-only Paper Serving
7. Artifact metadataと依存関係を管理する任意のPostgreSQL catalog

## 重要な設計原則

### 因果性とリーク防止

行`t`の判断には、行`t`のBar closeまでに利用可能な情報だけを使います。注文は最短でも`t + 1`のOpen以降で処理されます。

Datasetは、価格や特徴量だけでなく、次の情報もIdentityへ含めます。

- Feature availabilityとstaleness
- Fee、Maker/Taker fee、Spread、Impact
- Participation、Volume unit、Contract multiplier
- Tick size、Lot size、Minimum notional
- Borrow、Funding、Cash rate
- Mark price、Index price
- Corporate action、上場・廃止、売買制限

Published artifactは任意ID、Symlink、Root escape、未宣言ファイルを拒否します。

### 再現可能なArtifact

Market datasetは、検証済みの不変Directoryとして保存されます。

```text
my-dataset/
├── manifest.json
└── arrays.npz
```

正式APIは次の3つです。

- `write_market_dataset_files`: 決定論的なStaging fileを作成
- `publish_market_dataset_artifact`: 排他的にAtomic publish
- `load_market_dataset_artifact`: IdentityとFile closureを検証して読込

学習Run、Walk-forward結果、Checkpoint、Model、Evaluation evidenceも不変Artifactとして公開されます。失敗Runは`failed/<run-id>`へ隔離され、正常な`latest.json`を上書きしません。

### ベースライン比較

学習方策とShadow baseline bookを独立して更新します。ゼロResidual行動はベースラインと同じPortfolioになりますが、Rewardはゼロではなく、そのベースライン戦略の絶対資産成長を反映します。

候補方策の選択と評価では、Servingが実際に読み込む決定論的Mean ensembleを使用します。

## Action・Observation・Reward

### Action

維持対象のEnvironmentは、次のAction modeを扱います。

- Fast、Slow、Risk controlを持つResidual action
- 任意のAlpha scaleと因果Factor residual
- 銘柄順序をIdentityへ固定した`target_weight:<symbol>`直接出力

Alphaが無効な場合は、意味のないAlpha actionをAction spaceへ残しません。Action名とActionSpec digestは、学習ArtifactとServing bundleへ結合されます。

### Observation

Flat observationの正本は`baseline_residual_observation_v5`です。

主な内容:

- Feature、Availability、Staleness
- Baseline、Factor、現在Portfolio、要求Portfolio
- Cash、Net/Gross exposure、Margin、Drawdown
- Fill、Fee、Capacity、Previous action
- Persistent pending order state

Pending orderは銘柄ごとに、残Notional比率、Order type、Status、Age、Eligible delay、Trigger状態、Expiry距離の7座標を持ちます。

NormalizerはFoldのTrain capabilityだけでFitし、その後Freezeします。Categorical maskは連続値として変形せず、厳密に保持します。

### Reward

維持対象は**Reward schema v4**です。

主目的はHybrid bookの絶対対数資産成長です。Baseline-relative growthは補助的な非劣後判定として使い、独立した第二の主目的にはしません。

DrawdownとBaseline劣後は、許容幅を超えて悪化した増分だけを段階的に罰します。

## 約定シミュレーション

通常のEnvironment遷移、Baseline reward pre-roll、Compatibility target execution、Sensitivity replay、Deterministic replayは、同じ状態付き注文Engineを使用します。

扱う要素:

- Market、Limit、Stop-market
- LatencyとEligible delay
- GTCなどのTime in Force
- Partial-fill carry
- Cancel-and-replace
- Trigger状態とExpiry
- Shared processing-bar capacity
- Maker/Taker cost、Spread、Impact、Slippage
- Lot/Tick rounding、Minimum notional
- Borrow、Funding、Margin、Liquidation
- Corporate actionとDelisting settlement

Selected-finalへの昇格には、保守的なOHLC path、処理バー容量、Partial-fill carry、完全なOrder evidence、期待する`execution_policy_digest`が必要です。

ただしOHLCVから、真のIntrabar順序、Queue position、Hidden liquidity、Auction、Adverse selection、L2 depthを復元することはできません。このSimulatorは保守的な研究近似であり、実取引所と同等の約定を証明するものではありません。

## 学習と評価

対応Algorithm:

```text
ppo
sac
td3
tqc
```

Nested walk-forwardは、次の段階を分離します。

1. Fold-local training
2. Checkpoint validation
3. Configuration selection
4. 一度だけ開くSealed outer test
5. Execution sensitivity
6. Release gate

Independent foldは、Median、Weighted mean、Win rate、Worst foldなどの分布として報告します。連続期間と検証済みAccount-state handoffがない限り、複数Foldを一つの連続Portfolio returnやDrawdownとして扱いません。

Production statusは、十分なOOS期間、正のPaired block-bootstrap下限、署名済みFresh confirmation、完全な保守的Execution evidence、実Paper reconciliationなどの必須Gateが揃うまで`NO-GO`です。

## Trade RL Studio

`studio/`にはReact + Vite + TypeScript製のローカル研究画面があります。

主な用途:

- 検証済みDatasetの確認
- Training configとJob管理
- Run artifact、比較結果、Evidence chainの確認
- Read-only Paper Serving状態の確認
- Live Trainingによる探索行動の観察

起動方法:

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .

# 別ターミナル
npm ci --prefix studio
npm run dev --prefix studio
```

Live Trainingでは、学習中の探索をSeed単位・Episode単位の市場リプレイとして表示します。BUY／SELL MarkerはTarget exposureの変化であり、取引所注文ではありません。

契約上、Live Trainingは`not exchange activity`、`not model-selection evidence`、`not sealed evaluation`、`not profitability evidence`です。Telemetryは診断専用で、Checkpoint選択、Artifact identity、Serving承認、注文実行には使用しません。

詳細は[studio/README.md](studio/README.md)を参照してください。

## ServingとRelease

Serving bundle v5の正本は`serving_bundle_v5`です。

Bundleは、選択済みPolicy、Dataset、Environment、Normalizer、Execution policy、Training run、Evaluation evidenceなどを不変Identityへ結合します。Approval秘密情報はBundle内へ埋め込まず、別のEd25519 `ReleaseAttestation`で次を結合します。

- Bundle digest
- Training run
- Selection proposalとAuthorization
- Walk-forwardとGate evidence
- Fresh confirmation
- Paper reconciliation
- Selected policy
- Source commit
- Dependency provenance
- ApproverとExpiry

RuntimeとRegistryには目的別Public keyだけを渡します。Private keyはOffline CLIでのみ使用します。

Unsigned、期限切れ、Unknown key、改変済み、Evidence不足、Execution identity不一致は、Activation前にFail closedします。

Framework非依存Serving層は`PolicyLoader`を受け取り、Stable-Baselines3向けの正式Adapterとして`trade_rl.integrations.StableBaselines3PolicyLoader`を提供します。

## PostgreSQL artifact catalog

PostgreSQLは任意の**metadata catalog**です。

保存するもの:

- Artifact identityとDigest
- Canonical cache key
- LocationとSize
- Dependency edge
- Lifecycle status
- Persistent sealed-test reservation

NumPy配列、Dataset、Checkpoint、Model、Run evidenceは不変の**filesystem artifact**として保存し、PostgreSQLを数値計算の正本にはしません。

起動方法:

```bash
cp .env.example .env
docker compose up -d postgres

uv sync --extra postgres
export TRADE_RL_DATABASE_URL=postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
uv run trade-rl catalog migrate
uv run trade-rl catalog health
```

DBを設定しなくても、通常のFilesystem運用は可能です。ただし、Processを跨ぐSealed-testの一意性保証には永続Ledgerが必要です。

## Binance Public Data

Public Binance dataから決定論的Datasetを構築できます。

```bash
uv run trade-rl data binance \
  --market usds-m \
  --symbol BTCUSDT \
  --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --transport vision \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --minimum-notional 5 \
  --listed-at 2019-09-08T00:00:00Z \
  --output artifacts/datasets/binance-btcusdt
```

SpotとUSDⓈ-MのLinear productを扱います。現行Accounting modelがLinearであるため、COIN-M inverse futuresはFail closedします。

Metadata mode:

- `historical_signed`: 署名済みPoint-in-time metadataを使う最高Integrity mode
- `frozen_snapshot`: Current official payloadをByte単位で固定し、非Point-in-time証拠であることを明示
- `conservative_static`: Versioned static payloadを使う保守的近似

Current値を過去の真実として暗黙に遡及適用することはありません。

詳細は[docs/BINANCE.md](docs/BINANCE.md)を参照してください。

## Docker GPU完全リサーチ実行

CUDA対応環境では、Binance Multi-Timeframe研究WorkflowをContainerで実行できます。

```bash
docker compose -f compose.training.yaml build trainer
docker compose -f compose.training.yaml run --rm trainer
```

CUDA preflight、学習、評価、Research gateのいずれかが失敗すると非ゼロ終了します。正常終了はPipelineとResearch evidenceであり、収益性やProduction readinessを意味しません。

運用手順は[Docker GPU完全学習](docs/operations/docker-gpu-full-training.md)を参照してください。

## 品質確認

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

Export検証を含める場合:

```bash
uv sync --extra dev --extra train-sb3 --extra export
```

## リポジトリ構成

```text
trade_rl/
├── artifacts/      # Serialization、Hash、Staging、Atomic publish
├── catalog/        # Artifact catalog契約とPostgreSQL adapter
├── cli/            # trade-rlコマンド
├── data/           # Market data、Calendar、Feature、Dataset検証
├── domain/         # 不変IdentityとDomain contract
├── evaluation/     # Metrics、Walk-forward、Gate、Paired inference
├── integrations/   # Stable-Baselines3などの外部Adapter
├── learning/       # Teacher、Behavior cloning、Supervised contract
├── release/        # Offline署名とAttestation
├── risk/           # Pre-trade riskとEmergency deleverage
├── rl/             # Action、Observation、Reward、Environment、Training
├── serving/        # Bundle、Registry、Fail-closed runtime
├── simulation/     # Order lifecycle、Fill、Cost、Accounting
├── strategies/     # 決定論的な因果Baseline
├── studio/         # Local GUIとTyped read model
├── telemetry/      # Append-only診断Event
└── workflows/      # TrainingとEvaluationのApplication orchestration
```

依存方向の正本は`.importlinter`です。

## 関連ドキュメント

- [学習クイックスタート](START.md)
- [アーキテクチャ](docs/ARCHITECTURE.md)
- [研究状態とProduction gate](docs/RESEARCH_STATUS.md)
- [Binance Public Data Workflow](docs/BINANCE.md)
- [Docker GPU完全学習](docs/operations/docker-gpu-full-training.md)
- [Trade RL Studio](studio/README.md)
- [2026-07-23 Architecture Audit Closeout](docs/verification/2026-07-23-architecture-audit-closeout.md)

## 現在の非対応範囲

このリポジトリには、次の本番取引機能はありません。

- 取引所WebSocketによる常時接続
- 認証済みAccount access
- 注文送信、取消、訂正
- Broker reconciliationの実運用Connector
- Production secret管理
- Venue kill switch
- Operational alerting

これらを実装するまで、また実証的なGateを通過するまで、Production statusは**NO-GO**です。
