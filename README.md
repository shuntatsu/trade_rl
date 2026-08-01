# Trade RL

Trade RLは、暗号資産を中心としたポートフォリオ配分を、**因果性・再現性・現実的な約定・証拠保存**を重視して検証する研究用の強化学習基盤です。

> **現在の判定**
>
> - 研究用の学習・評価・可視化: 利用可能
> - 外部署名付きのRead-only Paper Serving: 条件付きで利用可能
> - 取引所への注文送信: 未実装
> - Production status: **NO-GO**
> - 収益性の主張: なし

CI、GPU実行、Paper Servingが成功しても、利益、実資金投入、実取引所と同等の約定、運用認可は保証されません。

## 最短で試す

Pythonは`>=3.12,<3.13`です。

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

このQuickstartはPipeline確認用です。デモデータと短時間学習を、収益性評価やモデル選択には使用できません。詳しい手順は[START.md](START.md)を参照してください。

## 何ができるか

- 因果的な市場データとMulti-Timeframe特徴量artifactの構築
- PPO、CostCriticPPO、LagrangianPPO、SAC、TD3、TQCによる学習
- OracleまたはTrend teacherを使ったBehavior CloningからPPOへの初期化
- Nested walk-forward、Seed分布、Checkpoint選択、Sealed outer test
- Market・Limit・Stop-market注文を扱う状態付きOHLCV約定シミュレーション
- Trade RL Studioでの学習中探索・TensorBoard診断のRead-only表示
- 不変Artifact、外部Attestation、Fail-closed Serving
- 任意のPostgreSQL metadata catalog

直接取引所へ注文する機能、認証済みAccount接続、Production secret管理は含みません。

## 全体像

```text
市場データ
  -> 因果特徴量・Availability・Staleness
  -> 不変Dataset artifact
  -> Exploratory / Selected-final training
  -> Checkpoint選択・Walk-forward・Execution sensitivity
  -> Sealed評価・Fresh confirmation・Paper reconciliation
  -> Release attestation
  -> Read-only Paper Serving
```

Flat observationの正本は`baseline_residual_observation_v5`、Serving bundleの正本は`serving_bundle_v6`です。市場Dataset、学習Run、Checkpoint、評価Evidence、Serving bundleは、内容Digestと宣言済みFile closureで結合されます。途中失敗したRunは隔離され、正常な`latest.json`を上書きしません。

## Observation encoder

`training_run_config_v4`では、`observation_encoder`を1つだけ選びます。

| 値 | 用途 |
|---|---|
| `flat_mlp` | 単純なFlat observation |
| `asset_set` | 銘柄別tokenを使う非系列Encoder |
| `hierarchical_sequence_v2` | 15m・1h・4h・1dの因果系列と銘柄間依存 |

`hierarchical_sequence_v2`の経路は次のとおりです。

```text
時間足別Causal TCN
  -> Context + 15m + 1h + 4h + 1d
  -> Gated Cross-Timeframe Attention
  -> 銘柄別token
  -> Gated Cross-Asset Attention
  -> Actor / Critic
```

時間足Attentionと銘柄Attentionは、Head数、Layer数、FFN倍率、Gate biasを別々に設定します。実際に組み立てられた構造、銘柄順、Action順からArchitecture digestを生成し、BC、PPO、CostCriticPPO、LagrangianPPO、Checkpoint、構造化Export、Servingで一致を要求します。

詳細は[設定リファレンス](docs/CONFIGURATION.md)と[アーキテクチャ](docs/ARCHITECTURE.md)を参照してください。

## 因果性と約定

行`t`の判断には、行`t`のBar closeまでに利用可能な情報だけを使います。注文処理は最短でも`t + 1`のOpen以降です。

Dataset identityには価格と特徴量だけでなく、Availability、Staleness、Fee、Spread、Impact、Participation、Tick/Lot、Minimum notional、Funding、Borrow、Mark/Index price、Corporate action、上場・廃止期間などを含めます。

状態付き約定Engineは、Latency、Partial fill、Time in Force、Cancel/Replace、Gap、Funding、Borrow、Margin、Liquidationを扱います。ただしOHLCVからQueue position、Hidden liquidity、Auction、L2 depthを復元することはできません。

維持対象Rewardは**Reward schema v4**です。絶対対数資産成長を主目的にし、Baseline-relative growthは補助的な非劣後判定として扱います。

## 学習診断とStudio

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .

# 別ターミナル
npm ci --prefix studio
npm run dev --prefix studio
```

Live Trainingは`not exchange activity`、`not model-selection evidence`、`not sealed evaluation`、`not profitability evidence`です。BUY／SELL表示はTarget exposureの変化であり、取引所注文ではありません。

TensorBoard診断では、損失やKLに加えて、時間足Attention比率、Attention entropy、Gate飽和、欠損率、系列BlockのGradient normを確認できます。診断値は選択Evidenceとして使用しません。

詳細は[Studio README](studio/README.md)を参照してください。

## ServingとExport

Flat policyのExportと、構造化系列PolicyのExportは別契約です。`hierarchical_sequence_v2`は`structured_policy_export_v2`を使い、Canonical input順、Shape、Dtype、Parity corpus、Policy identity、Architecture digestをManifestへ固定します。

Serving bundleの正本は`serving_bundle_v6`です。Bundleは「Baselineか学習済みPolicyか」を表す`policy_mode`と、「ResidualかTarget-weightか」を表す`action_mode`を別々に固定します。構造化Loaderは、Bundle、Export manifest、Model digest、Observation schema、Architecture digestが一致しない場合、Policy実行前にFail closedします。

## 主要ドキュメント

- [ドキュメント一覧](docs/README.md)
- [最初の学習](START.md)
- [アーキテクチャ](docs/ARCHITECTURE.md)
- [設定リファレンス](docs/CONFIGURATION.md)
- [研究状態とProduction gate](docs/RESEARCH_STATUS.md)
- [Binance Public Data](docs/BINANCE.md)
- [Docker GPU運用](docs/operations/docker-gpu-full-training.md)
- [Trade RL Studio](studio/README.md)

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

## 非対応範囲

- 取引所WebSocketの常時接続
- 認証済みAccount access
- 注文送信、取消、訂正
- Broker reconciliationの実運用Connector
- Production secret管理
- Venue kill switchとOperational alerting

これらの実装と実証Evidenceが揃うまで、Production statusは**NO-GO**です。

## 維持対象の契約バージョン

維持対象の学習設定は`training_run_config_v4`、構造化Policy exportは`structured_policy_export_v2`、Serving bundleは`serving_bundle_v6`です。Quickstartを含む維持対象設定は、Executionの全Fieldと`require_full_reward_preroll: true`を明示し、dataclass既定値の変更で学習意味が静かに変わることを防ぎます。

データの因果性はFeature契約だけでなく、Raw Barのavailability、MarketDatasetのeconomic arrays、SequenceObservationのstalenessまで多層で検証します。constraint costは報酬と分離されていますが、hard safetyとLagrangian soft budgetは同義ではありません。
