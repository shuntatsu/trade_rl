# Trade RL

Trade RLは、暗号資産を中心とした**単一InstrumentごとのTarget exposure**を、因果性・再現性・現実的な約定・証拠保存を重視して検証する研究用の強化学習基盤です。

> **現在の判定**
>
> - 研究用の学習・評価・可視化: 利用可能
> - 外部署名付きのRead-only Paper Serving: 条件付きで利用可能
> - Direct exchange routing / 取引所への注文送信: **未実装**
> - Production status: **NO-GO**
> - Profitability claim / 収益性の主張: **なし**

CI、GPU実行、Paper Servingの成功は、利益、実資金投入、実取引所と同等の約定、Production認可を保証しません。実装済み、CI検証済み、実証評価済み、収益性、運用認可は別の状態として扱います。

## Maintained contract

```text
one maintained run
  = one instrument
  = one target-weight action
  = one checkpoint and evidence chain
```

初期Maintained presetはBinance USDS-M perpetualの`BTCUSDT`です。複数銘柄へ資金を配分するときは、銘柄ごとに独立RunとBudgetを作り、Repository外のCapital allocatorで配分します。Maintained single-symbol contractの詳細は[SINGLE_SYMBOL.md](docs/SINGLE_SYMBOL.md)を参照してください。

Universal U3-U6は別の**研究用共有Policy学習経路**です。複数のtrain symbolから1つのPolicyを学習しますが、各episodeと推論対象は常に1つのconcrete instrumentです。BTC・ETH・BNBなどを同時にportfolio配分するPolicyではありません。現行契約は[UNIVERSAL_TRAINING.md](docs/UNIVERSAL_TRAINING.md)を参照してください。

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

このQuickstartはPipeline確認用です。デモデータと短時間学習を、収益性評価やモデル選択には使用できません。環境構築、入力Artifact、Troubleshootingを含む手順は[START.md](START.md)を参照してください。

## できること

- 因果的な市場データとMulti-Timeframe特徴量Artifactの構築
- PPO、CostCriticPPO、LagrangianPPO、SAC、TD3、TQCによる研究学習
- train-only `causal_alpha_ridge` teacherを使うUniversal Behavior Cloning、critic warm start、PPO-family研究
- Nested walk-forward、Seed分布、Checkpoint選択、Sealed outer test
- Market、Limit、Stop-marketを扱う状態付きOHLCV約定シミュレーション
- Trade RL StudioとTensorBoardによるRead-only研究診断
- 不変Artifact、外部Attestation、Fail-closed Paper Serving
- 任意のPostgreSQL **metadata catalog**

数値データの正本は**filesystem artifact**です。PostgreSQLはArtifact metadata、location、dependency、lifecycleなどを管理する任意Catalogであり、ModelやDatasetの数値BLOBを計算上の正本にはしません。

## Research flow

```text
市場データ
  -> 因果特徴量とDataset identity
  -> 学習 / Checkpoint validation
  -> Configuration selection
  -> Sealed evaluation
  -> Execution sensitivity / Paper reconciliation
  -> Release evidence
  -> Read-only Paper Serving
```

行`t`の判断には行`t`のBar closeまでに利用可能な情報だけを使い、注文処理は最短でも`t + 1`のOpen以降です。Normalizer、Teacher、Checkpoint、Evaluation、Servingの詳細な因果・Identity契約は[ARCHITECTURE.md](docs/ARCHITECTURE.md)、[CONFIGURATION.md](docs/CONFIGURATION.md)、[UNIVERSAL_TRAINING.md](docs/UNIVERSAL_TRAINING.md)が正本です。

約定EngineはLatency、Partial fill、Time in Force、Cancel/Replace、Gap、Funding、Borrow、Margin、Liquidationを扱いますが、OHLCVからQueue position、Hidden liquidity、Auction、L2 depthを復元することはできません。制約と限界は[EXECUTION_ROBUSTNESS.md](docs/EXECUTION_ROBUSTNESS.md)を参照してください。

## Studioと診断

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .

# 別ターミナル
npm ci --prefix frontend
npm run dev --prefix frontend
```

Live Trainingは`not exchange activity`、`not model-selection evidence`、`not sealed evaluation`、`not profitability evidence`です。LONG／SHORT／CLOSE表示はTarget exposureの変化であり、取引所注文ではありません。UIの責務は[frontend/README.md](frontend/README.md)を参照してください。

## Repository map

```text
trade_rl/        Python本体
frontend/        React/Vite研究UI
scripts/         CI・検証・運用補助スクリプト
tests/           契約・単体・統合・E2Eテスト
docs/            現行Reference、Runbook、研究状態、履歴資料
examples/        Quickstartと再現可能な実行例
docker/          Training imageとCompose定義
LICENSES/        Third-party notice等のライセンス資料
```

文書の正本と履歴資料の区別は[ドキュメント一覧](docs/README.md)に集約しています。

## 主要ドキュメント

- [ドキュメント一覧](docs/README.md)
- [最初の学習](START.md)
- [Maintained single-symbol workflow](docs/SINGLE_SYMBOL.md)
- [Universal U3-U6 training](docs/UNIVERSAL_TRAINING.md)
- [アーキテクチャ](docs/ARCHITECTURE.md)
- [設定リファレンス](docs/CONFIGURATION.md)
- [研究状態とProduction gate](docs/RESEARCH_STATUS.md)
- [Reward objective](docs/REWARD_OBJECTIVE.md)
- [Execution robustness](docs/EXECUTION_ROBUSTNESS.md)
- [Binance Public Data](docs/BINANCE.md)
- [Docker GPU運用](docs/operations/docker-gpu-full-training.md)
- [Licensing](docs/LICENSING.md)
- [Licensing provenance](docs/LICENSING_PROVENANCE.md)

## 品質確認

Repositoryの標準品質ゲートには、Pythonのlint/format/type/import境界、全pytestとcoverage、Frontend test/typecheck/build、Windows/Linux compatibility、Training image、Nautilus capability、PostgreSQL integrationを含みます。正確なCI構成はRepositoryのworkflowを正本とし、古い実行結果を現在HEADの成功として扱いません。

## 非対応範囲

- 取引所WebSocketの常時接続
- 認証済みAccount access
- 注文送信、取消、訂正
- Broker reconciliationの実運用Connector
- Production secret管理
- Venue kill switchとOperational alerting
- 複数Run間のCapital allocator

これらの実装と実証Evidence、運用認可が揃うまでProduction statusは**NO-GO**です。

## License

Maintained Trade RL source at and after the license transition is distributed under `LGPL-3.0-or-later`. Historical revisions already distributed under MIT keep their historical MIT grant. Exact boundaries and notices are documented in [docs/LICENSING.md](docs/LICENSING.md), [docs/LICENSING_PROVENANCE.md](docs/LICENSING_PROVENANCE.md), `LICENSE`, `LICENSES/`, and `LICENSES/THIRD_PARTY_NOTICES.md`.

NautilusTrader is an external upstream dependency developed by Nautech Systems. Trade RL is an independent project and is not affiliated with, endorsed by, sponsored by, or an official work of Nautech Systems.