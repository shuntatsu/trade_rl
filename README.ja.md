# Trade RL

Trade RLは、ポートフォリオ配分を対象とした研究用の強化学習基盤です。維持対象の方策は、決定論的な因果ベースラインへ制約付きResidual行動を加える方式と、銘柄順序を明示的に固定したTarget Weight方式を扱います。比較用のShadow baseline bookは学習方策と独立して更新されます。

> 能力判定は **研究実行可能**、**外部Attestation付きPaper Serving可能** です。取引所への直接注文送信は **NO-GO** のままです。テストや構造改善は、収益性や実資金投入を保証しません。

## 現在の維持対象

現在のリポジトリは、次の役割を明確に分離しています。

- 因果的な市場データとMulti-Timeframe特徴量artifact
- Exploratory学習と、承認付きSelected-final学習
- Nested walk-forward、Checkpoint選択、永続Sealed-test ledger
- 保守的なOHLCV状態付き約定シミュレータ
- 学習中の探索を観察するTrade RL Studio
- 外部署名で検証されるRead-only Paper Serving

PostgreSQLは、再計算不要なartifactの検索、cache identity、依存関係、状態、Sealed-test予約を保存する任意のメタデータカタログです。NumPy配列、dataset、checkpoint、model、run evidenceは従来どおり不変filesystem artifactに保存し、DBのBLOBを数値計算の正本にはしません。

通常のEnvironment遷移では、Market、Limit、Stop-market注文を状態として保持します。注文にはLatency、Time in Force、残数量、Trigger状態、Cancel-and-replaceの対応、決定論的OrderEvent証拠が含まれます。約定容量は処理対象バーの出来高を使い、OHLC内の経路は明示した仮定として記録します。これは板復元や実取引所と同等の約定を意味しません。

## 責務の分離

- `domain`: Dataset、Policy、Selection、Releaseなどの不変Identity
- `artifacts`: Canonical serialization、Hash、Staging、Atomic publish
- `release`: Offline署名と外部Attestationの検証契約
- `evaluation`: Return、Risk、Paired inference、Walk-forward、Gate
- `catalog`: Framework非依存のartifact catalog契約とPostgreSQL adapter
- `data`: Market calendar、特徴量、Execution data、Dataset検証
- `strategies`: 決定論的な因果Baseline
- `simulation`: Order lifecycle、Liquidity、約定、Cost、Carry、Margin、Accounting
- `risk`: Pre-trade制約、Portfolio risk、Emergency deleverage
- `rl`: Action、Observation、Normalizer、Reward、Environment、Training protocol
- `learning`: Teacher、Behavior cloning、Supervised data契約
- `serving`: Candidate bundle、Registry、Fail-closed runtime
- `integrations`: Stable-Baselines3など外部Framework adapter
- `workflows`: Training、Walk-forward、Artifact publicationのApplication orchestration
- `studio`: Local GUI用の型付きRead modelとJob管理API
- `cli`: 単一の`trade-rl`入口

依存方向の正本は`.importlinter`です。`trade_rl.telemetry`は標準ライブラリ中心の診断契約ですが、現時点ではLayer stackへ明示配置されていません。この点は[最新アーキテクチャ監査](docs/verification/2026-07-22-documentation-and-architecture-audit.md)に記録しています。

## Trade RL Studio

固定レイアウトのReact + Vite + TypeScript画面から、検証済みdataset、config、exploratory training job、run、比較結果、Evidence chain、read-only paper-serving状態を確認できます。Live Trainingでは、学習中の探索をSeed単位の市場リプレイとして表示します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .

# 別ターミナル
npm ci --prefix studio
npm run dev --prefix studio
```

BUY／SELL表示はTarget exposureの変化であり、取引所注文ではありません。Telemetryは診断専用で、モデル選択、artifact identity、Serving承認、注文実行には使いません。詳細は[`studio/README.md`](studio/README.md)を参照してください。

## 因果的な実データ入力

正式なデータ経路では、規則的なUnion clockを保持し、上場・廃止期間、実現した取引可否、情報利用可能時刻、特徴量ごとのAvailabilityとStalenessを別々に記録します。行`t`の方策判断は、行`t`のCloseまでに利用可能な情報だけを使い、最短でも`t + 1`のOpen以降で約定します。

`dataset_id`は、FeatureとAvailabilityだけでなく、Fee、Maker/Taker fee、Spread、Participation、Lot/Tick、Minimum notional、Borrow/Funding、Buy/Sell制限、Mark/Index価格、配当・分割・上場廃止、Cash rate、Volume unit、Contract multiplierを含む実行・会計配列から再計算されます。正式artifactは任意ID、Symlink、Root escape、未宣言ファイルを拒否します。

正式APIは次の3つです。

- `write_market_dataset_files`: 決定論的Staging file作成
- `publish_market_dataset_artifact`: 排他的なAtomic publish
- `load_market_dataset_artifact`: IdentityとFile closureを検証した読込

## Observation、学習、Servingの整合性

Observation schema v3は、Feature、Availability、Staleness、Baseline、Factor、現在・要求Portfolio、Fill/Cost/Capacity、Pending order、Cash/Net/Gross/Margin、Previous actionを含みます。Multi-Timeframe方策は、15m、1h、4h、1dのNative sequenceをDict observationとして受け取り、各時間足専用の因果TCNとCross-asset attentionを使用します。

学習EnvironmentとServingは同じObservation builder、Normalizer、Sequence adapter、Symbol/Feature順序を検証します。Candidate bundleは **Serving bundle v5** です。Bundle自体にApproval秘密情報は含めず、別の`ReleaseAttestation`がBundle、Selected-final run、Walk-forward、Gate、Fresh confirmation、Source commit、Dependency provenance、Approver、Expiryを結合します。

Runtimeは、署名、Identity、File closure、Action shape、有限値、Bounds、Observation schema、Normalizer、Pending-order state、ServingStateSnapshotが一致しない場合、Policy実行前またはActivation前にFail closedします。

## Rewardと評価

維持対象は **Reward schema v4** です。主目的は絶対対数資産成長で、Baseline-relative growthは補助目的です。DrawdownとBaseline劣後は、許容幅を超えて悪化した増分だけを段階的に罰します。

Nested walk-forwardでは、Train、Checkpoint validation、Configuration selection、Sealed outer testを分離します。NormalizerはFold train capabilityだけでFitし、その後Freezeします。Outer testは選択完了後に一度だけ開き、Servingと同じ決定論的Mean ensembleを評価します。

Production statusは、GPU検証、十分なOOS期間、Paired block-bootstrap、Fresh confirmation、Paper reconciliationなどの必須Gateが揃うまで`NO-GO`です。

## セットアップと最初の学習

Python 3.12が必要です。

```bash
uv sync --extra dev --extra train-sb3
uv run trade-rl --version
```

最短手順は[START.md](START.md)を参照してください。

```bash
uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset

uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-001
```

## PostgreSQL artifact catalog

```bash
cp .env.example .env
docker compose up -d postgres

uv sync --extra postgres
export TRADE_RL_DATABASE_URL=postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
uv run trade-rl catalog migrate
uv run trade-rl catalog health
```

DB未設定時はFilesystemだけで動作します。Market dataset publish時にDBが利用可能なら、Dataset ID、Artifact digest、Canonical cache key、Location、Sizeなどを登録します。

停止してVolumeを保持する場合:

```bash
docker compose down
```

Volumeも削除する場合だけ:

```bash
docker compose down -v
```

## Docker GPU完全リサーチ実行

```bash
docker compose -f compose.training.yaml build trainer
docker compose -f compose.training.yaml run --rm trainer
```

CUDA preflight、学習、評価、Research gateのいずれかが失敗すると非ゼロ終了します。成功はPipelineとResearch evidenceであり、利益を保証しません。Metadata modeの既定は`frozen_snapshot`で、Current official payloadをPoint-in-time履歴として偽装しません。最高Integrityの`historical_signed`は、期間を覆う署名済みMetadataが必要です。

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

詳細は[アーキテクチャ](docs/ARCHITECTURE.md)、[研究結果の扱い](docs/RESEARCH_STATUS.md)、[Binance Public Data Workflow](docs/BINANCE.md)、[2026-07-22アーキテクチャ監査](docs/verification/2026-07-22-documentation-and-architecture-audit.md)を参照してください。
