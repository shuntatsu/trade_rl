# Trade RL 学習クイックスタート

このページは、リポジトリ取得直後に **データartifact生成 → PPO学習 → 成果物確認 → Studioで探索観察** までを実行する手順です。

最初の例は決定論的なデモ相場を使います。デモデータと短時間学習はPipelineの動作確認用であり、収益性評価には使用できません。

> Production status: **NO-GO**  
> 学習、テスト、Studio表示、Paper Servingの成功は、本番資金での運用許可や利益を保証しません。

## 1. 環境を準備する

Pythonは`>=3.12,<3.13`です。リポジトリのルートで実行してください。

```bash
python -m pip install uv
uv sync --extra dev --extra train-sb3
uv run trade-rl --version
```

`train run`はStable-Baselines3とPyTorchを使うため、`--extra train-sb3`が必要です。Studio、PostgreSQL、ONNXを使う場合だけ、対応するExtraを追加します。

```bash
# Studio
uv sync --extra dev --extra train-sb3 --extra studio

# PostgreSQL catalog
uv sync --extra dev --extra train-sb3 --extra postgres

# ONNX/TorchScript export検証
uv sync --extra dev --extra train-sb3 --extra export
```

## 2. デモ用の市場データを作る

```bash
uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset
```

生成物:

```text
var/quickstart/dataset/
├── manifest.json
└── arrays.npz
```

`manifest.json`はDataset identity、配列名、Shape、Dtype、Digestを保持します。`arrays.npz`やManifestが変更・破損すると、学習前に拒否されます。

バー数を増やす場合:

```bash
uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset-large \
  --bars 4096
```

Published datasetは不変です。同じ出力先を上書きせず、新しいDirectoryへ生成してください。

## 3. 学習を開始する

```bash
uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-001
```

Quickstart設定はCPU、PPO、1 seed、512 timestepsの小規模Smokeです。初期設定のResidual actionは次の3次元です。

```text
fast_tilt
slow_tilt
risk_tilt
```

AlphaとFactorは無効です。通常のEnvironment遷移は、保守的なOHLC path、処理バーの出来高容量、Partial-fill carryを使う状態付き注文経路で実行されます。

成功時、CLIはMachine-readable JSONを1行出力します。概念例:

```json
{
  "artifact_path": "var/quickstart/artifacts/runs/quickstart-001",
  "production_status": "NO-GO",
  "run_id": "quickstart-001",
  "status": "published"
}
```

## 4. 学習成果物を確認する

```bash
cat var/quickstart/artifacts/latest.json
find var/quickstart/artifacts/runs/quickstart-001 -maxdepth 6 -type f | sort
```

主な成果物:

```text
var/quickstart/artifacts/
├── latest.json
└── runs/
    └── quickstart-001/
        ├── run.json
        ├── training-config.json
        ├── dataset-reference.json
        ├── environment.json
        ├── ensemble.json
        ├── policy-loader.json
        └── members/
            └── member-000/
                ├── policy.zip
                └── telemetry/
                    └── training-telemetry.jsonl
```

- `run.json`: 宣言済み成果物のPath、Size、SHA-256を束ねる最終Manifest
- `training-config.json`: 解決後の学習設定
- `dataset-reference.json`: 学習対象DatasetのIdentity
- `environment.json`: Action、Risk、Reward、Trend、Execution設定
- `ensemble.json`: Seed memberを束ねるManifest
- `policy.zip`: Stable-Baselines3の正式なRecovery/再学習形式
- `training-telemetry.jsonl`: Studio用のSeed単位・Append-only診断Telemetry
- `latest.json`: 最後に正常PublishされたRunへのPointer

Telemetryは探索観察用で、Checkpoint選択、Outer-test、Artifact identity、Serving承認、注文実行には使いません。

## 5. Studioで学習中の探索を見る

Python APIを起動します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
```

別ターミナル:

```bash
npm ci --prefix studio
npm run dev --prefix studio
```

`http://127.0.0.1:5173`を開き、`Live Training`からJobとSeedを選択します。`バッファ再生`は人間が追える速度、`ほぼライブ`は最新受信位置への追従です。BUY／SELL MarkerはWeight変化であり、取引所注文ではありません。

## 6. 再学習する

Runは不変Artifactです。同じ`--run-id`を再利用しないでください。

```bash
uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-002
```

失敗したRunは`failed/<run-id>`へ隔離され、正常な`latest.json`は変更されません。

## 7. 本格的な学習設定へ変更する

`examples/quickstart/training.json`の`training` Sectionを変更します。例:

```json
{
  "timesteps": 102400,
  "seeds": [0, 1, 2],
  "n_steps": 2048,
  "batch_size": 64,
  "n_epochs": 10,
  "device": "cpu",
  "policy_net_arch": [128, 128]
}
```

設定全体を上の断片で置換せず、既存JSON内の対応Sectionだけを変更してください。

GPU確認:

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

`True`なら`training.device`を`"cuda"`へ変更できます。GPU利用率ではなく、Throughput、再現性、Sealed OOS evidenceを評価してください。

対応Algorithm:

```text
ppo
sac
td3
tqc
```

Algorithm比較ではDataset range、Action contract、Reward、Cost、Execution policy、Seedを固定します。

## 8. 実データへ置き換える

`train run`が受け取るのは検証済みMarket dataset artifactです。

```text
my-dataset/
├── manifest.json
└── arrays.npz
```

最低限の因果契約:

- 行`t`のFeatureは行`t`のBar closeまでに利用可能な情報だけで作る
- 行`t`の判断は最短でも`t + 1`のOpen以降で約定する
- Future return、将来High/Low、後日改訂値をFeatureへ混入させない
- 欠損値だけでなくAvailabilityとStalenessを保存する
- Split、Dividend、Delisting、Funding、Borrow、Session gapを対象市場に合わせる
- Tick、Lot、Minimum notional、Fee、Spread、ParticipationなどのExecution metadataをIdentityへ含める
- Slow lookback、Sequence window、Reward pre-roll、Episode、評価Rangeより十分長い期間を用意する

公開Binance経路は[docs/BINANCE.md](docs/BINANCE.md)を参照してください。

## 9. PostgreSQL catalogを使う

PostgreSQLは任意です。DatasetやModel本体をDBへ移さず、Metadataと再利用Identityを登録します。

```bash
cp .env.example .env
docker compose up -d postgres
uv sync --extra postgres
export TRADE_RL_DATABASE_URL=postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
uv run trade-rl catalog migrate
uv run trade-rl catalog health
```

DB未設定時はFilesystemだけで動作します。

## 10. よくあるエラー

### `No module named stable_baselines3`

```bash
uv sync --extra dev --extra train-sb3
```

### `run already exists`

新しい`--run-id`を指定してください。Published runは上書きしません。

### `dataset digest mismatch`

`arrays.npz`または`manifest.json`が変更されています。元データからArtifactを再生成してください。

### Episodeを開始できない

Dataset期間が短い可能性があります。Bar数を増やすか、Episode、Trend lookback、Sequence window、Reward baseline windowを見直してください。

### CUDA関連で失敗する

`training.device`を`cpu`へ戻し、まずCPU SmokeでPipelineを確認してください。

### 学習途中で失敗した

```bash
find var/quickstart/artifacts/failed -maxdepth 5 -type f | sort
```

## 11. 次の段階

単一Runの動作確認後はNested walk-forwardを使用します。

```bash
uv run trade-rl walk-forward run \
  --config configs/walk-forward.json \
  --dataset artifacts/datasets/btc-usdt \
  --output artifacts/research \
  --run-id btc-usdt-wf-001
```

最終判断には、複数Seed、複数Fold、複数AUM、Fee/Spread/Impact/Funding/Borrow、Conservative stateful execution、未使用Outer test、Fresh confirmation、Paper reconciliationが必要です。詳細は[Architecture](docs/ARCHITECTURE.md)、[Research Status](docs/RESEARCH_STATUS.md)、[最新監査](docs/verification/2026-07-22-documentation-and-architecture-audit.md)を参照してください。
