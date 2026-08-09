# Trade RL 学習クイックスタート

この手順では、**デモDataset生成 → PPO学習 → Artifact確認 → Studio表示**までを実行します。

> デモ相場と短時間学習は動作確認用です。Production statusは`NO-GO`であり、収益性評価には使用できません。

## 1. 環境を準備する

Repository rootで実行します。Python本体は`trade_rl/`、Web UIは`frontend/`にあります。Python import名とCLIは`trade_rl.*` / `trade-rl`です。

```bash
python -m pip install uv
uv sync --extra dev --extra train-sb3
uv run trade-rl --version
```

用途に応じてExtraを追加します。

```bash
# Studio API/backend
uv sync --extra dev --extra train-sb3 --extra studio

# PostgreSQL metadata catalog
uv sync --extra dev --extra train-sb3 --extra postgres

# Flat policyのONNX/TorchScript検証
uv sync --extra dev --extra train-sb3 --extra export
```

## 2. デモDatasetを作る

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

Published datasetは不変です。別条件で作り直す場合は、新しいDirectoryを指定してください。

## 3. 学習を実行する

```bash
uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-001
```

QuickstartはCPU、PPO、1 seed、短いRolloutです。成功は学習とArtifact publishのPipelineが動いたことだけを意味します。

成功時のMachine-readable出力例:

```json
{
  "artifact_path": "var/quickstart/artifacts/runs/quickstart-001",
  "production_status": "NO-GO",
  "run_id": "quickstart-001",
  "status": "published"
}
```

## 4. 成果物を確認する

```bash
cat var/quickstart/artifacts/latest.json
find var/quickstart/artifacts/runs/quickstart-001 -maxdepth 6 -type f | sort
```

主な成果物:

```text
var/quickstart/artifacts/
├── latest.json
└── runs/quickstart-001/
    ├── run.json
    ├── training-config.json
    ├── dataset-reference.json
    ├── environment.json
    ├── ensemble.json
    ├── policy-loader.json
    └── members/member-000/
        ├── policy.zip
        └── telemetry/training-telemetry.jsonl
```

- `run.json`: 宣言済みFile、Size、SHA-256を束ねる最終Manifest
- `training-config.json`: 解決済み`training_run_config_v4`
- `environment.json`: Action、Observation、Reward、Risk、Execution identity
- `policy.zip`: Stable-Baselines3のRecovery形式
- `training-telemetry.jsonl`: Studio用の診断Stream
- `latest.json`: 最後に正常PublishされたRunへのPointer

失敗Runは`failed/<run-id>`へ隔離され、正常なPointerは変更されません。

## 5. Studioで探索を見る

Python APIを起動します。

```bash
uv sync --extra studio --extra train-sb3
uv run trade-rl studio start --project-root .
```

別ターミナルでWeb UIを起動します。

```bash
npm ci --prefix frontend
npm run dev --prefix frontend
```

`http://127.0.0.1:5173`を開きます。Live TrainingのBUY／SELL表示はWeight変化であり、取引所注文ではありません。

## 6. `training_run_config_v4`を使う

Top-levelの`schema_version`は必ず次の値です。

```json
{
  "schema_version": "training_run_config_v4"
}
```

旧Schemaは自動移行されず、明示的に拒否されます。

### Encoderを選ぶ

Quickstartの既定は`asset_set`です。系列Policyを使う場合は、`training` Sectionを次のように設定します。

```json
{
  "training": {
    "algorithm": "ppo",
    "policy": "MultiInputPolicy",
    "observation_encoder": "hierarchical_sequence_v2",
    "sequence_tcn_capacity": "standard",
    "sequence_d_model": 336,
    "sequence_timeframe_attention_heads": 8,
    "sequence_timeframe_attention_layers": 2,
    "sequence_timeframe_ffn_multiplier": 3,
    "sequence_timeframe_gate_bias": -2.0,
    "sequence_asset_attention_heads": 8,
    "sequence_asset_attention_layers": 2,
    "sequence_asset_ffn_multiplier": 3,
    "sequence_asset_gate_bias": -2.0,
    "sequence_dropout": 0.05
  }
}
```

設定全体をこの断片で置換せず、既存JSONの対応Sectionだけを変更してください。詳しくは[設定リファレンス](docs/CONFIGURATION.md)を参照してください。

## 7. BCからPPOへ進む

PPO-familyでは、`behavior_cloning_epochs`を正数にすると、同じFeature extractorとArchitecture identityを使ってBC warm startを行います。

```json
{
  "training": {
    "algorithm": "ppo",
    "behavior_cloning_epochs": 10,
    "behavior_cloning_teacher": "oracle",
    "behavior_cloning_batch_size": 256
  }
}
```

BC、PPO、CostCriticPPO、LagrangianPPOは、組み立て済みPolicyのArchitecture identityを共有します。異なる構造のCheckpointはResume時に拒否されます。

## 8. TensorBoard診断を有効にする

```json
{
  "training": {
    "tensorboard_enabled": true,
    "tensorboard_log_interval": 1
  }
}
```

Studioの`学習診断`からScalarを確認できます。AttentionやGradientの診断は、Checkpoint選択や収益性Evidenceには使用しません。

## 9. GPUを確認する

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

`True`の場合だけ`training.device`を`"cuda"`へ変更します。系列Policyでは、対応環境で次も利用できます。

```json
{
  "training": {
    "sequence_compile": true,
    "sequence_compile_mode": "reduce-overhead",
    "sequence_transfer_mode": "pinned_non_blocking"
  }
}
```

Dockerによる完全実行は[GPU運用手順](docs/operations/docker-gpu-full-training.md)を参照してください。

## 10. 実データへ置き換える

`train run`が受け取るのは検証済みMarket dataset artifactです。

最低限の因果契約:

- 行`t`のFeatureは行`t`のBar closeまでに利用可能な情報だけで作る
- 行`t`の判断は最短でも`t + 1`のOpen以降で処理する
- Future return、将来High/Low、後日改訂値をFeatureへ混入させない
- AvailabilityとStalenessを値とは別に保存する
- Fee、Spread、Participation、Tick/Lot、Funding、BorrowをIdentityへ含める
- Sequence window、Reward pre-roll、Episode、評価Rangeに必要な履歴を用意する

Public Binance経路は[docs/BINANCE.md](docs/BINANCE.md)を参照してください。

## 11. よくあるエラー

### `No module named stable_baselines3`

```bash
uv sync --extra dev --extra train-sb3
```

### `run already exists`

新しい`--run-id`を指定してください。Published runは上書きしません。

### Training config schemaの移行が必要

設定の`schema_version`を`training_run_config_v4`へ変更し、旧Encoder Booleanを`observation_encoder`へ置き換えてください。必要なv4 Fieldは[設定リファレンス](docs/CONFIGURATION.md)で確認してください。

### `dataset digest mismatch`

`arrays.npz`または`manifest.json`が変更されています。元データからArtifactを再生成してください。

### Episodeを開始できない

Dataset期間が短い可能性があります。Episode、Trend lookback、Sequence window、Reward pre-rollを確認してください。

### CUDA関連で失敗する

`training.device`を`cpu`へ戻し、まずCPU Quickstartを完走させてください。

## 12. 次の段階

単一Runの確認後はNested walk-forwardへ進みます。

```bash
uv run trade-rl walk-forward run \
  --config examples/binance-multitimeframe/walk-forward-full.json \
  --dataset artifacts/datasets/binance-multitimeframe \
  --output artifacts/research \
  --run-id binance-wf-001
```

最終判断には、複数Seed、複数Fold、複数AUM、保守的Execution、未使用Outer test、Fresh confirmation、Paper reconciliationが必要です。[研究状態](docs/RESEARCH_STATUS.md)を参照してください。
