# Trade RL 学習クイックスタート

このページは、リポジトリを取得した直後に、**データartifact生成 → PPO学習 → 成果物確認**までを最短で実行する手順です。

最初の例では決定論的なデモ相場を使います。デモデータは学習パイプラインの動作確認用であり、収益性の検証には使用できません。

> Production status: **NO-GO**  
> 学習やテストの成功は、本番資金での運用許可や利益を保証しません。

## 1. 環境を準備する

Python 3.12以上が必要です。リポジトリのルートで実行してください。

```bash
python -m pip install uv
uv sync --extra dev
uv run trade-rl --version
```

GPUやONNX/TorchScript exportを初回から使わない場合、`--extra export`は不要です。

## 2. デモ用の市場データを作る

```bash
uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset
```

成功すると、次の2ファイルが作られます。

```text
var/quickstart/dataset/
├── manifest.json
└── arrays.npz
```

`manifest.json`にはデータセット識別子、配列名、shape、dtype、SHA-256が保存されます。`arrays.npz`が変更・破損すると、学習前の検証で拒否されます。

バー数を増やす場合は次のように指定します。

```bash
uv run python examples/quickstart/create_demo_dataset.py \
  --output var/quickstart/dataset \
  --bars 4096
```

## 3. 学習を開始する

最小設定は`examples/quickstart/training.json`にあります。

```bash
uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-001
```

この設定は、CPU、PPO、1 seed、512 timestepsの小規模な動作確認です。行動空間は次の3次元です。

```text
fast_tilt
slow_tilt
risk_tilt
```

alphaとfactorは初回設定では無効です。

成功時、CLIは1行のJSONを出力します。

```json
{
  "artifact_path": "var/quickstart/artifacts/runs/quickstart-001",
  "production_status": "NO-GO",
  "run_id": "quickstart-001",
  "status": "published"
}
```

実際のJSONにはdataset、run、policyのdigestも含まれます。

## 4. 学習成果物を確認する

```bash
cat var/quickstart/artifacts/latest.json
find var/quickstart/artifacts/runs/quickstart-001 -maxdepth 4 -type f | sort
```

主な成果物は次のとおりです。

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
                └── policy.zip
```

- `run.json`: 全成果物のpath、size、SHA-256を束ねた最終manifest
- `training-config.json`: 実際に使用された学習設定
- `dataset-reference.json`: 学習対象データの識別情報
- `environment.json`: action、risk、reward、trend、execution設定
- `ensemble.json`: seedごとのpolicyを束ねるmanifest
- `policy.zip`: Stable-Baselines3の正式な保存形式
- `latest.json`: 最後に正常publishされたrunへのポインタ

## 5. 再学習する

runは不変artifactとして扱われるため、同じ`--run-id`を再利用しないでください。

```bash
uv run trade-rl train run \
  --config examples/quickstart/training.json \
  --dataset var/quickstart/dataset \
  --output var/quickstart/artifacts \
  --run-id quickstart-002
```

失敗したrunは`failed/<run-id>`へ隔離され、正常な`latest.json`は変更されません。

## 6. 本格的な学習設定へ変更する

`examples/quickstart/training.json`の`training`を変更します。

研究用の開始例:

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

設定全体を上の断片で置き換えるのではなく、既存JSONの`training`セクション内の値を変更してください。

### GPUを使う

まずCUDAが認識されるか確認します。

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

`True`の場合、設定の次の値を変更します。

```json
"device": "cuda"
```

CUDAが利用できない環境では`cpu`を使用してください。

### 学習アルゴリズムを変える

`algorithm`は次を選択できます。

```text
ppo
sac
td3
tqc
```

PPO以外では`buffer_size`、`learning_starts`、`train_freq`、`gradient_steps`も確認してください。アルゴリズム間の比較では、データ範囲、action contract、reward、cost、seedを固定します。

## 7. 実データへ置き換える

`trade-rl train run`が受け取るのは、次の形式の市場データartifactです。

```text
my-dataset/
├── manifest.json
└── arrays.npz
```

Python側では`MarketDataset`を作成し、正式writerで保存します。

```python
from pathlib import Path

from trade_rl.data import MarketDataset, write_market_dataset_files

# dataset = MarketDataset(...)
write_market_dataset_files(Path("artifacts/datasets/btc-usdt"), dataset)
```

最低限、次の配列が必要です。

- `timestamps`: bar close時刻。連続市場では完全に等間隔
- `features`: `(bars, symbols, features)`
- `global_features`: `(bars, global_features)`
- `open`, `high`, `low`, `close`, `volume`, `funding_rate`: `(bars, symbols)`
- `tradable`: `(bars, symbols)`
- `feature_available`: `features`と同じshape

重要な因果契約:

- 行`t`の特徴量は行`t`のbar close時点までに利用可能な情報だけで作る
- 行`t`の意思決定は最短でも行`t + 1`のopenで約定する
- future return、将来のhigh/low、後日改訂値を特徴量へ混入させない
- 欠損を0に置換するだけでなく、availabilityとstalenessを保存する
- split、dividend、delisting、funding、borrow、session gapを対象市場に合わせる
- データ期間はslow lookback、reward preroll、episode期間、評価範囲より十分長くする

実データartifactを作った後は、`--dataset`だけを差し替えます。

```bash
uv run trade-rl train run \
  --config configs/btc-usdt-ppo.json \
  --dataset artifacts/datasets/btc-usdt \
  --output artifacts/research \
  --run-id btc-usdt-ppo-001
```

## 8. alphaやfactorを使う

初回設定では無効です。利用する場合は、先にcontent-addressed signal artifactを用意し、設定を一致させる必要があります。

alphaの例:

```json
"alpha_artifact": "artifacts/signals/my-alpha",
"action": {
  "alpha_enabled": true,
  "n_factors": 0,
  "validation_mode": "clip"
}
```

factorの例では、`factor_artifact`と`action.n_factors`の両方を一致させます。不一致、symbol順序違い、dataset ID違い、digest不一致は学習前に拒否されます。

## 9. よくあるエラー

### `run already exists`

`--run-id`を新しい値へ変更してください。publish済みrunは上書きしません。

### `dataset digest mismatch`

`arrays.npz`または`manifest.json`が変更されています。元データからartifactを再生成してください。

### episodeを開始できない

データ期間が短い可能性があります。バー数を増やすか、次を短くしてください。

- `environment.episode_hours`
- `trend.fast_hours`
- `trend.base_hours`
- `trend.slow_hours`
- rewardのbaseline window

### CUDA関連で失敗する

`training.device`を`cpu`へ戻し、まずCPUで学習パイプラインを確認してください。

### 学習途中で失敗した

```bash
find var/quickstart/artifacts/failed -maxdepth 3 -type f | sort
```

失敗したstaging runは`failed/<run-id>`へ隔離されます。正常publish済みrunと`latest.json`は保持されます。

## 10. 次の段階

単一runの動作確認後は、nested walk-forwardを使用します。

```bash
uv run trade-rl walk-forward run \
  --config configs/walk-forward.json \
  --dataset artifacts/datasets/btc-usdt \
  --output artifacts/research \
  --run-id btc-usdt-wf-001
```

最終判断には、手数料・spread・impact・funding・borrowを含むsealed out-of-sample評価と、別期間・複数seed・複数AUMでの再現性確認が必要です。
