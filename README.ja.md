# Trade RL

Trade RLは、ポートフォリオ配分を対象とした**ベースライン固定型Residual強化学習**の研究コアです。方策がポートフォリオ比率を直接自由に決めるのではなく、決定論的なTrend baselineに対して、制約された残差行動だけを加えます。報酬と評価では、独立したShadow baseline bookとの差分を使用します。

> 本番運用判定は **NO-GO** です。ソフトウェア構造を改善したことは、過去の実データ検証で失敗した必須Gateを通過したことを意味しません。

## 再構築内容

旧`mars_lite`、Direct-action PPO、旧CLIスクリプト、重複した評価計算、旧テスト群は互換層を残さず削除しました。Git履歴が旧実装のアーカイブです。

現在の責務は次のように分離されています。

- `domain`: Dataset、Signal、Policy ensemble、Selection、Releaseの不変条件
- `artifacts`: canonical JSON、SHA-256、staging、atomic publish
- `data`: 実データsource、因果的特徴量builder、MarketDataset検証
- `strategies`: 決定論的Trend baseline
- `risk`: Gross、銘柄上限、Turnover、Drawdown縮小
- `simulation`: 約定、コスト、Funding、会計
- `evaluation`: Return、Sharpe、Sortino、Drawdown、paired比較、bootstrap、Gate
- `rl`: Residual action、Observation、Reward、Environment、PPO adapter
- `workflows`: 型付き設定を受け取るApplication orchestration
- `serving`: Bundle検証、Registry、Runtime hot-swap
- `cli`: 単一の`trade-rl`入口

## 因果的な実データ入力

正式なデータ経路は、銘柄ごとのCSV実データから`MarketDatasetBuilder`で`MarketDataset`を生成します。全銘柄の共通時刻を欠損ごと削除するのではなく、規則的なunion clockを保持し、上場・廃止期間、バー単位の取引可否、特徴量ごとの利用可否と鮮度を別々に記録します。

```python
from datetime import datetime, timezone

from trade_rl.data import (
    CsvMarketDataSource,
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    MarketDatasetBuilder,
)

config = MarketBuildConfig(
    base_timeframe="1h",
    features=(
        FeatureSpec(name="ret_1", kind=FeatureKind.LOG_RETURN),
        FeatureSpec(name="funding_bps", kind=FeatureKind.FUNDING_BPS),
    ),
)
contracts = (
    InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
    ),
)
dataset = MarketDatasetBuilder(config).build(
    CsvMarketDataSource("data/market"),
    contracts,
)
```

CSVの必須列は`timestamp,open,high,low,close,volume`で、`available_at`、`funding_rate`、`tradable`は任意です。`timestamp`は市場イベント時刻、`available_at`はその行を知り得た最初の時刻です。遅延到着した行を、同時刻に知っていた情報として過去の特徴量へ遡及注入しません。新しい因果的特徴量が利用可能になるまでは、直前の特徴量を鮮度低下付きで保持できます。

Trend、Alpha、pre-trade targetは同じpoint-in-time eligibility maskを使用します。要求されたlookback全体で、上場中・取引可能・情報利用可能である銘柄だけを対象にします。翌バー以降の取引可否はExecutorが実現状態として処理し、現在のtarget生成には使用しません。

`volume`の意味は銘柄契約ごとにbase asset数量、quote notional、契約枚数のいずれかを明示し、契約枚数にはbase asset換算用のcontract multiplierを設定します。

`dataset_id`には、特徴量生成設定、正規化設定と生成結果、銘柄順、上場・廃止期間、volume単位、contract multiplier、`available_at`を含む全配列の内容を含めます。同じIDである限り学習入力が同一であり、どれか一つでも変わればIDも変わります。

## 決定論的Dataset artifact

厳格なJSON設定から正式なCLI経路を実行できます。

```json
{
  "source_root": "data/market",
  "base_timeframe": "1h",
  "features": [
    {"name": "ret_1", "kind": "log_return", "lookback": 1}
  ],
  "instruments": [
    {
      "symbol": "BTCUSDT",
      "listed_at": "2019-09-01T00:00:00Z",
      "volume_unit": "quote_notional"
    }
  ]
}
```

```bash
uv run trade-rl data build --config market-build.json --output output/dataset
```

出力はcanonicalな`manifest.json`と決定論的な`arrays.npz`です。再読込時には両方のdigestを検証します。同一の設定とsource内容からは、同一のdataset IDとartifact digestを生成します。

## ObservationとServingの整合性

学習環境とServingは同じ`ObservationBuilder`を使用します。Serving bundleは`dataset_id`とaction schemaに加え、銘柄・特徴量・global featureの順序、正規化状態、vector長を含むobservation schema digestを保持します。入力dataset ID、observation schema、vector長のいずれかがactive bundleと一致しなければ、policyを呼び出す前にfail-closedで拒否します。

## セットアップ

```bash
uv sync --extra dev
```

## 使用例

```bash
uv run trade-rl --version

uv run trade-rl data build \
  --config market-build.json --output output/dataset

uv run trade-rl train config \
  --timesteps 1024 --gamma 0.5 --seed 0 --seed 1

uv run trade-rl walk-forward plan \
  --bars 220 --train-bars 80 --checkpoint-bars 10 \
  --selection-bars 10 --test-bars 20 --purge-bars 2 --max-folds 2
```

## 品質確認

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

詳細は[アーキテクチャ](docs/ARCHITECTURE.md)と[研究結果の扱い](docs/RESEARCH_STATUS.md)を参照してください。
