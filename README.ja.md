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

CSVの必須列は`timestamp,open,high,low,close,volume`で、`funding_rate`と`tradable`は任意です。`volume`の意味は銘柄契約ごとにbase asset数量、quote notional、契約枚数のいずれかを明示し、契約枚数にはbase asset換算用のcontract multiplierを設定します。

`dataset_id`には、特徴量生成設定、正規化設定と生成結果、銘柄順、上場・廃止期間、volume単位、contract multiplier、全配列の内容を含めます。同じIDである限り学習入力が同一であり、どれか一つでも変わればIDも変わります。

方策観測は時刻`t`までの情報だけで作られます。`tradable[t + 1]`は次バー執行の実績としてsimulation内部では使えますが、方策には渡されません。学習環境とServingは同じ`ObservationBuilder`を使用します。

## セットアップ

```bash
uv sync --extra dev
```

## 使用例

```bash
uv run trade-rl --version

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
