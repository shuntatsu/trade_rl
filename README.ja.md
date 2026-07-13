# Trade RL

Trade RLは、ポートフォリオ配分を対象とした**ベースライン固定型Residual強化学習**の研究コアです。方策がポートフォリオ比率を直接自由に決めるのではなく、決定論的なTrend baselineに対して、制約された残差行動だけを加えます。報酬と評価では、独立したShadow baseline bookとの差分を使用します。

> 本番運用判定は **NO-GO** です。ソフトウェア構造を改善したことは、過去の実データ検証で失敗した必須Gateを通過したことを意味しません。

## 再構築内容

旧`mars_lite`、Direct-action PPO、旧CLIスクリプト、重複した評価計算、旧テスト群は互換層を残さず削除しました。Git履歴が旧実装のアーカイブです。

現在の責務は次のように分離されています。

- `domain`: Dataset、Signal、Policy ensemble、Selection、Releaseの不変条件
- `artifacts`: canonical JSON、SHA-256、staging、atomic publish
- `data`: Shapeと時系列を検証したMarketDataset
- `strategies`: 決定論的Trend baseline
- `risk`: Gross、銘柄上限、Turnover、Drawdown縮小
- `simulation`: 約定、コスト、Funding、会計
- `evaluation`: Return、Sharpe、Sortino、Drawdown、paired比較、bootstrap、Gate
- `rl`: Residual action、Observation、Reward、Environment、PPO adapter
- `workflows`: 型付き設定を受け取るApplication orchestration
- `serving`: Bundle検証、Registry、Runtime hot-swap
- `cli`: 単一の`trade-rl`入口

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
