# Project: trade_rl

## Architecture
- `mars_lite/`: コアロジックライブラリ。
- `tests/`: ユニットテスト。
- `scripts/`: 学習およびユーティリティスクリプト。

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: 依存管理とCI構築 | R1, R2 の実装と環境整備 | none | DONE |
| 2 | M2: リスク制御とマニフェスト | R3, R4 の実装。注文前リスク制限およびモデルシリアライズ | M1 | DONE |
| 3 | M3: Regime FSMとリファクタリング | R5, R6 の実装。8状態ステートマシン構築および train_portfolio.py の分割 | M2 | DONE |
| 4 | M4: 評価・監視シミュレータ | R7, R8, R9 の実装。リプレイシミュレータ、ドリフト監視、ブートストラップ評価 | M3 | DONE |
| 5 | M5: モデルレジストリ | R10 の実装。モデル管理、アクティブ化、ロールバックサーバー | M4 | DONE |
| 6 | M6: デプロイメント・監査 | R11 の実装。3段階デプロイパイプライン CI ゲートと運用ドキュメント | M5 | DONE |

## Interface Contracts
- `mars_lite/trading/pre_trade_risk.py`: PreTradeRiskVerifier クラス。発注指示前のリスク検証。違反時に PreTradeRejection 例外をスロー。
- `output/portfolio/model_manifest.json`: 学習メタデータ（コミットハッシュ、パラメータ等）の記録。
- `mars_lite/learning/regime_fsm.py`: 8状態ステートマシン RegimeFSM クラス。
- `mars_lite/eval/replay_sim.py`: 約定シミュレータ ReplaySimulator クラス。
- `mars_lite/eval/drift_monitor.py`: 特徴量・予測ドリフト監視モジュール。
- `mars_lite/eval/bootstrap_eval.py`: ブートストラップ評価機能。
- `mars_lite/server/model_registry.py`: モデルレジストリ API/CLI。

## Code Layout
- `pyproject.toml` : 依存管理の一元化。
- `mars_lite/trading/pre_trade_risk.py` : 事前リスク管理。
- `mars_lite/learning/regime_fsm.py` : 8状態 Regime FSM。
- `mars_lite/learning/regime_calibrator.py` : regime 較正ロジック。
- `mars_lite/pipeline/` : リファクタリング後の分割モジュール。
- `mars_lite/eval/replay_sim.py` : リプレイ実行シミュレータ。
- `mars_lite/eval/drift_monitor.py` : ドリフト監視モジュール。
- `mars_lite/eval/bootstrap_eval.py` : Sharpe ブートストラップ評価。
- `mars_lite/server/model_registry.py` : モデルレジストリサーバー・CLI。
