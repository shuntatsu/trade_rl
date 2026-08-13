# Trade RL ドキュメント

このDirectoryの**Top-level文書は現在のmainで維持している契約の正本**です。`implementation-plans/`は設計・実装時点の履歴資料であり、現行運用の正本ではありません。監査ログや一時検証記録はGit履歴または該当Pull Requestから参照します。

## 読む順番

| 目的 | 文書 |
|---|---|
| まず動かす | [START.md](../START.md) |
| Maintained 1 Run＝1 Instrument契約を確認する | [SINGLE_SYMBOL.md](SINGLE_SYMBOL.md) |
| Universal U3-U6の現行学習契約を確認する | [UNIVERSAL_TRAINING.md](UNIVERSAL_TRAINING.md) |
| 全体構造を理解する | [ARCHITECTURE.md](ARCHITECTURE.md) |
| 設定項目を調べる | [CONFIGURATION.md](CONFIGURATION.md) |
| 研究結果を正しく解釈する | [RESEARCH_STATUS.md](RESEARCH_STATUS.md) |
| Multi-TimeframeのPhase境界 | [MULTITIMEFRAME_RESEARCH.md](MULTITIMEFRAME_RESEARCH.md) |
| Binance Public Dataを作る | [BINANCE.md](BINANCE.md) |
| DockerとCUDAで完全実行する | [operations/docker-gpu-full-training.md](operations/docker-gpu-full-training.md) |
| Causal Scenario C3を実行する | [operations/causal-scenario-c3-execution.md](operations/causal-scenario-c3-execution.md) |
| 4070 Ti SUPER向け設定を確認する | [performance/4070ti-super-full-training.md](performance/4070ti-super-full-training.md) |
| Frontendを使う | [frontend/README.md](../frontend/README.md) |

Repositoryの概要は[README.md](../README.md)が正本です。

## 文書の責務

- `README.md`: 状態、Quickstart、能力境界、主要リンク
- `START.md`: 実行手順とTroubleshooting
- `SINGLE_SYMBOL.md`: Maintained 1 Run＝1 Instrument、Action、Identity、Legacy互換
- `UNIVERSAL_TRAINING.md`: Universal runtime、causal teacher、holdout/admission、U5/U6共有、monitoring
- `ARCHITECTURE.md`: 現行実装の責務、Data flow、Identity、Fail-closed境界
- `CONFIGURATION.md`: 現行Schemaと設定値
- `RESEARCH_STATUS.md`: 実装済み能力、未取得Evidence、Production gate
- `MULTITIMEFRAME_RESEARCH.md`: Develop、Selected-final、Finalizeの外部承認境界
- `BINANCE.md`: Public market dataとMetadata mode
- `operations/`: 実行Runbook
- `performance/`: Hardware別の測定・候補設定
- `frontend/README.md`: UIと診断Telemetryの境界
- `implementation-plans/`: 設計・実装時点の履歴。現行契約の正本ではない

同じ説明を複数ファイルへ複製せず、責務を持つ文書へLinkします。

## 更新ルール

1. Top-level維持文書は現在の実装を記述し、開発経緯は記述しない。
2. Schema名、CLI、Path、設定名はCodeまたは維持対象Exampleと一致させる。
3. 実装済み、検証済み、収益性、Production認可を分ける。
4. `NO-GO`を、Pipeline失敗やモデル失敗と混同しない。
5. 一時的なCommit SHAやActions run IDは、維持文書へ固定しない。
6. Relative linkを使い、文書契約Testで解決可能性を確認する。
7. 完了済みPlanやSpecを現行契約の根拠として引用せず、必要な内容はTop-level維持文書へ反映する。

## 履歴資料

`implementation-plans/`配下のPlan/Specや削除済み文書は履歴資料です。特定時点の設計判断を調べる場合は、該当文書、Commit、Pull Requestの変更履歴を使用してください。現在の動作や運用判断ではTop-level維持文書を優先します。
