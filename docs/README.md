# Trade RL ドキュメント

このDirectoryのTop-level文書は、Trade RLの**維持対象コードベースに対する現行契約の正本**です。`operations/`は現在実行できるRunbook、`performance/`はHardware固有の測定資料、`implementation-plans/`は設計・実装時点の履歴資料です。

現在の動作を確認するときはSource、維持対象Configuration/ExampleとこのTop-level文書を優先してください。履歴資料は当時の設計意図や検証経緯を追跡するためのもので、現行runtime contractの正本ではありません。

## 読む順番

| 目的 | 正本文書 |
|---|---|
| Repositoryの状態と最短入口を知る | [README.md](../README.md) |
| 最初の学習を実行する | [START.md](../START.md) |
| Maintained 1 Run = 1 Instrument契約を確認する | [SINGLE_SYMBOL.md](SINGLE_SYMBOL.md) |
| Universal U3-U6の学習契約を確認する | [UNIVERSAL_TRAINING.md](UNIVERSAL_TRAINING.md) |
| 実行artifactからLLMなしの貼付用reportを生成する | [RUN_REPORTING.md](RUN_REPORTING.md) |
| System構造と責務境界を理解する | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Schema・設定値・互換境界を調べる | [CONFIGURATION.md](CONFIGURATION.md) |
| 実装済み・CI検証済み・未実証を区別する | [RESEARCH_STATUS.md](RESEARCH_STATUS.md) |
| Rewardの意味を確認する | [REWARD_OBJECTIVE.md](REWARD_OBJECTIVE.md) |
| 約定Modelの能力と限界を確認する | [EXECUTION_ROBUSTNESS.md](EXECUTION_ROBUSTNESS.md) |
| Multi-Timeframe研究Phase境界を確認する | [MULTITIMEFRAME_RESEARCH.md](MULTITIMEFRAME_RESEARCH.md) |
| Binance Public Dataを構築する | [BINANCE.md](BINANCE.md) |
| Nautilus移行・互換境界を確認する | [NAUTILUS_MIGRATION.md](NAUTILUS_MIGRATION.md) |
| License境界を確認する | [LICENSING.md](LICENSING.md) |
| License provenanceを追跡する | [LICENSING_PROVENANCE.md](LICENSING_PROVENANCE.md) |
| Docker/CUDAで維持対象Trainingを実行する | [operations/docker-gpu-full-training.md](operations/docker-gpu-full-training.md) |
| Causal Scenario C3を実行する | [operations/causal-scenario-c3-execution.md](operations/causal-scenario-c3-execution.md) |
| 4070 Ti SUPER向け測定・候補設定を見る | [performance/4070ti-super-full-training.md](performance/4070ti-super-full-training.md) |
| Frontendと診断Telemetryを使う | [frontend/README.md](../frontend/README.md) |

## 文書の責務

- `../README.md`: Repository entry point。状態、能力境界、Quickstart、主要リンクだけを持つ。
- `../START.md`: 維持対象の導入・実行・Troubleshooting。
- `SINGLE_SYMBOL.md`: Maintained one-run/one-instrument、Action、Identity、Legacy互換。
- `UNIVERSAL_TRAINING.md`: Universal runtime、causal teacher、selection checkpoint、holdout/admission、BC/critic warm start、U5/U6共有、monitoring。
- `RUN_REPORTING.md`: persisted research/training artifactから生成するread-only・fact-onlyのMachine Run Report契約とCLI。
- `ARCHITECTURE.md`: 現行実装の責務、依存方向、Data flow、Identity、Training/Evaluation/ServingのFail-closed境界。
- `CONFIGURATION.md`: 現行Schema、設定値、拒否する旧設定。
- `RESEARCH_STATUS.md`: Software、CI、Empirical evidence、Profitability、Production authorizationの状態。
- `REWARD_OBJECTIVE.md`: Rewardとconstraint costの意味。
- `EXECUTION_ROBUSTNESS.md`: Stateful executionの能力、近似、限界。
- `MULTITIMEFRAME_RESEARCH.md`: Develop、Selected-final、Finalizeなど研究Phaseの分離。
- `BINANCE.md`: Public market dataとMetadata mode。
- `NAUTILUS_MIGRATION.md`: 維持されるNautilus migration/compatibility契約。
- `LICENSING.md` / `LICENSING_PROVENANCE.md`: 現在のLicenseと履歴上のLicense境界。
- `operations/`: **現在実行可能なRunbookだけ**。
- `performance/`: Hardware固有の測定値と候補設定。一般契約の正本ではない。
- `implementation-plans/`: 設計、実装計画、完了済みhardening verificationなどの履歴。
- `../frontend/README.md`: UI、Studio APIとの責務分離、診断Telemetryの解釈。

同じ規範説明を複数ファイルへ複製せず、概念の所有文書へLinkします。

## Current operations

`docs/operations/`で維持するRunbookは次の2本です。

1. [Docker GPU full training](operations/docker-gpu-full-training.md)
2. [Causal Scenario C3 execution](operations/causal-scenario-c3-execution.md)

設計案、実装順序、特定hardening時点のverification noteはoperationsではありません。これらは[implementation-plans/](implementation-plans/README.md)へ保存します。

## 履歴資料

[implementation-plans/README.md](implementation-plans/README.md)を入口とし、主に次へ分類します。

- `implementation-plans/specs/`: 設計時点のDesign/Spec。
- `implementation-plans/plans/`: 実装計画、統合Review、完了済みverification記録。

履歴資料には当時のPath、Schema、Pull Request、移行前提が含まれる場合があります。それらを現在のruntime authorityとして使用しません。現在の挙動と異なる場合は、Source、維持対象Example、Top-level正本文書を優先します。

## 更新ルール

1. Top-level維持文書は**現在の契約**を記述し、実装 chronology は履歴資料へ分離する。
2. Schema名、CLI、Path、Config、Example名はSourceまたは維持対象Fixtureと一致させる。
3. 「実装済み」「CI検証済み」「実証評価済み」「収益性あり」「Production認可済み」を別状態として記述する。
4. `NO-GO`をPipeline failureや個別Model failureと混同しない。
5. 一時的なCommit SHA、Actions run ID、Pull Request番号を現行の規範根拠にしない。
6. Relative Markdown linkを使い、契約Testで解決可能性を検査する。
7. 必須の現行Pathは不存在を黙って無視せず、documentation/layout contractでFail closedする。
8. 完了済みPlan/Specの有効な契約はTop-level正本へ反映し、Plan/Spec自体をruntime source of truthにしない。
9. Security、provenance、licensing、migration、sealed evaluation、causal boundaryを「古い」という理由だけで削除しない。

## Production / Research boundary

Trade RLのProduction statusは現在**NO-GO**です。CI成功は収益性、実資金投入、Direct exchange routing、運用認可を意味しません。正確な状態区分は[RESEARCH_STATUS.md](RESEARCH_STATUS.md)を正本とします。