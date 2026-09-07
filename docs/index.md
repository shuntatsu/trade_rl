---
title: Trade RL Documentation
doc_type: index
topics:
  - documentation
  - navigation
description: Current documentation portal and source-of-truth map for Trade RL.
---

# Trade RL Documentation

このPortalは、人間とcoding agentが**現在の正本、実行手順、研究状態、履歴資料を混同せずに読むための入口**です。現在のruntime behaviorを判断するときは、Source/Testと`reference/`のcanonical documentを優先します。

## Start here

- [Getting started](getting-started/index.md) — 最初の学習を動かす。
- [Guides](guides/index.md) — Data取得などのtask-oriented guidance。
- [Reference](reference/index.md) — 現在のArchitecture、Schema、Training/Execution contractの正本。
- [Research](research/index.md) — 実装済み・CI検証済み・実証済み・Production認可を区別する。
- [Operations](operations/index.md) — 現在実行可能なRunbook。
- [Performance](performance/index.md) — Hardware固有の測定と候補設定。
- [Legal](legal/index.md) — Licenseとprovenance。
- [History](history/index.md) — 過去のSpec/Plan/ADR/Evidence。**current runtime authorityではありません**。

Frontend固有の責務とStudio UIは[`frontend/README.md`](../frontend/README.md)を参照してください。

## Authority

優先順位は次です。

1. 実行可能なSource、Schema、Contract Test
2. `authority: canonical`のcurrent document
3. current Guide / Runbook
4. Performance / Evidence
5. History

Source/Testとcurrent canonical documentが矛盾する場合は、都合の良い側を採用せず**documentation defectとして両者を照合**します。

## Agent routing

Repository変更前はroot [`AGENTS.md`](../AGENTS.md)を読み、必要な文書が明らかでない場合は次を使います。

```bash
python scripts/docs/context.py --path trade_rl/path/to/file.py
python scripts/docs/context.py --topic universal-rl
```

Historyは既定では返りません。設計経緯を調査するときだけ`--include-history`を明示します。
