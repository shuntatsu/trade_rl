# Trade RL documentation

このtreeは**現行の技術仕様と研究判断の正本**だけを保持する。実装の真偽はsource・public API・contract testsと照合する。過去の計画・migration・実験経緯はGit history、GitHub Issue/PR、再現可能な研究evidenceで追跡し、current docsへ作業日誌を追記し続けない。

## 正本の入口

- [AGENTS.md](AGENTS.md) — Agentの読み順序、更新matrix、検証・保持規則
- [architecture/lean-core.md](architecture/lean-core.md) — causal data、strategy/risk、execution、会計、artifact不変条件
- [architecture/package-boundaries.md](architecture/package-boundaries.md) — package責務、public API、依存方向
- [architecture/controlled-experiment-loop.md](architecture/controlled-experiment-loop.md) — Study/Experiment/EvidenceSet、lineage、freeze
- [architecture/final-evaluation-authorization.md](architecture/final-evaluation-authorization.md) — sealed unused-future認可境界
- [architecture/research-assurance.md](architecture/research-assurance.md) — G0–G5、反証条件、研究証拠の妥当性
- [architecture/portfolio-decision-trace.md](architecture/portfolio-decision-trace.md) — opt-in shared-portfolioでの行動・risk承認・約定と口座変化のtrace
- [research/current-status.md](research/current-status.md) — 現行の研究判断・証拠リンク・未完了gate

## Active work（完了までは暫定）

- [Issue #810](https://github.com/shuntatsu/trade_rl/issues/810) — 非LLM売買研究の親計画。進捗・依存はIssue/PR、現行判断はresearch/current-status.mdを正本とする。
- Prospective carry: [specs/funding-carry-paper.md](specs/funding-carry-paper.md) / [plans/funding-carry-paper.md](plans/funding-carry-paper.md)
- Directional exits: [specs/reduce-only-exits.md](specs/reduce-only-exits.md) / [plans/reduce-only-exits.md](plans/reduce-only-exits.md)

## 情報の所有先と更新

`architecture/`は恒久仕様、`research/`は現行判断、`specs/`・`plans/`は独立して必要な`Status: Active`の作業だけ。完了した暫定文書は残存契約を恒久仕様に反映して削除し、`docs/history`・`docs/archive`は作らない。履歴の正本はGit history。

コードの意味が変われば該当architectureとtestsを同一PRで更新する。研究の判定は結果・Dataset消費履歴を再分類せずresearch/current-status.mdを**差分更新**する。進捗履歴やTask/branch管理はGitHub Issue/PRとAgent Coordination Plane v1の責務であり、docsへ全文複製しない。

[Human Guide](../guide/README.md) は非正本の説明UI。内容変更は対象pageとsource fingerprint / code referenceを照合し、必要なtopicだけrefreshする。Pages公開状態は実際のGitHub設定・deploymentで確認する。

`LICENSE`・`LICENSES/`内のライセンス、provenance、third-party noticeおよびimmutable研究artifactは通常のdocs削除対象外。
