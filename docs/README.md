# Trade RL documentation

## 結論

この `docs/` tree は**現在のシステムを理解・変更するための正本だけ**を置く。過去の設計、完了済みmigration plan、完了済みspecを保存する場所ではない。過去資料が必要な場合は Git history を参照する。

現在読むべき恒久文書は次の5つである。

- [`AGENTS.md`](AGENTS.md) — Agent向けの読む順序、更新先、保持・削除ルール、verification contract
- [`architecture/lean-core.md`](architecture/lean-core.md) — 現行lean coreの目的、データ・strategy・risk・execution/accounting・artifactの不変条件
- [`architecture/package-boundaries.md`](architecture/package-boundaries.md) — 現在のpackage配置、責務境界、依存方向、public API方針
- [`architecture/controlled-experiment-loop.md`](architecture/controlled-experiment-loop.md) — development Study、Canonical M2 bootstrap preparation、EvidenceSet、controlled factor、lineage、FAILED/INVALID、freezeの恒久契約
- [`research/current-status.md`](research/current-status.md) — 現在の研究目的、比較候補、M1/M2/M3の状態、canonical bootstrap、development/final評価手順と未検証事項

現在Activeなimplementation plan:

- [`plans/2026-09-11-simulation-sha256-authority-plan.md`](plans/2026-09-11-simulation-sha256-authority-plan.md) — simulation内のSHA-256 lexical validationを既存authorityへ収束させる限定変更。完了後はcurrent treeから削除する。

Rootの `README.md` はRepository概要と実行入口、rootの `AGENTS.md` はAgentの最短routingを担当する。詳細な現行契約はこの `docs/` 以下を正本とする。

## フォルダ方針

現在の恒久フォルダは `architecture/` と `research/` である。`specs/` と `plans/` はActive design/implementationが存在する間だけ保持する。

- `architecture/`: 現行コードが満たすべき構造・責務・不変条件
- `research/`: 現在の研究仮説、比較条件、証拠プロトコル、研究状態
- `specs/`: 未実装またはreview中の独立normative designだけ
- `plans/`: 実装中の独立implementation planだけ

将来、未実装で独立してnormativeな設計がある間だけ `specs/` を作ってよい。実装中の独立した作業計画が必要な間だけ `plans/` を作ってよい。実装が終わり、耐久的な内容が `architecture/` または `research/` に反映されたら、完了済みspec/planはcurrent treeから削除する。

`docs/history` と `docs/archive` は作らない。Git historyが履歴の正本である。

## 何を更新するか

- package、責務、依存方向、公開境界を変えた → `architecture/package-boundaries.md`
- causal data、strategy/risk分離、execution/accounting、artifact invariantを変えた → `architecture/lean-core.md`
- Study/Experiment/EvidenceSet、Canonical M2 bootstrap、controlled factor、lineage、freeze契約を変えた → `architecture/controlled-experiment-loop.md`
- 候補、fit/evaluation scope、評価期間、bootstrap実行状態、判定手順、研究状態を変えた → `research/current-status.md`
- docsのrouting/保持ルールを変えた → `AGENTS.md` と必要に応じてこのREADME

コードとdocsが食い違った場合は、現行source・architecture test・public contractを確認して真の契約を確定し、docsだけを放置しない。

## 法務・provenance

`LICENSE` と `LICENSES/` はcurrent docsの整理対象ではない。ライセンス本文、provenance、third-party noticeは永久保持するcompliance materialであり、通常の完了済みplan/specと同じ削除ルールを適用しない。
