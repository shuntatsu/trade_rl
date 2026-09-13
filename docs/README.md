# Trade RL documentation

## 結論

この `docs/` tree は**現在のシステムを理解・変更するための正本だけ**を置く。過去の設計、完了済みmigration plan、完了済みspecを保存する場所ではない。過去資料が必要な場合は Git history を参照する。

現在読むべき恒久文書は次の5つである。

- [`AGENTS.md`](AGENTS.md) — Agent向けの読む順序、更新先、保持・削除ルール、verification contract
- [`architecture/lean-core.md`](architecture/lean-core.md) — 現行lean coreの目的、データ・strategy・risk・execution/accounting・artifactの不変条件
- [`architecture/package-boundaries.md`](architecture/package-boundaries.md) — 現在のpackage配置、責務境界、依存方向、public API方針
- [`architecture/controlled-experiment-loop.md`](architecture/controlled-experiment-loop.md) — development Study、Canonical M2 bootstrap preparation、EvidenceSet、controlled factor、lineage、FAILED/INVALID、freezeの恒久契約
- [`research/current-status.md`](research/current-status.md) — 現在の研究目的、比較候補、M1/M2/M3の状態、canonical bootstrap、development/final評価手順と未検証事項

人間向けの説明UIは root [`guide/`](../guide/README.md) に置く。Interactive Guideはこの `docs/` の正本と現行source/testsから派生した**非正本の説明層**であり、技術仕様・研究状態のauthorityにはしない。Guideは日本語を主表示し、実identifierを副表示するコード連動型実装エクスプローラである。Markdown section fingerprintとPython symbol/source digestを別々にfail-closed検証し、source linkはbuildしたexact revisionへ固定する。公開版は <https://shuntatsu.github.io/trade_rl/> で提供する。

現在Activeな独立spec:

- [`specs/2026-09-13-guide-markdown-first-design.md`](specs/2026-09-13-guide-markdown-first-design.md) — Human Guideをvisualization-firstからMarkdown-first technical document + secondary implementation explorerへ再設計するactive normative design。implementationは未開始。

新しい未実装designや実装中の独立planが必要になった場合だけ `specs/` / `plans/` を作り、完了後は恒久契約へ昇格してcurrent treeから削除する。

Interactive Human Guide Pagesと日本語コード連動型Guideの恒久運用契約は `.github/workflows/deploy-guide.yml`、`guide/README.md`、`docs/AGENTS.md`、architecture/contract testsへ昇格済みである。Issue #494のportable Dataset numerics contractは `architecture/lean-core.md` と `architecture/package-boundaries.md` へ、portable canonical lineageのcurrent stateは `research/current-status.md` へ昇格済みである。Agent Repository Control Plane v1、Agent Coordination Plane v1の実装済み契約も `AGENTS.md`、`architecture/package-boundaries.md`、Repository tooling source、architecture/contract testsへ昇格済みであり、完了済みspec/planはcurrent treeから除去する。外部GitHub設定や独立Agent実験のようなGit tree外の未完了作業はGitHub Issueで追跡し、完了済みimplementation planをcurrent docsへ残さない。

GitHub PagesのSource設定と公開siteの到達性はGit tree外の状態である。公開済み状態はworkflow実装やCI Greenだけでなく、Pages state read-back、deployment、public smokeで検証する。

Rootの `README.md` はRepository概要と実行入口、rootの `AGENTS.md` はAgentの最短routingを担当する。詳細な現行契約はこの `docs/` 以下を正本とする。

## フォルダ方針

現在の恒久フォルダは `architecture/` と `research/` である。`specs/` と `plans/` はActive design/implementationが存在する間だけ保持する。

- `architecture/`: 現行コードが満たすべき構造・責務・不変条件
- `research/`: 現在の研究仮説、比較条件、証拠プロトコル、研究状態
- `specs/`: 未実装またはreview中の独立normative designだけ
- `plans/`: 実装中または外部verification待ちで、独立したcurrent workが残るplanだけ

将来、未実装で独立してnormativeな設計がある間だけ `specs/` を作ってよい。実装中または外部verification待ちの独立作業が必要な間だけ `plans/` を作ってよい。実装・verificationが終わり、耐久的な内容が `architecture/`、`research/`、`AGENTS.md` 等のcurrent authorityへ反映されたら、完了済みspec/planはcurrent treeから削除する。

`docs/history` と `docs/archive` は作らない。Git historyが履歴の正本である。

## 何を更新するか

- package、責務、依存方向、公開境界を変えた → `architecture/package-boundaries.md`
- causal data、strategy/risk分離、execution/accounting、artifact invariantを変えた → `architecture/lean-core.md`
- Study/Experiment/EvidenceSet、Canonical M2 bootstrap、controlled factor、lineage、freeze契約を変えた → `architecture/controlled-experiment-loop.md`
- 候補、fit/evaluation scope、評価期間、bootstrap実行状態、判定手順、研究状態を変えた → `research/current-status.md`
- Guideが参照する正本sectionまたはPython symbolを変えた → 対応する `guide/content/topics/*.json` を再確認し、`--refresh` / `--refresh-code` を対象topicだけ実行して `guide/tools/content_contract.py --check` を通す
- Guide Pages deployment / public URLを変えた → `.github/workflows/deploy-guide.yml`、`guide/README.md`、Pages state read-backを確認する
- docsの入口・保持ルールを変えた → `AGENTS.md` と必要に応じてこのREADME

コードとdocsが食い違った場合は、現行source・architecture test・public contractを確認して真の契約を確定し、docsだけを放置しない。

## 法務・provenance

`LICENSE` と `LICENSES/` はcurrent docsの整理対象ではない。ライセンス本文、provenance、third-party noticeは永久保持するcompliance materialであり、通常の完了済みplan/specと同じ削除ルールを適用しない。