# Agent documentation contract

## 目的

この文書は、AgentがTrade RLを変更するときに**何を先に読み、どの文書を更新し、何をcurrent treeに残さないか**を定義する。

## 読む順序

1. root `AGENTS.md`
2. `main/current HEAD`、作業branch、open PRのhead/base、recent commitsを確認し、進行中の同目的の変更・後継実装・廃止予定の作業線を識別する
3. `docs/README.md`
4. `docs/architecture/lean-core.md`
5. `docs/architecture/package-boundaries.md`
6. Study/Experiment/EvidenceSetやdevelopment研究workflowに関わる作業なら `docs/architecture/controlled-experiment-loop.md`
7. 研究判断・候補・評価・データscopeに関わる作業なら `docs/research/current-status.md`
8. 対象source、public facade、nearest tests、CIを照合する

文書だけを根拠にsourceを推測しない。現行source、public API、`tests/architecture/`、関連contract testsとdocsを突き合わせる。

## 更新matrix

| 変更 | 必ず確認・更新する場所 |
|---|---|
| causal data、availability、artifact identity | `architecture/lean-core.md`, data tests |
| strategy interface、quantity hold、strategy/risk責任 | `architecture/lean-core.md`, strategy/replay tests |
| execution/accounting/fill/funding/borrow/liquidation | `architecture/lean-core.md`, simulation/risk/evaluation tests |
| package移動、責務境界、依存方向、public facade | `architecture/package-boundaries.md`, `tests/architecture/` |
| Study/Experiment/EvidenceSet、controlled factor、lineage、freeze | `architecture/controlled-experiment-loop.md`, experiment contract/workflow tests |
| 候補strategy/control、fit scope、evaluation scope | `research/current-status.md`, candidate/strategy tests |
| M1/M2/M3状態、development/final/stress手順 | `research/current-status.md` |
| docsの入口・保持ルール | `docs/README.md`, `docs/AGENTS.md`, root `AGENTS.md` |
| license/provenance/third party | `LICENSE`, `LICENSES/`, package metadata。通常cleanupとは分離する |

## フォルダと保持ルール

- `architecture/`: current authoritative architectureのみ。
- `research/`: current research status/evidence protocolのみ。
- `specs/`: 未実装またはreview中で、独立してnormativeなdesignが存在するときだけ作る。
- `plans/`: 実装中の独立planが必要なときだけ作る。
- `specs/` / `plans/` に置く各Markdownは先頭付近に `Status: Active` を明記する。Activeでないfileをcurrent treeに残さない。
- 完了した `specs/` / `plans/` は、耐久的な内容を `architecture/` / `research/` へ反映した後に削除する。
- `docs/history` と `docs/archive` は作らない。
- 過去の設計・migration経緯・削除済み文書は **Git history** から取得する。

削除する前に、非再生成可能なsource data、未記録の実験結果、exact reproductionに必要なcommit/lock/config、legal noticeを失わないことを確認する。

## 法務の境界

`LICENSES/` は通常docsではない。license text、licensing policy、provenance、third-party noticeは永久保持するcompliance materialであり、完了済みplan/specと同じ理由で削除しない。

## 実装時の原則

- 現行の良い責務境界を優先し、将来使うかもしれない抽象化を追加しない。
- private old pathを残すだけのcompatibility shimは、明示的public contractの証拠がない限り作らない。
- 変更前にTest Oracleとfailure modeを決める。
- refactor/bugfixはTDDでREDを観測してからproductionを変える。
- Greenにするためにassertionを弱めたり、skipや古いforwarderを残したりしない。
- 移動だけのrefactorでは、可能ならAST/serialization/public facade等の独立oracleでsemantic driftを反証する。

## Verification contract

完了報告前に、変更内容に応じたtargeted testと全体quality gateを確認する。現在の標準gateは次である。

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run mypy tests/architecture/imports.py tests/architecture/distribution.py
uv run pytest -q tests
```

さらに `uv build`、tracked production Python sourceとsdist/direct wheel/sdist再build wheelのpath・bytes一致、checkout外での非editable installとpublic import/CLI smoke、package identity、関連architecture/contract test、GitHub Actionsの**同一final HEAD**の結果を確認する。古いHEADのGreenを現在HEADの証拠にしない。

テストGreenだけでは正しさを宣言しない。最終diff、削除物、public API、重要failure mode、未検証事項、残存riskを再確認する。

## 作業線の終了と変更の証拠

不要な作業線は、mainへのancestor関係、同一tree、または後継実装が固有差分を包含する独立した証拠で判断する。open PRのhead/baseと進行中の実行は保護し、削除直前のexact SHA確認と削除後の一覧確認を行う。名称が古いことだけでは削除しない。未統合の実験設定・非再生成可能な証拠は残す。superseded PRは理由と後継の正本を会話へ記録し、旧設計を再mergeしない。

ソース移動やPython docstring変更でも、Runのimplementation manifestは相対パスとraw bytesにbindするため実装digestが変化する。同じStudyの比較途中へ移行を混ぜず、元のsource/runtimeを保持する。計算の意味保存と、期待されるprovenance変化は別々に検証する。
