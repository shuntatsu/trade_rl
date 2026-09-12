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

## Local repository tooling

Repository-localなsource-derived inspectionには次を使ってよい。

```bash
uv run python -m tools.agent_repo preflight --base main
uv run python -m tools.agent_repo context trade_rl/evaluation/runs
uv run python -m tools.agent_repo impact trade_rl/evaluation/runs/config.py
uv run python -m tools.agent_repo diff --base main
uv run python -m tools.agent_repo verify --base main
uv run python -m tools.agent_repo eval-list
```

- `preflight`: local branch/HEAD/base/worktree/Active docs/workflow rosterを表示する。
- `context`: capability、facade、imports/consumers、近傍test/docs、schema/effect signalをsourceから導出する。
- `impact`: 複数pathのcontextを決定的順序で表示する。
- `diff`: public/data-shape/schema/dependency/effectのreview signalを表示する。signal自体を仕様違反判定には使わない。
- `verify`: Fast / Required final / Extended / Coverage signalを分けて表示する。開発途中の無駄を減らすためのroutingであり、final full gateを置き換えない。
- `eval-list` / `eval-show` / `eval-score`: versionedなAgent-UX task promptとgeneric rubricを表示・Evaluator入力から採点する。task-specificなEvaluator answer keyはchecked-in corpusへ保存しない。

これらはnetwork-free local toolingであり、GitHub上のopen PR/branch overlapは別途確認する。出力は一時情報であり、生成JSON/Markdown reportをcurrent treeへcommitしない。

## Agent Eval protocol

Agent向けRepository構造の改善を評価する場合、baselineとcomparisonで同一suite/rubricを使い、**評価対象変更以外のRepository差分を混ぜない**。

1. 評価pairを作る時点のcurrent PR base/current `main` SHAをbaselineとし、comparisonにはそのexact baselineをancestorとして含むPR HEADを使う。実装開始時の古いbaseを、後からmain更新を取り込んだHEADと比較しない。
2. pairの実行完了前に`main`が進んだ場合、その比較証拠をstaleとして破棄し、current baseを含むcomparison HEADへ同期してbaseline/comparisonの両側を組み直す。
3. baseline側とcomparison側の各taskは、それぞれexact snapshotから作ったcleanな使い捨てbranch/worktreeで独立実行する。前taskの変更を次taskへ持ち越さない。
4. task-specificな過去会話を持たないfresh Agent sessionを開始し、通常のRepository accessだけを与える。
5. Agentへはtask promptと通常のRepository instructionsだけを渡す。generic rubricを明示的な攻略指示として追加せず、task-specificなsemantic goal / expected owner / critical failure / review question等のanswer keyはevaluated checkoutへ置かない。
6. 完了後にdiff、選択したcommands/tests、public/private imports、残存riskを観測する。
7. 全generic rubric dimensionを短いobservable evidence付きで採点する。critical failureはgeneric critical dimensionsの0点から決定し、task-specificな隠し正解をRepository内へ追加しない。
8. aggregate score、critical failure、changed-path count、目立つ不要探索だけをPR/Issue summaryへ記録する。
9. transcript、model reasoning、run-specific log、generated patchをcurrent treeへcommitしない。

Agent Evalはv1ではExtended verificationでありPR hard gateではない。点数上昇だけを品質証明にせず、critical failureやauthority/boundary/scopeの回帰を優先して見る。benchmarkへ過適合するためにprompt/rubricを弱めない。

## Git / PR integration safety

Agent実装は専用branchまたはworktreeで行い、PRを通常の統合経路とする。`main` への直接変更を通常経路にしない。mergeはユーザーの明示許可が必要であり、Agentはforce-push、history rewrite、`main`削除を行わない。

Integration invariant: tested PR head contains current `main`. PRの成功証拠は**現在のfinal HEAD**に束縛し、merge直前のcurrent `main` SHAをそのtested PR headがancestorとして包含していることも確認する。PR headが動いた後の古いCIだけでなく、`main` が進んだ後の古いPR-head Greenも現在の統合証拠として扱わない。`main` が進んだ場合は、non-force merge/rebase等でcurrent `main` を含む新しいPR HEADを作り、その新HEADでpermanent CIを再実行する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等のoracleとしてよい。

Branch protection / rulesetはGit treeとは別のGitHub設定である。保護を導入・変更した場合は、GitHubから設定をread-backし、required check、force-push/deletion、PR requirement、maintainer/admin bypass挙動を確認してから有効化済みと報告する。設定変更surfaceが利用できない場合は、proseやarchitecture testを代替にせず**未設定/未検証**と報告する。

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
- `plans/`: 実装中の独立planが必要な間だけ作る。
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
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run pytest -q tests
```

さらに `uv build`、tracked production Python sourceとsdist/direct wheel/sdist再build wheelのpath・bytes一致、checkout外での非editable installとpublic import/CLI smoke、package identity、関連architecture/contract test、GitHub Actionsの**同一final HEAD**の結果を確認する。古いHEADのGreenを現在HEADの証拠にしない。merge直前は同時に、tested PR head contains current `main` を満たすことを再確認する。

Coverageはbranch coverage 80%を目標signalとするが、80%未満だけを理由にmerge不可とはしない。重要Failure Mode、変更行、Error/Retry/Timeout/Fallback、Assertion品質を優先し、数値を上げるだけの低価値testを追加しない。

テストGreenだけでは正しさを宣言しない。最終diff、削除物、public API、重要failure mode、未検証事項、残存riskを再確認する。

## 作業線の終了と変更の証拠

不要な作業線は、mainへのancestor関係、同一tree、または後継実装が固有差分を包含する独立した証拠で判断する。open PRのhead/baseと進行中の実行は保護し、削除直前のexact SHA確認と削除後の一覧確認を行う。名称が古いことだけでは削除しない。未統合の実験設定・非再生成可能な証拠は残す。superseded PRは理由と後継の正本を会話へ記録し、旧設計を再mergeしない。

ソース移動やPython docstring変更でも、Runのimplementation manifestは相対パスとraw bytesにbindするため実装digestが変化する。同じStudyの比較途中へ移行を混ぜず、元のsource/runtimeを保持する。計算の意味保存と、期待されるprovenance変化は別々に検証する。
