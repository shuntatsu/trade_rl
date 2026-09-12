# Agent Repository Control Plane v1

Status: Active

Related: #452

## 1. Objective

Trade RL を「Agentが読みやすいRepository」から、**Agentが発見・変更・検証・統合の各段階で誤りにくいRepository**へ進める。

既存の Domain → Capability 構造、current-only docs、architecture tests、public facade、artifact integrity、distribution closure を正本として維持し、その上に次の5機能を追加する設計とする。

1. **Agent Eval** — Repository構造が実際にAgentの正答率・探索効率を改善しているか測る。
2. **Preflight / Context / Impact Inspector** — source/Gitから現在の変更対象・入口・依存・関連検証をその場で導出する。
3. **Merge Safety** — mainへの変更経路をPR + current-main-inclusive tested-head CI中心にし、Agentの誤操作をRepository側で止める。
4. **Semantic Diff** — Git diffだけでは見えにくい新しいauthority・public surface・schema・dependency/effectをreview signalとして可視化する。
5. **Risk-based Verification Planner** — 開発中は変更リスクに近い検査を選び、最終段階では現行full quality gateへ収束する。

このcontrol planeは**第二のdomain authorityを作らない**。出力はsource/Git/current docsから都度導出し、生成catalogをcurrent treeへ保存しない。

## 2. Context

現在のRepositoryは既に次を持つ。

- root `AGENTS.md` による topology-first routing;
- `docs/architecture/` / `docs/research/` のcurrent-only authority;
- `artifacts / data / integrations / risk / simulation / strategies / evaluation` のtop-level ownership;
- capability public facadeとdirect-owner import policy;
- static import ownership checker;
- artifact/tamper/state/concurrency/property tests;
- Git tracked source / sdist / direct wheel / sdist-rebuilt wheel closure;
- checkout外clean-install smoke;
- full pytest / Ruff / format / Mypy / build / package identity gate。

したがってv1で必要なのは、新しいmetadata systemではなく、**既存契約をAgentが低コストに発見・利用できる操作面**である。

設計時点のmainは `5e300a30cb0620b7d79bd2c24f66dc0512012747`。Open PRは0件。main branch protection / required status checksは未設定である。

## 3. Non-goals

v1では次を行わない。

- 全class / dataclass / schemaの永続catalogを作らない。
- #435型のYAML metadata / docs site / generated manifest systemを復活させない。
- `trade_rl` runtime packageへAgent専用APIを混ぜない。
- application/runtime behavior、trading semantics、research resultを変更しない。
- persisted artifact schemaを変更しない。
- private moduleをpublic compatibility shim化しない。
- file size、export count、coverage率だけをhard gateにしない。
- 毎PRで全OS、全optional dependency、mutation test、external provider canaryを実行しない。
- Agentによる自動mergeを導入しない。
- Agent Evalの結果や生成contextをcurrent treeへ蓄積しない。

## 4. Approaches considered

### A. Heavy metadata / registry revival

`source_of_truth_for`、YAML metadata、生成indexをRepository全体に付与する。

**利点:** machine discoveryは明示的。

**欠点:** sourceとは別のauthorityが増え、stalenessとcontext costが大きい。#445のcurrent-only / lean原則と衝突しやすい。

**判断:** 採用しない。

### B. Documentation and conventions only

AGENTS/docs/package facadeだけを強化し、追加toolingを作らない。

**利点:** 最小変更。

**欠点:** Agentが毎回HEAD、diff、dependency、tests、artifact/public surfaceを手作業で再構築する。multi-agentや大規模変更で探索コストと見落としが残る。

**判断:** 現状より良いが不足。

### C. Lightweight source-derived control plane — Recommended

Repository-local developer toolingを置き、Git/source/architecture contractから一時的にcontext・impact・semantic diff・verification planを生成する。Agent Evalは別の評価層として扱い、生成結果は保存しない。

**利点:** #445のlean原則を維持しつつ、#435のdeterministic discoveryの利点だけ回収できる。

**判断:** v1として採用する。

## 5. Core design principles

### 5.1 Source remains authoritative

Control planeは以下を読む側であり、domain authorityにはならない。

```text
Git / current HEAD
current source tree
package facades
architecture tests
current architecture/research docs
pyproject / CI
```

手書きのtype registry、schema registry、owner registryは作らない。

### 5.2 Ephemeral outputs

`context`、`impact`、`semantic diff`、`verification plan`はstdout/CI summary等へ出す。生成JSON/MarkdownをGit管理しない。

### 5.3 Review signal vs hard gate

新しいclass、schema、file write、export等は即failさせずreview signalにする。

既存architecture contract違反、forbidden dependency、distribution closure破壊等は既存hard gateを維持する。

### 5.4 Local fast loop, final full gate

開発途中は変更箇所に近いtestを優先する。完了判定は現行full quality gateへ戻る。

Coverageは**80%を目標signal**とするが、単一のhard completion gateにはしない。重要Failure Mode、変更行、Assertion品質を優先する。

## 6. Repository-local tooling boundary

Agent control planeの実装はruntime packageから分離し、top-level `tools/` 配下のrepository toolingとして持つことを推奨する。

初期案:

```text
tools/
  __init__.py
  agent_repo/
    __init__.py
    __main__.py
    git_state.py
    source_index.py
    semantic_diff.py
    verification.py
```

これは最終folder数を固定する契約ではない。実装時に責務が小さければ統合する。逆に1巨大fileへ集約しない。

`setuptools.packages.find.include = ["trade_rl*"]` を維持し、`tools/` は**installed production wheelへ混入させない**。sdistへの開発用tooling inclusionはproduction Python-source closureとは別問題なので、実装時に明示的に確認し、含まれる/含まれないを推測で断定しない。

CLI想定:

```bash
uv run python -m tools.agent_repo preflight
uv run python -m tools.agent_repo context trade_rl/evaluation/runs
uv run python -m tools.agent_repo impact trade_rl/evaluation/runs/config.py
uv run python -m tools.agent_repo diff --base main
uv run python -m tools.agent_repo verify --base main
```

v1はnetworkを必要としない。Open PR overlapはroot `AGENTS.md`に従いGitHub側で確認し、local inspectorがGitHub API accessを暗黙要求しない。

## 7. Preflight

`preflight` は少なくとも次を表示する。

- current branch;
- current HEAD;
- merge-base / base ref when resolvable;
- dirty / untracked state;
- changed production/test/docs/config paths;
- active `docs/specs` / `docs/plans` files;
- current permanent CI workflow roster。

これは「安全である」と判定するtoolではなく、**Agentが最初に確認すべき事実を一度に提示するtool**である。

GitHub open PR / branch overlapは別途GitHub queryが必要であり、offline preflightの結果だけで「競合なし」と宣言しない。

## 8. Context inspector

`context <path>` は対象pathから、次の事実をsource-derivedで示す。

- top-level domain / nearest capability;
- nearest package facade and literal/static `__all__` where resolvable;
- direct imports and semantic imports;
- direct known callers / reverse imports;
- related architecture docsへのpath;
- mirrored/nearby test paths;
- relevant architecture tests;
- detected persisted schema constants/public exports;
- detected filesystem/network effect signals;
- optional dependency / pyproject references where relevant。

「owner」は推測で断定しない。architecture docsやfacadeで明示される場合のみauthorityとして表示し、それ以外は`candidate owner` / `consumer`として区別する。

## 9. Impact inspector

`impact <path...>` は変更対象からreverse dependencyと契約面を導出する。

最低限:

```text
direct consumers
semantic consumers via static re-export
public facade exposure
related tests
architecture constraints
artifact/schema/public-surface signals
optional-dependency/config surfaces
```

Impact出力は完全なruntime reachabilityの証明ではない。dynamic import、reflection、runtime rebindingは限界として明示する。

## 10. Semantic diff

`diff --base <ref>` はGitのtext diffとは別に、review用semantic surfaceを列挙する。

v1のsignal候補:

- new/removed public exports;
- new/removed `dataclass`, `TypedDict`, `NamedTuple`, `Protocol`, `Enum`等のdata-shape declarations;
- schema/version constantの追加・削除・値変更;
- production dependency edgeの追加・削除;
- optional dependency / project script / CI surface変更;
- 新しいURL literal / network call site候補;
- 新しいfilesystem mutation call site候補;
- new/removed CLI/public facade surface。

重要: field集合の類似だけで「duplicate type」と自動判定しない。同形でも意味が異なるconceptを統合しないため、duplicate候補はreview signal止まりとする。

## 11. Risk-based verification planner

`verify --base <ref>` は変更path + semantic diffから検証を3層に分類する。

### Fast targeted

開発反復用。変更Capabilityに近いunit/contract/architecture/type checks。

### Required final

完了判定前に現行CI相当へ収束する。

```text
Ruff
Format
Mypy
architecture tooling type check
full pytest
build
distribution closure
clean installed smoke
package identity
```

### Extended conditional

変更Failure Modeが該当するときのみ提案する。

- Windows/filesystem locking;
- optional dependency installed integration;
- targeted mutation/falsification;
- long property/model-based test;
- performance benchmark;
- external provider canary;
- historical artifact compatibility corpus。

Plannerはtest成功を保証しない。また「このplanだけ実行すれば常に十分」と主張しない。最終reportで未実行Extended checkを明示する。

## 12. Verification profile policy

変更タイプから最低限のOracleを選ぶ。

| Change | Targeted oracle |
|---|---|
| private refactor | local behavior + semantic equivalence where valuable |
| package move | architecture/import/public API + semantic equivalence + distribution |
| public facade | object identity/export snapshot + clean installed import |
| Config/Spec | direct constructor + parse + resolve + malformed/unknown |
| artifact | roundtrip + tamper + atomicity + compatibility as applicable |
| network | retry/fallback/error category + offline/frozen boundary |
| state machine | transition + property/model-based + concurrency where applicable |
| accounting/risk | independent oracle + boundary/precision/property |
| causality | future-mutation/prefix invariance metamorphic test |
| optional dependency | missing dependency + installed capability smoke |
| bug fix | reproducing RED + regression + adjacent failure mode |

この表は永続的な全ケース列挙ではなく、v1のrouting contractである。

## 13. Coverage policy

Coverageは補助signalとして扱う。

- branch coverage **80%を目標値**とする。
- 80%未満だけを理由にmerge不可とはしない。
- 数値を上げるためだけの低価値testを追加しない。
- 重要なFailure Mode、変更行、Error/Retry/Timeout/Fallback、Assertion品質を優先する。
- Coverage計測をCI hard gateにするかは、baseline計測と実行コストを見て別途決定する。
- 設定だけ存在して実際には計測していない状態を最終形にしない。

## 14. Agent Eval

AI-first Repositoryの改善は、通常testだけでは測れない。実Agentを対象にした小さなbenchmark corpusを用意する。

v1ではmodel outputやrun logをcurrent treeへ保存しない。Repositoryへ保存するのは**task definitionとrubricだけ**に限定する。

候補ケース:

1. Candidate Runに新しい設定を追加する。
2. SHA-256 digest fieldのvalidationを追加する。
3. PPOのdataset-bound feature/symbol selectionを変更する。
4. persisted Run artifactへ互換性を壊さずfield/evidenceを追加する想定を検討する。
5. Binance transport fallback bugを修正する。

評価軸:

```text
correct capability/owner discovery
existing authority reuse
duplicate authority creation
public/private boundary compliance
scope discipline
appropriate test discovery
verification selection
artifact compatibility awareness
unnecessary file/context reads
```

Agent Evalは初期段階ではPR hard gateにしない。構造変更前後の比較、回帰調査、設計判断の根拠として用いる。

## 15. Multi-agent collision handling

v1では中央lock registryを作らない。

- root `AGENTS.md` の topology-first checkを維持する。
- local preflightはbranch/HEAD/changed pathsを示す。
- GitHub-sideでは作業開始時にopen PR head/base/diff overlapを確認する。
- 将来、重複変更が実際に繰り返し発生する場合のみ、PR overlap warningの自動化を検討する。

不要なcoordination metadataをRepositoryへ追加しない。

## 16. Merge safety

mainは設計時点でunprotectedである。AI Agent運用では、Repository内部の強いCIだけでは誤操作を防ぎきれない。

v1導入時のintegration invariantは **tested PR head contains current `main`** である。merge authorization時点のcurrent `main` commitがtested PR headのancestorであり、その同一PR HEADにpermanent CI successが存在することをcurrent integration evidenceとする。`main` が進んだ後の古いPR-head Greenは、PR head自体が変わっていなくてもstale evidenceとして扱う。

GitHub側で次を目標とする。

- main direct mutationを通常経路にしない;
- PR経由を標準化;
- tested PR headがcurrent `main` を含むことをmerge条件にする;
- その同一tested PR HEADのpermanent CI successをmerge条件にする;
- force-push/history rewriteを禁止;
- main deletionを禁止;
- auto-mergeは明示的に採用しない。

required approval count等の具体設定は、solo-maintainer workflowを壊さないよう実装時にGitHubの利用可能なruleset/branch-protection機能を確認して決める。strict branch-up-to-date / merge queue等でcurrent-main containmentをGitHub側に強制できる場合はそれを利用する。設定surfaceがその条件を表現できない場合でもchecked-in process invariantを弱めず、merge直前にread-only compare/merge-baseでcurrent `main` containmentを明示的に検証する。

Protection設定を変更した後はread-backし、期待したruleが有効か検証する。設定に失敗した場合、Repositoryが保護されたと主張しない。protectionが未設定の状態でdirect-main rejectionを確認するための書き込みprobeは、失敗時に`main`を実際に変更し得るため行わない。

## 17. PR quality contract

AgentがPRを書くとき最低限次を明示する。

```text
Objective
Non-goals
Acceptance Criteria
Invariants
Failure Modes
Test Oracle
Changed authorities / public surfaces
Tests / verification
Falsification
Unverified items
Residual risk
```

テンプレートは短く保ち、単純変更に不要な長文を強制しない。

## 18. Invariants

v1実装中も次を維持する。

- production runtime/economic behaviorは変えない。
- `trade_rl` public APIをcontrol-plane都合で拡張しない。
- persisted artifact schema/identityを変更しない。
- current-only docs policyを維持する。
- generated context/diff/reportをcommitしない。
- architecture testsの既存hard gateを弱めない。
- full final quality gateをfast local plannerで置き換えない。
- optional/extended checksを無条件に常時実行へ昇格しない。
- merge authorization時のtested PR headはthen-current `main` を包含し、same-head permanent CI evidenceを持つ。

## 19. Failure modes

### FM-1: Control plane becomes second authority

手書きowner mapやschema registryがsourceと乖離する。

**Mitigation:** source-derivedを原則とし、手書きpolicyは検証routing等の最小ルールだけに限定する。

### FM-2: False confidence from static analysis

Agentが`impact`出力を完全なruntime reachabilityと思う。

**Mitigation:** dynamic import/reflectionの限界を出力し、最終full testを維持する。

### FM-3: Alert fatigue

semantic diffが大量の低価値signalを出す。

**Mitigation:** review signalのprecisionを優先し、heuristicをhard failにしない。不要signalは削る。

### FM-4: Verification planner becomes expensive checklist

変更と無関係なplatform/optional/mutation testまで常時要求する。

**Mitigation:** Fast / Required final / Extended conditionalの3層を維持する。

### FM-5: Agent Eval overfits Repository to benchmark

数個のpromptだけ通る構造へ歪む。

**Mitigation:** task familyを複数持ち、個別expected pathではなくauthority reuse/scope/test selection等を評価する。

### FM-6: Branch protection blocks maintainer recovery

rulesetを厳しくしすぎて緊急修正不能になる。

**Mitigation:** solo-maintainer運用を踏まえて設定を事前レビューし、read-backする。force-push禁止等の高価値ruleを優先する。

### FM-7: Tooling leaks into production distribution

`tools/` がinstalled production wheelへ混入する。

**Mitigation:** existing distribution closureをoracleとして使い、production `trade_rl/**/*.py` rosterとの契約を維持する。sdist inclusionは別途観測し、production wheelの境界と混同しない。

### FM-8: Green PR head is stale against current main

PR HEAD自身はGreenでも、その後`main`が進み、実際の統合treeが未検証になる。

**Mitigation:** merge authorization直前にtested PR head contains current `main` をread-only compare/merge-baseで確認する。満たさなければcurrent `main` をnon-forceで取り込んだ新HEADを作り、permanent CIを再実行する。古いGreenを再利用しない。

## 20. Risk

全体リスクは **Medium**。

production runtimeを変更しないため経済ロジックへの直接リスクは低い。一方、Agentがcontrol-plane出力を過信すると誤ったscope/test selectionへ誘導するため、static-analysisの限界とfull final gateを明確にする必要がある。

GitHub protection変更はRepository運用へ直接影響するため、code PRとは分けて確認可能な手順として扱う。

## 21. Test Oracle

control plane自身にも独立Oracleを持つ。

- synthetic mini-treeでimport/public facade/re-export/relative importを検証;
- known diff fixturesで追加type/schema/export/dependency signalをexact確認;
- false-positive fixtureで無関係なprivate helperをpublic changeと誤認しないことを確認;
- verification planner table-driven testでchanged paths→recommended layersを検証;
- generated outputがfilesystem/current docsへ永続化されないことを確認;
- distribution closureで`tools/`がproduction wheelへ混入しないことを確認;
- tested PR headとcurrent `main` のcompare/merge-baseでintegration evidenceのfreshnessを確認;
- branch protectionはGitHub read-backで検証;
- Agent Evalはfresh-agent runを用い、rubricでauthority discovery / duplicate creation / scope / verificationを評価。

## 22. Required test layers

v1 implementationでは少なくとも次を要求する。

- Unit: parser/classifier/planner pure logic;
- Contract: CLI output semantics / exit status / fail-closed parsing;
- Architecture: control-plane dependencyとproduction-package non-dependency;
- Integration: temporary Git repositoryを用いたpreflight/diff/impact;
- Distribution: existing source/sdist/wheel closure;
- Regression: existing full suite;
- Falsification: semantic-diffやplannerの誤分類を意図的に作り、対応testが検出するか確認。

Agent Evalは通常pytestとは分離したExtended verificationとする。

## 23. Acceptance Criteria

1. Agentが一つのlocal commandでcurrent branch/HEAD/diff状態を確認できる。
2. pathを与えるとsource-derivedのcapability/facade/dependency/tests/docs候補を得られる。
3. changed pathについてreverse impactとpublic/schema/dependency surfaceを確認できる。
4. base refとの差からsemantic review signalを再現可能に生成できる。
5. verification plannerがFast / Required final / Extended conditionalを分離する。
6. generated outputをGit管理しない。
7. `trade_rl` runtime public API、artifact schema、economic behaviorを変更しない。
8. control planeはproduction wheelへ混入しない。
9. Agent Eval task/rubricが存在し、少なくとも構造変更前後を比較できる。
10. merge authorization時はtested PR head contains current `main` を満たし、その同一PR HEADのpermanent CI successをcurrent integration evidenceとして確認できる。main protection導入時はGitHub設定もread-backする。
11. Coverage 80%は目標signalとして扱い、不要なtestを増やすhard gateにしない。
12. current full CI / architecture / distribution gateを弱めない。

## 24. Quality Gate

完了扱いするには、該当implementation PRのfinal HEADで次を満たす。

- Acceptance Criteria対応表を確認;
- targeted tests Green;
- architecture toolingのself-tests Green;
- Ruff / format / Mypy Green;
- full pytest Green;
- build / source-sdist-wheel closure Green;
- clean installed public smoke Green;
- semantic-diff / verification-planner falsification mutationsを実施;
- final diffにgenerated reports、debug data、一時workflowがない;
- Agent Evalで少なくともbaseline比の悪化がないことを確認、または未実行理由を明記;
- merge authorization直前にfinal tested PR head contains current `main` とsame-head permanent CIを確認;
- branch protectionを実施したphaseではGitHub read-backを確認;
- 未検証事項・残存riskを明記。

## 25. Implementation sequencing

一つの巨大PRにはしない。以下を独立させる。

### Phase 0 — Baseline Agent Eval

現行mainを数個のfresh-agent taskで測定し、探索失敗・重複authority・不要検査のbaselineを得る。コード変更なし。

### Phase 1 — Preflight / Context / Impact core

source/Git read-only tooling。既存architecture analysis primitiveを重複実装しない。必要ならprivate test toolingをcanonical repository-tooling ownerへ移し、testsがそれを検証する。

### Phase 2 — Semantic Diff

public/data-shape/schema/dependency/config/effect signalを追加。review signalのみ。

### Phase 3 — Verification Planner

risk routingを追加。local fast loopを改善するが、final full CIを維持する。

### Phase 4 — Merge Safety

GitHub ruleset/branch-protectionを確認し、PR + current-main-inclusive tested-head CIを標準統合経路にする。Repository setting変更はコード差分と別に検証可能にする。

### Phase 5 — Agent Eval rerun / architecture decision

同一task familyで再測定し、context footprint・authority reuse・scope adherenceが改善したか確認する。改善しないtoolingは恒久化しない。

## 26. Review questions

実装計画へ進む前に特にレビューしてほしい点:

1. `tools/agent_repo` をruntime package外のrepository tooling ownerとする境界は妥当か。
2. v1をnetwork-free local inspectorとし、GitHub PR overlap automationを後回しにする判断は妥当か。
3. Semantic Diffのsignal集合は多すぎないか。最初に減らすべきsignalはあるか。
4. Verification Plannerが現行full CIを弱めず、開発途中の無駄だけを減らす設計になっているか。
5. Agent Evalのtask/rubricが特定モデルや現在のfolder名へ過適合していないか。
6. main protectionはsolo-maintainer運用を壊さず、Agent誤操作対策として十分か。
7. #452で扱う構造整理を、このcontrol planeの計測結果より先に広げる必要が本当にあるか。

このspec承認後にのみimplementation planへ進む。
