# evaluation/runs capability facade design

Status: Active
Date: 2026-09-11
Related: #452

## 結論

`trade_rl.evaluation.runs` を **Tier-2 capability public API** として明示し、`evaluation/runs/` 外のproduction codeは原則として `runs.config`、`runs.artifact`、`runs.execute`、`runs.provenance`、`runs.candidate_suite` の物理submoduleへ直接依存しない。

内部module分割は維持する。今回の目的はファイル統合や大規模移動ではなく、既に分離済みのRun Coreについて「どこが意味上の入口か」を一意にすることである。

`trade_rl.evaluation` のtop-level public APIは増やさない。`trade_rl.evaluation.runs` がcapability境界を所有し、必要な既存symbolだけを再公開する。

## 背景

現在の `evaluation/runs/` は責務自体は既に分離されている。

```text
runs/
  __init__.py
  artifact.py
  candidate.py
  candidate_suite.py
  config.py
  execute.py
  provenance.py
```

一方、`runs/__init__.py` はownership docstringだけで、他capabilityがRun Coreを使う際は物理module pathを知る必要がある。実際に `evaluation/experiments/` は `runs.config`、`runs.artifact`、`runs.execute`、`runs.provenance` へ直接依存している。また `evaluation/__init__.py` も `runs.candidate_suite` から直接importしている。

この状態では、内部ファイル分割がそのまま外部API構造として露出し、AgentがRun Coreの権威を探索する際に複数moduleを横断する必要がある。

Issue #452 の目的は「すべてを一つのファイルにする」ことではなく、Domain → Capability → internal responsibility の順で一つの意味上の入口を発見できるようにすることである。

## Objective

1. `trade_rl.evaluation.runs` をRun Coreの明示的なcapability facadeにする。
2. Run Core利用者が内部module配置を知らなくても必要なsemantic contractを発見できるようにする。
3. production codeのcapability外direct-submodule importsをarchitecture testで検出する。
4. 既存object identity、artifact schema、digest、execution behavior、CLI behavior、top-level `trade_rl.evaluation` APIを維持する。
5. facadeを第二の実装authorityにせず、各内部moduleを引き続き実装ownerとする。

## Non-goals

- `evaluation/runs/` のファイル統合・rename・package再編はしない。
- `CandidateRunConfig`、`ResolvedCandidateRunSpec`、artifact schema、provenance schemaの意味を変更しない。
- `candidate.py` CLI adapterをcapability public APIへ昇格させない。
- `PROVENANCE_SCHEMA` やprivate helperなど、内部module間だけで必要なsymbolをfacadeへ公開しない。
- `trade_rl.evaluation.__all__` を拡大しない。
- testsから内部moduleをimportすることを全面禁止しない。implementation-level testは内部ownerを直接検査してよい。
- runtimeでprivate importを不可能にする仕組みは追加しない。これはsource architecture contractで管理する。
- deprecated forwarding moduleや互換shimを新設しない。

## API tier

今回の変更では以下のtierを採用する。

```text
Tier 1: trade_rl.evaluation
  既存の広域evaluation public API。今回export増加なし。

Tier 2: trade_rl.evaluation.runs
  Run Core capability public API。外部capabilityはここを入口にする。

Tier 3: trade_rl.evaluation.runs.<module>
  Run Core内部実装owner。runs package内およびimplementation-level testsから利用可能。

Tier 4: persisted candidate-run artifact/schema contracts
  Python import pathとは独立して互換性を維持する。
```

## Tier-2 facade surface

`trade_rl.evaluation.runs.__init__` は以下を公開する。

### Config / resolve

- `CandidateRunConfig`
- `ResolvedCandidateRunSpec`
- `load_candidate_run_config`
- `parse_candidate_run_config`
- `resolve_candidate_run_spec`

### Candidate suite

- `LeanCandidateConfig`
- `run_lean_candidate_suite`

### Execute

- `CandidateRunResult`
- `execute_candidate_run`

### Artifact

- `CandidateRunArtifactIdentity`
- `LoadedCandidateRun`
- `PublishedCandidateRun`
- `inspect_candidate_run_artifact`
- `load_candidate_run_artifact`
- `publish_candidate_run`

### Provenance

- `build_candidate_run_provenance`

`runtime_environment_manifest`、`PROVENANCE_SCHEMA`、CLI `main`、`run_candidate_artifact` は今回のTier-2 surfaceには含めない。

理由:

- runtime manifest/schema constantは現時点でcapability外のsemantic consumerを持たない。
- CLI adapterはfilesystem entrypointであり、Run Core contractそのものではない。
- facadeは「便利そうなsymbol一覧」ではなく、現在のcross-capability boundaryを表す必要最小限のsurfaceとする。

## Dependency rule

production sourceについて、`trade_rl/evaluation/runs/` の外から以下への直接importを禁止する。

```text
trade_rl.evaluation.runs.artifact
trade_rl.evaluation.runs.candidate_suite
trade_rl.evaluation.runs.config
trade_rl.evaluation.runs.execute
trade_rl.evaluation.runs.provenance
```

許可する入口は:

```python
from trade_rl.evaluation.runs import ...
```

`candidate.py` はCLI adapterであり、通常のproduction dependencyとして利用すべきではない。ただし今回のhard gateでは、実際の利用状況とfalse positiveを避けるため、まず上記Run Core owner module群を対象とする。

`trade_rl/evaluation/runs/` 内部ではsubmodule direct importを許可する。package内部責務をfacade経由で循環させない。

architecture testはproduction sourceを対象とし、tests自身のimplementation-level importsは対象外とする。

## Import migration

主なmigration対象は次の通り。

```text
trade_rl/evaluation/__init__.py
trade_rl/evaluation/experiments/analysis.py
trade_rl/evaluation/experiments/delta.py
trade_rl/evaluation/experiments/evidence.py
trade_rl/evaluation/experiments/workflow.py
trade_rl/evaluation/experiments/codec.py
trade_rl/evaluation/experiments/bootstrap/config.py
trade_rl/evaluation/experiments/bootstrap/workflow.py
```

実際の変更前にrepository-wide import inventoryを再取得し、上記以外のproduction callerがあれば同じ契約で分類する。

migrationはimport pathだけを変え、呼出し順、引数、戻り値、exception handlingを変更しない。

## Object identity contract

facadeは既存ownerから直接re-exportする。

例:

```python
from trade_rl.evaluation.runs import CandidateRunConfig
from trade_rl.evaluation.runs.config import CandidateRunConfig as InternalCandidateRunConfig

assert CandidateRunConfig is InternalCandidateRunConfig
```

新しいwrapper class、subclass、adapter function、proxy objectは作らない。

これによりtype identity、dataclass field定義、serialization behavior、exception semanticsを維持する。

## Persisted artifact / provenance contract

今回の変更では以下を不変とする。

- candidate result schema
- candidate artifact identity schema
- provenance schema
- required artifact files
- summary/returns/provenance encoding
- semantic artifact digest
- exact-file digest/size evidence
- implementation manifestの定義

注意: `build_candidate_run_provenance()` はpackage内Python sourceのpathとraw bytesをimplementation identityへ含める。そのため `runs/__init__.py` の変更およびimport source変更は **新しいimplementation digestを生成することが正しい**。

既存Studyやartifactの過去implementation identityを新sourceと同一視してはならない。これはbehavior compatibilityとimplementation identityを区別する既存契約である。

## Architecture test design

既存のAST import toolingを再利用し、別の簡易parserを作らない。

hard gateは少なくとも次を検証する。

1. capability外production moduleが禁止submoduleをabsolute importするとfailする。
2. `from trade_rl.evaluation.runs.config import ...` がfailする。
3. `import trade_rl.evaluation.runs.artifact` がfailする。
4. `from trade_rl.evaluation.runs import CandidateRunConfig` はpassする。
5. `evaluation/runs/` 内部のdirect submodule importはpassする。
6. tests directoryはこのproduction dependency ruleの対象外である。
7. facade exportが既存internal ownerとobject identityを共有する。
8. `trade_rl.evaluation.__all__` の公開symbol集合は変更しない。

architecture gateは「内部moduleが存在してはいけない」とは判定しない。内部moduleは正規ownerとして必要である。

## Error / recovery behavior

この変更はerror taxonomyやrecovery policyを変更しない。

- config validation error
- dataset binding error
- artifact integrity error
- publication failure
- provenance mismatch
- candidate execution error

はいずれも既存ownerで発生し、facadeはcatch/translate/wrapしない。

## Acceptance Criteria

1. `trade_rl.evaluation.runs` からTier-2 surfaceをimportできる。
2. facade symbolは既存submodule symbolと同一objectである。
3. capability外production codeにRun Core owner moduleへのdirect importが残っていない。
4. architecture testが新しいdirect import regressionを検出する。
5. `trade_rl.evaluation.__all__` は変更しない。
6. candidate-run artifact schema、digest、provenance schema、CLI behaviorは変更しない。
7. existing run/experiment/bootstrap testsがGreenである。
8. full repository pytestがGreenである。
9. Ruff / Format / Mypy / architecture-tooling MypyがGreenである。
10. build、tracked-source closure、sdist rebuild、clean installed smoke、package identityがGreenである。
11. exact final HEADのCI成功を確認する。
12. final diffにtemporary workflow、helper、generated artifact、unrelated refactorが残らない。
13. `docs/architecture/package-boundaries.md` に耐久的なTier-2 facade ruleを反映する。
14. 実装完了後、このActive specをcurrent treeから削除する。

## Invariants

- Run Coreのsemantic ownersは既存内部moduleのまま維持する。
- `runs/__init__.py` にbusiness logicを置かない。
- facade importはside effectを追加しない。
- candidate execution結果は変えない。
- persisted artifact formatを変えない。
- historical artifact reader behaviorを変えない。
- provenanceを弱めない。
- top-level evaluation APIを増やさない。
- CLI adapter contractを変えない。

## Failure Modes

### Circular import

facadeが内部moduleをre-exportすることで既存internal dependency graphにcycleを導入する可能性がある。

対策:

- package内部はfacadeを使わずowner moduleを直接importする。
- facadeだけがowner moduleを外向きに束ねる。
- clean import testで確認する。

### Over-export

内部実装detailまで公開するとTier-2 APIが新たなdumping groundになる。

対策:

- actual cross-capability usageを基準にsurfaceを固定する。
- convenienceだけを理由にexportを追加しない。

### False architecture violation

implementation testsやpackage内部importまで禁止すると、内部責務テストや正常な依存が壊れる。

対策:

- rule scopeをproduction sourceかつruns package外に限定する。

### Behavioral drift during import migration

import整理と同時にrefactorを行うとbehavior差分が混入する。

対策:

- migrationはimport statementとfacade/architecture tests/docsに限定する。
- unrelated cleanupは別変更にする。

### Provenance confusion

source path/bytes変更によるimplementation digest変化をregressionと誤認する可能性がある。

対策:

- semantic behavior preservationとimplementation identity preservationを分離する。
- 新sourceは新implementation digestになることを明示する。

## Test Oracle

正しさは「importできた」だけでは判定しない。

観測対象:

- facade/internal symbol object identity
- architecture gateのallowed/forbidden import判定
- production import inventory
- `trade_rl.evaluation.__all__` snapshot
- existing config parse/resolve outputs
- candidate execution outputs
- artifact load/inspect/publish behavior
- artifact semantic/file identity
- provenance schema validation
- CLI smoke
- distribution source closure
- clean-installed package import origin
- exact-head CI result

## Required Test Layers

- Static architecture tests
- Unit tests for facade identity/export surface
- Existing config/execute/artifact/provenance tests
- Existing experiments/bootstrap integration tests
- Existing CLI tests
- Ruff
- Format check
- Mypy production
- Mypy architecture tooling
- Full pytest
- Build
- sdist/direct-wheel/source-rebuilt-wheel closure
- Clean installed smoke
- Package identity
- Final diff/self review
- Falsification review

## Falsification review

実装後、少なくとも以下を意図的に試す。

- `experiments` から `runs.config` を再導入するとarchitecture testが落ちるか。
- facadeのexportがwrapperに置き換わった場合identity testが落ちるか。
- top-level `evaluation.__all__` を誤って増やすとtestが検出するか。
- internal `runs` module同士の正常importを誤検出していないか。
- testsのimplementation-level direct importを誤検出していないか。
- artifact digest/schemaが変わっていないか。
- clean wheel上でもfacade importが成立するか。

## Quality Gate

次をすべて満たすまで完成扱いしない。

- Acceptance Criteriaを満たす。
- architecture hard gateが意図したdirect dependency regressionを検出できる。
- existing functional/integration testsがGreen。
- full testsがGreen。
- static/type/style checksがGreen。
- build/distribution/clean-install checksがGreen。
- final exact HEAD CIがGreen。
- final diffをレビューし、unrelated changesがない。
- falsification reviewを実施する。
- PR comments/review threadsを確認する。
- `package-boundaries.md` に耐久契約を反映する。
- Active specを削除する。
- 未検証事項と残存リスクを最終報告する。

## Docs lifecycle

実装中は本specを `docs/specs/` に保持する。

実装完了時:

1. `docs/architecture/package-boundaries.md` にTier-2 capability facadeとdependency ruleの耐久部分を反映する。
2. `docs/README.md` のActive spec/plan一覧を更新する。
3. 本specを削除する。
4. 完了履歴はGit historyとPRを正本とする。

`docs/history`、`docs/archive`、永続的なcompleted-spec directoryは作らない。
