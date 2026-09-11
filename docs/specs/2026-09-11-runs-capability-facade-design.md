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
- `load_candidate_run_config` をTier-2へ公開しない。現時点では `candidate.py` 内部だけが使用している。
- `PROVENANCE_SCHEMA`、`runtime_environment_manifest`、private helperなど、内部module間だけで必要なsymbolをfacadeへ公開しない。
- `trade_rl.evaluation.__all__` を拡大しない。
- testsから内部moduleをimportすることを全面禁止しない。implementation-level testは内部ownerを直接検査してよい。
- runtimeでprivate importを不可能にする仕組みは追加しない。これはsource architecture contractで管理する。
- deprecated forwarding moduleや互換shimを新設しない。

## API tier

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

`trade_rl.evaluation.runs.__init__` は以下だけを公開する。

### Config / resolve

- `CandidateRunConfig`
- `ResolvedCandidateRunSpec`
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

このsurfaceは、現在のcross-capability利用と公開functionの戻り型に必要なcontractへ限定する。convenienceだけを理由にexportを増やさない。

## Dependency rule

production sourceについて、`trade_rl/evaluation/runs/` の外から以下への**直接綴られたimport**を禁止する。

```text
trade_rl.evaluation.runs.artifact
trade_rl.evaluation.runs.candidate_suite
trade_rl.evaluation.runs.config
trade_rl.evaluation.runs.execute
trade_rl.evaluation.runs.provenance
```

許可する入口は:

```python
from trade_rl.evaluation.runs import CandidateRunConfig
```

次のようなmodule importは、facade経由に見えても内部moduleを直接選択しているため禁止対象とする。

```python
from trade_rl.evaluation.runs import config
```

`candidate.py` はCLI adapterであり通常のcross-capability dependencyとして利用すべきではない。ただし今回のhard gateでは、実際にsemantic ownerとして使われている上記5 moduleを対象とし、CLI module自体の存在は禁止しない。

`trade_rl/evaluation/runs/` 内部ではsubmodule direct importを許可する。package内部責務をfacade経由で循環させない。

architecture ruleはproduction sourceを対象とし、tests自身のimplementation-level importsは対象外とする。

## Architecture tooling design

既存 `tests/architecture/imports.py` の `ImportCollector.collect()` は、static re-exportを追跡してsemantic ownerまで展開する。この性質は既存dependency boundary検査には有用だが、今回の「callerがどのpathを直接綴ったか」というhard gateにはそのまま使えない。

例えば:

```python
from trade_rl.evaluation.runs import CandidateRunConfig
```

を `collect()` すると、facadeからre-exportされた `trade_rl.evaluation.runs.config` まで依存として観測され得る。これをdirect-import禁止に使うと正しいfacade利用を誤検出する。

したがって別parserを新設せず、既存 `ImportCollector` に **direct spelling専用API** を追加する。

候補名:

```python
ImportCollector.collect_direct(path: Path) -> set[str]
```

契約:

- relative importは既存 `_base()` でabsolute moduleへ解決する。
- `import a.b.c` は `a.b.c` を返す。
- `from a.b import Symbol` は `a.b` を返す。
- `from a.b import child_module` で `a.b.child_module` が物理moduleとして存在する場合は、そのchild moduleも返す。
- re-exportされたclass/functionのownerまでは追跡しない。
- `collect()` の既存semantic/transitive behaviorは変更しない。

これにより今回のdirect-spelling gateと既存semantic dependency gateを混同しない。

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

実装前にrepository-wide import inventoryを再取得し、上記以外のproduction callerがあれば同じcontractで分類する。

migrationはimport pathだけを変え、呼出し順、引数、戻り値、exception handlingを変更しない。

`evaluation/runs/` 内部moduleはfacadeを経由せず既存owner moduleを直接importする。

## Object identity contract

facadeは既存ownerから直接re-exportする。

```python
from trade_rl.evaluation.runs import CandidateRunConfig
from trade_rl.evaluation.runs.config import CandidateRunConfig as InternalCandidateRunConfig

assert CandidateRunConfig is InternalCandidateRunConfig
```

wrapper class、subclass、adapter function、proxy objectは作らない。

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

`build_candidate_run_provenance()` はpackage内Python sourceのrelative pathとraw bytesをimplementation identityへ含める。したがって `runs/__init__.py` の変更やimport source変更により **新しいimplementation digestになることは正しい**。

既存Studyやartifactの過去implementation identityを新sourceと同一視してはならない。behavior compatibilityとimplementation identity preservationは別contractである。

## Architecture test design

hard gateは少なくとも次を検証する。

1. capability外production moduleが禁止submoduleをabsolute importするとfailする。
2. `from trade_rl.evaluation.runs.config import ...` がfailする。
3. `import trade_rl.evaluation.runs.artifact` がfailする。
4. `from trade_rl.evaluation.runs import CandidateRunConfig` はpassする。
5. `from trade_rl.evaluation.runs import config` はchild module direct selectionとしてfailする。
6. relative import解決でも同じdirect-spelling contractになる。
7. `evaluation/runs/` 内部のdirect submodule importはpassする。
8. tests directoryはこのproduction dependency ruleの対象外である。
9. facade exportが既存internal ownerとobject identityを共有する。
10. exact `runs.__all__` が設計surfaceと一致する。
11. `trade_rl.evaluation.__all__` の公開symbol集合は変更しない。
12. 既存 `ImportCollector.collect()` のre-export追跡behaviorを壊さない。

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

1. `trade_rl.evaluation.runs` から設計済みTier-2 surfaceをimportできる。
2. facade symbolは既存submodule symbolと同一objectである。
3. capability外production codeにRun Core owner moduleへのdirect importが残っていない。
4. `collect_direct()` がrelative importとactual child-module importを正しく判定し、re-export ownerは追跡しない。
5. architecture testが新しいdirect import regressionを検出する。
6. 既存 `ImportCollector.collect()` のsemantic/transitive behaviorは維持される。
7. `trade_rl.evaluation.__all__` は変更しない。
8. candidate-run artifact schema、digest、provenance schema、CLI behaviorは変更しない。
9. existing run/experiment/bootstrap testsがGreenである。
10. full repository pytestがGreenである。
11. Ruff / Format / Mypy / architecture-tooling MypyがGreenである。
12. build、tracked-source closure、sdist rebuild、clean installed smoke、package identityがGreenである。
13. exact final HEADのCI成功を確認する。
14. final diffにtemporary workflow、helper、generated artifact、unrelated refactorが残らない。
15. `docs/architecture/package-boundaries.md` に耐久的なTier-2 facade ruleを反映する。
16. 実装完了後、このActive specをcurrent treeから削除する。

## Invariants

- Run Coreのsemantic ownersは既存内部moduleのまま維持する。
- `runs/__init__.py` にbusiness logicを置かない。
- facade importはnetwork/filesystem/mutation side effectを追加しない。
- candidate execution結果は変えない。
- persisted artifact formatを変えない。
- historical artifact reader behaviorを変えない。
- provenanceを弱めない。
- top-level evaluation APIを増やさない。
- CLI adapter contractを変えない。
- 既存architecture import collectorのsemantic dependency検出能力を弱めない。

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

- actual cross-capability usageとpublic return typeを基準にsurfaceを固定する。
- convenienceだけを理由にexportを追加しない。

### False architecture violation by re-export expansion

既存 `collect()` はfacadeのre-export先まで追跡するため、direct import gateへ流用すると正しいfacade利用を違反扱いし得る。

対策:

- `collect_direct()` と `collect()` のoracleを分離する。
- direct-spelling regression testを追加する。

### False architecture violation of internal/tests imports

implementation testsやpackage内部importまで禁止すると正常な責務テストが壊れる。

対策:

- rule scopeをproduction sourceかつruns package外に限定する。

### Behavioral drift during import migration

import整理と同時にrefactorを行うとbehavior差分が混入する。

対策:

- migrationはimport statement、facade、architecture tooling/tests、必要docsに限定する。
- unrelated cleanupは別変更にする。

### Provenance confusion

source path/bytes変更によるimplementation digest変化をregressionと誤認する可能性がある。

対策:

- semantic behavior preservationとimplementation identity preservationを分離する。
- 新sourceは新implementation digestになることを明示する。

## Test Oracle

正しさは「importできた」だけでは判定しない。

観測対象:

- exact Tier-2 `__all__`
- facade/internal symbol object identity
- `collect_direct()` のdirect import set
- 既存 `collect()` のsemantic/transitive import set
- architecture gateのallowed/forbidden判定
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

- Unit tests for `ImportCollector.collect_direct()`
- Static architecture tests for Run Core facade boundary
- Unit tests for facade identity/export surface
- Existing architecture import-tooling tests
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

実装後、少なくとも以下を意図的に確認する。

- `experiments` から `runs.config` を再導入するとarchitecture testが落ちるか。
- `from trade_rl.evaluation.runs import config` を違反として検出できるか。
- facadeの正規class/function importをre-export先追跡で誤検出しないか。
- relative importでhard gateを迂回できないか。
- facade exportをwrapperへ置き換える誤実装をidentity testが検出するか。
- top-level `evaluation.__all__` の誤変更を検出するか。
- internal `runs` module同士の正常importを誤検出しないか。
- testsのimplementation-level direct importを誤検出しないか。
- 既存semantic dependency testが `collect_direct()` 追加によって弱くなっていないか。
- artifact schema/semantic digest behaviorが変わっていないか。
- clean wheel上でもfacade importが成立するか。

## Quality Gate

次をすべて満たすまで完成扱いしない。

- Acceptance Criteriaを満たす。
- architecture hard gateが意図したdirect dependency regressionを検出できる。
- 既存semantic import collector behaviorが維持される。
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
