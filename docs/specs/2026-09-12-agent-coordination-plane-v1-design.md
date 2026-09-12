# Agent Coordination Plane v1

Status: Active

Related: #500, #475

## 1. Objective

Trade RL で複数のAI Agentが同時に作業するとき、単なるparallel executionではなく、**task decomposition / ownership / dependency / collision / review / verification / integration / recoveryを一貫したprotocolで管理する**。

v1の目的は、複数Agentを増やしても次を機械的に保てることにある。

- 同じTaskを複数Workerが重複実装しない。
- 独立Taskだけを安全に並列化する。
- fileが異なっても同じsemantic authorityを壊す変更を競合として扱う。
- Task仕様、PR HEAD、current mainが変わったとき古いreview/CI evidenceを再利用しない。
- Agent/sessionが失われてもGitHub上のbranch / PR / status evidenceから作業を復元できる。
- Worker自身の「完了」主張ではなく、independent review / verification / integration evidenceで完了を決める。
- 親Issueはchild Taskの完了数だけでcloseせず、親Acceptance Criteriaを最後に再評価する。

この仕組みを **Agent Coordination Plane** と呼ぶ。

## 2. Relationship to Agent Repository Control Plane

PR #475 の Agent Repository Control Plane は、Repositoryをsource-derivedに理解・検証する層である。

```text
preflight
context
impact
semantic diff
verification routing
Agent Eval
integration-safety inspection
```

Agent Coordination Planeはこれを置き換えない。

責務を3層に分ける。

```text
┌────────────────────────────────────────────┐
│            Coordination Plane              │
│ Task DAG / scheduling / lease / conflict   │
│ heartbeat / handoff / dashboard            │
└─────────────────────┬──────────────────────┘
                      │ consumes facts
┌─────────────────────▼──────────────────────┐
│         Repository Control Plane           │
│ preflight / context / impact / diff        │
│ verify / source-derived inspection         │
└─────────────────────┬──────────────────────┘
                      │ produces evidence
┌─────────────────────▼──────────────────────┐
│            Integration Plane               │
│ review / verify / current-main sync        │
│ merge authorization / post-merge oracle    │
└────────────────────────────────────────────┘
```

Repository Control Planeは「何がRepository上の事実か」を読む。
Coordination Planeは「誰が、何を、いつ実行できるか」を決める。
Integration Planeは「その変更を現在のmainへ統合してよいか」を判断する。

Coordination Planeはdomain authority、research authority、artifact authorityにはならない。

## 3. Dependency on #475

#500は#475のsource inspection primitiveを再実装しない。

実装時点で#475がmainへ統合済みなら、そのcanonical `tools.agent_repo` APIを利用する。

#475が未統合なら、#500 implementationは次のどちらか一方にする。

1. #475を先にcurrent mainへ同期・検証・統合する。
2. #475 HEADを明示的stack baseとするstacked implementationを行い、#475統合後にcurrent mainへrebase/mergeして全evidenceを取り直す。

mainに存在しない#475実装をcopy/pasteして別ownerを作ることは禁止する。

## 4. Non-goals

v1では次を行わない。

- Agent同士の自由なpeer-to-peer negotiationをcoordination authorityにしない。
- Redis、database、distributed lock service等の常設coordination backendを導入しない。
- runtime `trade_rl/**` にAgent orchestration APIを混ぜない。
- checked-in global mutable task ledgerを作らない。
- generated coordination reportをcurrent treeへ蓄積しない。
- Agentが自律的に無制限にTaskを再帰分割する仕組みを作らない。
- automatic merge authorizationを導入しない。
- reviewをWorkerの自己レビューだけで代替しない。
- test GreenだけでTaskをcompleteにしない。
- approximate equalityや「ほぼ同じ」をidentity/evidence contractの代替にしない。
- #475のRepository analysis機能をcoordination都合でproduction packageへ移さない。

## 5. Design principles

### 5.1 Single-writer coordination

Taskのauthoritative coordination stateを書き換えるのはCoordinatorだけとする。

Worker / Reviewer / Verifier / Integratorは事実や成果物を生成するが、Taskのlease、phase、condition、dependency resolutionを直接確定しない。

これにより、二人のAgentが同時に`running`へ変更するようなmulti-writer raceを避ける。

### 5.2 Durable evidence, ephemeral derivation

Durable truthは既存のGitHub/Git primitiveに置く。

```text
Issue contract
Issue status record
branch
commit SHA
PR
review
CI run
artifact
main commit
```

Conflict graph、ready-set、scheduling priority等はこれらから再計算できる一時的derived stateとする。

Coordinator再起動後もGitHub evidenceから復元できなければならない。

### 5.3 Fail closed

Task state、dependency、lease、HEAD binding等が欠けている場合、「たぶん安全」と推測してdispatch/mergeしない。

不明なTaskはblocked/stale相当に扱い、必要なfactを再取得する。

### 5.4 Evidence is exact-snapshot bound

Review、CI、verificationは「Task一般」に対してvalidではない。

最低限、次へ束縛する。

```text
task_id
task_revision
base_sha
head_sha
evidence kind / identifier
```

どれかが変われば必要なevidenceをstaleにする。

## 6. Task granularity

GitHub Taskへ分ける単位は、次の3条件を満たすものとする。

1. 別Agentへhandoff可能。
2. 独立したAcceptance Criteria / Test Oracleを持てる。
3. 独立Reviewerがapprove/rejectできる。

単純なhelper rename、1 import修正、同一TDD loop内のRED/GREEN stepまではTask Issueへ分割しない。

原則として、Taskは「fresh reviewer gateに値する最小単位」とする。

## 7. Task Packet v1

Task contractはversionedなTask Packetとして表現する。

Normative logical schema:

```yaml
schema: agent_task_v1

task_id: T500-03
task_revision: 1
parent_issue: 500

title: Implement lease and ownership state machine

objective:
  Prevent duplicate task ownership and make abandoned work recoverable.

non_goals:
  - no distributed lock server
  - no automatic merge

dependencies:
  hard: [T500-01]
  evidence: []
  integration: []

write_scope:
  allow:
    - tools/agent_repo/**
    - tests/agent_repo/**
  deny:
    - trade_rl/**

resource_keys:
  - authority:agent-coordination-task-state
  - workflow:agent-task-claim

acceptance_criteria:
  - duplicate claim is rejected
  - expired lease is not reassigned before reconciliation
  - stale epoch cannot mutate ownership

test_oracle:
  - deterministic state-machine transitions
  - race simulation with stale epochs

base_sha: <exact commit>
risk: medium

deliverable:
  type: pull_request
```

これはlogical schemaであり、v1実装でYAML fileをRepositoryへ保存することを要求しない。Issue bodyやGitHub structured fieldsから同じcontractを再構築してよい。

## 8. Task revision

`task_revision`はTask contract変更時に単調増加するintegerとする。

Revisionを上げる変更の例:

- objective変更;
- acceptance criteria変更;
- write scope変更;
- semantic resource変更;
- dependency変更;
- test oracle変更;
- non-goalを破るようなscope変更。

表現上の誤字修正などsemantic contractに影響しない編集はrevisionを上げなくてよい。

Workerがrevision 2で作業中にTaskがrevision 3へ変わった場合、そのlease/work/review evidenceは自動的にstaleとなる。

Coordinatorは旧revisionの成果物を破棄せず、再利用可能性を新revision against diffで評価するが、旧evidenceをそのままvalidにはしない。

## 9. Phase and Condition

Task状態は1個のenumに押し込めず、`phase`と`condition`の直積で表す。

### 9.1 Phase

```text
planned
ready
executing
review
verification
integration
complete
```

意味:

- `planned`: Taskは定義済みだがhard dependency未解決。
- `ready`: dependency/conflict上、着手可能。
- `executing`: active leaseを持つWorkerが作業中。
- `review`: reviewable exact HEADがあり、独立review中。
- `verification`: accepted review HEADに対してtest/falsification/quality evidenceを収集中。
- `integration`: current-main containment、competing PR、merge authorization、post-merge準備を評価中。
- `complete`: integrationとpost-merge completion oracleを満たした。

### 9.2 Condition

```text
healthy
blocked
stale
failed
conflicted
```

意味:

- `healthy`: phase進行を妨げる既知条件なし。
- `blocked`: 外部依存、permission、upstream result等を待つ。
- `stale`: task revision、HEAD、base/main等の移動により既存evidenceが古い。
- `failed`: phase固有oracleが失敗し、修正が必要。
- `conflicted`: active Taskとのhard conflictまたはintegration conflictがある。

例:

```text
review / stale
executing / blocked
verification / failed
integration / stale
```

が正当な状態になる。

## 10. State transition rules

通常経路:

```text
planned
  ↓ dependencies satisfied
ready
  ↓ lease granted
executing
  ↓ reviewable checkpoint submitted
review
  ↓ independent review accepted
verification
  ↓ acceptance + quality evidence accepted
integration
  ↓ merge + post-merge oracle
complete
```

代表的な逆遷移:

```text
review       → executing     changes requested
verification → executing     implementation defect
integration  → verification  main/head movement requiring re-evidence
integration  → executing     merge conflict requiring semantic change
```

Conditionはphaseと独立して変わる。

`complete`だけはterminalであり、complete後の新問題は元Taskを書き換えず新Issue/Taskとして扱う。ただし誤closeだったことが明らかな場合はGitHub Issueのreopen semanticsを使う。

## 11. Lease protocol

### 11.1 Purpose

Leaseは同一Taskへの二重実装を防止し、Agent/session消失時に安全にownershipを回収するために使う。

### 11.2 Lease record

最低限:

```yaml
owner: agent-A
epoch: 3
task_revision: 2
base_sha: abc123
branch: fix/T500-03-lease-state
head_sha: def456
last_heartbeat_at: <timestamp>
expires_at: <timestamp>
```

### 11.3 Single writer

Lease recordを更新するのはCoordinatorだけ。

WorkerはTask Issueのlease fieldを直接書き換えない。

### 11.4 Epoch

Leaseを新しいWorkerへgrantするたびに`epoch`を増加させる。

古いWorkerが後から復帰しても、旧epochのresultをcurrent ownershipとしてacceptしない。

### 11.5 Expiry is not immediate reassignment

Lease expiryだけでは別Workerへ即grantしない。

必ずreconciliationを行う。

```text
branch existence
remote HEAD
open PR
CI runs
review comments
artifacts
unmerged commits
```

を確認し、旧Workerのside effectをinventoryする。

その後にのみ、resume / salvage / supersede / reassignを決める。

## 12. Heartbeat

Heartbeatの目的はAgentが生存していることだけでなく、**そのleaseがまだ同じTask revision / base / branchに対して意味を持つか**を確認することにある。

Heartbeat更新はCoordinatorが観測した次のeventから行える。

- Workerからのexplicit progress event;
- branch HEAD advance;
- Draft PR creation/update;
- CI start/finish;
- reviewable checkpoint publication。

単なる時間経過だけを進捗とみなさない。

v1では固定heartbeat intervalをrepository contractにしない。runtime/environmentごとに設定可能とする。ただしlease expiry semantics自体はdeterministic test対象とする。

## 13. Work isolation

Default invariant:

```text
1 Task
= 1 active owner
= 1 isolated writable worktree
= 1 task branch
= normally 1 PR
```

Worker同士が同じwritable working treeを共有してはならない。

Read-only checkout/cacheの共有は、その共有が変更stateを持たず結果へ影響しない場合のみ許可する。

複数Taskを1PRへまとめるのは、Task間が実際には独立reviewできず同一atomic changeであるとCoordinatorが再分類した場合だけとする。

## 14. Dependency graph

Task dependencyには3種類を持つ。

### Hard dependency

上流Taskがcompleteまたは指定phaseを満たさない限り下流Taskを実行できない。

### Evidence dependency

実装は並列可能だが、上流evidenceなしでは下流Taskをverification/completeにできない。

### Integration dependency

実装・reviewは独立だがmerge順序がある。

DAGにcycleがある場合、Coordinatorはfail closedしdispatchしない。

## 15. Resource Keys and conflict graph

File overlapだけではsemantic collisionを検出できないため、TaskはResource Keyを持つ。

v1 key namespace:

```text
file:<repository-path-or-prefix>
authority:<semantic-owner>
identity:<content-addressed-identity>
schema:<schema-or-contract>
workflow:<workflow-or-integration-path>
artifact:<artifact-family>
```

例:

```text
authority:market-feature-computation
identity:market-dataset
schema:candidate-run-v2
workflow:canonical-m2-bootstrap
artifact:evidence-set
```

Resource Keyは可能な限りRepository Control Planeのsource-derived context/impactから導出する。導出できないsemantic concernだけTask contractで明示する。

## 16. Conflict classification

Task pairは`hard / soft / none`のいずれかに分類する。

### Hard

同時実行自体を禁止する。

例:

- 同一file regionを両方がwriteする;
- 同一schema version authorityを別方針で変更する;
- 同一content identity algorithmを同時に変更する。

### Soft

並列実装は可能だが、一方のintegration後に他方のreview/verificationを再実行する。

例:

- 同一package boundaryの別file変更;
- 同一public facadeへ別exportを追加;
- 共通verification workflowを別々に触る。

### None

独立して実装・review・integration可能。

Coordinatorは「Agentが空いているから」ではなく、次でready-setを計算する。

```text
ready = dependency-satisfied
      ∩ non-hard-conflicting
      ∩ unleased
      ∩ valid-task-revision
```

## 17. Scheduler

Coordinator refresh cycleは概念的に次とする。

```text
1. refresh current main / open PR / task contracts / leases / evidence
2. invalidate stale task/review/CI/integration evidence
3. reconcile expired leases
4. resolve dependency states
5. derive/update resource keys
6. build conflict graph
7. compute ready-set
8. rank ready tasks
9. grant leases
10. dispatch Worker / Reviewer / Verifier as appropriate
11. publish dashboard projection
```

v1 scheduling priorityは複雑なAI scoringにしない。

推奨deterministic priority:

1. blocking upstream Tasks;
2. critical-path Tasks;
3. Tasks that unblock the largest number of descendants;
4. lower-risk independent Tasks;
5. stable tie-break by task id。

## 18. Roles

### Coordinator

- Task decomposition;
- Task revision management;
- dependency graph;
- Resource Key / conflict graph;
- phase/condition authority;
- lease ownership;
- stale evidence invalidation;
- scheduling;
- status/dashboard projection。

Coordinatorは原則production implementationを行わない。

### Worker

- Task Packet理解;
- isolated worktree/branch;
- TDD;
- focused implementation;
- meaningful checkpoints;
- self-review;
- Draft PR / reviewable evidence package。

Workerは自分でTaskを`complete`にしない。

### Reviewer

Workerとは独立したcontextで、最低限次だけをauthorityとして読む。

```text
Task Packet
current source
PR diff
relevant tests
CI/evidence
```

Workerの説明やreasoningを正解として引き継がない。

Verdict:

```text
PASS
CHANGES_REQUIRED
BLOCKING_FINDING
```

Review verdictはexact HEADへbindする。

### Verifier

- Acceptance Criteria;
- Invariants;
- Failure Modes;
- Test Oracle;
- full quality evidence;
- falsification;
- unverified items / residual risk

をexact HEADに対して確認する。

### Integrator

- current main SHA;
- PR base/head;
- tested HEAD contains current main;
- competing PR / semantic conflicts;
- exact-head review;
- exact-head CI;
- merge authorization;
- expected-head merge CAS;
- post-merge main exact-head verification

を担当する。

Small deploymentではCoordinatorとIntegratorを同一Agentにしてよい。ただしmaterial changeでWorkerとReviewerを同一Agentにしないことを推奨する。

## 19. Worker completion package

`executing → review`の前にWorkerは最低限次を満たす。

- current Task revisionを再確認;
- allowed/denied write scopeを確認;
- reproducing REDを観測;
- minimal GREENを観測;
- relevant refactor後もGREEN;
- final diff self-review;
- debug/tmp/generated residueなし;
- branchへcommit/push済み;
- Draft PRまたは同等のimmutable review targetあり;
- known limitation / unverified item明記。

Workerの自然言語「できた」はstate transition oracleにしない。

## 20. Checkpoint policy

Agent/session lossに備え、meaningful state boundaryでrecoverable checkpointを作る。

例:

```text
RED reproduced
root cause evidence captured
minimal GREEN reached
refactor complete
external long CI before wait
reviewable HEAD ready
```

機械的な時間間隔だけでcheckpointを作らない。

長時間external jobを開始する前は、原則としてそのjobが参照するcommitをremoteへpushしてimmutable SHAを確定する。

## 21. Handoff packet

Worker変更時はformal handoffを作る。

Logical schema:

```yaml
task_id: T500-03
task_revision: 2
lease_epoch: 4
last_good_head: abc123

verified:
  - RED reproduction confirmed
  - target unit tests green

pending:
  - full CI
  - independent review

known_failures:
  - none

do_not_repeat:
  - root-cause diagnostic already completed

evidence:
  - pr: 512
  - ci_run: 123456
```

Handoffはprivate chat historyを必要とせず、branch/PR/evidenceだけで後任が再開できることをoracleとする。

## 22. Evidence model

Evidence recordのlogical minimum:

```yaml
evidence_kind: review | test | ci | artifact | integration | post_merge
task_id: T500-03
task_revision: 2
base_sha: aaa111
head_sha: bbb222
identifier: <stable external identifier>
result: pass | fail
```

Task revisionまたはHEADが変わればreview/verification evidenceをstaleにする。

Current mainが変わった場合、implementation reviewまで常にstaleにする必要はないが、**integration evidenceは必ずstale**にする。

v1は安全側に倒し、semantic merge/rebaseでHEADが変わった場合はreviewとverificationも再取得する。

## 23. Current-main advancement

次をintegration invariantとする。

```text
tested PR head contains then-current main
+
that exact PR head has required review/CI evidence
```

PR HEADがGreenでも、その後mainが進んだ場合:

```text
phase=integration
condition=stale
reason=current_main_advanced
```

とする。

Current mainをnon-forceで取り込んだ新HEADを作り、必要なexact-head evidenceを再取得する。

古いGreenを再利用しない。

## 24. Integration as a first-class task

MergeはCoordinatorの隠れた最後の操作ではなく、独立したIntegration phase/taskとして扱う。

Integration oracle:

```text
current main resolved
no unresolved hard conflict
PR head contains current main
Task revision current
review exact HEAD PASS
verification exact HEAD PASS
required CI exact HEAD Green
final diff residue clean
merge authorization present
merge uses expected HEAD
post-merge main HEAD observed
post-merge required oracle Green
```

PR merge成功だけでは`complete`にしない。

## 25. Parent issue completion

Parent Issueはchild Taskのstatusを単純ANDしない。

完了条件:

1. required child Tasks complete;
2. parent-level Acceptance Criteria再評価;
3. cross-task invariants再評価;
4. integration orderingの結果を含むfinal evidence確認;
5. residual risks / unverified items明記。

したがって:

```text
all children complete
≠ automatically parent complete
```

である。

## 26. GitHub durable representation

推奨projection:

| Information | Durable location |
|---|---|
| Task contract | Issue body |
| parent/dependency | Issue body / structured references |
| phase | label or machine-readable status record |
| condition | label or machine-readable status record |
| Worker assignment | assignee + status record |
| lease/revision/base/head | Coordinator-owned status record |
| implementation | branch / PR |
| review | PR review |
| verification | CI / artifact / PR status evidence |
| completion | Issue closed/completed after post-merge oracle |

Task IssueにはCoordinator-owned status commentを原則1つ持ち、更新して使う。

例:

```text
<!-- agent-coordination:T500-03 -->
Task-Revision: 2
Phase: executing
Condition: healthy
Owner: agent-A
Lease-Epoch: 4
Base: aaa111
Branch: fix/T500-03
Head: bbb222
Checkpoint: RED confirmed
Blocking: none
```

Worker/Reviewerがこのcommentを直接編集しない。

## 27. Dashboard

Human-facing dashboardはderived viewでありauthorityではない。

最低限:

```text
current main
active parent issues
child progress
phase / condition
owner
PR
blocking dependency
stale reason
```

例:

```text
#500 Agent Coordination Plane
2 / 6 complete

T01 state model       complete
T02 conflict model    complete
T03 lease protocol    executing / healthy   Agent-A  PR #...
T04 GitHub projection planned               waits T03
T05 review/falsify    planned               waits T03,T04
T06 integration       planned               waits T05
```

## 28. Recovery scenarios

### Agent lost before push

Recoverable evidenceがない。Taskをfailed/blocked相当にし、旧local stateを成功扱いしない。新Workerはlast durable checkpointから再実行する。

### Agent lost after pushed checkpoint

Coordinatorはremote branch/PR/CIをreconcileし、同じbranchをresumeするかnew branchへsalvageする。

### Stale Worker pushes after reassignment

旧lease epoch由来のpushをcurrent Task completion evidenceとしてacceptしない。必要なら新owner branchへexplicit cherry-pick/reviewする。

### Coordinator restart

GitHub Issues/PRs/branches/CI/status recordsからTask graph/leases/evidenceを再構築する。Coordinator memoryのみの情報が必要なら設計失敗とする。

## 29. Security and trust boundaries

Coordination PlaneはAgent generated textを無条件に信用しない。

- Worker summaryよりGit SHA/diff/test outputを優先;
- Reviewer summaryよりreview target HEADを確認;
- CI proseよりrun/head bindingを確認;
- Issue labelだけでmerge可否を決めない;
- malformed machine-readable statusはfail closed;
- write scope違反をreview signalではなくTask contract violationとして扱う;
- secrets/token値をTask Packet/status recordへ保存しない。

## 30. Failure modes

### FM-1 Duplicate claim

二つのWorkerが同じTaskを実装する。

Mitigation: Coordinator single writer + lease epoch + one active owner invariant。

### FM-2 Stale lease holder side effects

Lease expiry後に旧Workerがpushする。

Mitigation: epoch binding + reassignment前reconciliation + old epoch evidence rejection。

### FM-3 Stale task specification

Workerが古いAcceptance Criteriaで実装を続ける。

Mitigation: task revision + heartbeat/review transitionでrevision再確認。

### FM-4 File-disjoint semantic collision

別fileなのでparallel扱いしたが同一authority/identityを変更する。

Mitigation: Resource Keys + Control Plane impact + hard/soft conflict graph。

### FM-5 Review stale after new commit

Approve後にWorkerが追加commitする。

Mitigation: exact HEAD evidence binding。HEAD movementでreview invalidation。

### FM-6 Green PR stale against main

mainが先に進む。

Mitigation: integration stale + current-main containment + reverify。

### FM-7 Coordinator becomes second domain authority

Task metadataがsource/schema/research contractを上書きする。

Mitigation: coordination metadataはexecution authorityのみ。domain truthはexisting source/docs/artifacts。

### FM-8 Global task ledger merge conflicts

全Agentが同一checked-in JSON/YAMLを編集する。

Mitigation: mutable global ledgerをGit treeへ置かない。

### FM-9 Agent death loses work

private local changesしかない。

Mitigation: meaningful pushed checkpoints。未push stateはverified progressとして扱わない。

### FM-10 Child completion masks parent failure

すべてのsubtaskがGreenでもcross-task invariantが壊れる。

Mitigation: parent final acceptance re-evaluation。

## 31. Test strategy

### Unit

Pure state logic:

- phase/condition transition;
- task revision invalidation;
- dependency DAG;
- cycle rejection;
- resource key normalization;
- hard/soft/none conflict classification;
- lease epoch/CAS semantics;
- evidence staleness rules;
- scheduler ready-set。

### Property / model-based

State machineについて:

- at most one active lease per Task;
- epoch monotonicity;
- complete Task has required integration/post-merge evidence;
- stale evidence cannot become valid without new matching evidence;
- hard-conflicting Tasks cannot both hold active executing leases。

### Integration

Temporary Git repository/worktreeで:

- isolated worktrees;
- branch ownership;
- HEAD movement;
- main movement;
- merge-base/current-main containment;
- checkpoint recovery;
- handoff from remote commit。

### GitHub contract

Live GitHub operationを必要とする部分は可能な限りread-only/fake boundaryでunit化し、actual API integrationはdedicated extended testで確認する。

少なくとも:

- Issue contract/status parse;
- PR exact head;
- CI head binding;
- review head binding;
- status comment update/reconstruction;
- stale-main detection

を検証する。

### Falsification

意図的に次を作り、検出できることを確認する。

- duplicate lease;
- stale epoch mutation;
- cyclic dependencies;
- semantic conflict without file overlap;
- approved-old-HEAD + new commit;
- Green-old-main + advanced main;
- malformed status record;
- parent AC failure with all children complete。

## 32. Implementation boundary

#475がcanonicalになった場合、Coordination Planeは同じrepository-tooling ownerへ追加するのが第一候補である。

想定責務分割:

```text
tools/agent_repo/
  coordination/
    model.py          # immutable Task/phase/condition/evidence types
    state.py          # transition and stale-invalidation rules
    dependencies.py   # DAG and readiness
    conflicts.py      # Resource Keys and conflict graph
    leases.py         # lease/epoch/reconciliation model
    scheduler.py      # deterministic ready-set / allocation
    github_state.py   # durable GitHub projection boundary
    dashboard.py      # derived human-readable view
```

ファイル名は実装時の既存#475構造に合わせて調整してよいが、責務境界は維持する。巨大な`coordination.py`一枚へ集約しない。

`trade_rl/**`からこのtoolingへ依存してはならない。

## 33. CLI direction

#475のCLI ownerを継続できる場合、別CLI binaryを増やすより既存entrypointへnamespaceを追加する。

Conceptual commands:

```text
python -m tools.agent_repo task show <id>
python -m tools.agent_repo task graph <parent>
python -m tools.agent_repo task ready <parent>
python -m tools.agent_repo task claim <id> --owner <agent>
python -m tools.agent_repo task reconcile <id>
python -m tools.agent_repo task submit <id>
python -m tools.agent_repo task dashboard [<parent>]
```

`claim`等のwrite operationはv1実装でGitHub write surfaceを持つか、Coordinator library APIだけにするかをimplementation plan時に現在利用可能なconnector/permission boundaryから決める。ただしstate machine contractは同一とする。

## 34. Dogfood plan

Coordination Plane自身のcontract testがGreenになる前に、本番multi-Agent orchestrationのauthorityとして使わない。

最初のdogfood候補は、semantic scopeが明確に分離できるTask pairとする。

例:

```text
#494 feature numerical portability
#476 PPO aggregation contract
```

ただしdogfood開始時点の実際のIssue/PR状態を再確認し、既に完了・競合・仕様変更していれば別Taskを選ぶ。

Dogfoodで確認するのは「速く終わったか」だけではない。

- duplicate work zero;
- scope violation zero;
- stale evidenceを再利用しなかった;
- handoff可能だった;
- parallel化したTaskがsemantic conflictを起こさなかった;
- integration後のremaining Task evidenceが正しくstale化された。

## 35. Acceptance Criteria

1. Independent Taskを2つ以上、shared writable worktreeなしでconcurrentにleaseできる。
2. 同一Taskへactive leaseを二重grantできない。
3. Lease expiry後はside-effect reconciliationなしにreassignできない。
4. Task revision変更で旧lease/review/verification evidenceがdeterministically staleになる。
5. File overlapがなくてもsemantic Resource Key collisionをhard/soft conflictとして表現できる。
6. Hard-conflicting Tasksをschedulerが同時executingへしない。
7. PR HEAD変更後、旧review/CI evidenceをvalidとして扱わない。
8. Main advancement後、integration evidenceがstaleになりcurrent-main-inclusive HEADの再検証を要求する。
9. Agent/session loss後、durable checkpointからprivate conversationなしでhandoff/recoveryできる。
10. WorkerはTaskをcompleteにできず、review/verification/integration protocolを通る。
11. All child completeだけではparent closeせず、parent Acceptance Criteriaを再評価する。
12. Coordinator restart後、GitHub durable evidenceからcoordination stateを復元できる。
13. Repository Control PlaneとCoordination Planeのauthorityが分離される。
14. `trade_rl/**` runtime/public API/artifact semanticsをcoordination都合で変更しない。
15. Existing final quality gatesを弱めない。

## 36. Quality Gate

Implementationをcompleteと判断するには:

- Task-state unit tests Green;
- lease/race/property tests Green;
- dependency/conflict/scheduler tests Green;
- temp-Git/worktree integration tests Green;
- GitHub projection parsing/reconstruction tests Green;
- stale-HEAD/stale-main falsification Green;
- parent completion falsification Green;
- Ruff / Format / Mypy;
- repository-tooling type checks;
- full pytest;
- build/distribution closure;
- clean installed production package checks;
- final diff review;
- no generated coordination state/debug artifact committed;
- exact final HEAD CI Green;
- current-main containment checked before merge;
- post-merge main oracle checked;
- dogfood result reviewed with residual risks documented。

Tests Greenだけではcompletionとしない。

## 37. Implementation sequencing

### Phase 0 — dependency normalization

#475のcurrent statusを確認し、Control Plane primitiveをcanonicalにする順序を確定する。

### Phase 1 — pure coordination model

Task Packet representation、phase/condition、revision/evidence invalidation、dependency DAG、Resource Keysをnetwork-free pure logicとして実装する。

### Phase 2 — lease and scheduler

Lease epoch、reconciliation requirement、ready-set、hard/soft conflict schedulingを実装する。

### Phase 3 — Git/GitHub projection

Task Issue/status record、branch/PR/head/CI evidenceからstateをreconstructするboundaryを実装する。

### Phase 4 — role workflow

Worker submission、Reviewer verdict、Verifier evidence、Integration gate、handoff/recoveryを接続する。

### Phase 5 — dashboard and operator UX

Humanがactive/blocked/stale/owner/dependencyを一目で把握できるderived dashboardを追加する。

### Phase 6 — adversarial verification

race/stale/recovery/main-movement/parent-acceptanceを壊すfalsificationを行う。

### Phase 7 — dogfood

独立した実Taskで2+ Workerを並列運用し、coordination contractを実環境で検証する。

## 38. Completion lifecycle

このspecはimplementation中だけ`docs/specs/`に保持する。

実装完了後:

- durable responsibility/invariantを`docs/AGENTS.md`および必要なarchitecture docsへ移す;
- completed implementation planとこのActive specをcurrent treeから削除する;
- historical designはGit history / PR / Issueを参照する。

これはRepositoryのcurrent-only docs policyに従う。
