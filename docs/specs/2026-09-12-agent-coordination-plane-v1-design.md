# Agent Coordination Plane v1

Status: Active

Related: #500, #475

## 1. Objective

Trade RL で複数のAI Agentが同時に作業するとき、単なるparallel executionではなく、**task decomposition / ownership / dependency / collision / review / verification / integration / recoveryを一貫したprotocolで管理する**。

v1は次を機械的に保つためのcoordination layerである。

- 同一Taskを複数Workerが重複実装しない。
- 独立Taskだけを安全に並列化する。
- fileが異なっても同じsemantic authority / identity / side effectを変更するTaskを競合として扱う。
- Task contract、PR HEAD、current mainの変化で古いevidenceを再利用しない。
- Agent/session消失後もGitHub上のdurable evidenceから復元できる。
- read-only investigation/reviewは不要なbranchを作らず軽量に並列化する。
- write Taskはlease epochごとにwritable worktree/branchを隔離する。
- Worker自身の「完了」ではなくindependent review / verification / integration evidenceで完了を決める。
- child Task完了数だけでparent Issueをcloseせず、parent Acceptance Criteriaを最後に再評価する。

この仕組みを **Agent Coordination Plane** と呼ぶ。

## 2. Three-plane architecture

PR #475 の Agent Repository Control Plane は、Repositoryをsource-derivedに理解・検証する層である。

```text
┌────────────────────────────────────────────┐
│            Coordination Plane              │
│ Task DAG / lease / scheduling / conflict   │
│ capability / heartbeat / handoff / status  │
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
│ review / verification / main sync          │
│ merge authorization / post-merge oracle    │
└────────────────────────────────────────────┘
```

Repository Control Planeは「Repository上の事実」を読む。
Coordination Planeは「誰が、何を、どの権限で、いつ実行できるか」を決める。
Integration Planeは「現在のmainへ統合してよいか」を判断する。

Coordination Planeはdomain / research / artifact authorityにはならない。

## 3. Dependency on #475

#500は#475のsource inspection primitiveを再実装しない。

実装時に#475がmainへ統合済みならcanonical `tools.agent_repo` APIを利用する。未統合なら次のどちらかにする。

1. #475をcurrent mainへ同期・検証・統合してから#500を実装する。
2. #475 HEADを明示的stack baseにし、#475統合後にcurrent mainへ同期して全integration evidenceを取り直す。

mainに存在しない#475実装をcopy/pasteして別ownerを作ることは禁止する。

## 4. Non-goals

v1では次を行わない。

- Agent同士のfree-form peer-to-peer negotiationをcoordination authorityにしない。
- Redis / DB / distributed lock serviceを導入しない。
- runtime `trade_rl/**` にorchestration APIを混ぜない。
- checked-in global mutable task ledgerを作らない。
- generated coordination reportをcurrent treeへ蓄積しない。
- unlimited autonomous recursive task decompositionを許可しない。
- automatic merge authorizationを導入しない。
- Worker自己レビューだけでindependent reviewを代替しない。
- Tests Greenだけでcompleteにしない。
- approximate equalityをidentity/evidence contractの代替にしない。
- 複数Coordinatorのhigh-availability / split-brain consensusをv1で解決しない。

v1のoperational invariantは**同時にactive Coordinatorは1つだけ**である。Coordinator takeoverは旧Coordinator sessionが停止済みであることを確認してから行う。

## 5. Authority and persistence model

### 5.1 Single-writer coordination

Taskのauthoritative coordination stateを書き換えるのはCoordinatorだけ。

Worker / Reviewer / Verifier / Integratorは成果物とevidenceを生成するが、lease、phase、condition、dependency resolutionを直接確定しない。

### 5.2 Durable evidence

Coordinator再起動後に必要な状態は次のGit/GitHub evidenceから復元可能でなければならない。

```text
Issue Task contract
Coordinator-owned Task status record
branch / commit SHA
PR
review
CI run
artifact
main commit
```

Conflict graph、ready-set、priority等はdurable evidenceから再計算するderived stateとする。

### 5.3 Labels and dashboards are projections

GitHub labelsやdashboard表示はoperator UX用projectionであり、state authorityではない。Issue Task contract、Coordinator status record、Git/PR/CI evidenceから再構成する。

### 5.4 Fail closed

Task contract、lease、dependency、contract digest、HEAD binding等が欠ける・壊れる場合、推測でdispatch/mergeしない。

## 6. Task granularity

Task Issueへ分離する単位は次をすべて満たすものとする。

1. 別Agentへhandoff可能。
2. 独立Acceptance Criteria / Test Oracleを持てる。
3. fresh Reviewerが単独でapprove/rejectできる。

helper renameや同一TDD loop中のRED/GREEN stepはTaskへ分割しない。

## 7. Task Packet v1

Logical schema:

```yaml
schema: agent_task_v1

task_id: T500-03
task_revision: 1
parent_issue: 500

title: Implement lease and ownership state machine
objective:
  Prevent duplicate task ownership and make abandoned work recoverable.

execution_mode: write

non_goals:
  - no distributed lock server
  - no automatic merge

dependencies:
  hard:
    - task_id: T500-01
      requires_phase: complete
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
  - side-effect:github-task-status

capabilities:
  - repository_read
  - lease_branch_write
  - draft_pr_write

acceptance_criteria:
  - duplicate claim is rejected
  - expired lease is not reassigned before reconciliation
  - stale epoch cannot become current ownership

test_oracle:
  - deterministic state-machine transitions
  - stale-epoch race simulation

base_sha: <exact commit>
risk: medium

deliverable:
  type: pull_request
```

これはlogical schemaであり、checked-in YAML task ledgerを要求しない。Issue body等から同一contractを再構築してよい。

## 8. Execution mode

Taskは`read_only`または`write`を明示する。

### read_only

調査、review、evidence verification等。

- immutable checkout/snapshotを共有してよい。
- writable worktree/branchを原則要求しない。
- source mutationを行わない。
- GitHub review/comment等、Task Packetで明示したside effectだけ許可する。

### write

source/docs/config/Git refs等を変更するTask。

- active lease必須。
- lease epoch固有のwritable worktree/branch必須。
- write scope / Resource Key / capability contract必須。

これにより「parallel investigationは軽く、implementationは強く隔離」を両立する。

## 9. Task contract identity

`task_revision`だけではmanual editのrevision-bump漏れを検知できない。そのため**canonical Task Contract digest**を導入する。

Canonical payloadはruntime statusやpresentation-only fieldを除外し、UTF-8 JSON / sorted keys / stable separatorsで一意にserializeしてSHA-256を取る。

```text
task_contract_digest = sha256(canonical_task_packet)
```

Lease、review、verification、integration evidenceは最低限次へbindする。

```text
task_id
task_revision
task_contract_digest
base_sha
head_sha when applicable
```

Issue本文が変更されrevisionが同じでもdigestが変われば`stale`としてfail closedする。semantic変更ならrevisionを増やし、presentation-only変更ならcanonical payloadを変えない。

## 10. Task revision

`task_revision`はsemantic contract変更時に単調増加するintegerとする。

Revisionを上げる例:

- objective / non-goal変更;
- execution mode変更;
- dependency変更;
- write scope / capability変更;
- Resource Key変更;
- Acceptance Criteria / Test Oracle変更。

Workerがrevision 2で作業中にrevision 3へ変われば、旧lease/review/verificationはstaleとなる。

## 11. Phase + Condition state model

Task状態は単一enumではなく2軸で表す。

### Phase

```text
planned
ready
executing
review
verification
integration
complete
```

### Condition

```text
healthy
blocked
stale
failed
conflicted
```

例:

```text
review / stale
executing / blocked
verification / failed
integration / stale
```

通常遷移:

```text
planned
  ↓ dependencies satisfied
ready
  ↓ dispatch / lease grant when write
executing
  ↓ reviewable result submitted
review
  ↓ independent review PASS
verification
  ↓ acceptance + quality evidence PASS
integration
  ↓ merge + post-merge oracle when applicable
complete
```

代表的な逆遷移:

```text
review       → executing     CHANGES_REQUIRED
verification → executing     implementation defect
integration  → verification  re-evidence only
integration  → executing     semantic merge conflict
```

Read-only Taskでmergeが存在しない場合、verificationからcompleteへ進める。ただしTask Packetが要求するdurable evidenceを満たすこと。

`complete`はterminal。後発問題は原則new Task/Issueとして扱う。

## 12. Lease and fencing protocol

Leaseは`write` Taskのduplicate work防止とabandoned work回収のためのownership fenceである。

### Lease record

```yaml
owner: agent-A
epoch: 3
task_revision: 2
task_contract_digest: <sha256>
base_sha: abc123
lease_branch: agent/T500-03/e0003-agent-a
head_sha: def456
last_heartbeat_at: <timestamp>
expires_at: <timestamp>
```

Lease recordを更新するのはCoordinatorだけ。

### Epoch fencing

新しいownership grantごとに`epoch`を増加する。旧epoch resultをcurrent ownershipとしてacceptしない。

### Branch per epoch

**Reassignment後に同じwritable branchを再利用しない。**

```text
agent/T500-03/e0003-agent-a
agent/T500-03/e0004-agent-b
```

旧Workerがold branchへlate pushしてもnew Owner branchを直接汚さない。旧成果物を再利用する場合はnew epoch branchへexplicit importし、改めてreviewする。

Same-owner resumeでownership/epochが変わらない場合のみ同一lease branchを継続できる。

### Expiry is not reassignment

Lease expiry後は必ずreconciliationする。

```text
old epoch branch HEAD
open PR
CI runs
reviews/comments
artifacts
unmerged commits
```

をinventoryし、`resume / salvage / supersede / reassign`を決めてからnew epochをgrantする。

## 13. Heartbeat

Heartbeatは単なるlivenessではなく、leaseが同じrevision/digest/base/branchに対して意味を持つか確認するeventである。

Coordinatorは次をprogress evidenceとして利用できる。

- explicit Worker progress event;
- lease branch HEAD advance;
- Draft PR create/update;
- CI start/finish;
- reviewable checkpoint publication。

固定intervalはrepository contractにしないが、expiry/reconciliation semanticsはdeterministic test対象とする。

## 14. Capability model

Roleごとにleast privilegeをTask Packet上で明示する。

Conceptual capabilities:

```text
repository_read
lease_branch_write
draft_pr_write
issue_comment_write
review_write
ci_read
artifact_read
integration_write
merge_write
```

Default:

- Worker(write): repository_read + lease_branch_write + draft_pr_write。
- Worker(read_only): repository_read。必要な外部side effectだけ追加。
- Reviewer: repository_read + review_write。production branch writeなし。
- Verifier: repository_read + ci_read + artifact_read。原則source writeなし。
- Integrator: repository_read + integration_write。merge_writeは明示authorization時のみ。

Runtime/toolingが物理的permission isolationを提供できない場合でも、protocol contractとして違反を検出・報告する。v1は利用可能でない権限制御を実装済みと主張しない。

## 15. Work isolation

Write Task default invariant:

```text
1 active Task lease
= 1 owner
= 1 lease epoch
= 1 isolated writable worktree
= 1 lease branch
= normally 1 PR
```

Worker同士はwritable worktreeを共有しない。

Read-only Taskは同一immutable commit snapshotを共有可能。ただしgenerated mutable cacheが結果を変える場合は分離する。

## 16. Dependency graph

依存関係は3種類。

### Hard dependency

上流Taskが指定`requires_phase`を満たすまで下流実行不可。

### Evidence dependency

実装はparallel可能だが指定上流evidenceなしではverification/complete不可。

### Integration dependency

実装/reviewは独立だがintegration順序を持つ。

DAG cycleはfail closedする。依存先Task revisionが変わり、edgeの前提が崩れた場合もdependent Taskを再評価する。

## 17. Resource Keys

File overlapだけでsemantic collisionを判定しない。

v1 namespace:

```text
file:<repository-path-or-prefix>
authority:<semantic-owner>
identity:<content-addressed-identity>
schema:<schema-or-contract>
workflow:<workflow-or-integration-path>
artifact:<artifact-family>
side-effect:<external-or-repository-resource>
```

例:

```text
authority:market-feature-computation
identity:market-dataset
schema:candidate-run-v2
workflow:canonical-m2-bootstrap
artifact:evidence-set
side-effect:github-issue-500-status
```

可能な限り#475のcontext/impactからderiveし、sourceから判定できないsemantic/side-effect concernだけTask contractで明示する。

## 18. Conflict graph

Task pairは`hard / soft / none`へ分類する。

### Hard

同時executing禁止。

- same writable file region;
- same schema/identity authorityを別方針で変更;
- same mutable external side-effect resourceを同時更新。

### Soft

parallel implementation可。ただし片方のintegration後、他方のrelevant review/verificationをstaleにする。

- same package/public facadeの別surface;
- shared verification workflow変更;
- common architecture contract変更。

### None

独立parallel可。

Ready set:

```text
ready = dependency-satisfied
      ∩ non-hard-conflicting
      ∩ unleased when write
      ∩ current contract digest
```

## 19. Scheduler

Coordinator refresh cycle:

```text
1. refresh current main / task contracts / status / PRs / evidence
2. recompute Task Contract digests
3. invalidate stale contract/head/main/conflict evidence
4. reconcile expired write leases
5. resolve dependency states
6. derive/update Resource Keys
7. build conflict graph
8. compute ready-set
9. rank ready Tasks deterministically
10. dispatch role work / grant write leases
11. publish status/dashboard projection
```

v1 priority:

1. blocking upstream Task;
2. critical path;
3. largest descendant-unblock count;
4. lower-risk independent Task;
5. stable task-id tie-break。

複雑なAI priority scoreはv1に入れない。

## 20. Roles

### Coordinator

Task decomposition、revision/digest、DAG、conflict graph、phase/condition、lease、stale invalidation、scheduling、status projectionを所有する。原則production implementationを行わない。

### Worker

Task Packet、TDD、focused implementation/investigation、checkpoint、self-review、reviewable resultを担当する。Write Taskではepoch worktree/branchを使う。WorkerはTaskをcompleteにできない。

### Reviewer

Workerとは独立したcontextで、Task Packet / current source / exact diff/result / tests/evidenceからadversarial reviewする。Worker reasoningをauthorityとして引き継がない。

Verdict:

```text
PASS
CHANGES_REQUIRED
BLOCKING_FINDING
```

### Verifier

Acceptance Criteria、Invariants、Failure Modes、Test Oracle、quality gate、falsification、unverified itemsをexact snapshotに対して確認する。

### Integrator

current main、tested-head containment、competing work、exact-head review/CI、expected-head merge、post-merge oracleを担当する。

Small deploymentではCoordinator/Integrator併任可。material write TaskではWorker/Reviewerを分離する。

## 21. Worker submission gate

Write Taskの`executing → review`前に最低限:

- current task revision/digest確認;
- write scope/capability確認;
- reproducing RED観測;
- minimal GREEN観測;
- refactor後GREEN;
- final diff self-review;
- debug/tmp/generated residueなし;
- epoch branchへcommit/push;
- immutable review targetあり;
- known limitations / unverified items明記。

Read-only TaskではRED/GREENやbranch要件のうち非該当項目を除外し、Task固有Test Oracle/evidenceを満たす。

自然言語「できた」はtransition oracleではない。

## 22. Checkpoint and handoff

Meaningful boundaryでrecoverable checkpointをdurable化する。

```text
RED reproduced
root cause evidence captured
minimal GREEN
refactor complete
long external CI before wait
reviewable HEAD/result
```

Worker交代時のhandoff minimum:

```yaml
task_id: T500-03
task_revision: 2
task_contract_digest: <sha256>
lease_epoch: 4
last_good_head: abc123
verified:
  - RED reproduction confirmed
pending:
  - full CI
known_failures: []
do_not_repeat:
  - completed root-cause diagnostic
evidence:
  - pr: 512
  - ci_run: 123456
```

Private conversation historyなしで再開できることをoracleとする。

## 23. Evidence binding and invalidation

Evidence logical minimum:

```yaml
evidence_kind: review | test | ci | artifact | integration | post_merge
task_id: T500-03
task_revision: 2
task_contract_digest: <sha256>
base_sha: aaa111
head_sha: bbb222
identifier: <stable id>
result: pass | fail
```

Read-only evidenceでHEADが不要な場合も、評価したimmutable source snapshot SHAを`head_sha`相当としてbindする。

Invalidation rules:

- task revision/digest change → lease/review/verification/integration stale;
- source/PR HEAD change → review/verification/integration stale;
- current main change → integration stale;
- hard/soft conflict integration → affected Task evidence stale according to edge;
- malformed/missing binding → invalid, not unknown-pass。

v1は安全側に倒し、HEADが変わればreview/verificationを取り直す。

## 24. Current-main and Integration

Integration invariant:

```text
tested PR head contains then-current main
+
exact head has required review and CI evidence
```

main advance後は:

```text
phase=integration
condition=stale
reason=current_main_advanced
```

とする。

Integrationはfirst-class phaseであり、merge successだけではcompleteではない。

Required oracle:

```text
current main resolved
no unresolved hard conflict
current Task Contract digest
PR head contains current main
review exact HEAD PASS
verification exact HEAD PASS
required CI exact HEAD Green
final diff/status clean
merge authorization present
merge uses expected HEAD
post-merge main HEAD observed
post-merge required oracle Green
```

## 25. Parent completion

Parent Issue completion:

1. required child Tasks complete;
2. parent Acceptance Criteria再評価;
3. cross-task invariants再評価;
4. integration ordering/final tree確認;
5. residual risks / unverified items明記。

```text
all children complete
≠ automatically parent complete
```

## 26. GitHub projection

推奨durable mapping:

| Information | Location |
|---|---|
| Task contract | Issue body |
| phase/condition | Coordinator status record; labels are projection |
| owner | assignee + status record |
| revision/digest/lease/base/head | Coordinator status record |
| implementation | epoch-specific branch / PR |
| review | PR review |
| verification | CI / immutable artifact |
| completion | Issue closed/completed after required oracle |

Coordinator-owned status commentはTaskごとに原則1つをupdate-in-placeする。

```text
<!-- agent-coordination:T500-03 -->
Schema: agent_coordination_status_v1
Task-Revision: 2
Task-Contract-Digest: sha256:...
Phase: executing
Condition: healthy
Owner: agent-A
Lease-Epoch: 4
Base: aaa111
Lease-Branch: agent/T500-03/e0004-agent-a
Head: bbb222
Checkpoint: RED confirmed
Blocking: none
```

Worker/Reviewerはこのrecordを直接編集しない。

Human-facing dashboardはこのevidenceから生成するderived viewとする。

## 27. Recovery and fencing scenarios

### Worker lost before push

Durable evidenceなし。未push stateをprogressとして扱わずlast durable checkpointから再実行する。

### Worker lost after push

Remote branch/PR/CIをreconcileし、same epoch resumeまたはnew epoch salvageを選ぶ。

### Stale Worker returns after reassignment

旧Workerは旧epoch branchへしか成果物を出さない。旧epoch resultをcurrent stateとしてacceptしない。必要ならnew epoch branchへexplicit importして再reviewする。

### Coordinator restart

Issue/status/branch/PR/CI evidenceからstateを再構築する。Coordinator memoryだけに必要情報があれば設計違反。

### Coordinator split-brain

v1では自動解決しない。複数Coordinatorを同時activeにしないことをoperational invariantとし、takeover前に旧session停止を要求する。

## 28. Security / trust boundary

- Worker summaryよりGit SHA/diff/test outputを優先する。
- Reviewer proseよりreview target SHAを確認する。
- CI textよりrun/head bindingを確認する。
- labelだけでmerge可否を決めない。
- malformed status/contractはfail closedする。
- write scope/capability違反はTask contract violation。
- secrets/token値をTask Packet/statusへ保存しない。
- Reviewer/Verifierへ不要なwrite capabilityを与えない。

## 29. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Duplicate claim | single Coordinator + one active lease + epoch |
| Stale Worker after reassign | branch-per-epoch fencing + old epoch rejection |
| Issue edited without revision bump | canonical Task Contract digest |
| Worker uses stale Task spec | revision/digest transition checks |
| Read-only task creates unnecessary mutable state | execution mode contract |
| File-disjoint semantic collision | Resource Keys + conflict graph |
| External side-effect race | `side-effect:` Resource Key |
| Review stale after commit | exact snapshot evidence binding |
| Green PR stale after main advance | integration invalidation + current-main containment |
| Global ledger merge conflict | no checked-in mutable ledger |
| Agent death loses work | meaningful durable checkpoints |
| Child completion masks parent failure | parent final acceptance re-evaluation |
| Coordinator restart loses state | GitHub evidence reconstruction |
| Coordinator split-brain | explicit single-active-Coordinator invariant |
| Role overreach | least-capability contract + violation detection |

## 30. Test strategy

### Unit

- Task Packet canonicalization/digest;
- execution mode/capability validation;
- phase/condition transitions;
- task revision invalidation;
- typed dependency DAG and cycle rejection;
- Resource Key normalization;
- hard/soft/none conflict classification;
- lease epoch/fencing;
- evidence invalidation;
- deterministic ready-set/scheduling。

### Property / model-based

- at most one active write lease per Task;
- epoch monotonicity;
- epoch branch uniqueness;
- hard-conflicting write Tasks never both executing;
- stale evidence never satisfies completion;
- complete implies required verification/integration evidence;
- digest change with unchanged revision never remains healthy;
- read-only Task never requires or mutates lease branch state。

### Integration

Temporary Git repo/worktreeで:

- isolated write worktrees;
- shared immutable read-only snapshots;
- epoch branches;
- old-worker late push isolation;
- HEAD movement;
- main movement;
- merge-base/current-main containment;
- checkpoint recovery/handoff。

### GitHub contract

- Task/status parse;
- status reconstruction;
- exact PR head;
- review/CI head binding;
- stale-main detection;
- malformed status fail-closed;
- side-effect Resource Key collision。

### Falsification

意図的に:

- duplicate lease;
- stale epoch result;
- Issue contract mutation without revision bump;
- read-only Task attempting source write;
- cyclic dependency;
- semantic conflict without file overlap;
- same external side effect without file overlap;
- approved-old-HEAD + new commit;
- Green-old-main + advanced main;
- all children complete + parent AC failure

を作り、systemが拒否することを確認する。

## 31. Implementation boundary

#475がcanonicalになった場合、同じrepository-tooling ownerへ追加する。

Conceptual responsibility split:

```text
tools/agent_repo/
  coordination/
    model.py
    contract.py
    state.py
    dependencies.py
    conflicts.py
    leases.py
    scheduler.py
    github_state.py
    dashboard.py
```

- `model.py`: immutable domain-neutral coordination types
- `contract.py`: Task Packet canonicalization/digest/capabilities
- `state.py`: phase/condition and invalidation
- `dependencies.py`: typed DAG
- `conflicts.py`: Resource Keys/conflict graph
- `leases.py`: epoch/fencing/reconciliation state
- `scheduler.py`: deterministic ready-set/allocation
- `github_state.py`: GitHub durable projection boundary
- `dashboard.py`: human derived view

Exact filenamesは実装時の#475 structureに合わせて調整可能だが責務分離は維持し、巨大single fileへ集約しない。

`trade_rl/**`からこのtoolingへ依存しない。

## 32. CLI direction

#475のentrypointを継続できる場合、別binaryではなくnamespaceを追加する。

```text
python -m tools.agent_repo task show <id>
python -m tools.agent_repo task graph <parent>
python -m tools.agent_repo task ready <parent>
python -m tools.agent_repo task reconcile <id>
python -m tools.agent_repo task dashboard [<parent>]
```

Write commandsを直接CLIへ公開するかCoordinator API専用にするかは、implementation時点のGitHub permission surfaceを確認して決める。Read-only model/state contractsはwrite capabilityに依存させない。

## 33. Dogfood

Coordination Plane自身のcontract testsがGreenになる前に本番coordination authorityとして使わない。

最初のdogfoodは開始時点のcurrent stateを確認し、semantic scopeが独立した2+ Taskを選ぶ。候補例は#494と#476だが、完了/仕様変更済みなら別Taskを選ぶ。

評価:

- duplicate work zero;
- scope/capability violation zero;
- stale evidence reuse zero;
- successful recovery/handoff;
- no hidden semantic/side-effect collision;
- integration後のsoft-conflicting remaining Taskが正しくstale化。

## 34. Acceptance Criteria

1. Independent read-only/write Taskを2つ以上、安全にconcurrent dispatchできる。
2. Read-only Taskに不要なwritable branch/worktreeを要求しない。
3. 同一write Taskへactive leaseを二重grantできない。
4. Lease expiry後、side-effect reconciliationなしにreassignできない。
5. New lease epochはunique writable branchを持ち、old Worker late pushがnew owner branchを直接汚さない。
6. Task revisionまたはcanonical contract digest変更で旧lease/review/verificationがstaleになる。
7. File overlapなしのsemantic/side-effect Resource Key collisionをhard/soft conflictとして表現できる。
8. Hard-conflicting write Tasksを同時executingにしない。
9. PR/source snapshot変更後、旧review/CI evidenceをvalid扱いしない。
10. Main advancement後、integration evidenceをstaleにしcurrent-main-inclusive exact HEADを再検証する。
11. Agent/session loss後、durable checkpointからprivate chatなしでhandoffできる。
12. WorkerはTaskをcompleteにできずreview/verification/integration protocolを通る。
13. All child completeだけでparent closeせずparent ACを再評価する。
14. Coordinator restart後、durable evidenceからstateを再構築できる。
15. Repository Control PlaneとCoordination Planeのauthorityを分離する。
16. Role capability violationを検出・報告できる。
17. `trade_rl/**` runtime/public API/artifact semanticsをcoordination都合で変更しない。
18. Existing final quality gatesを弱めない。
19. Multi-Coordinator consensusを実装したと誤って主張せず、v1 single-active-Coordinator invariantを明記する。

## 35. Quality Gate

Implementation completeには最低限:

- Task contract digest tests Green;
- execution-mode/capability tests Green;
- state/revision/lease/fencing/property tests Green;
- dependency/conflict/scheduler tests Green;
- temp-Git/worktree integration Green;
- old-worker late-push falsification Green;
- GitHub projection/reconstruction tests Green;
- stale-HEAD/stale-main falsification Green;
- parent completion falsification Green;
- Ruff / Format / Mypy;
- repository-tooling type checks;
- full pytest;
- build/distribution closure;
- clean-installed production package checks;
- final diff review;
- generated coordination/debug residueなし;
- exact final HEAD CI Green;
- current-main containment before merge;
- post-merge main oracle;
- dogfood result and residual risks documented。

Tests Greenだけではcompletionとしない。

## 36. Implementation sequencing

### Phase 0 — dependency normalization

#475のcurrent statusを再確認し、Control Plane primitiveをcanonicalにする順序を決める。

### Phase 1 — pure model

Task Packet canonical digest、execution mode/capability、phase/condition、revision/evidence invalidation、DAG、Resource Keysをnetwork-freeで実装する。

### Phase 2 — lease/fencing/scheduler

Epoch branch fencing、reconciliation、ready-set、hard/soft conflict schedulingを実装する。

### Phase 3 — Git/GitHub projection

Issue/status/branch/PR/head/CIからstateをreconstructするboundaryを実装する。

### Phase 4 — role protocol

Worker submission、Reviewer、Verifier、Integrator、handoff/recoveryを接続する。

### Phase 5 — dashboard

active/blocked/stale/owner/dependencyを一目で確認できるderived viewを作る。

### Phase 6 — adversarial verification

race、stale worker、contract mutation、capability violation、side-effect collision、main movement、parent acceptanceを壊す。

### Phase 7 — dogfood

独立Taskを2+ Workerで並列運用して実環境検証する。

## 37. Completion lifecycle

このspecはimplementation中だけ`docs/specs/`へ保持する。

実装完了後はdurable responsibility/invariantを`docs/AGENTS.md`と必要なarchitecture docsへ移し、completed planとActive specをcurrent treeから削除する。Historical designはGit history / PR / Issueを参照する。
