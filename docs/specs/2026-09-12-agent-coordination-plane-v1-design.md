# Agent Coordination Plane v1

Status: Active

Related: #500, #475

## 1. Objective

Trade RL で複数のAI Agentが同時に作業するとき、単なるparallel executionではなく、**task decomposition / ownership / dependency / collision / review / verification / integration / recoveryを一貫したprotocolで管理する**。

v1は次を機械的に保証するためのcoordination layerである。

- 同一Taskを複数Workerが重複実装しない。
- 独立Taskだけを安全に並列化する。
- fileが異なっても同じsemantic authority / identityを変更するTaskを競合として扱う。
- Task contract、PR HEAD、current mainの変化で古いevidenceを再利用しない。
- Agent/session消失後もGitHub上のdurable evidenceから復元できる。
- Worker自身の「完了」ではなくindependent review / verification / integration evidenceで完了を決める。
- child Task完了数だけでparent Issueをcloseせず、parent Acceptance Criteriaを最後に再評価する。

この仕組みを **Agent Coordination Plane** と呼ぶ。

## 2. Three-plane architecture

PR #475 の Agent Repository Control Plane は、Repositoryをsource-derivedに理解・検証する層である。

```text
┌────────────────────────────────────────────┐
│            Coordination Plane              │
│ Task DAG / lease / scheduling / conflict   │
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
│ review / verification / main sync          │
│ merge authorization / post-merge oracle    │
└────────────────────────────────────────────┘
```

Repository Control Planeは「Repository上の事実」を読む。
Coordination Planeは「誰が、何を、いつ実行できるか」を決める。
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

### 5.3 Labels are projections

GitHub labelsやdashboard表示はoperator UX用projectionであり、state authorityではない。IssueのTask contract、Coordinator status record、Git/PR/CI evidenceを正本として再構成する。

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

## 8. Task contract identity

`task_revision`だけではmanual editのrevision-bump漏れを検知できない。そのためTask Packetには**canonical contract digest**を導入する。

Canonical payloadはpresentation-only fieldsを除外し、UTF-8 JSON / sorted keys / stable separators等の実装で一意にserializationし、SHA-256でdigestする。

Conceptual identity:

```text
task_contract_digest = sha256(canonical_task_packet_without_runtime_status)
```

Lease、review、verification、integration evidenceは最低限次へbindする。

```text
task_id
task_revision
task_contract_digest
base_sha
head_sha
```

Issue本文が変更され、revisionが同じでもdigestが変わった場合、Coordinatorは`stale`としてfail closedする。semantic変更ならrevisionを増やし、presentation-only変更ならcanonical payloadを変えない。

## 9. Task revision

`task_revision`はsemantic contract変更時に単調増加するintegerとする。

Revisionを上げる例:

- objective / non-goal変更;
- acceptance criteria / test oracle変更;
- write scope変更;
- resource key変更;
- dependency変更。

Workerがrevision 2で作業中にrevision 3へ変われば、旧lease/review/verificationはstaleとなる。

## 10. Phase + Condition state model

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
  ↓ lease granted
executing
  ↓ reviewable HEAD submitted
review
  ↓ independent review PASS
verification
  ↓ acceptance + quality evidence PASS
integration
  ↓ merge + post-merge oracle
complete
```

代表的な逆遷移:

```text
review       → executing     CHANGES_REQUIRED
verification → executing     implementation defect
integration  → verification  re-evidence only
integration  → executing     semantic merge conflict
```

`complete`はterminal。後発問題は原則new Task/Issueとして扱う。

## 11. Lease and fencing protocol

### 11.1 Purpose

Leaseはduplicate work防止とabandoned work回収のためのownership fenceである。

### 11.2 Lease record

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

### 11.3 Epoch is a fencing token

新しいownership grantごとに`epoch`を増加する。

旧epochから返ってきたresultはcurrent ownershipとしてacceptしない。

### 11.4 Branch per lease epoch

**Reassignment後に同じwritable branchを再利用しない。**

各lease epochは固有branch/worktreeを持つ。

```text
agent/T500-03/e0003-agent-a
agent/T500-03/e0004-agent-b
```

これにより、復帰した旧Workerがold branchへpushしてもnew Ownerのbranchを直接汚さない。旧成果物を再利用する場合は、新epoch branchへexplicit cherry-pick/mergeし、改めてreviewする。

通常のsame-owner resumeでownership/epochが変わらない場合のみ同一lease branchを継続できる。

### 11.5 Expiry is not reassignment

Lease expiry後は必ずreconciliationする。

```text
old lease branch HEAD
open PR
CI runs
review comments
artifacts
unmerged commits
```

をinventoryし、`resume / salvage / supersede / reassign`を決めた後にnew epochをgrantする。

## 12. Heartbeat

Heartbeatは単なるlivenessではなく、leaseが同じrevision/digest/base/branchに対して有効か確認するeventである。

Coordinatorは次をprogress evidenceとして利用できる。

- explicit Worker progress event;
- lease branch HEAD advance;
- Draft PR create/update;
- CI start/finish;
- reviewable checkpoint publication。

固定intervalはrepository contractにしないが、expiry/reconciliation semanticsはdeterministic test対象とする。

## 13. Work isolation

Default invariant:

```text
1 active Task lease
= 1 owner
= 1 lease epoch
= 1 isolated writable worktree
= 1 lease branch
= normally 1 PR
```

Worker同士はwritable worktreeを共有しない。

Read-only cache共有は変更stateを持たず結果へ影響しない場合のみ許可する。

## 14. Dependency graph

依存関係は3種類。

### Hard dependency

上流が指定条件を満たすまで下流実行不可。

### Evidence dependency

実装はparallel可能だが上流evidenceなしではverification/complete不可。

### Integration dependency

実装/reviewは独立だがmerge順序を持つ。

DAG cycleはfail closedする。

## 15. Resource Keys

File overlapだけでsemantic collisionを判定しない。

v1 namespace:

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

可能な限り#475のcontext/impactからderiveし、sourceから判定できないsemantic concernだけTask contractで明示する。

## 16. Conflict graph

Task pairは`hard / soft / none`へ分類する。

### Hard

同時executing禁止。

- same writable file region;
- same schema authorityを別方針で変更;
- same content identity algorithmを同時変更。

### Soft

parallel implementation可。ただし片方のintegration後、他方のreview/verificationをstaleにする。

- same package/public facadeの別surface;
- shared verification workflow変更;
- common architecture contract変更。

### None

独立parallel可。

Ready set:

```text
ready = dependency-satisfied
      ∩ non-hard-conflicting
      ∩ unleased
      ∩ current task contract
```

## 17. Scheduler

Coordinator refresh cycle:

```text
1. refresh current main / active task contracts / leases / PRs / evidence
2. recompute task contract digests
3. invalidate stale contract/head/main evidence
4. reconcile expired leases
5. resolve dependency states
6. derive/update resource keys
7. build conflict graph
8. compute ready-set
9. rank ready tasks deterministically
10. grant leases / dispatch role work
11. publish status/dashboard projection
```

v1 priority:

1. blocking upstream Task;
2. critical path;
3. largest descendant-unblock count;
4. lower-risk independent Task;
5. stable task-id tie-break。

複雑なAI priority scoreはv1に入れない。

## 18. Roles

### Coordinator

Task decomposition、revision/digest、DAG、conflict graph、phase/condition、lease、stale invalidation、scheduling、status projectionを所有する。

原則production implementationをしない。

### Worker

Task Packet、isolated worktree、TDD、focused implementation、checkpoint、self-review、Draft PR/evidence packageを担当する。

Worker自身はTaskをcompleteにできない。

### Reviewer

Workerとは独立したcontextで、Task Packet / current source / exact PR diff / tests / evidenceからadversarial reviewする。Worker reasoningをauthorityとして引き継がない。

Verdict:

```text
PASS
CHANGES_REQUIRED
BLOCKING_FINDING
```

### Verifier

Acceptance Criteria、Invariants、Failure Modes、Test Oracle、quality gate、falsification、unverified itemsをexact HEADに対して確認する。

### Integrator

current main、tested-head containment、competing work、exact-head review/CI、expected-head merge、post-merge oracleを担当する。

Small deploymentではCoordinator/Integrator併任可。material changeではWorker/Reviewerを分離する。

## 19. Worker submission gate

`executing → review`前に最低限:

- current task revision/digest確認;
- write scope確認;
- reproducing RED観測;
- minimal GREEN観測;
- refactor後GREEN;
- final diff self-review;
- debug/tmp/generated residueなし;
- lease branchへcommit/push;
- immutable review targetあり;
- known limitations / unverified items明記。

自然言語「できた」はtransition oracleではない。

## 20. Checkpoint and handoff

Meaningful boundaryでrecoverable checkpointをpushする。

```text
RED reproduced
root cause evidence captured
minimal GREEN
refactor complete
long external CI before wait
reviewable HEAD
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

## 21. Evidence binding and invalidation

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

Invalidation rules:

- task revision/digest change → lease/review/verification/integration stale;
- PR/branch HEAD change → review/verification/integration stale;
- current main change → integration stale;
- hard/soft conflict integration → affected downstream evidence stale according to conflict edge;
- malformed/missing binding → invalid, not unknown-pass。

v1は安全側に倒し、HEADが変わればreview/verificationを取り直す。

## 22. Current-main and Integration

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
current task contract digest
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

## 23. Parent completion

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

## 24. GitHub projection

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
| completion | Issue closed/completed after post-merge oracle |

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

## 25. Recovery and fencing scenarios

### Worker lost before push

Durable evidenceなし。未push stateをprogressとして扱わずlast durable checkpointから再実行する。

### Worker lost after push

Remote branch/PR/CIをreconcileし、same epoch resumeまたはnew epoch salvageを選ぶ。

### Stale Worker returns after reassignment

旧Workerは旧epoch branchへしか成果物を出さない。旧epoch resultをcurrent stateとしてacceptしない。必要ならnew epoch branchへexplicitly importして再reviewする。

### Coordinator restart

Issue/status/branch/PR/CI evidenceからstateを再構築する。Coordinator memoryだけに必要情報があれば設計違反。

### Coordinator split-brain

v1では自動解決しない。複数Coordinatorを同時activeにしないことをoperational invariantとし、takeover前に旧session停止を要求する。

## 26. Security / trust boundary

- Worker summaryよりGit SHA/diff/test outputを優先する。
- Reviewer proseよりreview target HEADを確認する。
- CI textよりrun/head bindingを確認する。
- labelだけでmerge可否を決めない。
- malformed status/contractはfail closedする。
- write scope違反はTask contract violation。
- secrets/token値をTask Packet/statusへ保存しない。

## 27. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Duplicate claim | single Coordinator + one active lease + epoch |
| Stale Worker after reassign | branch-per-epoch fencing + old epoch rejection |
| Issue edited without revision bump | canonical task contract digest |
| Worker uses stale task spec | revision/digest heartbeat and transition checks |
| File-disjoint semantic collision | Resource Keys + conflict graph |
| Review becomes stale after commit | exact HEAD evidence binding |
| Green PR becomes stale after main advance | integration invalidation + current-main containment |
| Global ledger merge conflict | no checked-in mutable ledger |
| Agent death loses work | meaningful pushed checkpoints |
| Child completion masks parent failure | parent final acceptance re-evaluation |
| Coordinator restart loses state | durable GitHub reconstruction |
| Coordinator split-brain | explicit v1 single-active-Coordinator invariant |

## 28. Test strategy

### Unit

- Task Packet canonicalization/digest;
- phase/condition transitions;
- task revision invalidation;
- dependency DAG and cycle rejection;
- Resource Key normalization;
- hard/soft/none conflict classification;
- lease epoch/fencing;
- evidence invalidation;
- deterministic ready-set/scheduling。

### Property / model-based

- at most one active lease per Task;
- epoch monotonicity;
- epoch branch uniqueness;
- hard-conflicting Tasks never both executing;
- stale evidence never satisfies completion;
- complete implies required integration/post-merge evidence;
- digest change with unchanged revision never remains healthy。

### Integration

Temporary Git repo/worktreeで:

- isolated worktrees;
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
- malformed status fail-closed。

### Falsification

意図的に:

- duplicate lease;
- stale epoch result;
- Issue contract mutation without revision bump;
- cyclic dependency;
- semantic conflict without file overlap;
- approved-old-HEAD + new commit;
- Green-old-main + advanced main;
- all children complete + parent AC failure

を作り、systemが拒否することを確認する。

## 29. Implementation boundary

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
- `contract.py`: Task Packet canonicalization/digest
- `state.py`: phase/condition and invalidation
- `dependencies.py`: typed DAG
- `conflicts.py`: Resource Keys/conflict graph
- `leases.py`: epoch/fencing/reconciliation state
- `scheduler.py`: deterministic ready-set/allocation
- `github_state.py`: GitHub durable projection boundary
- `dashboard.py`: human derived view

Exact filenamesは実装時の#475 structureに合わせて調整可能だが責務分離は維持し、巨大なsingle fileへ集約しない。

`trade_rl/**`からこのtoolingへ依存しない。

## 30. CLI direction

#475のentrypointを継続できる場合、別binaryではなくnamespaceを追加する。

```text
python -m tools.agent_repo task show <id>
python -m tools.agent_repo task graph <parent>
python -m tools.agent_repo task ready <parent>
python -m tools.agent_repo task reconcile <id>
python -m tools.agent_repo task dashboard [<parent>]
```

Write commandsを直接CLIへ公開するかCoordinator API専用にするかは、implementation時点のGitHub permission surfaceを確認して決める。Read-only model/state contractsはwrite capabilityに依存させない。

## 31. Dogfood

Coordination Plane自身のcontract testsがGreenになる前に本番coordination authorityとして使わない。

最初のdogfoodは実際のcurrent stateを確認して、semantic scopeが独立した2+ Taskを選ぶ。候補例は#494と#476だが、開始時に完了/仕様変更済みなら別Taskを選ぶ。

評価:

- duplicate work zero;
- scope violation zero;
- stale evidence reuse zero;
- successful recovery/handoff;
- no hidden semantic collision;
- integration後のsoft-conflicting remaining Taskが正しくstale化。

## 32. Acceptance Criteria

1. Independent Taskを2つ以上shared writable worktreeなしでconcurrent leaseできる。
2. 同一Taskへactive leaseを二重grantできない。
3. Lease expiry後、side-effect reconciliationなしにreassignできない。
4. New lease epochはunique writable branchを持ち、old Worker late pushがnew owner branchを直接汚さない。
5. Task revisionまたはcanonical contract digest変更で旧lease/review/verificationがstaleになる。
6. File overlapなしのsemantic Resource Key collisionをhard/soft conflictとして表現できる。
7. Hard-conflicting Tasksを同時executingにしない。
8. PR HEAD変更後、旧review/CI evidenceをvalid扱いしない。
9. Main advancement後、integration evidenceをstaleにしcurrent-main-inclusive exact HEADを再検証する。
10. Agent/session loss後、durable checkpointからprivate chatなしでhandoffできる。
11. WorkerはTaskをcompleteにできずreview/verification/integrationを通る。
12. All child completeだけでparent closeせずparent ACを再評価する。
13. Coordinator restart後、durable evidenceからstateを再構築できる。
14. Repository Control PlaneとCoordination Planeのauthorityを分離する。
15. `trade_rl/**` runtime/public API/artifact semanticsをcoordination都合で変更しない。
16. Existing final quality gatesを弱めない。
17. Multi-Coordinator consensusを実装したと誤って主張せず、v1 single-active-Coordinator invariantを明記する。

## 33. Quality Gate

Implementation completeには最低限:

- Task contract digest tests Green;
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

## 34. Implementation sequencing

### Phase 0 — dependency normalization

#475のcurrent statusを再確認し、Control Plane primitiveをcanonicalにする順序を決める。

### Phase 1 — pure model

Task Packet canonical digest、phase/condition、revision/evidence invalidation、DAG、Resource Keysをnetwork-freeで実装する。

### Phase 2 — lease/fencing/scheduler

Epoch branch fencing、reconciliation、ready-set、hard/soft conflict schedulingを実装する。

### Phase 3 — Git/GitHub projection

Issue/status/branch/PR/head/CIからstateをreconstructするboundaryを実装する。

### Phase 4 — role protocol

Worker submission、Reviewer、Verifier、Integrator、handoff/recoveryを接続する。

### Phase 5 — dashboard

active/blocked/stale/owner/dependencyを一目で確認できるderived viewを作る。

### Phase 6 — adversarial verification

race、stale worker、contract mutation、main movement、parent acceptanceを壊す。

### Phase 7 — dogfood

独立Taskを2+ Workerで並列運用して実環境検証する。

## 35. Completion lifecycle

このspecはimplementation中だけ`docs/specs/`へ保持する。

実装完了後はdurable responsibility/invariantを`docs/AGENTS.md`と必要なarchitecture docsへ移し、completed planとActive specをcurrent treeから削除する。Historical designはGit history / PR / Issueを参照する。
