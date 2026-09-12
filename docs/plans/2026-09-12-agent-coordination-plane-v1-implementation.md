# Agent Coordination Plane v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the network-light, repository-local coordination primitives defined by #500 so multiple agents can be scheduled, isolated, reviewed, recovered, and integrated without duplicate ownership or stale evidence.

**Architecture:** Extend the existing `tools.agent_repo` owner from stacked base PR #475. Keep the core model pure and deterministic; keep GitHub projection/serialization at the boundary; keep `trade_rl/**` completely independent. The implementation PR is stacked on `feat/agent-repository-control-plane-v1` until #475 becomes canonical, at which point it must be synchronized with current `main` and all integration evidence regenerated.

**Tech Stack:** Python 3.12, stdlib dataclasses/enums/hashlib/json/datetime/pathlib, pytest, existing `tools.agent_repo` CLI and repository-tooling CI.

**Spec:** `docs/specs/2026-09-12-agent-coordination-plane-v1-design.md`

## Global Constraints

- Do not add coordination dependencies to `trade_rl/**` or the production wheel.
- Do not introduce a checked-in mutable global task ledger or a database/Redis lock service.
- Keep one active Coordinator as the v1 operational invariant; do not claim multi-Coordinator consensus.
- `read_only` tasks do not require writable branch/worktree state; `write` tasks do.
- Task contracts bind evidence through `task_id + task_revision + task_contract_digest + base/source SHA + exact HEAD when applicable`.
- Lease reassignment increments epoch and uses a new lease branch/worktree; old-epoch output cannot become current ownership without explicit salvage/review.
- File-disjoint semantic/identity/side-effect collisions must be representable.
- Main movement invalidates integration evidence; HEAD movement invalidates exact-HEAD review/verification evidence.
- Existing final CI gates remain unchanged in strength.
- This stacked implementation is not integration-complete until #475 is canonical and the final tested #500 HEAD contains then-current `main`.

---

### Task 1: Task contract, state, and evidence identity

**Files:**
- Create: `tools/agent_repo/coordination/__init__.py`
- Create: `tools/agent_repo/coordination/model.py`
- Create: `tools/agent_repo/coordination/state.py`
- Create: `tests/agent_repo/test_coordination_model.py`

**Interfaces:**
- Produces `ExecutionMode`, `TaskPhase`, `TaskCondition`, `RiskLevel`, `DependencyKind`, `EvidenceKind`, `EvidenceResult` enums.
- Produces immutable `TaskDependency`, `WriteScope`, `TaskPacket`, `TaskStatus`, and `EvidenceRecord` dataclasses.
- `TaskPacket.canonical_payload() -> dict[str, object]` excludes runtime status/presentation-only fields.
- `TaskPacket.contract_digest() -> str` returns SHA-256 of canonical UTF-8 JSON with sorted keys and compact separators.
- `validate_transition(current: TaskPhase, target: TaskPhase) -> None` rejects illegal phase transitions.
- `evidence_is_current(evidence, packet, *, head_sha, current_main_sha=None) -> bool` enforces exact binding; integration/post-merge evidence additionally requires current-main binding.

- [ ] **Step 1: Write failing contract/evidence tests**

Cover exact digest determinism under equivalent tuple/list construction, digest change on semantic fields, no digest change from absent runtime state, invalid SHA/revision/task id rejection, `read_only` vs `write` scope validation, legal/illegal phase transitions, exact-head evidence invalidation, and integration evidence invalidation when current main changes.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent_repo/test_coordination_model.py`

Expected: import/attribute failures because coordination model does not exist.

- [ ] **Step 3: Implement minimal immutable model/state layer**

Use frozen dataclasses and string enums. Validate full lowercase 40-hex Git SHAs where a commit identity is required. Canonicalize sequence fields to tuples in `__post_init__`; reject duplicate dependency task IDs or duplicate resource/capability entries rather than silently collapsing them.

- [ ] **Step 4: Run GREEN + adjacent types**

Run:
`uv run pytest -q tests/agent_repo/test_coordination_model.py`
`uv run mypy tools/agent_repo/coordination`
`uv run ruff check tools/agent_repo/coordination tests/agent_repo/test_coordination_model.py`

- [ ] **Step 5: Commit**

`git commit -m "feat: add agent coordination task model"`

---

### Task 2: Dependency DAG, semantic conflicts, and deterministic scheduling

**Files:**
- Create: `tools/agent_repo/coordination/dependencies.py`
- Create: `tools/agent_repo/coordination/conflicts.py`
- Create: `tools/agent_repo/coordination/scheduler.py`
- Create: `tests/agent_repo/test_coordination_scheduler.py`

**Interfaces:**
- `DependencyGraph(packets: Sequence[TaskPacket])` validates referenced task IDs and rejects cycles.
- `DependencyGraph.dependencies_satisfied(task_id, statuses, evidence) -> bool` distinguishes hard/evidence/integration dependencies.
- `ConflictLevel = NONE | SOFT | HARD`.
- `classify_conflict(left: TaskPacket, right: TaskPacket) -> ConflictLevel` returns NONE when both tasks are read-only immutable-snapshot work; otherwise file-prefix/identity/schema/side-effect collisions are HARD and authority/workflow/artifact collisions are SOFT unless a stronger shared key exists.
- `ready_tasks(packets, statuses, leases, evidence) -> tuple[str, ...]` returns only healthy, dependency-satisfied, unleased, non-hard-conflicting tasks in deterministic priority/order.

- [ ] **Step 1: Write failing DAG/conflict/scheduler tests**

Include missing dependency, hard cycle, `requires_phase`, evidence dependency, integration dependency, `file:docs/**` vs `file:docs/x.md`, same identity/schema/side-effect HARD, same authority/workflow/artifact SOFT, read-only sharing NONE, and two write tasks with a hard collision never both ready/executing.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent_repo/test_coordination_scheduler.py`

- [ ] **Step 3: Implement deterministic DAG/conflict/scheduler logic**

Avoid Git/network access. Normalize resource keys once, fail closed on unknown namespaces, and use stable task-id tie-breaking after descendant/critical-path ranking.

- [ ] **Step 4: Run GREEN + Task 1 regression**

Run: `uv run pytest -q tests/agent_repo/test_coordination_model.py tests/agent_repo/test_coordination_scheduler.py`

- [ ] **Step 5: Commit**

`git commit -m "feat: schedule nonconflicting agent tasks"`

---

### Task 3: Lease epochs, branch fencing, heartbeat, and reconciliation

**Files:**
- Create: `tools/agent_repo/coordination/leases.py`
- Create: `tests/agent_repo/test_coordination_leases.py`

**Interfaces:**
- Frozen `LeaseRecord` binds task/revision/digest/base/owner/epoch/branch/head/heartbeat/expiry.
- Frozen `LeaseReconciliation` binds the expired epoch to observed remote side effects and a decision: `resume | salvage | supersede | reassign`.
- `lease_branch_name(task_id, epoch, owner) -> str` yields deterministic `agent/<task>/eNNNN-<owner>` fencing branch names with safe slug validation.
- `grant_lease(packet, owner, *, epoch, now, ttl, previous=None, reconciliation=None) -> LeaseRecord` rejects read-only tasks, active duplicate leases, non-monotonic epochs, mismatched reconciliation, and reuse of a previous epoch branch on reassignment.
- `lease_is_current(lease, packet, *, now) -> bool` includes revision/digest/base/expiry checks.
- `heartbeat(lease, *, head_sha, observed_at) -> LeaseRecord` is monotonic in time and preserves ownership/epoch.

- [ ] **Step 1: Write failing lease/fencing tests**

Cover duplicate claim, active lease rejection, expiry alone not sufficient for reassignment, matching reconciliation required, epoch monotonicity, branch uniqueness, stale epoch output rejection, task revision/digest change invalidation, head heartbeat update, and read-only task no-write-lease rule.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent_repo/test_coordination_leases.py`

- [ ] **Step 3: Implement minimal lease protocol**

Use timezone-aware UTC datetimes only. `ttl` must be positive. Reconciliation data records branch existence/head plus stable identifiers for PR/CI/artifacts; it does not perform remote mutation.

- [ ] **Step 4: Run GREEN + scheduler regression**

Run: `uv run pytest -q tests/agent_repo/test_coordination_model.py tests/agent_repo/test_coordination_scheduler.py tests/agent_repo/test_coordination_leases.py`

- [ ] **Step 5: Commit**

`git commit -m "feat: fence agent task ownership with lease epochs"`

---

### Task 4: Durable GitHub projection, handoff, recovery, and dashboard

**Files:**
- Create: `tools/agent_repo/coordination/github_state.py`
- Create: `tools/agent_repo/coordination/dashboard.py`
- Create: `tests/agent_repo/test_coordination_projection.py`

**Interfaces:**
- `TaskStatusRecord` is a strict machine-readable projection of packet identity + phase/condition + owner/lease/head/checkpoint/blocking fields.
- `render_status_comment(record) -> str` emits exactly one marker `<!-- agent-coordination:<task-id> -->` and one fenced canonical JSON object.
- `parse_status_comment(text) -> TaskStatusRecord` fails closed on missing/duplicate marker, invalid JSON, unknown fields/enums, malformed SHA/digest, or inconsistent lease fields.
- Frozen `HandoffPacket` binds task/revision/digest/lease epoch/last-good HEAD + verified/pending/known-failures/do-not-repeat/evidence identifiers.
- `validate_handoff(handoff, packet, lease) -> None` rejects stale revision/digest/epoch/head.
- `render_dashboard(parent_title, main_sha, snapshots) -> str` is deterministic and displays progress, phase/condition, owner, PR, dependency/blocking/stale reason without becoming state authority.

- [ ] **Step 1: Write failing projection/recovery tests**

Include round-trip status comment, malformed/partial status fail-closed, duplicate marker rejection, Coordinator restart reconstruction from serialized records, stale handoff rejection, and deterministic dashboard ordering.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent_repo/test_coordination_projection.py`

- [ ] **Step 3: Implement strict serialization boundary**

Use stdlib JSON only; reject unknown keys rather than silently ignoring them. Never serialize secrets/tokens. Labels remain derived and are not parsed as authority.

- [ ] **Step 4: Run GREEN + full coordination tests**

Run: `uv run pytest -q tests/agent_repo/test_coordination_*.py`

- [ ] **Step 5: Commit**

`git commit -m "feat: project agent coordination state to GitHub records"`

---

### Task 5: CLI/operator surface and architecture boundaries

**Files:**
- Modify: `tools/agent_repo/__main__.py`
- Modify: `tests/agent_repo/test_cli.py`
- Create: `tests/architecture/test_agent_coordination_boundary.py`
- Modify: `docs/AGENTS.md`
- Modify: `docs/architecture/package-boundaries.md`

**Interfaces:**
- Extend the existing CLI with a nested `task` command, not a new binary.
- v1 network-free commands:
  - `task digest <packet-json>`
  - `task ready <snapshot-json>`
  - `task status-render <status-json>`
  - `task status-parse <status-text-file>`
  - `task dashboard <snapshot-json>`
- CLI write operations such as live GitHub claim/comment mutation are deliberately deferred; the library exposes deterministic state rules and external Coordinators/connectors perform authorized writes.
- Architecture test asserts `trade_rl/**` does not import `tools.agent_repo` and coordination tooling is not in the production wheel.

- [ ] **Step 1: Write failing CLI/boundary tests**

Add one end-to-end JSON fixture that constructs two independent tasks plus one hard-conflicting task and verifies digest/ready/dashboard output. Assert malformed packet/status inputs return exit code 2 and no stdout. Assert CLI invocation does not mutate the temporary Git repo.

- [ ] **Step 2: Run RED**

Run: `uv run pytest -q tests/agent_repo/test_cli.py tests/architecture/test_agent_coordination_boundary.py`

- [ ] **Step 3: Implement CLI dispatch and durable docs**

Follow the existing `_parser` / `_dispatch` / `_write_json` pattern. Keep network calls out of the CLI. Document source-derived Control Plane vs Coordination Plane responsibilities and the stacked dependency on #475.

- [ ] **Step 4: Run GREEN + repository-tooling checks**

Run:
`uv run pytest -q tests/agent_repo tests/architecture/test_agent_coordination_boundary.py`
`uv run mypy tools/agent_repo tests/architecture/distribution.py`
`uv run ruff check tools tests/agent_repo tests/architecture/test_agent_coordination_boundary.py`
`uv run ruff format --check tools tests/agent_repo tests/architecture/test_agent_coordination_boundary.py`

- [ ] **Step 5: Commit**

`git commit -m "feat: expose agent coordination operator commands"`

---

### Task 6: Adversarial verification and stacked integration readiness

**Files:**
- Create or extend: `tests/agent_repo/test_coordination_falsification.py`
- Modify only if needed: `docs/specs/2026-09-12-agent-coordination-plane-v1-design.md`
- Modify only if needed: `docs/plans/2026-09-12-agent-coordination-plane-v1-implementation.md`

**Interfaces:**
- No new production interface unless a falsification finding proves one necessary.
- Final stacked HEAD must be reviewable independently of private conversation state.

- [ ] **Step 1: Add adversarial tests before any required fix**

Exercise: two workers racing one task, stale old-epoch result after reassignment, task body/digest drift without revision bump, approved old HEAD followed by new commit, Green old-main with advanced main, file-disjoint shared identity conflict, all-child-complete with parent acceptance false, malformed status reconstruction, and read-only task attempting write lease.

- [ ] **Step 2: Verify falsification suite catches seeded wrong states**

Run: `uv run pytest -q tests/agent_repo/test_coordination_falsification.py`

If a demonstrated defect requires production changes, preserve TDD: keep the failing case, apply the smallest fix, rerun targeted tests.

- [ ] **Step 3: Run full stacked quality gate on exact final HEAD**

Run equivalent to permanent CI:
`uv run ruff check trade_rl tests tools`
`uv run ruff format --check trade_rl tests tools`
`uv run mypy trade_rl`
`uv run mypy tools/agent_repo tests/architecture/distribution.py`
`uv run pytest -q tests`
`uv build`
plus existing distribution closure, clean-installed smoke, and package identity steps from `.github/workflows/ci.yml`.

- [ ] **Step 4: Final diff/self-review**

Confirm no `trade_rl/**` behavior/schema changes, no coordination generated state checked in, no temporary workflows/debug artifacts, no duplicate #475 authority, and only active spec/plan remain as current work.

- [ ] **Step 5: Stacked PR disposition**

Record that implementation evidence is valid for the exact stacked HEAD only. Do not claim final integration while #475 is Draft. Once #475 is canonical, retarget/synchronize #500 to current main, invalidate stale integration evidence, rerun exact-head review/full CI, then perform post-merge main verification before closing #500.

- [ ] **Step 6: Commit**

`git commit -m "test: falsify agent coordination failure modes"`
