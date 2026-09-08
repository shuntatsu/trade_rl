Status: Active

# Controlled Experiment Loop v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an append-only development Study that executes multi-seed Study-owned candidate evidence, proves one declared semantic factor changed, records paired/seed evidence and decisions, and freezes WINNER or NO_WINNER without opening final unused data.

**Architecture:** `trade_rl.evaluation.experiments` sits above the verified PR1 Run Core. Study contracts freeze dataset/evaluation/provenance/seed policy; EvidenceSet executes one Candidate Run per pre-registered PPO seed; delta verification proves actual resolved changes and unaffected-strategy invariance; workflow artifacts are append-only and lineage/budget constrained. Existing paired/bootstrap/seed robustness primitives remain the statistical authorities.

**Tech Stack:** Python 3.12, NumPy, standard-library `fcntl`/`msvcrt` advisory locking, existing canonical/content digest + atomic file helpers, existing evaluation comparison primitives, pytest/Hypothesis, Ruff, Mypy.

**Spec:** `docs/specs/2026-09-09-controlled-experiment-loop-v1-design.md`

**Prerequisite:** Verified PR1 Candidate Run Core final HEAD. The exact interfaces from PR1 plan are required: `CandidateRunConfig`, `ResolvedCandidateRunSpec`, `execute_candidate_run`, `publish_candidate_run`, `load_candidate_run_artifact`, `inspect_candidate_run_artifact`, `build_candidate_run_provenance`.

## Global Constraints

- `evaluation/experiments` must not import sealed final-test authorization.
- Reuse Run Core for all candidate execution/config resolution/artifact publication.
- No external Run import/registration in v1.
- Study `ppo_seeds` has at least 2 unique non-negative values; only `ppo_seed` varies inside an EvidenceSet.
- Existing 5 candidates + 3 controls are fixed.
- No aggregate inference that treats symbols as independent samples.
- No automatic winner selection.
- All research artifacts append-only; existing final target is never overwritten.
- Study mutation is single-writer / multi-reader with process-safe OS advisory lock.
- Study ends at WINNER / NO_WINNER; M3 unused future/stress remains unreachable.
- Completed spec/plans are removed only after durable current docs are updated at final PR2 gate.

---

### Task 1: Establish experiments boundary, errors, and immutable contracts

**Files:**
- Create: `trade_rl/evaluation/experiments/__init__.py`
- Create: `trade_rl/evaluation/experiments/errors.py`
- Create: `trade_rl/evaluation/experiments/contracts/__init__.py`
- Create: `trade_rl/evaluation/experiments/contracts/study.py`
- Create: `trade_rl/evaluation/experiments/contracts/experiment.py`
- Create: `trade_rl/evaluation/experiments/contracts/decision.py`
- Create: `tests/evaluation/experiments/__init__.py`
- Create: `tests/evaluation/experiments/test_contracts.py`
- Modify: `tests/architecture/test_lean_evaluation_layout.py`
- Modify: `tests/architecture/test_lean_dependency_boundaries.py` if required to forbid sealed-test dependency.

**Interfaces:**
- Produces:

```python
class ControlledExperimentError(Exception): ...
class ContractViolationError(ControlledExperimentError): ...
class ArtifactIntegrityError(ControlledExperimentError): ...
class InvalidExperimentStateError(ControlledExperimentError): ...
class UncontrolledDeltaError(ControlledExperimentError): ...
class ExperimentBudgetExceededError(ControlledExperimentError): ...
class StudyFrozenError(ControlledExperimentError): ...

class ControlledFactor(Enum):
    FEATURE_SET = "FEATURE_SET"
    RULE_SIGNAL = "RULE_SIGNAL"
    RULE_THRESHOLDS = "RULE_THRESHOLDS"
    FORECAST_THRESHOLDS = "FORECAST_THRESHOLDS"
    FIT_SYMBOL_SCOPE = "FIT_SYMBOL_SCOPE"
    PPO_TRAINING_BUDGET = "PPO_TRAINING_BUDGET"
    GROSS_BUDGET = "GROSS_BUDGET"
```

`StudyPlan`, `ExperimentDefinition`, `ExperimentDecision`, `StudyFreeze` are frozen dataclasses with `digest_payload()` + validated content digest helpers.

- [ ] **Step 1: Architecture RED**

For Task 1, require only `experiments/__init__.py`, `errors.py`, and `contracts/{__init__,study,experiment,decision}.py`; later tasks extend the required-layout tuple as each permanent owner is implemented, and Task 7 requires the complete final package. Replace PR1's `assert not (PACKAGE / "experiments").exists()` with this incremental contract.

Add AST dependency assertion that any module below `trade_rl.evaluation.experiments` importing the sealed-test authorization module/path is an offender.

- [ ] **Step 2: Contract RED**

Tests must cover:

```python
with pytest.raises(ContractViolationError):
    StudyPlan(..., ppo_seeds=(0,), ...)

with pytest.raises(ContractViolationError):
    StudyPlan(..., ppo_seeds=(0, 0), ...)

with pytest.raises(ContractViolationError):
    StudyPlan(..., max_experiments=0, ...)
```

Assert baseline config seed equals first Study seed, allowed factors unique, research question non-empty, `n_bootstrap > 0`, `bootstrap_seed >= 0`.

Decision tests require exactly ACCEPT_CANDIDATE / KEEP_BASELINE / INCONCLUSIVE; rationale/decided_by non-empty and decided_at timezone-aware.

Freeze tests require candidate-only strategy for WINNER and forbid selected fields for NO_WINNER.

- [ ] **Step 3: Run RED**

```bash
uv run pytest -q tests/evaluation/experiments/test_contracts.py tests/architecture/test_lean_evaluation_layout.py
```

Expected: missing experiments package/contracts.

- [ ] **Step 4: Implement minimal contracts**

Use existing `content_digest()` and canonical JSON-compatible payloads. Do not add filesystem behavior to contracts.

- [ ] **Step 5: Run GREEN/static**

```bash
uv run pytest -q tests/evaluation/experiments/test_contracts.py tests/architecture/test_lean_evaluation_layout.py
uv run ruff check trade_rl/evaluation/experiments tests/evaluation/experiments
uv run mypy trade_rl/evaluation/experiments
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/evaluation/experiments tests/evaluation/experiments tests/architecture
git commit -m "feat: define controlled experiment contracts"
```

---

### Task 2: Implement StudyStore append-only filesystem and mutation lock

**Files:**
- Create: `trade_rl/evaluation/experiments/store.py`
- Create: `tests/evaluation/experiments/test_store.py`
- Create: `tests/evaluation/experiments/test_store_concurrency.py`

**Interfaces:**
- Produces private/public filesystem helper:

```python
class StudyStore:
    def __init__(self, root: str | Path) -> None: ...
    @contextmanager
    def mutation_lock(self) -> Iterator[None]: ...
    def publish_json_once(self, relative: str | Path, value: object) -> Path: ...
    def read_json(self, relative: str | Path) -> dict[str, object]: ...
    def publish_directory_once(
        self,
        relative: str | Path,
        builder: Callable[[Path], None],
    ) -> Path: ...
```

Lock implementation is private and chooses `fcntl.flock` on POSIX / `msvcrt.locking` on Windows.

- [ ] **Step 1: Append-only RED**

Assert first `publish_json_once("plan.json", payload)` succeeds and second publication raises `InvalidExperimentStateError` without changing original bytes.

Assert `publish_directory_once()` removes staging if builder raises and does not create final target.

- [ ] **Step 2: Path/symlink RED**

Reject absolute paths, `..`, target symlink/non-regular file, and publication through a symlinked parent beneath Study root.

- [ ] **Step 3: Lock RED**

Use two processes on the current platform: process A acquires the Study lock and signals; process B attempts lock and must not enter critical section until A releases. Assert both exit cleanly and only one writes the protected sequence probe at a time.

- [ ] **Step 4: Run RED**

```bash
uv run pytest -q tests/evaluation/experiments/test_store.py tests/evaluation/experiments/test_store_concurrency.py
```

- [ ] **Step 5: Implement StudyStore**

The lock file `.mutation.lock` may persist but is not canonical research evidence. Use staging siblings under the Study root and atomic rename only after builder success.

- [ ] **Step 6: Run GREEN**

```bash
uv run pytest -q tests/evaluation/experiments/test_store.py tests/evaluation/experiments/test_store_concurrency.py
uv run mypy trade_rl/evaluation/experiments/store.py
```

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/experiments/store.py tests/evaluation/experiments/test_store.py tests/evaluation/experiments/test_store_concurrency.py
git commit -m "feat: add append-only study store"
```

---

### Task 3: Implement Study creation and multi-seed EvidenceSet execution

**Files:**
- Create: `trade_rl/evaluation/experiments/evidence.py`
- Create: `tests/evaluation/experiments/test_evidence.py`
- Modify: `trade_rl/evaluation/experiments/workflow.py` only if a minimal `create_study`/`run_baseline` entry is introduced here; otherwise create it in Task 6.

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EvidenceSet:
    fingerprint: str
    semantic_config_digest: str
    ppo_seeds: tuple[int, ...]
    run_digests: tuple[tuple[int, str], ...]
    analysis_digest: str


def execute_evidence_set(
    *,
    store: StudyStore,
    target: Path,
    dataset_root: str | Path,
    plan: StudyPlan,
    config: CandidateRunConfig,
) -> EvidenceSet: ...
```

The execution seam calls PR1 Run Core functions; tests monkeypatch `execute_candidate_run` only, not the evidence/digest/publication logic. Evidence loading is explicit:

```python
@dataclass(frozen=True, slots=True)
class LoadedEvidenceSet:
    evidence: EvidenceSet
    resolved_config: dict[str, object]
    runs: Mapping[int, LoadedCandidateRun]
    analysis: dict[str, object]

def load_evidence_set(root: str | Path) -> LoadedEvidenceSet: ...
```

- [ ] **Step 1: Shared Run Core resolution RED**

Patch/record PR1 `resolve_candidate_run_spec` inside EvidenceSet generation and prove every seed is resolved through the shared Run Core; CEL must not resolve feature/symbol/timestamps independently. Study creation pre-resolution is tested in Task 6 where `create_study()` exists.

- [ ] **Step 2: Multi-seed RED**

For Study seeds `(2, 5, 9)`, patch Run Core execution/publication to record configs and assert exactly three runs are generated in order and:

```python
assert [config.ppo_seed for config in seen] == [2, 5, 9]
assert all(other_fields_equal(seen[0], item) for item in seen[1:])
```

- [ ] **Step 3: Dataset/provenance mismatch RED**

Inspect dataset artifact and current `build_candidate_run_provenance()` before any seed execution; mismatch with StudyPlan raises `ArtifactIntegrityError` and the target EvidenceSet directory remains absent.

- [ ] **Step 4: Deterministic seed-invariance RED**

Build fake candidate artifacts for two seeds where `trend` raw returns differ by one value; expect `ArtifactIntegrityError` and no final EvidenceSet. Repeat with PPO-only difference and require success.

- [ ] **Step 5: Partial failure RED**

Make seed 2 succeed and seed 5 raise. Assert no final EvidenceSet target exists and subsequent retry can start cleanly after stale staging cleanup under lock.

- [ ] **Step 6: Implement EvidenceSet execution**

Use `CandidateRunConfig` cloning with only `ppo_seed` replaced. Resolve each seed through the shared Run Core, execute, build provenance, publish and inspect. Compare deterministic strategy metrics/diagnostics/raw arrays semantically across all seeds.

- [ ] **Step 7: Run GREEN**

```bash
uv run pytest -q tests/evaluation/experiments/test_evidence.py tests/evaluation/test_candidate_run_config.py tests/evaluation/test_candidate_run_artifact_identity.py
uv run mypy trade_rl/evaluation/experiments/evidence.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/evaluation/experiments/evidence.py tests/evaluation/experiments/test_evidence.py
git commit -m "feat: execute multi-seed study evidence"
```

---

### Task 4: Implement within-suite and baseline-vs-candidate analysis

**Files:**
- Create: `trade_rl/evaluation/experiments/analysis.py`
- Create: `tests/evaluation/experiments/test_analysis.py`

**Interfaces:**
- Consumes loaded PR1 candidate artifacts, `compare_paired_returns`, `summarize_seed_robustness`.
- Produces JSON-compatible deterministic analysis payload + digest.

```python
def analyze_evidence_set(
    loaded_runs: Mapping[int, LoadedCandidateRun],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]: ...


def compare_evidence_sets(
    baseline_runs: Mapping[int, LoadedCandidateRun],
    candidate_runs: Mapping[int, LoadedCandidateRun],
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]: ...
```

- [ ] **Step 1: Exact matrix RED**

Create small synthetic loaded artifacts with two symbols. Assert missing, duplicate, or extra `symbol × strategy` cells fail closed.

- [ ] **Step 2: Deterministic paired RED**

For `trend` vs `cash`, call `compare_paired_returns` independently in the test and assert the analysis payload matches all returned fields exactly.

Repeat for deterministic candidate-vs-candidate pair `trend` vs `ridge24`.

- [ ] **Step 3: PPO seed robustness RED**

For two PPO seed return series and deterministic `ridge24`, independently call `summarize_seed_robustness()` and assert payload equality, including worst seed, median return/difference, CI, p-value, block size.

- [ ] **Step 4: PPO factor-effect seed-matched RED**

Baseline/candidate use identical seeds `(0, 1)`. Independently compute `compare_paired_returns(candidate_ppo_seed_s, baseline_ppo_seed_s)` for each seed and assert all seed-wise evidence plus `positive_seed_count`, `negative_seed_count`, `median_excess_total_return`, `worst_excess_total_return`.

No combined p-value is allowed in this aggregate section.

- [ ] **Step 5: Cross-symbol descriptive RED**

Assert positive/negative counts, median/worst/best excess, median total return, worst drawdown, median turnover/cost are calculated from per-symbol evidence and no aggregate significance field exists.

- [ ] **Step 6: Implement analysis using existing primitives**

Do not duplicate bootstrap formulas. Derive `ReturnSeries` metadata from candidate artifact metrics (`return_kind`, `periods_per_year`) and raw arrays.

- [ ] **Step 7: Run GREEN**

```bash
uv run pytest -q tests/evaluation/experiments/test_analysis.py tests/evaluation/test_comparisons.py tests/evaluation/test_seed_robustness.py
uv run mypy trade_rl/evaluation/experiments/analysis.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/evaluation/experiments/analysis.py tests/evaluation/experiments/test_analysis.py
git commit -m "feat: add controlled experiment evidence analysis"
```

---

### Task 5: Implement controlled-factor resolved delta verification

**Files:**
- Create: `trade_rl/evaluation/experiments/delta.py`
- Create: `tests/evaluation/experiments/test_delta.py`
- Create: `tests/property/test_controlled_experiment_delta_properties.py`

**Interfaces:**
- Produces factor registry and verification payload:

```python
@dataclass(frozen=True, slots=True)
class FactorRule:
    allowed_paths: frozenset[tuple[str, ...]]
    unaffected_strategies: frozenset[str]

FACTOR_RULES: Mapping[ControlledFactor, FactorRule]


def verify_controlled_delta(
    *,
    plan: StudyPlan,
    definition: ExperimentDefinition,
    baseline: LoadedEvidenceSet,
    candidate: LoadedEvidenceSet,
) -> ControlledVerification: ...
```

- [ ] **Step 1: Factor mapping RED**

For every factor in the spec, construct baseline/candidate resolved config payloads and assert only its exact allowed paths are accepted. Add one extra unrelated changed path and require INVALID/uncontrolled result.

- [ ] **Step 2: No-op RED**

Same resolved payload under a declared factor must not become CONTROLLED.

- [ ] **Step 3: Study fixed-field RED**

Parameterize dataset digest, evaluation window, fit cutoff, initial capital, seed policy, implementation digest, runtime digest, strategy roster. Any drift prevents CONTROLLED.

- [ ] **Step 4: Unaffected-strategy RED**

For each factor, mutate one raw return in one required-unaffected strategy and assert violation. Mutate an affected strategy and prove this oracle alone does not reject it.

- [ ] **Step 5: Property test delta partition**

Using Hypothesis over small canonical dict trees, generate an allowed path change plus zero/one forbidden path. Oracle:

```python
assert controlled is (has_allowed_change and not has_forbidden_change)
```

with Study fixed fields held equal.

- [ ] **Step 6: Implement verification**

Compare resolved semantic payloads, not requested JSON strings. `UncontrolledDeltaError` may be private/internal; workflow later converts trustworthy violations to immutable INVALID verification.

- [ ] **Step 7: Run GREEN**

```bash
uv run pytest -q tests/evaluation/experiments/test_delta.py tests/property/test_controlled_experiment_delta_properties.py
uv run mypy trade_rl/evaluation/experiments/delta.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/evaluation/experiments/delta.py tests/evaluation/experiments/test_delta.py tests/property/test_controlled_experiment_delta_properties.py
git commit -m "feat: verify controlled experiment deltas"
```

---

### Task 6: Implement append-only Study workflow, decisions, lineage, and budget

**Files:**
- Create: `trade_rl/evaluation/experiments/workflow.py`
- Modify: `trade_rl/evaluation/experiments/__init__.py`
- Create: `tests/evaluation/experiments/test_workflow.py`
- Create: `tests/evaluation/experiments/test_lineage.py`

**Interfaces:**
- Produces high-level API:

```python
def create_study(... ) -> StudySnapshot: ...
def run_baseline(... ) -> StudySnapshot: ...
def define_experiment(... ) -> ExperimentDefinition: ...
def run_experiment(... ) -> EvidenceSet: ...
def verify_experiment(... ) -> ControlledVerification: ...
def compare_experiment(... ) -> ExperimentComparison: ...
def decide_experiment(... ) -> ExperimentDecision: ...
def inspect_study(root: str | Path) -> StudySnapshot: ...
```

- [ ] **Step 1: Create/baseline state RED**

Assert new Study contains only `plan.json` + lock housekeeping before baseline. `run_baseline` requires exact dataset/provenance identity and publishes baseline EvidenceSet once. Second baseline call cannot overwrite.

- [ ] **Step 2: Definition-before-execution RED**

`run_experiment(0001)` before `define_experiment` raises `InvalidExperimentStateError`. Definition publishes before candidate evidence and consumes attempt immediately.

- [ ] **Step 3: Sequence/budget RED**

With max=2: define 0001 and 0002, reject 0003 with `ExperimentBudgetExceededError`. Directory numbering is contiguous four-digit. Manual gap/tamper makes inspection/freeze fail.

- [ ] **Step 4: INVALID state RED**

Patch `verify_controlled_delta` to return a trustworthy uncontrolled result. Assert `verification.json` status INVALID is published, comparison/decision are forbidden, and the attempt remains counted.

Operational artifact integrity failure must raise and leave verification unpublished.

- [ ] **Step 5: Decision ordering RED**

Reject compare before CONTROLLED verification; reject decide before comparison; reject second decision. Accept exactly ACCEPT/KEEP/INCONCLUSIVE.

- [ ] **Step 6: Lineage RED**

After ACCEPT on 0001, its candidate EvidenceSet may baseline 0002. After KEEP/INCONCLUSIVE/INVALID, candidate fingerprint cannot be used as later baseline.

- [ ] **Step 7: Implement workflow**

Every mutating operation holds `StudyStore.mutation_lock()`, reconstructs state from disk, validates referenced digests, then publishes only the next immutable artifact.

- [ ] **Step 8: Run GREEN**

```bash
uv run pytest -q tests/evaluation/experiments/test_workflow.py tests/evaluation/experiments/test_lineage.py
uv run mypy trade_rl/evaluation/experiments/workflow.py
```

- [ ] **Step 9: Export public package API and commit**

```bash
git add trade_rl/evaluation/experiments tests/evaluation/experiments
git commit -m "feat: orchestrate append-only controlled studies"
```

---

### Task 7: Freeze, race/falsification tests, current docs, and final cleanup

**Files:**
- Modify: `trade_rl/evaluation/experiments/workflow.py`
- Create: `tests/evaluation/experiments/test_freeze.py`
- Extend: `tests/evaluation/experiments/test_store_concurrency.py`
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/research/current-status.md`
- Modify only if evidence contract merits it: `docs/architecture/lean-core.md`
- Delete at final verified tree: `docs/specs/2026-09-09-controlled-experiment-loop-v1-design.md`
- Delete at final verified tree: `docs/plans/2026-09-09-candidate-run-core-refactor.md`
- Delete at final verified tree: `docs/plans/2026-09-09-controlled-experiment-loop-v1.md`

**Interfaces:**
- Adds `freeze_study(...) -> StudyFreeze`.
- Durable docs become the current authority after active spec/plan deletion.

- [ ] **Step 1: Freeze RED**

Tests:

```python
freeze_study(..., outcome="WINNER", selected_strategy="cash", ...)
# -> ContractViolationError
```

WINNER requires ACCEPT-lineage EvidenceSet + one of five candidate strategies. NO_WINNER forbids selected strategy/fingerprint. Open nonterminal Experiment, sequence gap, broken digest, rejected candidate resurrection, existing freeze all reject.

- [ ] **Step 2: Post-freeze mutation RED**

After freeze, every mutating public API raises `StudyFrozenError`; `inspect_study` still works.

- [ ] **Step 3: Freeze/mutation race RED**

Two processes contend: one freezes while another tries to define/run. Oracle: exactly one mutation order is serialized; no state contains both an invalid post-freeze artifact and freeze. Reconstruct Study after processes exit.

- [ ] **Step 4: Tamper falsification RED**

After a valid controlled flow, change one byte in a seed `returns.npz`, provenance, analysis JSON, decision digest reference, or plan. `inspect_study` / next mutation / freeze must fail `ArtifactIntegrityError` before accepting derived state.

- [ ] **Step 5: Implement freeze/reconstruction**

Reconstruct all evidence from disk under lock and verify every digest/reference/terminal state/lineage before `freeze.json` publication.

- [ ] **Step 6: Update current docs**

`package-boundaries.md` must document `evaluation/experiments` ownership and dependency direction. `current-status.md` must state:

- CEL v1 infrastructure implemented;
- real-data M2 development Study still not run unless an actual canonical Study has been executed;
- candidate Run now has provenance evidence;
- multi-seed Study policy and controlled one-factor Experiment workflow;
- winner/no-winner freeze is development evidence only, not Production or M3 final proof.

Do not claim profitability or a winner without real Study evidence.

- [ ] **Step 7: Full quality gate**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Package identity:

```bash
expected="$(uv run python -c 'from importlib.metadata import version; print(version("trade-rl"))')"
module="$(uv run python -c 'import trade_rl; print(trade_rl.__version__)')"
test "$module" = "$expected"
```

- [ ] **Step 8: Requirements-first falsification review**

Attack at least:

- hidden second-factor delta;
- one missing PPO seed;
- lucky single-seed PPO evidence leak;
- deterministic candidate seed drift;
- external path/mtime changing identity;
- partial staging mistaken as final evidence;
- INVALID disappearing from budget;
- KEEP candidate entering lineage;
- control selected as winner;
- final/sealed-test import from experiments;
- concurrent sequence/freeze race;
- tampered digest reference accepted after restart.

Fix every in-scope Critical/High finding and rerun targeted then full gates.

- [ ] **Step 9: Reflect durable contracts then remove active docs**

Only after current docs contain all durable implemented contracts, delete this spec and both active plans. Verify `tests/architecture/test_current_docs_layout.py` passes and no `docs/history`/`docs/archive` is created.

- [ ] **Step 10: Final diff/self-review**

Review requirement compliance, public API, state transitions, error propagation, concurrency, artifact cleanup, dead/debug/temp files, backward compatibility. Confirm top-level `evaluation.__all__` was not expanded unless explicitly justified by a separate contract change.

- [ ] **Step 11: Commit final permanent changes**

```bash
git add -A
git commit -m "docs: finalize controlled experiment contracts"
```

- [ ] **Step 12: Exact-head CI gate**

Update/open Draft PR2 against PR1, ensure CI checks out exact final PR2 HEAD, and require Ruff / Format / Mypy / full tests / package identity success. Do not merge to main without explicit user authorization.


## Plan self-review result

- Spec coverage: every PR2 acceptance criterion maps to Tasks 1-7; Study creation pre-resolution is explicitly owned by Task 6 and multi-seed per-run resolution by Task 3.
- Placeholder scan: no TBD/TODO/"implement later" step remains.
- Type consistency: delta/workflow consume the `LoadedEvidenceSet` produced by Task 3; no undefined `EvidenceSetView` remains.
- Incremental architecture testing: each task can return its targeted layer to Green; Task 7 alone requires the complete final `evaluation/experiments` layout.
