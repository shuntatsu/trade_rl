# Agent Repository Control Plane Core Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a network-free repository-local control plane that gives Agents source-derived preflight, context, impact, semantic-diff, and risk-based verification guidance without creating a second domain authority.

**Architecture:** Move the existing static import-analysis primitive out of `tests/` into a repository-tooling owner under `tools/agent_repo/`, then build read-only Git/source inspection on top of it. Generated reports go only to stdout, `trade_rl` never depends on the tooling, and final verification remains the existing full CI/distribution gate.

**Tech Stack:** Python 3.12 stdlib (`ast`, `argparse`, `dataclasses`, `json`, `pathlib`, `subprocess`, `tomllib`), pytest, mypy, Ruff, existing GitHub Actions CI.

**Spec:** `docs/specs/2026-09-11-agent-repository-control-plane-v1-design.md`

## Global Constraints

- Do not change trading/runtime/economic behavior.
- Do not change persisted artifact schemas or identities.
- Do not add a permanent owner/type/schema registry.
- All context/diff/verification output is ephemeral; no generated report is committed.
- `trade_rl` must not import `tools.agent_repo`.
- `tools/agent_repo` must not be present in the installed production wheel.
- Development verification may be targeted; completion still requires the current full CI/distribution gate.
- Coverage branch target is 80% as a signal, not a standalone merge gate; do not add low-value tests to chase the number.
- Do not require network access from the local control plane.

---

## File map

Create:

```text
tools/__init__.py
tools/agent_repo/__init__.py
tools/agent_repo/__main__.py
tools/agent_repo/git_state.py
tools/agent_repo/source_index.py
tools/agent_repo/semantic_diff.py
tools/agent_repo/verification.py
tests/agent_repo/__init__.py
tests/agent_repo/test_git_state.py
tests/agent_repo/test_source_index.py
tests/agent_repo/test_cli.py
tests/agent_repo/test_semantic_diff.py
tests/agent_repo/test_verification.py
tests/architecture/test_agent_repo_tooling_boundary.py
```

Modify:

```text
tests/architecture/test_import_collector_contract.py
tests/architecture/test_lean_dependency_boundaries.py
.github/workflows/ci.yml
AGENTS.md
docs/AGENTS.md
docs/architecture/package-boundaries.md
pyproject.toml
```

Remove after all callers migrate:

```text
tests/architecture/imports.py
```

The existing production package tree remains unchanged.

---

### Task 1: Promote the static import collector to repository tooling

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/agent_repo/__init__.py`
- Create: `tools/agent_repo/source_index.py`
- Modify: `tests/architecture/test_import_collector_contract.py`
- Modify: `tests/architecture/test_lean_dependency_boundaries.py`
- Remove: `tests/architecture/imports.py`

**Interfaces:**
- Produces: `within_module(name: str, prefix: str) -> bool`
- Produces: `ImportCollector(package: Path)` with `collect_direct(path: Path) -> set[str]` and `collect(path: Path) -> set[str]`
- Constraint: source inspection must never import/execute `trade_rl` modules.

- [ ] **Step 1: Change one contract test to import the future owner and verify RED**

Replace the collector import in `tests/architecture/test_import_collector_contract.py` with:

```python
from tools.agent_repo.source_index import ImportCollector
```

Run:

```bash
uv run pytest -q tests/architecture/test_import_collector_contract.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tools.agent_repo'`.

- [ ] **Step 2: Create the new owner with the existing collector semantics unchanged**

Create `tools/__init__.py` and `tools/agent_repo/__init__.py` as package markers, then move the current contents of `tests/architecture/imports.py` into `tools/agent_repo/source_index.py` without changing the observable behavior of `within_module`, relative-import resolution, literal `__all__`, cycle handling, `collect_direct`, or `collect`.

- [ ] **Step 3: Migrate all architecture-tooling imports**

Update `tests/architecture/test_lean_dependency_boundaries.py` and any other `tests/**` caller discovered by:

```bash
git grep -n "tests\.architecture\.imports\|architecture/imports.py"
```

so they import the canonical repository-tooling owner:

```python
from tools.agent_repo.source_index import ImportCollector, within_module
```

- [ ] **Step 4: Run collector and dependency contracts**

```bash
uv run pytest -q \
  tests/architecture/test_import_collector_contract.py \
  tests/architecture/test_lean_dependency_boundaries.py
```

Expected: all pass with the same allowed/forbidden import behavior.

- [ ] **Step 5: Remove the old test-only owner and prove there is no forwarding shim**

Delete `tests/architecture/imports.py`, then run:

```bash
git grep -n "tests\.architecture\.imports" -- . ':!docs/plans/*'
uv run pytest -q tests/architecture/test_import_collector_contract.py tests/architecture/test_lean_dependency_boundaries.py
```

Expected: grep returns no current-code references; tests pass.

- [ ] **Step 6: Commit**

```bash
git add tools tests/architecture
 git commit -m "refactor: promote repository import inspector"
```

---

### Task 2: Add exact Git preflight facts

**Files:**
- Create: `tools/agent_repo/git_state.py`
- Create: `tests/agent_repo/__init__.py`
- Create: `tests/agent_repo/test_git_state.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class GitState:
    branch: str | None
    head: str
    base_ref: str | None
    merge_base: str | None
    dirty_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    active_docs: tuple[str, ...]
    workflow_paths: tuple[str, ...]


def read_git_state(repository: Path, *, base_ref: str | None = None) -> GitState: ...
```

- [ ] **Step 1: Write a failing temporary-repository test**

In `tests/agent_repo/test_git_state.py`, initialize a temporary Git repository, configure a local test identity, create `main`, commit `trade_rl/a.py`, `docs/specs/active.md` containing `Status: Active`, and `.github/workflows/ci.yml`, then create `feature`, modify `trade_rl/a.py`, and add an untracked file.

Assert:

```python
state = read_git_state(repo, base_ref="main")
assert state.branch == "feature"
assert len(state.head) == 40
assert state.merge_base is not None
assert state.dirty_paths == ("trade_rl/a.py",)
assert state.untracked_paths == ("scratch.txt",)
assert "trade_rl/a.py" in state.changed_paths
assert state.active_docs == ("docs/specs/active.md",)
assert state.workflow_paths == (".github/workflows/ci.yml",)
```

Run:

```bash
uv run pytest -q tests/agent_repo/test_git_state.py
```

Expected: import failure because `GitState/read_git_state` do not exist.

- [ ] **Step 2: Implement fail-closed Git command execution**

Use a single helper:

```python
def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
```

Use `symbolic-ref --short -q HEAD`, `rev-parse HEAD`, `merge-base`, `diff --name-only`, `status --porcelain=v1`, and `ls-files --others --exclude-standard`. Do not silently convert a failed Git invocation into an empty report.

- [ ] **Step 3: Add detached-HEAD and missing-base tests**

Verify detached HEAD produces `branch is None`, and an unknown explicit `base_ref` raises `subprocess.CalledProcessError` instead of claiming no changes.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/agent_repo/test_git_state.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/agent_repo/git_state.py tests/agent_repo
 git commit -m "feat: add repository preflight state"
```

---

### Task 3: Add source-derived context and impact inspection

**Files:**
- Modify: `tools/agent_repo/source_index.py`
- Create: `tests/agent_repo/test_source_index.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class PathContext:
    path: str
    module: str | None
    domain: str | None
    capability: str | None
    facade_module: str | None
    direct_imports: tuple[str, ...]
    semantic_imports: tuple[str, ...]
    direct_consumers: tuple[str, ...]
    semantic_consumers: tuple[str, ...]
    public_exports: tuple[str, ...]
    schema_constants: tuple[str, ...]
    test_candidates: tuple[str, ...]
    doc_references: tuple[str, ...]
    effect_signals: tuple[str, ...]


class SourceIndex:
    @classmethod
    def build(cls, repository: Path) -> "SourceIndex": ...
    def context(self, relative_path: str) -> PathContext: ...
    def impact(self, relative_paths: tuple[str, ...]) -> tuple[PathContext, ...]: ...
```

- [ ] **Step 1: Write a synthetic-tree RED test**

Create a temporary repository tree with:

```text
trade_rl/evaluation/runs/__init__.py
trade_rl/evaluation/runs/config.py
trade_rl/evaluation/experiments/workflow.py
tests/evaluation/test_candidate_run_config.py
docs/architecture/package-boundaries.md
```

Make the facade statically re-export `CandidateRunConfig`, and make `experiments/workflow.py` import it through the facade. Assert context for `runs/config.py` reports domain `evaluation`, capability `runs`, facade `trade_rl.evaluation.runs`, the workflow as a consumer, the nearby test as a candidate, and the architecture doc as a textual reference.

- [ ] **Step 2: Implement physical-tree indexing only**

Build the module map from `trade_rl/**/*.py`, reuse `ImportCollector`, and derive reverse edges by scanning current production source. Do not import production modules and do not store generated index files.

Capability derivation rule for v1:

```python
parts = PurePosixPath(relative_path).parts
# trade_rl/<domain>/<capability>/... -> domain/capability
# trade_rl/<domain>/<file>.py       -> domain with capability=None
```

The nearest `__init__.py` package is only called a facade when it has a statically resolvable public export; otherwise report `facade_module=None` rather than inventing ownership.

- [ ] **Step 3: Add public export, schema, and effect signal extraction**

Support only deterministic static signals:

```text
public_exports: literal __all__ names
schema_constants: module-level string constants whose name contains SCHEMA
effect_signals: call names matching known filesystem mutation or network primitives
```

Initial effect call names:

```python
FILESYSTEM_EFFECTS = {
    "write_text", "write_bytes", "replace", "rename", "unlink", "mkdir", "rmdir"
}
NETWORK_EFFECTS = {"urlopen"}
```

Label these as signals, not proof of runtime effects.

- [ ] **Step 4: Add false-positive tests**

Verify:

- a private helper class is not listed as a public export merely because it exists;
- a string containing `https://` without a network call does not become a network signal;
- a semantically unrelated test outside the same domain is not returned solely because its filename contains `config`.

- [ ] **Step 5: Run focused tests and current architecture contracts**

```bash
uv run pytest -q tests/agent_repo/test_source_index.py tests/architecture
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tools/agent_repo/source_index.py tests/agent_repo/test_source_index.py
 git commit -m "feat: add source context and impact inspection"
```

---

### Task 4: Add the network-free CLI surface

**Files:**
- Create: `tools/agent_repo/__main__.py`
- Create: `tests/agent_repo/test_cli.py`

**Interfaces:**

```bash
uv run python -m tools.agent_repo preflight [--base REF]
uv run python -m tools.agent_repo context PATH
uv run python -m tools.agent_repo impact PATH [PATH ...]
```

Each command prints deterministic pretty JSON to stdout and uses non-zero exit status on malformed input or inspection failure.

- [ ] **Step 1: Write CLI RED tests**

Call the module through `subprocess.run([sys.executable, "-m", "tools.agent_repo", ...])` inside a temporary Git repository. Parse stdout with `json.loads` and assert stable top-level keys rather than snapshotting whitespace.

- [ ] **Step 2: Implement argparse dispatch**

Use:

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ...
    return 0
```

At module bottom:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Serialize dataclasses with `dataclasses.asdict`, `sort_keys=True`, `indent=2`.

- [ ] **Step 3: Prove the CLI is read-only**

In a test, capture `git status --porcelain=v1` before and after each command and assert equality. Do not monkeypatch write methods; observe repository state.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/agent_repo/test_cli.py
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tools/agent_repo/__main__.py tests/agent_repo/test_cli.py
 git commit -m "feat: expose agent repository inspection CLI"
```

---

### Task 5: Add semantic diff review signals

**Files:**
- Create: `tools/agent_repo/semantic_diff.py`
- Create: `tests/agent_repo/test_semantic_diff.py`
- Modify: `tools/agent_repo/__main__.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True, order=True)
class SemanticSignal:
    kind: str
    change: str
    path: str
    name: str
    detail: str


def semantic_diff(repository: Path, *, base_ref: str) -> tuple[SemanticSignal, ...]: ...
```

CLI:

```bash
uv run python -m tools.agent_repo diff --base main
```

- [ ] **Step 1: Write a Git-backed RED fixture**

Commit a baseline containing a private helper, literal `__all__`, one schema constant, and one import. In the worktree add:

```python
@dataclass(frozen=True)
class NewConfig:
    value: int

NEW_SCHEMA = "new_schema_v1"
__all__ = ["Existing", "NewConfig"]
```

also add one new production import and one `Path(...).write_text(...)` call.

Assert exact signal kinds include:

```text
DATA_SHAPE
SCHEMA
PUBLIC_EXPORT
DEPENDENCY
FILESYSTEM_EFFECT
```

- [ ] **Step 2: Implement base/current source loading without checkout mutation**

For baseline content use `git show <base_ref>:<path>`. For current content use the worktree when present. Deleted files use baseline only; added files use current only. Parse Python with `ast.parse`; syntax errors fail closed.

- [ ] **Step 3: Implement intentionally narrow extractors**

Detect data-shape declarations only when one of these is statically visible:

- `@dataclass` decorated class;
- class base name ending in `Protocol`, `TypedDict`, `NamedTuple`, or `Enum`;
- literal `__all__` export changes;
- module-level string constant with `SCHEMA` in its name;
- direct production import edge changes from `ImportCollector.collect_direct`;
- effect-call candidate changes from Task 3.

Do not compare field similarity and do not emit `DUPLICATE_TYPE` automatically.

- [ ] **Step 4: Add false-positive tests**

Verify changing only a comment/docstring emits no data-shape/schema/dependency signal, and adding a private ordinary class emits no `PUBLIC_EXPORT` unless the facade actually exports it.

- [ ] **Step 5: Wire the `diff` CLI and run tests**

```bash
uv run pytest -q tests/agent_repo/test_semantic_diff.py tests/agent_repo/test_cli.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tools/agent_repo/semantic_diff.py tools/agent_repo/__main__.py tests/agent_repo
 git commit -m "feat: add semantic review diff"
```

---

### Task 6: Add risk-based verification planning without weakening final CI

**Files:**
- Create: `tools/agent_repo/verification.py`
- Create: `tests/agent_repo/test_verification.py`
- Modify: `tools/agent_repo/__main__.py`

**Interfaces:**

```python
@dataclass(frozen=True, slots=True, order=True)
class VerificationStep:
    tier: str  # fast | final | extended | signal
    command: str
    reason: str


def plan_verification(
    repository: Path,
    *,
    base_ref: str,
) -> tuple[VerificationStep, ...]: ...
```

CLI:

```bash
uv run python -m tools.agent_repo verify --base main
```

- [ ] **Step 1: Write table-driven RED tests**

Cover these cases:

```text
trade_rl/evaluation/runs/config.py changed
  -> nearby run tests + architecture/import checks in fast tier
  -> all final CI commands still present

network effect signal added
  -> integration/network/fallback review command in extended tier

artifact/schema signal added
  -> artifact tamper/compatibility guidance in extended tier

only docs changed
  -> docs/architecture targeted tests in fast tier
  -> full final gate still listed for completion
```

- [ ] **Step 2: Encode the existing final CI commands exactly**

The `final` tier must contain current permanent gates, including:

```text
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run pytest -q tests
uv build
```

and descriptive steps for the existing distribution-closure, clean-installed smoke, and package-identity shell blocks. The planner prints them; it does not execute them.

- [ ] **Step 3: Generate fast tests from actual context rather than a global hand-maintained test map**

Use `SourceIndex.context(...).test_candidates` for changed production paths. Add architecture tests when semantic signals touch imports/public exports/schemas. Deduplicate commands deterministically.

- [ ] **Step 4: Add conditional expensive checks conservatively**

Only suggest an extended check when a matching signal exists. Initial routing:

```text
NETWORK_EFFECT      -> relevant integration/network tests
SCHEMA              -> artifact/compatibility/tamper review
DEPENDENCY          -> architecture import tests
FILESYSTEM_EFFECT   -> atomicity/failure-injection review
optional-dep config -> installed optional capability smoke
```

Unknown high-impact changes get an explicit `extended` manual-review entry rather than a fabricated command.

- [ ] **Step 5: Add coverage as a non-blocking signal command**

When production Python changed, add tier `signal`:

```bash
uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-fail-under=0 -q tests
```

Reason text must say `target=80%; signal only; investigate important uncovered paths before adding tests`.

- [ ] **Step 6: Wire CLI and run focused tests**

```bash
uv run pytest -q tests/agent_repo/test_verification.py tests/agent_repo/test_cli.py
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add tools/agent_repo/verification.py tools/agent_repo/__main__.py tests/agent_repo
 git commit -m "feat: add risk based verification planner"
```

---

### Task 7: Enforce the repository-tooling boundary and type-check it

**Files:**
- Create: `tests/architecture/test_agent_repo_tooling_boundary.py`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`

**Interfaces:**
- `trade_rl/**` must not import `tools`.
- installed wheel must contain no Python source outside `trade_rl/**` (existing distribution checker already fails such a wheel).
- repository tooling is type-checked independently of production Mypy scope.

- [ ] **Step 1: Write architecture RED for a synthetic forbidden dependency**

Use `ImportCollector` on a temporary tree and verify a `trade_rl/example.py` containing `from tools.agent_repo import source_index` is classified as forbidden. Also scan the real production tree and assert no current offender.

- [ ] **Step 2: Add CI type-checking for tooling**

Change the architecture-tooling type step from the retired test helper to:

```yaml
- name: Repository tooling types
  run: uv run mypy tools/agent_repo tests/architecture/distribution.py
```

Do not add `tools` to the production `[tool.mypy].files = ["trade_rl"]` contract.

- [ ] **Step 3: Clarify the coverage target without turning it into PR blocking**

Keep the existing `fail_under = 80` value as the documented target signal, add a TOML comment immediately above it:

```toml
# Project target signal; permanent CI does not use this as a standalone merge gate.
fail_under = 80
```

The control-plane measurement command uses `--cov-fail-under=0` so measurement itself does not fail solely on the percentage.

- [ ] **Step 4: Run architecture/type checks**

```bash
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run pytest -q tests/agent_repo tests/architecture
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml tests/architecture tools
 git commit -m "test: enforce agent repository tooling boundary"
```

---

### Task 8: Document the durable usage contract and perform falsification

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/architecture/package-boundaries.md`

**Interfaces:**
- Agents may use the local inspector to reduce discovery cost, but it does not replace GitHub open-PR checks or final full verification.
- No generated report is committed.

- [ ] **Step 1: Add concise durable routing**

Add to root `AGENTS.md` after topology checks:

```text
Local repository tooling may be used to summarize preflight/context/impact and verification routing. Its output is source-derived guidance, not a replacement for source review, open-PR checks, or the final CI gate. Generated reports are not committed.
```

Add command examples to `docs/AGENTS.md` and a short `Repository tooling boundary` section to `docs/architecture/package-boundaries.md` stating that `tools/agent_repo` is development tooling outside the runtime package and must never be imported by `trade_rl`.

- [ ] **Step 2: Run docs/architecture tests**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py tests/architecture/test_agent_repo_tooling_boundary.py
```

Expected: pass.

- [ ] **Step 3: Falsification — intentionally create wrong changes and confirm tests kill them**

Perform each mutation separately, run the named test, then restore the file before the next mutation:

1. Make `trade_rl/__init__.py` import `tools.agent_repo` → `test_agent_repo_tooling_boundary.py` must fail.
2. Change semantic diff to treat every class as `PUBLIC_EXPORT` → false-positive test must fail.
3. Remove the full final pytest command from planner → verification contract test must fail.
4. Make `context` write a JSON report under the repository → CLI read-only test must fail.
5. Change `ImportCollector.collect_direct()` to follow semantic re-exports → existing import-collector direct-vs-semantic contract must fail.

After restoration:

```bash
git diff --check
git status --short
uv run pytest -q tests/agent_repo tests/architecture
```

Expected: no mutation remains; tests pass.

- [ ] **Step 4: Run final repository verification on one exact HEAD**

Run the same commands as permanent CI, including build, distribution closure, sdist-rebuilt wheel closure, clean-installed package/CLI smoke, and package identity. Also run the non-blocking coverage measurement once and record the observed percentage in the PR body without changing the pass/fail criterion.

- [ ] **Step 5: Commit durable docs**

```bash
git add AGENTS.md docs
 git commit -m "docs: route agent repository control plane"
```

- [ ] **Step 6: Final review**

Confirm:

```text
no generated context/diff/report files
no temporary workflow
no debug print
no trade_rl -> tools dependency
no runtime/public/artifact-schema change
all Acceptance Criteria for core tooling mapped to evidence
unverified dynamic/reflection limits explicitly reported
```
