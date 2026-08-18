# Training Capability Audit Operations Boundary Design

## Status and objective

This design moves the maintained training-capability audit implementation out of `scripts/run_training_capability_audit.py` and into the package-owned `trade_rl.operations` layer, while preserving the existing workflow command, audit schema, numerical behavior, report digest semantics, and output layout.

The current script is application-grade code: it defines the deterministic Gym audit environment, configures and runs PPO/SAC/TD3/TQC probes, validates network architecture, exercises checkpoint/replay resume, exports actors, constructs a synthetic residual-market dataset, runs residual-control and sequence-training probes, assembles `full_training_capability_audit_v1`, computes its digest, and writes `audit-report.json`. Keeping that implementation under `scripts/` leaves most of the audit logic outside package-level type/import/dead-code ownership checks and makes the script itself a de facto application layer.

The repository already has a maintained `trade_rl.operations` boundary for GPU smoke tests, benchmarks, preflight checks, and training monitoring. This change aligns the full training capability audit with that existing ownership model.

## Current integration context

- Base branch: `main` at `d27b4ecf35ec1ccf1647425640ca9936df1054f3`.
- PR #415 changes only walk-forward ownership and does not touch this audit surface.
- Open PRs #410, #411, and #414 do not touch `run_training_capability_audit.py` or `trade_rl.operations` audit files.
- `.github/workflows/full-training-capability-audit.yml` currently invokes `uv run python scripts/run_training_capability_audit.py --output var/training-capability-audit` and independently verifies the generated `full_training_capability_audit_v1` schema.

## Approaches considered

### A. Leave the implementation in `scripts/` and add more tests

This preserves the current layout but does not fix the ownership problem. The heavy audit implementation would remain outside the package boundary and could continue accumulating model, environment, and report logic in a script entry point.

Rejected.

### B. Move all audit logic into one public `trade_rl.operations.training_capability_audit` module

This would put the implementation under package-level gates, but the module would remain large and expose many implementation details that are only useful to the audit itself.

Acceptable but not preferred.

### C. Add a thin public operations facade plus a private implementation module

Add:

- `trade_rl/operations/training_capability_audit.py` as the public, stable operations API;
- `trade_rl/operations/_training_capability_audit_impl.py` as the implementation owner;
- keep `scripts/run_training_capability_audit.py` as an argparse/printing adapter only.

Selected. This follows the existing operations pattern used by GPU training smoke code, keeps the public surface small, and puts all application-grade audit behavior under package-level static and architecture gates.

## Quality contract

### Objective

Make `trade_rl.operations` the owner of training-capability audit behavior while preserving the existing externally observable audit contract.

### Non-goals

- no change to PPO, SAC, TD3, or TQC audit hyperparameters;
- no change to audit environment dynamics or synthetic market construction;
- no change to residual-action, alpha/factor, sequence, replay-resume, checkpoint-resume, or actor-export semantics;
- no change to `full_training_capability_audit_v1` fields, digest construction, or `audit-report.json` location;
- no change to the workflow invocation command or its independent result-schema assertions;
- no change to production training, serving, reward, risk, execution, selection, or research behavior;
- no general `scripts/` reorganization in this PR.

### Acceptance criteria

1. `trade_rl.operations.training_capability_audit` exposes the maintained package API used to execute the full audit.
2. The heavy implementation is owned by `trade_rl.operations._training_capability_audit_impl`, not by the script adapter.
3. `scripts/run_training_capability_audit.py` owns only argument parsing, invoking the package API, compact sorted-JSON stdout rendering, and process exit behavior.
4. The script no longer directly imports Gymnasium, NumPy, Torch, `MarketDataset`, SB3 training adapters, RL environment/action/export implementations, execution-cost models, or trend-strategy implementations.
5. The existing workflow command remains valid without changing `.github/workflows/full-training-capability-audit.yml` unless a contract test proves a change is required.
6. The generated report remains schema `full_training_capability_audit_v1` with the same field names and deterministic digest construction.
7. The report remains written to `<output>/audit-report.json`; a pre-existing output root is recursively removed and recreated as it is today. The persisted file remains `json.dumps(report, indent=2, sort_keys=True) + "\n"`.
8. Script stdout remains `print(json.dumps(report, sort_keys=True))`, preserving the existing compact sorted JSON plus print newline.
9. Existing audit probes continue to verify requested timesteps, policy publication, checkpoint publication, expected actor/critic widths, replay artifacts for off-policy algorithms, PPO checkpoint resume, SAC replay resume, deterministic actor export, residual controls, and sequence training.
10. Architecture tests prevent application-grade training implementation from returning to the script adapter.
11. Required repository quality gates pass on one exact final HEAD.

### Invariants

- `full_training_capability_audit_v1` is the schema authority for this audit.
- Audit report identity is derived from the same report payload fields as before.
- The audit remains a diagnostic/verification operation and does not become a training or promotion API used by maintained research workflows.
- Production training modules do not import from `trade_rl.operations.training_capability_audit`.
- The script remains directly executable with the current workflow command.
- No audit-only fake environment or synthetic dataset becomes a general production fixture/API.

## Architecture

### 1. Public operations API

`trade_rl/operations/training_capability_audit.py` is a small facade. It exposes the minimum maintained API:

```python
def run_training_capability_audit(output_root: Path) -> dict[str, object]: ...
```

The facade delegates to the private implementation. It does not re-export internal audit environment classes, model-loading helpers, synthetic providers, or per-probe functions.

The function name intentionally describes the operation rather than preserving the script-local generic `run_audit` name as a new public package contract.

### 2. Private implementation owner

`trade_rl/operations/_training_capability_audit_impl.py` receives the existing implementation with behavior-preserving changes only. It owns:

- `AuditEnv`;
- architecture inspection helpers;
- algorithm-specific audit configs;
- PPO/SAC/TD3/TQC training probes;
- PPO checkpoint resume and SAC replay resume;
- policy export verification;
- synthetic `MarketDataset` construction;
- audit alpha/factor providers;
- residual-control training verification;
- sequence-training verification;
- report assembly, digest computation, and `audit-report.json` persistence.

Implementation-only symbols remain private to the package. No new generic abstractions are introduced solely for hypothetical reuse.

### 3. Script adapter

`scripts/run_training_capability_audit.py` becomes a thin executable adapter:

1. parse `--output` with the same `Path("var/training-capability-audit")` default;
2. call `run_training_capability_audit(output_root)`;
3. execute `print(json.dumps(report, sort_keys=True))`;
4. return `0`.

The adapter must not know how individual algorithms are configured or trained.

### 4. Workflow boundary

`.github/workflows/full-training-capability-audit.yml` should continue invoking the script path. The workflow remains the end-to-end executable contract, while package tests can now directly exercise the operations API without importing a script module.

The workflow's separate schema validation remains valuable as an independent Test Oracle and should not be folded into the implementation under test.

## Data flow

```text
GitHub Actions / operator
        |
        v
scripts/run_training_capability_audit.py
        |
        v
trade_rl.operations.training_capability_audit
        |
        v
trade_rl.operations._training_capability_audit_impl
        |
        +--> training backends / environments / exports
        |
        +--> audit-report.json
        |
        +--> deterministic report payload + digest
        v
script prints compact sorted JSON
```

The dependency direction is one-way: script -> operations facade -> private implementation -> lower training/data/integration layers. No lower layer may import the operations audit.

## Failure modes

- Script still imports model/environment implementation modules after extraction: architecture test fails.
- Facade leaks private audit classes/functions: public-surface contract test fails.
- Report schema or keys drift during move: contract/behavior test fails.
- Digest construction accidentally includes/excludes a field: golden-equivalence or deterministic payload test fails.
- Persisted JSON or stdout serialization changes during extraction: exact serialization contract fails.
- Output-root replacement behavior changes: filesystem behavior test fails.
- One algorithm silently stops short, misses a policy/checkpoint/replay artifact, or changes architecture: existing audit runtime checks continue to fail closed.
- Resume/export/residual/sequence probe is accidentally omitted during extraction: report field/status contract tests fail.
- Workflow invocation breaks because script CLI changes: workflow contract and capability-audit execution fail.
- Circular dependency from lower training layers into operations: Import Linter/architecture tests fail.
- Moving code causes type/import/dead-code defects that were previously hidden outside the package: Mypy, Import Linter, Ruff, or vulture fail and must be fixed without weakening gates.

## Test oracle

Correctness is determined by observable contracts, not by successful import alone:

- script AST/import graph shows only CLI/serialization plus the public operations API, not training implementation dependencies;
- direct package API writes the required `audit-report.json` and returns the report payload;
- persisted report parses to exactly the returned payload and its bytes match the existing indented/sorted/newline representation;
- script stdout parses to exactly the returned payload and its text matches the existing compact sorted representation;
- schema remains `full_training_capability_audit_v1`;
- digest remains a valid deterministic content digest of the report's pre-digest payload;
- all expected report sections remain present and status-bearing;
- workflow's existing independent schema assertions continue to pass;
- runtime probes continue validating training artifacts, architecture, resume behavior, exports, residual controls, and sequence behavior.

## Required test layers

### Unit / contract

- public operations facade delegation and public-surface contract;
- report schema/digest/output path and exact serialization contract;
- script thin-adapter import/AST contract;
- script argument/default/stdout behavior with package API substituted at the public boundary.

### Integration

- targeted execution of the package audit with the existing complete audit dependencies;
- existing workflow's result-schema verification;
- resume/export/artifact side effects observed through the final report and filesystem.

### Static / architecture

- Ruff;
- Ruff format check;
- Mypy across the package;
- Import Linter;
- vulture/dead-code check;
- architecture tests preventing implementation ownership from returning to `scripts/`.

### Regression / packaging

- full pytest with branch coverage;
- critical coverage ratchet;
- Ubuntu/Windows compatibility where applicable;
- training image/package identity checks;
- Full training capability audit workflow on the exact final HEAD.

## TDD sequence

1. Add architecture/contract tests that require a package-owned audit API and a thin script adapter. They must fail against current `main` for the intended ownership reasons.
2. Add/adjust behavioral tests that pin the existing report schema, output file, digest relationship, exact serialization, and script delegation contract.
3. Move the implementation into the private operations module without semantic changes.
4. Add the public operations facade and reduce the script to a thin adapter.
5. Run the smallest targeted tests to GREEN.
6. Run static/type/import/dead-code gates and fix only extraction-related issues.
7. Run the full repository test/coverage gates.
8. Run/verify the Full training capability audit workflow and standard CI on the same final HEAD.
9. Re-review the diff from the original acceptance criteria and attempt to falsify ownership, report identity, missing probe, and workflow-compatibility assumptions.

## Quality gate

The change is not complete unless all of the following are true on the same final HEAD:

- acceptance criteria are represented by tests or independent structural evidence;
- valid RED evidence exists before the implementation move;
- targeted audit and architecture tests are Green;
- the script has no application-grade training implementation dependencies;
- report schema/digest/output/stdout contracts remain unchanged;
- Ruff and format checks pass;
- Mypy passes;
- Import Architecture passes;
- dead-code check passes;
- full pytest and branch-coverage gate pass;
- critical coverage ratchet passes;
- the Full training capability audit workflow executes successfully with its existing command and schema assertions;
- standard CI required jobs pass on the same exact final HEAD;
- final diff is reviewed for unrelated changes, debug code, generated artifacts, and accidental public API expansion;
- remaining unverified empirical conditions are disclosed.

## Falsification review targets

Before Ready status, explicitly attempt to prove the implementation wrong by checking:

- whether the script still contains or imports any algorithm/environment-specific implementation logic;
- whether a lower layer imports the new operations module;
- whether one report section was dropped or renamed while tests still pass;
- whether digest validation can pass when the persisted report and returned report differ;
- whether persisted file bytes or script stdout changed while JSON-equivalence tests still pass;
- whether the script's default `--output` changed;
- whether output-root replacement semantics changed;
- whether mocks hide failures in actual SB3 training, resume, replay, export, residual, or sequence probes;
- whether workflow success is being inferred from unit tests rather than the real workflow command;
- whether exact-final-HEAD CI evidence is stale after the last change.

## Remaining limitations

This refactor improves ownership, static coverage, and testability. It does not establish that the learning algorithms are economically useful, profitable, or production-ready. The capability audit remains a short deterministic implementation probe, not a full real-market research experiment or performance benchmark.
