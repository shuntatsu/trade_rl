# Warning Remediation and Branch Consolidation Design

## Objective

Remove the current Vite production-build chunk-size warning and the Python 3.12 multiprocessing/fork deprecation warnings without suppressing warnings or weakening tests, then keep `integration/cost-aware-causal-teacher-final2` as the single maintained integration branch containing all current work.

## Non-goals

- Do not merge to `main`.
- Do not force-push or rewrite history.
- Do not raise Vite's chunk-size warning limit merely to silence the warning.
- Do not filter, ignore, or reclassify the Python multiprocessing deprecation warning.
- Do not change trading, reward, risk, serving, causal, promotion, or sealed-evaluation semantics.
- Do not delete old Git refs unless an explicit safe delete-ref capability becomes available.

## Current evidence

At head `f2df9d30f596511c6311dc1c0e3599aa7fd1e9be`:

- Vite emits a warning for a `521.26 kB` minified main JavaScript chunk.
- `frontend/src/App.tsx` statically imports every workspace page, including `LiveTrainingPage`; that path statically imports `SynchronizedResearchChartWorkspace`, which imports `lightweight-charts`.
- The Python full suite emits 60 warnings, all summarized from `multiprocessing/popen_fork.py` in tests that exercise walk-forward/training paths.
- `trade_rl/workflows/_market_walk_forward_core.py::_collect_normalizer_matrix` explicitly selects `multiprocessing.get_context("fork")` for parallel normalizer collection.
- Python 3.12 documents `fork()` from a multi-threaded process as deprecated and recommends another multiprocessing start method such as `spawn` or `forkserver`.
- Vite documents dynamic imports as code-splitting boundaries and automatically optimizes preload dependencies for them.

## Design decision 1: workspace-level frontend code splitting

`DashboardPage` remains eager because it is the default landing workspace and needs the already-polled overview object immediately.

The non-default workspaces become lazy modules using `React.lazy` and are rendered under one stable `Suspense` boundary inside the existing `AppShell`:

- Data Lab
- Experiments
- Runs
- Live Training
- Compare
- Evidence
- Serving
- Settings

This creates semantic chunk boundaries at navigation boundaries instead of an arbitrary vendor split. In particular, `lightweight-charts` is no longer part of the initial Dashboard chunk and is loaded only when a workspace that needs the charting stack is requested.

The loading fallback remains inside the main workspace surface so the shell, navigation, status and viewport contract remain stable while an async page is loaded.

## Design decision 2: bundle-size contract

Add a build verification script under `frontend/scripts/` that inspects the completed `dist/assets/*.js` output and fails when any JavaScript chunk is greater than 500 KiB (`512000` bytes).

`npm run build` remains the actual Vite production build. A separate `check:bundle` script evaluates the output after build, and CI executes it immediately after the build as part of `Verify Studio frontend`.

This makes the previous warning a fail-closed quality gate rather than suppressing it. The contract is based on actual emitted files, not Vite warning text.

## Design decision 3: remove fork-only normalizer workers

The current parallel collector depends on a process-global `_NORMALIZER_ENVIRONMENT_FACTORY`, which works only because `fork` inherits live Python state. That prevents safe use of `forkserver` or `spawn` because the closure/environment factory is not picklable.

Replace that inheritance contract with an explicit serializable worker specification containing the minimum inputs needed to reconstruct the environment inside the child process. The worker constructs its own environment from this specification, collects its assigned interval, closes the environment, and returns a NumPy matrix.

Start-method policy:

- use `forkserver` when available;
- otherwise use `spawn`;
- never explicitly use `fork` for this path;
- retain the existing serial path when worker count is 1 or when finite-horizon semantics make partitioned collection invalid.

The parent and worker must produce the same ordered matrix as the existing serial collector for identical inputs.

## Acceptance Criteria

1. `npm run build --prefix frontend` emits no `Some chunks are larger than 500 kB` warning.
2. `npm run check:bundle --prefix frontend` succeeds and every emitted JavaScript chunk is `<= 512000` bytes.
3. Dashboard remains the default rendered workspace and workspace navigation still renders each requested page.
4. Existing fixed-viewport/layout checks remain green.
5. No production configuration increases `chunkSizeWarningLimit` above Vite's current 500 KiB threshold to hide the warning.
6. `_collect_normalizer_matrix` and its helper path no longer request the `fork` start method.
7. Parallel and serial normalizer collection produce equal ordered observations for the same deterministic test fixture.
8. The targeted warning-producing pytest set emits no `multiprocessing/popen_fork.py` deprecation warnings.
9. The full Python suite emits no multiprocessing/fork deprecation warnings. Any remaining warnings are enumerated separately rather than described as fixed.
10. Ruff, format, Mypy, Import Linter, frontend tests/typecheck/build/layout, Windows compatibility, Ubuntu compatibility, Training image, Nautilus Capability, PostgreSQL Catalog, full pytest and critical branch coverage pass on the same final head.
11. `codex/universal-real-data-training`, `integration/cost-aware-causal-teacher-final`, and `integration/cost-aware-causal-teacher-review` remain ancestors of the final `integration/cost-aware-causal-teacher-final2` head with no unique commits outside the maintained branch.
12. `main` remains unchanged and the PR remains unmerged until explicit user authorization.

## Invariants

- Trading/reward/risk/evaluation outputs are unchanged by frontend chunking.
- Normalizer row order and values are unchanged by multiprocessing start-method changes.
- Point-in-time data and sealed-evaluation contracts are unchanged.
- Finite-horizon normalizer collection remains serial where current semantics require it.
- Worker processes always close constructed environments.
- No warning is removed by pytest warning filters, `PYTHONWARNINGS`, Vite warning-limit inflation, or log filtering.

## Failure Modes

- Lazy page loading breaks tests because async modules are not awaited.
- Lazy page loading moves CSS or page initialization in a way that violates fixed viewport/layout behavior.
- Code splitting still leaves a vendor or entry chunk over 500 KiB.
- Bundle check passes because it inspects the wrong directory or ignores a chunk; therefore it must fail if `dist/assets` is missing and inspect every `.js` file.
- A worker specification is not picklable under `spawn`/`forkserver`.
- Reconstructed worker environments differ from the serial environment because a signal provider, dataset view, config, or action size is omitted.
- Parallel partitions overlap, leave gaps, or reorder observations.
- Child environments leak resources when a worker raises.
- Tests pass only because the parallel path is silently disabled; tests must assert the chosen start method and multi-partition execution independently from result parity.
- Branch consolidation is claimed from branch names rather than ancestry; compare results are the oracle.

## Test Oracle

Frontend:

- emitted production chunk file sizes;
- rendered heading/navigation after async workspace switch;
- existing viewport screenshot/layout scripts.

Python:

- exact normalizer matrix equality between serial and parallel collection;
- selected multiprocessing start method is never `fork`;
- warnings capture contains no fork deprecation warning;
- environment cleanup and exception propagation remain observable.

Git:

- compare status from each legacy work branch to final2 is `ahead` from the maintained branch perspective with `behind_by=0` for final2 relative to each old branch;
- main -> final2 remains `behind_by=0` for final2 relative to main;
- PR remains unmerged.

## Required Test Layers

- Frontend unit/component tests.
- Frontend production build and emitted bundle-size contract.
- Frontend fixed viewport/layout checks.
- Python unit/contract tests for start-method selection, picklability boundary, partition ordering and serial/parallel parity.
- Targeted integration tests that previously emitted fork warnings.
- Full pytest with branch coverage.
- Ruff, format, Mypy and Import Linter.
- Windows and Ubuntu compatibility.
- Training image packaged-runtime checks.
- Nautilus Capability and PostgreSQL Catalog workflows.
- Git ancestry/PR-state verification.

## Quality Gate

Do not mark the PR Ready or report completion unless all Acceptance Criteria have fresh evidence on the exact final head. A Green suite alone is insufficient: final diff, warning output, emitted bundle sizes, workflow results, ancestry, PR state, residual warnings and remaining risks must be reviewed explicitly.
