# Contract Hardening Status — 2026-07-31

## Implemented

- execution-policy identity includes all economic execution fields;
- multi-asset isolated margin is rejected until a per-symbol collateral ledger exists;
- execution promotion requires non-empty, content-bound order-event evidence;
- selected-final authorization binds the exact execution-evidence digest;
- checkpoint, replay-buffer, and structured TorchScript deserialization use private verified copies;
- symbol-triplet cursors bind the exact previous completion digest and advance under an exclusive compare-and-swap lock;
- artifact pointer publication distinguishes failure before replacement from durability failure after replacement;
- public lazy exports are checked against real module attributes.

## Local verification

- all Python sources and tests compile;
- focused contract, artifact, promotion, serving, and stage-cursor tests: 112 passed;
- four additional collection paths require the optional Gymnasium / Stable-Baselines3 environment provided by CI;
- canonical Ruff formatting was applied to every touched Python file;
- `git diff --check` passed.

## CI

The repository CI matrix is the final source of truth for formatting, Ruff, Mypy, Import Linter, dead-code reporting, full pytest/branch coverage, platform compatibility, container probes, and PostgreSQL integration. Automation-authored cleanup commits do not recursively start GitHub Actions, so final verification must run on a subsequent repository-authenticated head; exact results are recorded in the pull-request description before merge.
