# Walk-Forward E2E Design

## Goal

Provide one concrete, deterministic workflow that executes nested walk-forward research from typed fold boundaries through candidate training, checkpoint validation, configuration selection, sealed outer-OOS evaluation, chronological stitching, and gate-ready evidence.

## Scope

This slice does not add a production data downloader or claim profitable performance. It consumes validated data through injected policy-training and evaluation adapters. The workflow itself owns chronology, identity checks, selection isolation, and sealed-test discipline.

## Contracts

1. Each fold trains candidates only on the fold train range.
2. Checkpoint validation may select a checkpoint within a candidate configuration but may not select among configurations.
3. Configuration selection may compare frozen candidate configurations but may not access the outer test range.
4. The outer test range is evaluated exactly once for the selected configuration and identity baseline.
5. Baseline fallback is explicit when no residual candidate satisfies the predeclared selection rule.
6. Every fold result binds dataset identity, fold boundaries, selected configuration, policy digest, and evaluation digest.
7. Stitched OOS evidence is chronological, non-overlapping, and gate-ready.
8. Adapters receive immutable range-scoped requests; the workflow never hands them unrestricted fold data.

## Architecture

- `trade_rl/workflows/fold_runner.py`: concrete adapter-driven fold executor and immutable request/result records.
- `trade_rl/workflows/walk_forward.py`: orchestration over all folds and chronological selected/baseline stitching.
- `trade_rl/domain/evaluation.py`: final evaluation identity carried by Gate decisions.
- `trade_rl/domain/selection.py`: selection records remain the authoritative selected-mode identity.
- `trade-rl walk-forward plan`: validates pure fold boundaries; actual execution is exposed as a typed Python workflow so project-specific data and model adapters remain outside the domain layer.

## Selection rule

A residual configuration is selectable only when its selection score is finite and strictly above the baseline score by a configured minimum uplift. Ties are resolved deterministically by configuration name. If no candidate clears the threshold, select `baseline_only`.

## Gate identity

Configuration-selection evidence and final sealed-OOS evidence are distinct identities. A Gate binds the evaluated dataset, selected policy digest when a single policy is applicable, and the final evaluation digest. A Release records both the selection-evaluation digest and gate-evaluation digest and rejects dataset or policy mismatches.

## Non-goals

- Exchange/download integration
- Hyperparameter search generation
- Distributed training
- Production release activation
- Profitability claims
