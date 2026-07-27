# Causal Scenario C3 Core Design

## Goal

Implement the evaluation-only C3 core that measures whether the train-only causal scenario mechanism adds realized value over the maintained Trend baseline and whether the evidence is strong enough to permit a later Phase A teacher experiment.

Production remains `NO-GO`.

## Scope

This branch owns only the pure evaluation layer under `trade_rl.evaluation`:

- immutable C3 configuration, replay identity, persisted decision, realized outcome, query-comparison, and perfect-information contracts;
- deterministic persistence and reloading of the pre-replay decision artifact;
- predicted C1/C2 evidence reconstruction;
- same-period realized replay comparison for Trend, Scenario Oracle, deterministic PPO mean action, seeded random candidates, and optional Perfect-Information evidence;
- maintained adverse-scenario source validation and comparison;
- fold-local and aggregate evidence reports with paired moving-block inference;
- deterministic report artifacts;
- the pure Phase A entry gate.

The branch does not add a CLI command, walk-forward orchestration, training integration, Studio UI, Serving integration, promotion, release, or direct exchange routing.

## Trust boundaries

1. The selected scenario decision is created, atomically published, reloaded, and digest-verified before realized replay starts.
2. Realized returns, rankings, regret, and future bars never feed C1/C2 selection or policy fitting.
3. Every compared policy uses the same dataset, fold, environment, observation, action, execution, risk, AUM, initial-state, query, and realized-stop identity.
4. Every replay starts from an independent state clone.
5. Perfect-Information evidence is `not_comparable` unless its feasible set is proven to contain the evaluated policy set for the identical period and initial state.
6. Report loaders recompute aggregates from raw fold/query arrays and reject tampering, extra files, symlinks, dtype/shape drift, and digest mismatch.
7. The Phase A gate is a pure function over immutable aggregate evidence. Missing, non-finite, incompatible, or insufficient evidence fails closed.

## Gate

The gate passes only when all of the following hold:

1. no leakage, identity, replay, artifact, or determinism failure;
2. at least six independently reset folds and at least 180 selection days;
3. Scenario Oracle uplift over Trend is positive in at least four folds;
4. aggregate paired 95% lower bound for Scenario Oracle minus Trend daily log growth is strictly positive;
5. worst-fold drawdown is at most 20% and no more than two percentage points worse than Trend;
6. realized candidate selection beats the seeded random comparator with a strictly positive paired lower margin;
7. aggregate predicted-versus-realized Spearman correlation and its lower bound are strictly positive;
8. every asserted Perfect-Information comparison is compatible and ordered within tolerance;
9. nominal and all required adverse scenarios satisfy the declared uplift, cost, turnover, and drawdown limits.

## Dependencies

The core may depend on existing artifact hashing/canonical serialization, domain validation helpers, C1 value evidence, C2 frozen scenario evidence, stateful replay adapters, maintained paired moving-block bootstrap, NumPy, and Python standard-library modules. It must not depend on CLI, workflow orchestration, training, Serving, release, promotion, Studio, or exchange-routing modules.

## Verification

Focused C3 tests must pass first. The exact branch head must then pass Ruff, formatting, Mypy, import-linter, the complete pytest suite with branch coverage, critical-coverage checks, CLI smoke, Ubuntu and Windows compatibility, and the training-image job. No positive profitability or production claim may be made from software verification alone.
