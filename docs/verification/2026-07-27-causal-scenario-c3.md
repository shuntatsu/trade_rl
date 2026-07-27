# Causal Scenario C3 Verification

## Status

Implementation verification is in progress on PR #196. This document records only evidence produced by exact repository heads. Production status remains `NO-GO`.

## Implemented scope

- immutable C3 configuration, persisted-decision, realized-outcome, comparison, fold-report, aggregate-report, and Phase A gate contracts;
- deterministic C1-to-C3 decision construction;
- atomic two-file decision artifacts with exact file closure, idempotent identical publication, and tamper detection;
- realized same-period comparison for Trend, selected Scenario Oracle residual, deterministic PPO mean residual, deterministic random-candidate comparator, and explicit Perfect-Information comparability;
- realized candidate ranking, selected regret, random regret, predicted-versus-realized Spearman correlation, economic cost fields, turnover, drawdown, fill count, pending-order events, and terminal equity;
- fold-local and aggregate reports with deterministic paired bootstrap evidence;
- exactly nine fail-closed Phase A entry conditions, including six folds, 180 selection days, drawdown, regret, ranking, Perfect-Information compatibility, and required adverse evidence;
- deterministic aggregate-report and gate artifacts;
- evaluation-only batch workflow and machine-readable gate CLI;
- import boundary preventing C3 from entering maintained training, Serving, release, promotion, or direct execution paths;
- public evaluation and causal-scenario workflow exports.

## Evidence policy

The following must be recorded after exact-head verification completes:

- final commit SHA;
- focused C3 test result;
- Ruff, format, Mypy, import architecture, dead-code, full test, coverage, critical coverage, and CLI smoke results;
- Ubuntu and Windows compatibility results;
- training-image and non-root runtime result;
- PostgreSQL Catalog result;
- PR mergeability and final changed-file closure.

A software-valid C3 implementation does not imply that the empirical Phase A gate passes. The gate can only be evaluated from real frozen six-fold evidence covering at least 180 selection days.
