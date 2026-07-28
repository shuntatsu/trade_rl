# Documentation Readability Refresh Design

## Goal

Make the current repository understandable from `README.md` without requiring readers to inspect historical plans, audit logs, or implementation notes.

## Source of truth

- `README.md` is the only repository-level overview and the canonical Japanese entry point.
- `START.md` is the executable first-run guide.
- `docs/README.md` is the documentation map and maintenance policy.
- `docs/ARCHITECTURE.md` documents the current implementation, not its development history.
- `docs/CONFIGURATION.md` documents `training_run_config_v2`, `observation_encoder`, the separated timeframe/asset attention settings, checkpoint architecture identity, TensorBoard diagnostics, and structured export/serving.
- `docs/RESEARCH_STATUS.md` documents empirical status and release gates.
- `docs/BINANCE.md`, `docs/operations/docker-gpu-full-training.md`, and `studio/README.md` remain focused operational guides.

## Deletion policy

The following are removed from the working tree because Git history already preserves them:

- `README.ja.md`;
- completed files under `docs/superpowers/plans`, `docs/superpowers/specs`, `docs/superpowers/status`, except this design and its implementation plan;
- dated files under `docs/verification`, `docs/audits`, `docs/reviews`, and `docs/plans`;
- `.superpowers/sdd/task-8-report.md`;
- `docs/MULTITIMEFRAME_RESEARCH.md`, whose maintained content moves into architecture and configuration documents.

Specialized maintained operational references remain:

- `docs/operations/causal-scenario-c3-execution.md`;
- `docs/performance/4070ti-super-full-training.md`.

## Information architecture

Readers should follow one of four paths:

1. First run: `README.md` → `START.md`.
2. Model or configuration work: `README.md` → `docs/CONFIGURATION.md` → `docs/ARCHITECTURE.md`.
3. Data and GPU execution: `README.md` → `docs/BINANCE.md` or the GPU operations guide.
4. Research interpretation: `README.md` → `docs/RESEARCH_STATUS.md`.

README stays concise and links outward instead of duplicating complete subsystem contracts.

## Current model contract

Documentation must describe only the maintained v2 contract:

- schema: `training_run_config_v2`;
- encoders: `flat_mlp`, `asset_set`, `hierarchical_sequence_v2`;
- sequence model: timeframe-specific causal TCNs, gated cross-timeframe attention, then gated cross-asset attention;
- separate timeframe and asset attention heads, layers, FFN multipliers, and gate biases;
- one architecture identity shared by BC, PPO, CostCriticPPO, and LagrangianPPO and verified by checkpoints and serving;
- TensorBoard diagnostics are observational and never selection evidence;
- structured sequence policies use structured export and a fail-closed structured serving loader.

Old Boolean encoder fields and `training_run_config_v1` are documented only as rejected legacy input.

## Verification

The documentation contract test is rewritten around maintained files. It verifies:

- README.ja.md and historical documentation trees are absent;
- required documents exist;
- current schema and encoder names are documented;
- rejected legacy settings do not appear as active examples;
- Import Linter order matches architecture documentation;
- internal Markdown links resolve across all maintained Markdown files;
- README length and headings remain bounded for readability.

A temporary inventory workflow is removed before merge.
