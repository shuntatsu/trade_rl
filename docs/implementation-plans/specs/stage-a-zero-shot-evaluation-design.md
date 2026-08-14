# Stage A Zero-Shot Evaluation Design

## Problem

Stage A training is symbol-disjoint, but a valid zero-shot claim also needs a fail-closed evaluation boundary. A result is invalid when a declared candidate, fold, seed, unseen-symbol triplet, checkpoint, shared baseline, dataset/feature/execution/evaluation identity, or execution-evidence digest is missing or inconsistent. Sealed-test results must never influence candidate selection.

The original v1 contract proved Cartesian closure and JSON integrity, but it left three unsafe assumptions outside the public API: callers could construct a self-consistent forged validation selection, candidates could use different baselines for the same market cell, and an opaque execution-evidence digest was not tied to the plan identities. Version 2 removes those assumptions instead of preserving compatibility with the flawed schema.

## Evaluation plan

`StageAZeroShotEvaluationPlan` binds:

- symbol-disjoint source and triplet manifest digests;
- candidate configuration, final training completion, policy, checkpoint, and seed identities;
- dataset, feature, execution, and evaluation identities;
- exact fold, seed, validation-triplet, and test-triplet sets;
- one-sided fold-bootstrap confidence, resample count, and seed;
- validation and test thresholds for the lower confidence bound, worst unseen triplet, worst seed, and non-negative-triplet pass fraction.

Bootstrap resamples are limited to `1,000,000`. The implementation generates index draws in bounded chunks, so the plan cannot request an unbounded two-dimensional allocation.

## Observation and evidence closure

Each `StageAEvaluationObservation` binds one candidate, split, unseen-symbol triplet, fold, seed, retained checkpoint, dataset/feature/execution/evaluation identities, policy execution-evidence digest, baseline execution-evidence digest, and paired policy/baseline log growth.

An evidence artifact is valid only when its candidate × triplet × fold × seed Cartesian product is complete and duplicate-free. For each `(triplet, fold, seed)` cell, every candidate must reference exactly the same baseline evidence digest and baseline log growth. This prevents candidate-dependent baseline substitution.

Consumption revalidates every observation against the exact plan. A later runner must obtain these values from the canonical execution artifact; the v2 contract ensures that the resulting observation cryptographically binds the source digest to all relevant identities.

## Statistical unit and robustness

For each candidate:

- seed and triplet observations are averaged within each fold;
- a deterministic one-sided bootstrap resamples fold means;
- every candidate in the same evidence artifact receives the same derived bootstrap draw seed;
- unseen-triplet means and seed means are retained separately;
- the worst unseen-triplet mean, worst seed mean, and fraction of unseen triplets with non-negative excess growth are explicit gate inputs.

Folds remain the market-history resampling unit, so seeds do not masquerade as independent histories. The additional robustness gates prevent a profitable subset of triplets or seeds from hiding a failed unseen-symbol subset.

## Validation selection

Validation evidence must contain every declared candidate. Eligibility requires all predeclared validation thresholds. Selection is deterministic, ordered by:

1. highest lower confidence bound;
2. highest worst-triplet excess growth;
3. highest worst-seed excess growth;
4. highest mean excess growth;
5. lexical candidate ID.

Serialized selections are not authority. Loaders recompute them from the bound plan and validation evidence.

## Sealed test

The sealed-test function requires both validation evidence and the supplied validation selection. It recomputes the expected selection and rejects any mismatch before reading test statistics. Test evidence must contain exactly the already selected candidate. The test gate applies its independently predeclared lower-bound and robustness thresholds and cannot change the candidate.

A directly instantiated dataclass is only a value object, not an authorization token. The supported consumers are the recomputing loader and gate functions.

## Artifact publication

All Stage A plan, evidence, selection, and decision writers use a fully flushed temporary file followed by `os.replace`. A failed replacement preserves the previous destination and removes the temporary file.

## Non-goals

This module does not execute checkpoints, build market datasets, or consume the one-shot sealed-test ledger. The Stage A runner must use the canonical policy loader and execution artifact, then construct these v2 contracts and consume the existing ledger only when sealed data is accessed.
