# Stage A Zero-Shot Evaluation Design

## Problem

Stage A training is now symbol-disjoint, but the repository still needs an immutable evaluation boundary that proves the selected shared policy generalizes to symbols that never appeared in training. A result is not valid if a fold, seed, triplet, checkpoint, dataset identity, or execution-evidence identity is missing, or if sealed-test results influence candidate selection.

## Goal

Produce content-addressed evaluation evidence for every declared candidate, fold, seed, and unseen-symbol triplet; aggregate paired excess log growth at the fold level; select candidates using validation evidence only; and allow sealed-test evaluation for the one selected candidate only.

## Contracts

### Evaluation plan

The plan binds the symbol-disjoint source and triplet manifests, training completion evidence, policy/config/checkpoint identities, dataset/feature/execution/evaluation identities, the complete seed and fold sets, validation and test triplet identities, bootstrap parameters, and explicit validation/test lower-bound thresholds.

### Observation and evidence closure

Each observation binds one candidate, split, unseen-symbol triplet, fold, seed, retained checkpoint, dataset identity, execution evidence, policy log growth, and baseline log growth. An evidence artifact is valid only when its Cartesian product is complete and duplicate-free. Loading or consuming evidence requires revalidation against the exact plan.

### Statistical unit

For each candidate and fold, excess log growth is averaged across every seed and unseen-symbol triplet. The deterministic paired bootstrap resamples those fold means, so seeds do not masquerade as independent market histories. The one-sided lower bound and all bootstrap parameters are part of the resulting identity.

### Validation selection

Validation evidence must contain every declared candidate. Candidates pass only when their lower confidence bound meets the predeclared validation threshold. Selection is deterministic: highest lower bound, then highest mean excess, then lexical candidate ID.

### Sealed test

A sealed-test decision requires a passed validation selection and test evidence containing exactly the selected candidate. Any additional candidate, changed checkpoint, missing fold/seed/triplet, or test evidence presented to the validation selector fails closed. The final test gate uses the independently predeclared test threshold and cannot change the selected candidate.

### Artifact self-consistency

Selection and sealed-test loaders do not trust serialized winners, pass flags, or reasons. They recompute the complete result from the bound plan and evidence and reject any artifact whose payload differs from that deterministic recomputation.

## Non-goals

This change does not run checkpoints, build market datasets, alter PPO, or open the existing sealed-test ledger. A later runner will produce these pure artifacts and use the existing one-shot ledger when it actually accesses sealed data.
