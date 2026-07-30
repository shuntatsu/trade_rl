# Stage A Symbol-Disjoint Training Design

## Problem

The existing 319-stage training cycle partitions all `C(15, 3) = 455` triplets into train, validation, and test triplet sets. That balances triplets, but every symbol appears in every split. It therefore cannot establish zero-shot generalization to unseen symbols, even when a separate symbol-disjoint manifest exists.

## Goal

Make the maintained Stage A training command consume only triplets formed from `SymbolDisjointManifest.train_symbols`. Validation and test symbols must never appear in a training stage, dataset binding, transfer checkpoint stage, or training-plan identity.

## Manifest boundary

A new `SymbolDisjointTripletManifest` is derived from one immutable `SymbolDisjointManifest`.

It contains:

- the source symbol-disjoint manifest digest;
- the complete source universe and universe digest;
- the exact train, validation, and test symbol sets;
- all three-symbol combinations formed independently inside each split;
- deterministic, balanced ordering within each split;
- content-addressed triplet, member, slot, schedule, and manifest identities.

For a 9/3/3 split, the counts are 84 train, 1 validation, and 1 test triplet. No cross-split combination is representable.

## Training-plan boundary

`SymbolTripletTrainingPlan` is generalized so its stages-per-cycle count is derived from the bound manifest instead of the legacy hard-coded 319 count. Its schema and digest remain content-addressed; existing 319-stage plans continue to validate against the legacy all-triplet manifest, while new Stage A plans validate against the symbol-disjoint triplet manifest.

The plan still repeats one exact immutable cycle, uses stable `SLOT0..SLOT2` runtime bindings, and preserves the current checkpoint-transfer and cursor contracts.

## Operator command

The maintained Binance symbol-triplet stage command loads `SymbolDisjointTripletManifest` only. Supplying the legacy all-symbol manifest fails closed before metadata, PostgreSQL, or training is touched. The full source universe remains available as provenance vocabulary, while every active stage selects only train-split symbols.

## Testing

TDD coverage proves:

- train/validation/test slot symbols are subsets of their respective disjoint symbol sets;
- the default 9/3/3 partition produces 84/1/1 triplets;
- each train symbol appears equally often in a complete cycle;
- JSON round trips and tampering is rejected;
- the training plan derives 84 stages per cycle and repeats exactly;
- legacy plan tests remain valid;
- the operator command rejects the legacy manifest and accepts the new manifest without touching external systems when the plan is complete.
