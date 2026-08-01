# Serving Publication Integrity Hardening Design

## Scope

Harden the selected-final training-run to serving-bundle publication boundary without changing policy behavior, training behavior, evaluation semantics, or the active Studio and Stage A implementation lanes.

## Problem

The training-run manifest binds every artifact path to a SHA-256 digest and byte size, but release packaging validates the run and later copies files with `shutil.copy2` without rechecking the manifest identity during the copy. A source file can therefore change between validation and publication. Sequence-policy packaging also accepts a missing structured loader and defers failure until runtime activation.

## Design

Introduce one focused verified-copy helper owned by `trade_rl.workflows.release_packaging` or a small adjacent private module. The helper consumes a `RunFile`, rejects symlinks and root escapes, verifies the source size and digest before copying, copies through an opened source handle into a temporary destination, fsyncs the destination, atomically renames it, and verifies the published destination against the same manifest entry. Any mismatch removes the staging bundle and fails closed.

For `SEQUENCE_OBSERVATION_SCHEMA`, publication requires `structured-policy-loader.json` to be present in the training manifest and staging directory. The loader manifest must decode successfully, match the ensemble architecture digest, and match the action size. Flat policies keep the current explicit-loader behavior.

## Error handling

All integrity failures raise `ValueError` with stable messages identifying source mutation, destination mismatch, or missing structured loader. The existing outer staging cleanup remains responsible for removing partial bundles. Existing immutable output-root behavior is unchanged.

## Tests

Add focused tests that fail on current main:

1. A sequence-policy training run without a structured loader is rejected during packaging.
2. A manifest-bound artifact changed after initial validation is rejected before bundle publication.
3. A verified copy produces a byte-identical, manifest-bound serving artifact.
4. Structured loader action-size mismatch is rejected during packaging.

Run focused release-packaging tests first, then Ruff, MyPy, Import Linter, critical coverage, and the complete Python suite through GitHub Actions.

## Non-goals

This change does not alter action semantics, serving bundle schema v6, release authorization, confirmation evidence, PostgreSQL ledgers, process supervision, or runtime provenance. Those remain separate reviewable changes.