# Sequence Projection CI Guard Design

## Context

Current `main` already contains the `AUD-CI-002` numerical-test remediation: strict float64 historical equivalence and float32 semantic-gradient equivalence. This follow-up protects those contracts without changing production code.

## Design

Add a dependency-light AST test that parses `tests/rl/test_sequence_policy_core.py`. It requires these tests:

- `test_projection_after_selection_matches_legacy_in_float64`
- `test_projection_after_selection_preserves_float32_gradient_semantics`

It forbids restoration of:

- `test_projection_after_selection_matches_legacy_outputs_and_gradients`

Add a targeted GitHub Actions workflow for Ubuntu and Windows. It runs only when the sequence encoder, focused test module, AST contract, or workflow changes. Each operating system repeats the three focused contracts 10 times and stops on the first failure. Matrix fail-fast is disabled so both platform results remain visible.

The workflow uses read-only contents permission, exact-head checkout, pinned external Actions, per-ref concurrency cancellation, and diagnostic artifacts.

## Scope

Changed responsibilities:

- `tests/architecture/test_sequence_projection_stability.py`: stable test-name contract.
- `.github/workflows/sequence-projection-stability.yml`: narrowly triggered cross-platform repetition.
- verification documentation: immutable evidence and final run IDs.

No file under `trade_rl/` changes. The normal full CI remains authoritative; the new workflow is an additional focused guard rather than a replacement.

## Safety boundary

Production remains `NO-GO`. No model, training, serving, execution, selection, promotion, release, artifact, or exchange-routing behavior changes. The pull request remains Draft and unmerged.
