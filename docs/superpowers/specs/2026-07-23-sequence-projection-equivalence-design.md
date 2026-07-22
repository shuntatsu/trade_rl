# Sequence Projection Equivalence Stability Design

## Status

Approved continuation of the architecture remediation roadmap. This design addresses `AUD-CI-002` only.

## Problem

`tests/rl/test_sequence_policy_core.py::test_projection_after_selection_matches_legacy_outputs_and_gradients` compares the former implementation, which projects every timestep and selects afterward, against the maintained implementation, which selects the final available causal state before projection.

The test uses `float32` and requires every parameter-gradient element to match with `atol=1e-6`. One exact-head CI attempt failed with a maximum absolute difference of `4.112720489501953e-06`, while an unchanged rerun passed. The two computation graphs are mathematically equivalent but use different matrix shapes, so backend reduction order can introduce a few float32 ULPs of gradient noise.

This is a test-stability defect. No production output, causality, masking, or training defect has been reproduced.

## Goals

1. Preserve a direct test that selection-before-projection matches the historical selection-after-projection semantics.
2. Detect wrong availability indices, invalid-row handling, projection ordering, input-gradient routing, and parameter-gradient direction.
3. Avoid backend-sensitive per-element float32 gradient assertions as the sole invariant.
4. Keep the test deterministic on supported Linux and Windows CPU runners.
5. Change tests and verification documentation only.

## Non-goals

- Changing `CausalTimeframeEncoder` production code.
- Relaxing all Torch tests globally.
- Enabling nondeterministic algorithms.
- Hiding numerical drift with an unbounded or unexplained tolerance.
- Claiming GPU bitwise determinism.

## Approaches considered

### A. Increase `atol` only

Rejected. Raising the existing elementwise tolerance to exceed the observed failure would make CI pass but would weaken the only semantic invariant and would not explain which discrepancies are acceptable.

### B. Force deterministic algorithms only

Insufficient. `torch.set_num_threads(1)` is already used. Deterministic algorithm settings do not guarantee identical floating-point accumulation for two mathematically equivalent graphs with different matrix shapes.

### C. Split mathematical equivalence from float32 semantic gradients

Selected.

The test suite will use two complementary contracts:

1. A `float64` historical-equivalence test compares outputs, input gradients, and parameter gradients elementwise at a strict dtype-appropriate tolerance. Double precision makes the same reduction-order variation negligible while retaining the original graph comparison.
2. A `float32` semantic-gradient test checks finite gradients, near-identical direction by cosine similarity, bounded relative L2 error, zero output and zero input gradient for a fully unavailable row, and nonzero causal gradient flow for selected rows.

Neither test changes production computation.

## Float64 historical-equivalence contract

- Construct a small `CausalTimeframeEncoder` with dropout disabled.
- Convert the module and inputs to `torch.float64`.
- Use a fixed seed and one CPU thread.
- Compute the legacy path as `projection(forward_sequence(input))` followed by availability selection.
- Compute the maintained path through `encoder(input, available)` using a cloned input.
- Compare outputs, input gradients, and every named parameter gradient with `rtol=1e-9` and `atol=1e-10`.
- Verify the fully unavailable row is exactly zero.

The tolerance is tied to double precision and is materially stricter than the current float32 assertion.

## Float32 semantic-gradient contract

For each pair of corresponding gradients:

- both tensors must be finite;
- shape and dtype must match;
- cosine similarity must be at least `0.999999` when both norms are nonzero;
- relative L2 error `||a-b|| / max(||a||, ||b||, 1e-12)` must be at most `2e-5`;
- if both norms are zero, the pair is accepted exactly as a zero-gradient case.

Additional invariants:

- maintained and legacy outputs remain close with `rtol=1e-5`, `atol=2e-6`;
- the fully unavailable sample output is exactly zero;
- the fully unavailable sample input gradient is exactly zero;
- selected samples have finite, nonzero input gradients;
- all maintained parameters that receive a legacy gradient also receive a maintained gradient.

These bounds are substantially below model-scale drift and above the observed few-ULP backend noise.

## Test decomposition

The current flaky test will be replaced by:

- `test_projection_after_selection_matches_legacy_in_float64`
- `test_projection_after_selection_preserves_float32_gradient_semantics`

Private test helpers will own:

- encoder/input construction;
- historical forward path;
- named gradient capture;
- relative L2 and cosine checks.

The helpers remain in the test module; no production utility is introduced.

## Stability verification

A temporary exact-head workflow will run the two focused tests repeatedly on `ubuntu-latest` and `windows-latest`.

- At least 100 repetitions per operating system.
- Each repetition creates fresh seeded modules and inputs.
- The workflow stores the complete output as an artifact.
- The temporary workflow is removed before final verification.

The final branch must pass the normal full CI, PostgreSQL workflow, compatibility jobs, and training-image probe even though the effective change is test-only.

## Failure handling

- A failure of the float64 test indicates a genuine mathematical or autograd equivalence regression.
- A failure of cosine or relative L2 bounds indicates materially different float32 gradient semantics.
- A failure isolated to exact per-element float32 equality is no longer treated as a product regression.
- No automatic rerun is used to convert a failing final head into success; the repeated matrix provides explicit stability evidence.

## Safety boundary

- Production remains `NO-GO`.
- No model, policy, observation, reward, execution, selection, serving, or release behavior changes.
- No direct exchange routing is added.
- The PR remains Draft and is not merged without an explicit request.
