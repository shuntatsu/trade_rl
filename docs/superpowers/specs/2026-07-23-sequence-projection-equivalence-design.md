# Sequence Projection Equivalence Stability Design

## Status

Approved remediation for `AUD-CI-002` on current `main` baseline `b39103aee26f9f80ab3b908fbf374c21ca1604a0`.

## Problem

`tests/rl/test_sequence_policy_core.py::test_projection_after_selection_matches_legacy_outputs_and_gradients` compares two mathematically equivalent implementations:

1. project every encoded timestep and select the last available output;
2. select the last available encoded timestep and project once.

The test uses float32 and requires every parameter-gradient element to match with `atol=1e-6`. A previous CI attempt failed once with a maximum difference of `4.112720489501953e-06`, while an unchanged rerun passed. The two graphs use different matrix shapes, so backend accumulation order can introduce a few float32 ULPs without changing selection, masking, output, or gradient semantics.

## Goals

- Preserve direct historical graph-equivalence coverage.
- Detect wrong availability selection, projection order, mask behavior, input-gradient routing, and parameter-gradient direction.
- Remove backend-sensitive float32 per-element equality as the sole invariant.
- Keep Linux and Windows CPU verification deterministic and bounded.
- Change tests, targeted CI, and documentation only.

## Non-goals

- Changing `CausalTimeframeEncoder` or another production module.
- Increasing global Torch tolerances.
- Claiming GPU bitwise determinism.
- Hiding numerical drift with an unexplained tolerance.

## Selected design

The old test is replaced by two complementary contracts.

### Float64 historical equivalence

A small dropout-free encoder and cloned inputs are converted to float64. The historical and maintained paths compare:

- outputs;
- input gradients;
- every named parameter gradient;
- exact zero output and input gradient for a fully unavailable row.

Tolerance:

```text
rtol = 1e-9
atol = 1e-10
```

This is stricter than the former float32 assertion and retains direct mathematical equivalence coverage.

### Float32 semantic-gradient equivalence

The float32 contract checks:

- outputs with `rtol=1e-5`, `atol=2e-6`;
- finite matching gradient shapes and dtypes;
- cosine similarity `>= 0.999999` for nonzero gradient pairs;
- relative L2 error `<= 2e-5`;
- exact zero output and input gradient for a fully unavailable row;
- nonzero causal input-gradient flow for selected rows;
- identical named parameter-gradient sets.

These limits remain far below meaningful model drift while covering the observed backend accumulation noise.

## Continuous protection

A permanent architecture test requires both stable test names and forbids restoration of the old flaky test name.

A targeted workflow runs only when the sequence encoder, its focused tests, the architecture contract, or the workflow itself changes. It repeats the focused contracts 10 times on Ubuntu and Windows. Unrelated pull requests do not incur this cost.

## Evidence strategy

- Preserve the earlier expected RED source-contract run and artifact.
- Preserve the earlier focused GREEN run.
- Preserve the 100-repetition Ubuntu and Windows matrix.
- Reapply the same reviewed patch to the current `main` baseline.
- Run exact-head normal CI, PostgreSQL verification, and the permanent targeted workflow.

## Safety boundary

- Production remains `NO-GO`.
- No model, policy, observation, reward, environment, execution, selection, serving, promotion, release, or artifact behavior changes.
- No direct exchange routing is added.
- The pull request remains Draft and unmerged.
