# Causal Alpha V3 Signal Diagnostic Sidecar TDD State

Current phase: **exact-final-HEAD verification / independent review**. Production implementation is present, but the task is not complete until the exact final HEAD passes the required full CI and final review gates.

## TDD checkpoints

- Task 1 RED: the focused test failed because `causal_alpha_v3_weight_digest` / the diagnostic contract did not yet exist.
- Task 1 GREEN: ESS / weight-identity contract tests passed.
- Task 2 RED: the focused test failed because the paired Signal builder did not yet exist.
- Task 2 GREEN: canonical metric + sidecar paired computation tests passed.
- Task 3 RED: the strict diagnostic payload parser did not yet exist.
- Task 3 GREEN: strict codec and immutable sidecar store tests passed.
- Task 4 RED: pipeline tests failed because the pipeline still treated the paired build result as the old metric-only return value.
- Task 4 GREEN: paired resume states passed, including both-present reuse, neither-present build, one-side repair, and corrupt/stale fail-closed handling.

## Falsification and architecture review

The falsification suite checks that a sidecar with a self-consistent artifact digest but a forged `signal_metric_digest` is rejected, that the Signal Gate receives only exact `CausalAlphaV3SignalScopeMetric` instances, and that diagnostic fields do not leak into `signal/rejection.json`.

Architecture review found the first diagnostic module mixed artifact contracts and runtime construction in one large file. The implementation was split into contract/validation, builder, and strict codec modules without changing the canonical Gate calculation.

## Canonical metric oracle

A fixed cross-runner model/forecast digest was rejected as a test oracle after the exact same pre-sidecar commit and input produced different low-bit model/forecast/artifact digests on separate GitHub-hosted runners while all Gate observations remained identical. The design quality contract now explicitly records this pre-existing numerical-backend reproducibility boundary.

The maintained no-op oracle is an old-vs-new **same-runner** cross-tree test that compares the full canonical metric payload and artifact digest under the same dependency environment and fixed numerical thread settings. A completed comparison passed before the final documentation-only quality-contract refinement; the exact final HEAD must still be checked before completion.

## Current verification status

After the final type-only cleanup and formatter pass, a focused verification succeeded with:

- Ruff: passed;
- format check: passed;
- Mypy: `Success: no issues found in 485 source files`;
- targeted Signal regression suite: `41 passed`.

The Mypy fixes only disambiguate prediction/realized loop variable types and preserve the strict pre-validation of the diagnostic constant-mask payload. They do not change Signal calculations or parser rejection conditions.

The quality contract was then refined to require exact old-vs-new metric/digest equality within the same numerical execution environment, while treating cross-host low-bit ridge variation as a pre-existing residual risk that must fail closed during partial repair rather than overwrite evidence.

The next and final software verification gate is the full CI/check matrix plus the same-runner cross-tree oracle on the exact final HEAD, followed by final diff/status/PR review and independent acceptance-criteria/falsification review.

No CI result, targeted test result, or reviewer conclusion by itself is treated as completion. Final status requires the quality contract, full checks, final diff review, and residual-risk review to be satisfied on the same HEAD.
