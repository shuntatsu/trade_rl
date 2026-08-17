# Causal Alpha V3 Signal Diagnostic Sidecar TDD State

Current phase: **verification / falsification**. Production implementation is present, but the task is not complete until the exact final HEAD passes the required full CI and final review gates.

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

A fixed cross-runner model/forecast digest was rejected as a test oracle after the exact same pre-sidecar commit and input produced different low-bit model/forecast/artifact digests on separate GitHub-hosted runners while all Gate observations remained identical. The unit regression therefore fixes the stable Gate observations and scope identities.

Separately, an old-vs-new **same-runner** cross-tree test checks the full canonical metric payload and artifact digest under the same dependency environment and BLAS thread settings. That comparison passed for the pre-sidecar base and the implementation before the final type-only cleanup; it must be repeated against the final implementation HEAD before completion.

## Current verification status

A focused Signal regression run passed 41 tests together with Ruff and format checks before the final type-only cleanup. Full CI then found three Mypy errors caused by variable-name type inference and an already-validated boolean sequence not being narrowed for Mypy. The fixes do not change numerical calculations or parser rejection conditions. The formatter has been applied; Mypy and targeted tests are being re-run, followed by exact-final-HEAD full CI.

No CI result, targeted test result, or reviewer conclusion by itself is treated as completion. Final status requires the quality contract, full checks, final diff review, and residual-risk review to be satisfied on the same HEAD.
