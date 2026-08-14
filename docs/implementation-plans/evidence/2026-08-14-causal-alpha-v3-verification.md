# Causal Alpha V3 Verification Evidence

Status: IMPLEMENTATION VERIFICATION IN PROGRESS — NOT AN ECONOMIC ADMISSION OR PRODUCTION GO.

This record captures software-verification evidence for the research-only Causal Alpha V3 path. Canonical U6 remains unchanged and the V3 path remains non-promotable until the existing teacher-admission/evaluation gates are satisfied separately.

## TDD evidence

The temporary `Causal V3 TDD Red` workflow run `31803434539` executed the planned feature tests independently before implementation. It observed non-zero exits for all five planned areas:

- weighted ridge: missing `sample_weights` / `normalize_objective` API;
- V3 predictor/compiler: module absent;
- V3 workflow fit: module absent;
- anchored target residual: action type absent;
- DAgger: module absent.

The temporary workflow self-contained the expected-RED oracle and was removed before maintained verification.

Documentation contracts were also introduced before their implementation and failed only on the missing research-only boundary text.

## Falsification review

Independent re-reading from the acceptance criteria found two defects after the first Green pass:

1. `DaggerEpisodeRollout` did not copy/freeze batched observations, allowing post-construction mutation to diverge from the recorded digest.
2. Structured DAgger observations validated only one field's sample count, allowing another field to drift.

Regression tests reproduced both failures before the fix. The implementation now copies each flat/structured observation array, validates every structured field against the expected sample count, and marks stored arrays read-only. A separate structured-observation regression test covers field ordering, copying, and immutability.

The same review corrected two documentation inconsistencies without changing behavior:

- symbol balancing is defined as equal total eligible weight mass per train symbol; absolute global scale is irrelevant because the weighted V3 objective is normalized by total eligible weight;
- the conservative target compiler optimizes incremental `delta_weight`, with HOLD (`delta=0`) as objective zero, so already-paid execution costs are not charged again.

## Focused verification history

A focused verification run after the behavior changes reached:

- 120 focused tests passed;
- Ruff passed;
- Ruff format check passed;
- Mypy found one stale return-type annotation in `ResidualMarketEnv._parse_action`, which still omitted `AnchoredTargetResidualAction` from its union.

That Mypy finding was treated as a contract defect, not ignored. The environment import and return annotation were updated to include `AnchoredTargetResidualAction`.

## Exact-head gate

Pending at the time this evidence file was committed:

- rerun the focused V3 tests, Ruff, format, and Mypy including the type fix;
- remove temporary focused verification workflow;
- run the repository's normal exact-head CI, architecture/import checks, full pytest/coverage, compatibility jobs, training-image build, and other required checks;
- review `main...HEAD` and confirm no temporary workflows, debug code, generated junk, or secret material remain.

Passing these software checks will establish only the implemented software contracts. It will not establish positive gross/net alpha, teacher admission, RL uplift, profitability, or Production GO.
