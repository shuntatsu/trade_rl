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

After the DAgger fixes and anchored-action type-contract correction, focused run `31806707021` passed:

- 120 focused tests;
- Ruff;
- Ruff format check (`1195 files already formatted`);
- Mypy (`Success: no issues found in 11 source files`).

The focused workflow was then removed so it cannot remain as repository maintenance surface.

## Exact-head architecture regression

The first normal exact-head CI attempt at `7c72752b605aa3367d47c40bd159874ae66559ef` exposed an architecture regression in both Linux and Windows compatibility jobs. The compatibility suite result was `1 failed, 1036 passed`; the only failure was `test_environment_constructor_delegates_reward_execution_resources`, because `ResidualMarketEnv.__init__` had grown to 157 lines while the maintained architecture contract limits it to 150.

The architecture test was not weakened. Anchored alpha/action compatibility validation was extracted into `_validate_action_alpha_contract()`, leaving the constructor as orchestration while retaining the existing behavior test that rejects non-target-weight anchors. The normal exact-head CI must be rerun from a user-authored commit containing this decomposition before the software gate can be considered satisfied.

## Exact-head gate

Pending from the commit that records this evidence:

- rerun the repository's normal exact-head CI, architecture/import checks, full pytest/coverage, Linux/Windows compatibility jobs, training-image build, PostgreSQL catalog, Nautilus capability, and other required checks;
- review `main...HEAD` again and confirm no temporary workflows, debug code, generated junk, or secret material remain;
- keep PR #402 Draft unless the same final HEAD satisfies the required software quality gate.

Passing these software checks will establish only the implemented software contracts. It will not establish positive gross/net alpha, teacher admission, RL uplift, profitability, or Production GO.
