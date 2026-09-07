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

The architecture test was not weakened. Anchored alpha/action compatibility validation was first extracted into `_validate_action_alpha_contract()`. On the next exact-head attempt at `6029e8773ae07b2ab1b13424128335b94dbe878c`, the same compatibility suite again had only that architecture failure (`1 failed, 1036 passed`), with the constructor reduced to 151 lines. This second failure proved that the first refactor was insufficient even though the behavior remained correct.

The final correction replaces the two constructor statements `self.action_spec = ...` plus `_validate_action_alpha_contract()` with one validated assignment, `self.action_spec = self._validated_action_spec(...)`. `_validated_action_spec()` validates anchored target-residual compatibility against the target-weight alpha contract and returns the action spec. This preserves the behavior test, avoids weakening the architecture oracle, and reduces the constructor by the final required line to the maintained 150-line boundary.

## Exact-head gate

The commit containing this evidence is the next exact-head candidate. It must satisfy, on the same SHA:

- normal CI including Linux/Windows compatibility, architecture/import checks, full pytest/coverage, static analysis, format, Mypy, training-image build, and package/runtime identity checks;
- PostgreSQL Catalog;
- Nautilus Capability;
- final `main...HEAD` review confirming no temporary workflows, debug code, generated junk, secret material, or canonical U6 config drift.

PR #402 remains Draft while these checks are running. Passing these software checks will establish only the implemented software contracts. It will not establish positive gross/net alpha, teacher admission, RL uplift, profitability, or Production GO.
