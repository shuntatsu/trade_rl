# Causal Teacher Admission Hardening Design

## Objective

Strengthen the pre-BC causal teacher admission contract without turning it into a second BC bootstrap gate, and make the maintained common teacher economics gate the single base economic authority for both canonical U6 and the research-only Causal Alpha V3 lane.

## Non-goals

- Do not change the exactly-once per-symbol teacher holdout partition.
- Do not add bootstrap confidence intervals or additional holdout episodes to teacher admission.
- Do not change candidate selection, signal gate, reward, risk, execution, BC, critic warm start, or PPO semantics.
- Do not tune thresholds from teacher-admission holdout outcomes.
- Do not restore or preserve obsolete V3 admission persistence APIs solely for compatibility if the maintained runtime no longer calls them.

## Maintained admission contract

The common `evaluate_causal_alpha_teacher_admission` gate remains a lightweight pre-BC safety gate over one untouched holdout per train symbol. It rejects when any of the following is true:

1. aggregate gross return is negative;
2. aggregate after-cost net return is negative;
3. more than half of train symbols have negative gross return;
4. any symbol holdout has net return below `-0.05`;
5. total holdout trade count is zero.

The `-0.05` per-symbol net floor matches the existing maintained causal candidate-selection lower-tail floor. It is fixed in code before the teacher holdout is opened. The gate does not claim statistical significance.

## V3 responsibility split

V3 currently duplicates aggregate gross/net/majority logic and additionally checks hard-risk violations and unexplained execution rejections. After this change:

- V3 converts its immutable admission records to maintained `CausalAlphaTeacherHoldoutMetric` values;
- the common teacher admission evaluator owns all base economic checks;
- V3 layers only V3-specific hard-risk and unexplained-execution checks on top;
- V3 admission records remain the exactly-once durable source of truth and are not replayed merely because gate logic changes.

This removes duplicated economic decision logic while preserving V3-specific evidence.

## Artifact compatibility and fail-closed behavior

Changing an admission decision while retaining the old evidence schema would allow old `passed=true` evidence to masquerade as evidence under the stronger contract. Therefore:

- common teacher admission evidence moves from `causal_alpha_teacher_admission_v1` to `causal_alpha_teacher_admission_v2`;
- the Universal pretraining bundle accepts only the new common admission schema;
- V3 aggregate admission evidence moves to a new schema revision while per-symbol V2 admission records stay unchanged because their recorded observations are still complete and valid;
- immutable existing artifacts that do not match the new evidence payload/schema fail closed rather than being silently promoted.

## Invariants

- Each teacher-admission holdout symbol is evaluated at most once per bound run identity once a valid durable record exists.
- Holdout results do not flow back into signal fitting, candidate selection, model tuning, or threshold tuning.
- Existing V3 hard-risk and unexplained-execution rejection behavior remains at least as strict as before.
- BC economic/reconstruction gates remain unchanged and still run only after teacher admission passes.
- Common and V3 admission artifacts remain deterministic and content-addressed.

## Failure modes and test oracle

The change must detect:

- gross-positive but after-cost aggregate-net-negative teachers;
- one catastrophic symbol hidden by profitable peers;
- completely inactive zero-trade teachers;
- majority gross-negative teachers;
- V3 hard-risk violations;
- V3 unexplained execution rejections;
- stale common v1 admission evidence presented to the current pretraining bundle.

Correctness is observed through rejection reasons, `passed`, aggregate/worst-symbol/trade-count evidence, immutable schema/digest behavior, and preservation of exactly-once V3 admission-record reuse.

## Required verification layers

- Unit tests for the common admission evaluator.
- V3 workflow/contract tests proving common-gate delegation semantics and V3-specific additive checks.
- Integration tests for stored teacher-admission reuse and stale-schema rejection.
- Ruff, formatting, Mypy, import architecture, repository dead-code analysis, full pytest/coverage, build/compatibility checks, and exact-head GitHub Actions.

## Dead-code cleanup boundary

After the gate is connected, audit the touched admission/runtime graph for APIs that are no longer reachable from maintained runtime entrypoints. Remove obsolete V3 v1 admission record/store mechanisms when they have no maintained production call path, and remove any helper/property made redundant by the common-gate connection. Do not delete historical documentation or unrelated compatibility surfaces merely because they are not exercised by this feature.
