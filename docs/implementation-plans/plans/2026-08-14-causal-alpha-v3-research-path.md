# Causal Alpha V3 Research Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only causal-alpha V3 path that diagnoses historical selection evidence, fits overlap-aware weighted predictors, compiles uncertainty/cost-aware targets, supports teacher-anchored residual RL, and collects DAgger learner-state labels without changing canonical U6 defaults.

**Architecture:** Keep the maintained causal-alpha selection/admission path intact and add opt-in research primitives around it. Extend the low-level ridge solver only through default-off arguments so legacy fits remain unchanged. New diagnostics are explicitly non-promotable; new action semantics are a separate action mode that no canonical config selects.

**Tech Stack:** Python 3.12, NumPy, Gymnasium/SB3-compatible environment contracts, pytest, Ruff, Mypy, GitHub Actions.

## Global Constraints

- Scalar reward remains pure net log growth; do not add cost/drawdown/baseline shaping.
- Keep `max_position_to_market_notional=0.02` hard risk authoritative.
- No validation/test/teacher-admission holdout leakage into research fitting or selection.
- Historical checkpoint diagnostics must not be accepted by selection resume/promotion.
- Existing `residual` and `target_weight` action semantics/digests remain unchanged.
- Existing `fit_causal_alpha_ridge` behavior remains unchanged unless new optional arguments are explicitly supplied.
- Canonical Universal example configs remain unchanged in this plan.

---

### Task 1: Historical checkpoint diagnostics

**Files:**
- Create: `trade_rl/workflows/causal_alpha_research_diagnostics.py`
- Create: `scripts/analyze_causal_alpha_checkpoint.py`
- Create: `tests/workflows/test_causal_alpha_research_diagnostics.py`
- Create: `tests/scripts/test_analyze_causal_alpha_checkpoint.py`

**Interfaces:**
- Produces: `load_causal_alpha_diagnostic_checkpoint_v2(path: Path) -> CausalAlphaDiagnosticCheckpointV2`
- Produces: `build_causal_alpha_research_report(snapshot: CausalAlphaDiagnosticCheckpointV2) -> CausalAlphaResearchReport`
- Produces: `paired_candidate_delta(snapshot, left_digest, right_digest) -> CausalAlphaPairedCandidateDelta`

- [ ] Write tests that create a synthetic v2 JSONL with repeated signal evidence across two candidates and assert the diagnostic reader accepts an old generator digest, preserves it, and counts one unique prediction identity per exact signal/episode scope.
- [ ] Write a test that mixes generator/grid digests and assert diagnostic loading fails closed.
- [ ] Write a paired-comparison test where candidates have different replay coverage and assert only common `(symbol, episode_index)` scopes contribute to the delta.
- [ ] Run the focused tests and verify RED because the diagnostic module does not exist.
- [ ] Implement immutable diagnostic snapshot/report dataclasses, internal metric verification through `causal_alpha_candidate_metric_v2_from_payload`, prediction de-duplication, candidate summaries, and exact-scope paired deltas.
- [ ] Implement the CLI to emit canonical JSON to stdout or `--output`, clearly marking `promotion_eligible=false`.
- [ ] Run focused tests, Ruff, Mypy on the new modules and commit.

### Task 2: Weighted ridge without legacy drift

**Files:**
- Modify: `trade_rl/learning/causal_alpha_teacher.py`
- Create: `tests/learning/test_causal_alpha_weighted_ridge.py`

**Interfaces:**
- Extend: `fit_causal_alpha_ridge(..., sample_weights: object | None = None, normalize_objective: bool = False) -> CausalAlphaRidgeModel`

- [ ] Write a regression test that calls the old argument set and an explicit `sample_weights=None, normalize_objective=False` call and asserts identical model payload/digest/predictions.
- [ ] Write a weighted synthetic-regression test with one heavily down-weighted outlier and assert coefficients move toward the independently computed weighted least-squares solution.
- [ ] Write validation tests for negative, non-finite, zero-total and shape-mismatched weights.
- [ ] Write a normalized-objective test showing coefficient behavior is invariant to duplicating all rows and weights when `normalize_objective=True`.
- [ ] Run focused tests and verify RED for missing arguments/behavior.
- [ ] Implement weighted available-feature location/variance, normalized eligible weights, weighted Gram/RHS, and optional mean-objective normalization. Keep the exact old branch for default arguments where needed to avoid numerical drift.
- [ ] Run focused existing causal-alpha teacher tests plus new tests, Ruff/Mypy, and commit.

### Task 3: V3 overlap-aware fit and forecast bundle

**Files:**
- Create: `trade_rl/learning/causal_alpha_v3.py`
- Create: `tests/learning/test_causal_alpha_v3.py`

**Interfaces:**
- Produces: `CausalAlphaV3FitConfig`
- Produces: `causal_alpha_overlap_uniqueness_weights(...) -> np.ndarray`
- Produces: `build_causal_alpha_v3_symbol_balanced_weights(...) -> np.ndarray`
- Produces: `fit_causal_alpha_v3(...) -> CausalAlphaV3Fit`
- Produces: `CausalAlphaV3Fit.predict(...) -> CausalAlphaV3Forecast`

- [ ] Write a concurrency test where one label interval overlaps more peers and assert it receives lower uniqueness weight.
- [ ] Write a cutoff test proving rows whose label end is at/after the knowledge cutoff cannot influence weights, statistics or coefficients.
- [ ] Write a two-symbol test with different row counts and assert equal total eligible symbol weight after balancing.
- [ ] Write a forecast test asserting 72h prediction is divided by three before 24h-equivalent blending and uncertainty grows when horizons disagree.
- [ ] Run focused tests and verify RED because V3 module is absent.
- [ ] Implement interval uniqueness using per-symbol difference arrays, symbol balancing, weighted/objective-normalized 24h and 72h ridge fits, weighted residual RMSE evidence, and deterministic forecast bundling.
- [ ] Bind config, weight digests, model digests, residual scales and sample scope into immutable fit evidence.
- [ ] Run focused tests plus causal-alpha fitting tests, Ruff/Mypy, and commit.

### Task 4: Uncertainty-aware target compiler

**Files:**
- Extend: `trade_rl/learning/causal_alpha_v3.py`
- Extend: `tests/learning/test_causal_alpha_v3.py`

**Interfaces:**
- Produces: `CausalAlphaV3TargetConfig`
- Produces: `CausalAlphaV3TargetPath`
- Produces: `causal_alpha_v3_target_path(...) -> CausalAlphaV3TargetPath`

- [ ] Write a HOLD test where expected return is positive but conservative edge is below uncertainty+cost and assert target remains unchanged.
- [ ] Write long/short tests where confident positive/negative forecasts select the expected signed grid target.
- [ ] Write a cadence test where a non-emergency target change is blocked between alpha rebalance decisions.
- [ ] Write a liquidity contraction test where the cap falls below the current absolute target and assert immediate deterministic deleveraging even off cadence.
- [ ] Write a `max_target_delta` test and tie-break test (prefer lower turnover, then lower absolute exposure) for deterministic output.
- [ ] Run focused tests and verify RED.
- [ ] Implement candidate-grid construction including current target, zero and current cap; objective evaluation; deterministic tie breaking; rebalance/strong-reversal logic; and immutable path evidence with reason codes.
- [ ] Run focused tests, Ruff/Mypy, and commit.

### Task 5: Teacher-anchored residual action mode

**Files:**
- Modify: `trade_rl/rl/actions.py`
- Modify: `trade_rl/rl/environment_decision.py`
- Modify: `trade_rl/rl/environment.py`
- Create: `tests/rl/test_anchored_target_residual.py`
- Modify as needed: `tests/rl/test_environment_decision_service.py`

**Interfaces:**
- Add enum value: `ActionMode.ANCHORED_TARGET_RESIDUAL = "anchored_target_residual"`
- Add: `AnchoredTargetResidualAction`
- Composer contract: zero residual + target-weight alpha anchor returns the anchor before downstream risk projection.

- [ ] Write action-spec tests requiring positive `target_weight_count`, `alpha_enabled=true`, target-weight alpha semantics at environment construction, and `0 < residual_scale <= 1`.
- [ ] Write composer tests proving zero action reproduces the anchor, residual action is scaled and gross-normalized, and legacy modes produce unchanged results.
- [ ] Write decision-planner integration tests proving the anchored alpha target becomes the shadow baseline for this mode while one-decision execution delay remains unchanged.
- [ ] Run focused tests and verify RED for the missing mode.
- [ ] Implement the new parsed action type and composition path without altering existing mode branches.
- [ ] Update `baseline_action()` so zero residual is the exact policy-space baseline for anchored mode; update planner shadow target only for this mode.
- [ ] Bind the new mode into action-spec/environment identity. Do not add it to canonical configs.
- [ ] Run all action/environment tests, Ruff/Mypy, and commit.

### Task 6: DAgger learner-state collection

**Files:**
- Create: `trade_rl/learning/dagger.py`
- Create: `tests/learning/test_dagger.py`

**Interfaces:**
- Produces: `DaggerTeacher` protocol with immutable identity and `action_for(environment, observation)`.
- Produces: `collect_dagger_episode(...) -> DaggerEpisodeRollout`
- Produces: `merge_dagger_rollouts(base: EpisodeSupervisedPolicyDataset, rollouts: tuple[DaggerEpisodeRollout, ...]) -> EpisodeSupervisedPolicyDataset`

- [ ] Build a deterministic fake environment/model/teacher test where teacher actions differ from learner actions; assert environment `step()` receives learner actions while stored labels equal teacher actions.
- [ ] Write tests for decision-index alignment, observation schema mismatch, action-dimension mismatch, non-finite labels/actions and mixed teacher identities.
- [ ] Write merge tests proving original samples are preserved, DAgger episodes receive new contiguous IDs, and dataset/environment/action/teacher identities remain closed.
- [ ] Run focused tests and verify RED because the module is absent.
- [ ] Implement observation copying/stacking for flat and mapping observations, learner prediction, teacher labeling before step, rollout digest, and dataset merge using the existing `EpisodeSupervisedPolicyDataset` contract.
- [ ] Run focused BC/teacher artifact tests, Ruff/Mypy, and commit.

### Task 7: Maintained documentation and architecture gates

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Add/modify tests under `tests/test_current_documentation_contract.py` or a focused contract test if required.

**Interfaces:**
- Documentation must state V3 diagnostics/fitting/compiler/anchored residual/DAgger are research-only and do not bypass teacher admission.

- [ ] Write/adjust documentation contract tests first so they fail until the research-only boundary is documented.
- [ ] Update maintained docs with the new path, invariants and explicit canonical non-change.
- [ ] Verify canonical U6 example configs still use `action.mode=target_weight`, `behavior_cloning_teacher=causal_alpha_ridge`, existing reward weights, and existing risk limits.
- [ ] Run documentation/architecture tests and commit.

### Task 8: Falsification review and exact-head verification

**Files:** no behavior changes unless review finds a defect.

- [ ] Compare `main...HEAD` and review every changed file against the design acceptance criteria.
- [ ] Try to falsify: mixed checkpoint identity, duplicate-signal inflation, future-label weight influence, zero-weight/constant-feature cases, target tie nondeterminism, cost double counting, anchored residual saturation, DAgger teacher-forcing, and canonical config drift.
- [ ] For every defect found, add a failing regression test before the fix and rerun affected checks.
- [ ] Run repository-required Ruff/format/Mypy/import/architecture checks and full pytest through exact-head CI.
- [ ] Inspect GitHub Actions for the exact final HEAD; do not reuse green checks from older commits.
- [ ] Keep the PR Draft if any required check is red, skipped unexpectedly or unavailable. Mark Ready only after the exact-head quality gate is satisfied.
