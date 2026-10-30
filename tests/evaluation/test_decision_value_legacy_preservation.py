"""Independent pinned software counterexamples for pre-TD8 v1 semantics (#1025)."""

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_decision_value import (
    _conservative_model,
    _context,
    _model,
    _pending_order,
    _snapshot,
)
from trade_rl.evaluation.decision_value_fit import _fit_standardized_ridge
from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValuePolicy,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
)
from trade_rl.strategies.portfolio_policy_codec import ACTION_HOLD, ACTION_LONG

_HOLD = (ACTION_HOLD, ACTION_HOLD)


def test_legacy_v1_ridge_bias_is_penalized_and_td8_solver_is_not() -> None:
    # Literal original v1 reference: Z has only an intercept with four rows;
    # ridge argmin ||Z b - 2||² + 10*b² yields b=8/(4+10).
    rows = [np.array([1.0, 1.0], dtype=np.float64) for _ in range(4)]
    values = [2.0] * 4
    coef, _ = _fit_standardized_ridge(rows, values, 10.0, solver_version="legacy_v1")
    assert coef == pytest.approx((8.0 / 14.0, 0.0), abs=1e-12)
    modern, _ = _fit_standardized_ridge(rows, values, 10.0, solver_version="td8_v2")
    assert modern == pytest.approx((2.0, 0.0), abs=1e-12)


def test_legacy_linear_keeps_old_pending_decide_portfolio() -> None:
    model = _model()
    policy = LinearDecisionValuePolicy(model, runtime_context=_context(model))
    pending = replace(_snapshot(), active_orders=(_pending_order(),))
    # Main@8cab046a consumed pending states with the scoring copy stripped,
    # then made the *native proposal* from the real pending account.
    assert policy.decide_portfolio(pending).target_weights == (0.0, 0.0)
    assert policy.decide_portfolio(pending).hold_current_quantity_symbols == (1,)
    with pytest.raises(ValueError, match="pending"):
        policy.decide_action(pending)


def test_legacy_linear_keeps_hold_score_zero_even_when_invested() -> None:
    flat = _snapshot()
    s = replace(
        flat,
        account=replace(
            flat.account,
            exact_quantities=("4", "0"),
            quantities=(4.0, 0.0),
            weights=(0.4, 0.0),
            cash=600.0,
        ),
    )
    model = _model()
    policy = LinearDecisionValuePolicy(model, runtime_context=_context(model))
    assert policy.value_for_action(s, _HOLD) == 0.0
    assert policy.decide_action(s) == _HOLD
    assert policy.decide_portfolio(s).target_weights == (0.0, 0.0)
    assert policy.decide_portfolio(s).hold_current_quantity_symbols == (1,)


def test_legacy_conservative_keeps_original_hold_and_pending_proposals() -> None:
    model = _conservative_model(cov_diag_val=0.0)
    policy = ConservativeLinearDecisionValuePolicy(
        model, runtime_context=_context(_model())
    )
    flat = _snapshot()
    invested = replace(
        flat,
        account=replace(
            flat.account,
            exact_quantities=("4", "0"),
            quantities=(4.0, 0.0),
            weights=(0.4, 0.0),
            cash=600.0,
        ),
    )
    assert policy.conservative_value_for_action(invested, _HOLD) == 0.0
    assert policy.decide_action(invested) == _HOLD
    assert policy.decide_portfolio(invested).target_weights == (0.0, 0.0)
    assert policy.decide_portfolio(invested).hold_current_quantity_symbols == (1,)
    pending = replace(flat, active_orders=(_pending_order(),))
    assert policy.decide_portfolio(pending).target_weights == (0.0, 0.0)
    assert policy.decide_portfolio(pending).hold_current_quantity_symbols == (1,)


def test_pre_td8_fitted_models_match_original_main_numerical_contract() -> None:
    """Restore v1 numerical semantics; B2 floats vary with BLAS implementation."""
    import hashlib

    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.decision_value_fit import (
        fit_conservative_linear_decision_value,
        fit_synthetic_linear_decision_value,
        fit_synthetic_shallow_decision_value,
    )

    batch = _fit_batch()
    b1 = fit_synthetic_linear_decision_value(batch, candidate_actions=_CANDIDATES)
    b1c = fit_conservative_linear_decision_value(batch, candidate_actions=_CANDIDATES)
    b2 = fit_synthetic_shallow_decision_value(batch, candidate_actions=_CANDIDATES)
    exact_golden = {
        "B1": "1bb38fb3596fae4011aafcba90e5a03a5981b5530b64a0bd38b0e5052166f989",
        "B1c": "373c1df969040a1577461af495bfbc7f585d7eef66887e1a2b899f072539a61d",
    }
    for name, model in (("B1", b1), ("B1c", b1c)):
        assert (
            hashlib.sha256(canonical_json_bytes(model.coefficients)).hexdigest()
            == exact_golden[name]
        )
    assert (
        hashlib.sha256(canonical_json_bytes(b1c.covariance_matrices)).hexdigest()
        == "86ad5c317ee789b4ddf221c90135480790bec00be71318744778c650e1e75f10"
    )
    # These two rows are copied from pre-TD8 main@8cab046a, not recalculated
    # from the changed source. Different BLAS kernels produce slightly
    # different final float bits for the shallow Ridge solve even when its
    # design and solution are mathematically identical.
    original_b2_coefficients = (
        (
            -0.0009598933451838923,
            0.00312249472506639,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.0016069808687683973,
            -0.0024048453496372107,
            -0.0007077320195169316,
            0.0014840463898239775,
            0.001633232599425124,
            0.0010174408097366305,
            -0.0006456262755229596,
            9.984371747833404e-05,
        ),
        (
            -0.0009598933451838799,
            -0.003129997715992419,
            0.0,
            0.0,
            0.0,
            0.0,
            0.001610842256517563,
            0.002410623912758587,
            0.0007094326170791977,
            -0.001487612380349564,
            -0.0016371570670227287,
            -0.001019885601429841,
            0.0006471776402206857,
            -0.00010008363029540378,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(b2.coefficients),
        np.asarray(original_b2_coefficients),
        rtol=1e-9,
        atol=1e-11,
    )


def test_legacy_b2_matches_frozen_dense_ridge_normal_equations() -> None:
    """Reference v1 source equation, evaluated on the *same* BLAS backend."""
    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.evaluation.decision_value_fit import (
        _collect_training,
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import (
        expand_state,
        fixed_hidden_weights,
    )

    batch = _fit_batch()
    model = fit_synthetic_shallow_decision_value(batch, candidate_actions=_CANDIDATES)
    observed = _collect_training(batch, candidate_actions=_CANDIDATES)
    width = len(observed.rows[0][0])
    hidden = fixed_hidden_weights(width)
    for rows, values, coefficients in zip(
        observed.rows, observed.values, model.coefficients, strict=True
    ):
        design = np.stack([expand_state(row, hidden) for row in rows])
        targets = np.asarray(values, dtype=np.float64)
        # Exact original pre-TD8 recipe: penalize ALL columns, including bias.
        normal_matrix = design.T @ design + model.ridge_penalty * np.eye(
            design.shape[1]
        )
        expected = np.linalg.solve(normal_matrix, design.T @ targets)
        np.testing.assert_allclose(
            np.asarray(coefficients), expected, rtol=1e-12, atol=1e-14
        )


def test_td8_solver_produces_new_model_schemas_with_unchanged_v1_loaders() -> None:
    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.evaluation.decision_value_fit import (
        fit_conservative_linear_decision_value,
        fit_synthetic_linear_decision_value,
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value import (
        load_conservative_linear_decision_value,
        load_linear_decision_value,
        serialize_conservative_linear_decision_value,
        serialize_linear_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import (
        load_shallow_decision_value,
        serialize_shallow_decision_value,
    )

    batch = _fit_batch()
    for fitter, serializer, loader, versions in (
        (
            fit_synthetic_linear_decision_value,
            serialize_linear_decision_value,
            load_linear_decision_value,
            (
                "synthetic_linear_decision_value_v1",
                "synthetic_linear_decision_value_v2",
            ),
        ),
        (
            fit_conservative_linear_decision_value,
            serialize_conservative_linear_decision_value,
            load_conservative_linear_decision_value,
            (
                "conservative_linear_decision_value_v2",
                "conservative_linear_decision_value_v3",
            ),
        ),
        (
            fit_synthetic_shallow_decision_value,
            serialize_shallow_decision_value,
            load_shallow_decision_value,
            (
                "synthetic_shallow_decision_value_v1",
                "synthetic_shallow_decision_value_v2",
            ),
        ),
    ):
        legacy = fitter(batch, candidate_actions=_CANDIDATES)
        modern = fitter(batch, candidate_actions=_CANDIDATES, solver_version="td8_v2")
        assert (legacy.schema_version, modern.schema_version) == versions
        assert (
            loader(text=serializer(legacy)[0], expected_sha256=serializer(legacy)[1])
            == legacy
        )
        assert (
            loader(text=serializer(modern)[0], expected_sha256=serializer(modern)[1])
            == modern
        )


def test_v1_shallow_serialization_keeps_original_field_shape() -> None:
    """A fixed model pins old exact JSON; no backend-dependent fitted bytes."""
    import hashlib
    import json
    from dataclasses import asdict

    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.evaluation.decision_value_fit import (
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import (
        load_shallow_decision_value,
        serialize_shallow_decision_value,
    )

    model = fit_synthetic_shallow_decision_value(
        _fit_batch(), candidate_actions=_CANDIDATES
    )
    # A fixed numeric model removes LAPACK/BLAS variation from artifact
    # byte-level compatibility, while retaining all original metadata.
    fixed = replace(
        model,
        coefficients=tuple((0.0,) * len(row) for row in model.coefficients),
    )
    payload, digest = serialize_shallow_decision_value(fixed)
    # Computed independently on pre-TD8 main@8cab046a using the identical
    # all-zero coefficient fixture; preserved canonical v1 byte identity.
    assert digest == "2d54b5d61d3c3ef437d746402b3273ffd756ab512199398ccd5820239f673fd9"
    old_json = asdict(fixed)
    old_json.pop("input_means")
    old_json.pop("input_scales")
    expected = json.dumps(
        old_json,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert payload == expected
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == digest
    assert "input_means" not in json.loads(payload)
    assert "input_scales" not in json.loads(payload)
    assert load_shallow_decision_value(payload, expected_sha256=digest) == fixed


def test_td8_selector_uses_td8_native_equivalence_under_drawdown_stop() -> None:
    """A blocked risky short is the same native HOLD, not a fictitious profit."""
    from trade_rl.strategies.decision_value import (
        preserve_posture_aware_proposal,
        select_hold_relative_action,
    )
    from trade_rl.strategies.portfolio_policy_codec import ACTION_SHORT

    flat = _snapshot()
    exhausted = replace(
        flat,
        account=replace(
            flat.account,
            portfolio_value=750.0,
            peak_value=1_000.0,
        ),
        active_orders=(_pending_order(),),
    )
    hold = (ACTION_HOLD, ACTION_HOLD)
    short = (ACTION_SHORT, ACTION_HOLD)
    targets = (0.25, 0.25)
    # At risk stop the native proposal for SHORT cancels the pending LONG,
    # just like HOLD; no learned Q can create an executable trade here.
    assert preserve_posture_aware_proposal(
        hold, exhausted, targets, version="td8_v2"
    ) == preserve_posture_aware_proposal(short, exhausted, targets, version="td8_v2")
    selected = select_hold_relative_action(
        (hold, short),
        exhausted,
        lambda _action: 100.0,
        target_abs_weights=targets,
        eval_one_way_cost=0.0,
        margin_multiplier=0.0,
    )
    assert selected == hold


def test_versioned_models_refuse_cross_mode_value_semantics() -> None:
    """A trusted v1 model digest must not silently change its decision objective."""
    from tests.strategies.test_decision_value import _conservative_model
    from trade_rl.strategies.decision_value import ConservativeLinearDecisionValuePolicy

    linear = _model()
    context = _context(linear)
    modern = replace(linear, schema_version="synthetic_linear_decision_value_v2")
    with pytest.raises(ValueError, match="schema|hold_relative|model"):
        LinearDecisionValuePolicy(linear, runtime_context=context, hold_relative=True)
    with pytest.raises(ValueError, match="schema|hold_relative|model"):
        LinearDecisionValuePolicy(modern, runtime_context=context, hold_relative=False)
    assert (
        LinearDecisionValuePolicy(
            modern, runtime_context=context, hold_relative=True
        ).value_for_action(_snapshot(), _HOLD)
        == 0.0
    )

    conservative = _conservative_model()
    context = _context(linear)
    modern_cons = replace(
        conservative, schema_version="conservative_linear_decision_value_v3"
    )
    with pytest.raises(ValueError, match="schema|hold_relative|model"):
        ConservativeLinearDecisionValuePolicy(
            conservative, runtime_context=context, hold_relative=True
        )
    with pytest.raises(ValueError, match="schema|hold_relative|model"):
        ConservativeLinearDecisionValuePolicy(
            modern_cons, runtime_context=context, hold_relative=False
        )


def test_v1_shallow_loader_rejects_td8_fields_under_old_schema() -> None:
    import hashlib
    import json

    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.evaluation.decision_value_fit import (
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import (
        load_shallow_decision_value,
        serialize_shallow_decision_value,
    )

    v1 = fit_synthetic_shallow_decision_value(
        _fit_batch(), candidate_actions=_CANDIDATES
    )
    payload, _ = serialize_shallow_decision_value(v1)
    wrong = json.loads(payload)
    wrong["input_means"] = None
    wrong["input_scales"] = None
    tampered = json.dumps(
        wrong, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    sha = hashlib.sha256(tampered.encode()).hexdigest()
    with pytest.raises(ValueError, match="schema|field"):
        load_shallow_decision_value(tampered, expected_sha256=sha)


def test_old_and_new_paired_proposals_are_separately_versioned_on_pending_state() -> (
    None
):
    """The TD8 paired teacher must not replace the original v1 money oracle."""
    from tests.evaluation.test_paired_portfolio_action import (
        _CAPITAL,
        _COST,
        _RISK,
        _TARGET_WEIGHT,
        _dataset,
        _EnterThenHold,
    )
    from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
    from trade_rl.strategies.decision_value import preserve_posture_aware_proposal
    from trade_rl.strategies.portfolio_policy_codec import portfolio_action_to_proposal

    baseline = (ACTION_HOLD, ACTION_HOLD)
    candidate = (ACTION_LONG, ACTION_HOLD)
    weights = (_TARGET_WEIGHT, _TARGET_WEIGHT)
    results = {}
    for projection in ("legacy_v1", "posture_v2"):
        pair = run_paired_portfolio_action(
            _dataset(),
            reference_policy_factory=_EnterThenHold,
            start_index=0,
            stop_index=7,
            intervention_index=1,
            baseline_action=baseline,
            candidate_action=candidate,
            target_abs_weights=weights,
            initial_capital=_CAPITAL,
            execution_cost=_COST,
            risk=_RISK,
            intervention_projection=projection,
        )
        assert pair.intervention_projection == projection
        assert pair.baseline_snapshot.account.exact_quantities == ("1", "0")
        assert pair.baseline_snapshot.active_orders
        for action, arm in (
            (baseline, pair.baseline_result),
            (candidate, pair.candidate_result),
        ):
            actual = arm.decisions[1]
            expected = (
                portfolio_action_to_proposal(action, pair.baseline_snapshot, weights)
                if projection == "legacy_v1"
                else preserve_posture_aware_proposal(
                    action, pair.baseline_snapshot, weights, version="td8_v2"
                )
            )
            assert actual.proposal_weights == expected.target_weights
            assert (
                actual.requested_hold_symbols == expected.hold_current_quantity_symbols
            )
        results[projection] = pair.baseline_result.decisions[1].proposal_weights
    assert results["legacy_v1"] != results["posture_v2"]


def test_shallow_model_requires_matching_scoring_mode() -> None:
    from tests.evaluation.test_decision_value_fit import _CANDIDATES, _fit_batch
    from trade_rl.evaluation.decision_value_fit import (
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import ShallowDecisionValuePolicy

    batch = _fit_batch()
    v1 = fit_synthetic_shallow_decision_value(batch, candidate_actions=_CANDIDATES)
    v2 = fit_synthetic_shallow_decision_value(
        batch, candidate_actions=_CANDIDATES, solver_version="td8_v2"
    )
    # Runtime contexts must match the fitted source and feature digests.
    context = LinearDecisionValueRuntimeContext(
        source_scope=v1.source_scope,
        feature_config_digest=v1.feature_config_digest,
        normalization_digest=v1.normalization_digest,
        execution_policy_digest=v1.execution_policy_digest,
        reference_policy_digest=v1.reference_policy_digest,
    )
    with pytest.raises(ValueError, match="schema"):
        ShallowDecisionValuePolicy(v1, runtime_context=context, hold_relative=True)
    with pytest.raises(ValueError, match="schema"):
        ShallowDecisionValuePolicy(v2, runtime_context=context, hold_relative=False)
    assert ShallowDecisionValuePolicy(v2, runtime_context=context, hold_relative=True)
