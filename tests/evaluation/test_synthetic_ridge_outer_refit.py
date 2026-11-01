import inspect
import json
import math
from dataclasses import replace

import numpy as np
import pytest

import trade_rl.evaluation.runs.synthetic_ridge_outer_refit as outer_refit
from trade_rl.evaluation.runs.synthetic_ridge_outer_refit import (
    SyntheticFixtureVariant,
    default_synthetic_ridge_outer_refit_spec,
    run_synthetic_ridge_outer_refit,
)


def test_default_harness_runs_two_vintages_on_one_native_account() -> None:
    spec = default_synthetic_ridge_outer_refit_spec()
    result = run_synthetic_ridge_outer_refit(spec=spec)

    assert len(result.fit_events) == 2
    assert len(result.replay.decisions) == len(result.decisions) == 17
    assert result.fit_events[0].model.n_samples == 14
    assert result.fit_events[1].model.n_samples == 30
    assert not np.allclose(
        result.fit_events[0].model.coefficients,
        result.fit_events[1].model.coefficients,
    )
    assert not math.isclose(
        result.fit_events[0].unit_signal_prediction,
        result.fit_events[1].unit_signal_prediction,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.decisions[7].index == 16
    assert result.decisions[7].vintage_id == result.fit_events[0].vintage_id
    assert result.decisions[8].index == 17
    assert result.decisions[8].vintage_id == result.fit_events[1].vintage_id

    native = result.replay.ledger_evidence
    assert native is not None
    ledger = native.intervals
    assert ledger.terminal_exact_quantities == ("0", "0")
    assert ledger.active_order_remainders == ()
    fills = [
        event
        for interval in ledger.intervals
        for event in interval.order_events
        if event.filled_quantity != 0.0
    ]
    assert fills
    assert all(event.execution_price is not None for event in fills)
    gross_fill_notional = math.fsum(
        abs(event.filled_quantity * float(event.execution_price)) for event in fills
    )
    signed_fill_cash_flow = math.fsum(
        event.filled_quantity * float(event.execution_price) for event in fills
    )
    expected_fee = gross_fill_notional * 0.001
    expected_spread_cost = gross_fill_notional * 0.0005
    expected_total_cost = expected_fee + expected_spread_cost
    expected_cash = 1_000.0 - signed_fill_cash_flow - expected_total_cost
    assert ledger.final_total_cost == pytest.approx(expected_total_cost)
    assert ledger.final_cash == pytest.approx(expected_cash)
    assert ledger.final_portfolio_value == pytest.approx(expected_cash)
    assert 0.0 <= ledger.final_max_drawdown <= spec.bound.objective.maximum_drawdown
    assert all(event.fit_elapsed_seconds >= 0.0 for event in result.fit_events)

    activation_snapshot = result.decisions[8]
    preceding_interval = next(
        interval for interval in ledger.intervals if interval.start_index == 16
    )
    assert (
        activation_snapshot.exact_quantities
        == preceding_interval.exact_quantities_after
    )
    assert activation_snapshot.cash == preceding_interval.cash_after
    assert activation_snapshot.equity == preceding_interval.portfolio_value_after
    assert activation_snapshot.max_drawdown == preceding_interval.max_drawdown_after
    assert len(activation_snapshot.active_order_facts) == 2
    assert all(facts[1] > 0.0 for facts in activation_snapshot.active_order_facts)

    active_by_id: dict[str, tuple[int, float]] = {}
    for interval in ledger.intervals:
        if interval.next_index > activation_snapshot.index:
            break
        for event in interval.order_events:
            if event.event_type in {"filled", "rejected", "cancelled", "expired"}:
                active_by_id.pop(event.order_id, None)
            else:
                active_by_id[event.order_id] = (
                    event.symbol_index,
                    event.remaining_quantity,
                )
    assert len(active_by_id) == 2
    for symbol_index, remaining, _, _ in activation_snapshot.active_order_facts:
        assert any(
            active_symbol == symbol_index
            and active_remaining == pytest.approx(remaining)
            for active_symbol, active_remaining in active_by_id.values()
        )

    payload = result.to_mapping()
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    decoded = json.loads(encoded)
    assert json.dumps(decoded, sort_keys=True, allow_nan=False) == encoded


def test_vintage_artifact_roundtrip_and_fail_closed_identity_checks() -> None:
    artifact = run_synthetic_ridge_outer_refit().fit_events[0]
    payload = json.loads(json.dumps(artifact.to_mapping(), allow_nan=False))

    restored = outer_refit.RidgeVintageArtifact.from_mapping(payload)
    assert restored.to_mapping() == artifact.to_mapping()

    mismatched_cutoff_index = dict(payload)
    mismatched_cutoff_index["fit_cutoff_index"] = artifact.fit_cutoff_index - 1
    with pytest.raises(
        ValueError,
        match="fit cutoff index and timestamp do not match fixture clock",
    ):
        outer_refit.RidgeVintageArtifact.from_mapping(mismatched_cutoff_index)

    wrong_source = dict(payload)
    wrong_source["coordinator_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="coordinator source digest mismatch"):
        outer_refit.RidgeVintageArtifact.from_mapping(wrong_source)

    wrong_vintage = dict(payload)
    wrong_vintage["vintage_id"] = "f" * 64
    with pytest.raises(ValueError, match="causal vintage identity mismatch"):
        outer_refit.RidgeVintageArtifact.from_mapping(wrong_vintage)

    wrong_fixture = dict(payload)
    wrong_fixture["fixture_dataset_id"] = "f" * 64
    with pytest.raises(ValueError, match="content digest mismatch"):
        outer_refit.RidgeVintageArtifact.from_mapping(wrong_fixture)

    missing_runtime = dict(payload)
    del missing_runtime["runtime_digest"]
    with pytest.raises(ValueError, match="field closure mismatch"):
        outer_refit.RidgeVintageArtifact.from_mapping(missing_runtime)


def test_runner_admits_only_the_closed_synthetic_fixture_variants() -> None:
    assert set(inspect.signature(run_synthetic_ridge_outer_refit).parameters) == {
        "fixture_variant",
        "spec",
    }
    with pytest.raises(ValueError, match="not an allowed synthetic variant"):
        run_synthetic_ridge_outer_refit(fixture_variant="external_dataset")  # type: ignore[arg-type]


def test_future_tail_mutation_preserves_causal_vintages_and_pre_tail_actions() -> None:
    baseline = run_synthetic_ridge_outer_refit()
    mutated = run_synthetic_ridge_outer_refit(
        fixture_variant=SyntheticFixtureVariant.FUTURE_TAIL_MUTATED
    )

    assert baseline.fixture_dataset_id != mutated.fixture_dataset_id
    for original, changed in zip(baseline.fit_events, mutated.fit_events, strict=True):
        assert original.training_prefix_digest == changed.training_prefix_digest
        assert original.vintage_id == changed.vintage_id
        assert original.model.n_samples == changed.model.n_samples
        np.testing.assert_array_equal(
            original.model.feature_mean, changed.model.feature_mean
        )
        np.testing.assert_array_equal(
            original.model.feature_scale, changed.model.feature_scale
        )
        np.testing.assert_array_equal(
            original.model.coefficients, changed.model.coefficients
        )
        assert original.model.intercept == changed.model.intercept
        assert original.unit_signal_prediction == changed.unit_signal_prediction
        assert original.artifact_digest != changed.artifact_digest

    original_prefix = [
        decision for decision in baseline.decisions if decision.index < 20
    ]
    changed_prefix = [decision for decision in mutated.decisions if decision.index < 20]
    assert [decision.target_weights for decision in original_prefix] == [
        decision.target_weights for decision in changed_prefix
    ]
    assert [decision.vintage_id for decision in original_prefix] == [
        decision.vintage_id for decision in changed_prefix
    ]


def test_refit_schedule_rejects_duplicate_cutoff_and_backdated_activation() -> None:
    spec = default_synthetic_ridge_outer_refit_spec()
    changed_bound = replace(
        spec.bound,
        clock=replace(spec.bound.clock, gamma=0.99),
    )
    with pytest.raises(
        ValueError,
        match="bound clock must match the fixed synthetic contract",
    ):
        run_synthetic_ridge_outer_refit(spec=replace(spec, bound=changed_bound))

    duplicate_cutoff = replace(
        spec,
        schedule=(spec.schedule[0], replace(spec.schedule[1], fit_cutoff_index=8)),
    )
    with pytest.raises(ValueError, match="fit cutoffs must strictly increase"):
        run_synthetic_ridge_outer_refit(spec=duplicate_cutoff)

    backdated = replace(
        spec,
        schedule=(spec.schedule[0], replace(spec.schedule[1], activation_index=16)),
    )
    with pytest.raises(ValueError, match="fit must complete before its activation"):
        run_synthetic_ridge_outer_refit(spec=backdated)

    out_of_fixture = replace(
        spec,
        schedule=(spec.schedule[0], replace(spec.schedule[1], fit_cutoff_index=30)),
    )
    with pytest.raises(ValueError, match="fit cutoff must identify a fixture bar"):
        run_synthetic_ridge_outer_refit(spec=out_of_fixture)

    skipped_readiness = replace(
        spec,
        schedule=(
            spec.schedule[0],
            outer_refit.RefitActivation(12, 12, 20),
            outer_refit.RefitActivation(14, 15, 21),
        ),
    )
    with pytest.raises(ValueError, match="readiness must follow the prior activation"):
        run_synthetic_ridge_outer_refit(spec=skipped_readiness)


def test_scheduled_fit_failure_aborts_without_old_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = outer_refit.fit_ridge_forecast
    original_init = outer_refit._ScheduledRidgePortfolioPolicy.__init__
    fit_calls = 0
    policies: list[outer_refit._ScheduledRidgePortfolioPolicy] = []

    def capture_init(
        policy: outer_refit._ScheduledRidgePortfolioPolicy,
        *args: object,
        **kwargs: object,
    ) -> None:
        original_init(policy, *args, **kwargs)
        policies.append(policy)

    def fail_second_fit(*args: object, **kwargs: object):
        nonlocal fit_calls
        fit_calls += 1
        if fit_calls == 2:
            raise RuntimeError("synthetic scheduled fit failed")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(outer_refit, "fit_ridge_forecast", fail_second_fit)
    monkeypatch.setattr(
        outer_refit._ScheduledRidgePortfolioPolicy,
        "__init__",
        capture_init,
    )
    with pytest.raises(RuntimeError, match="synthetic scheduled fit failed"):
        run_synthetic_ridge_outer_refit()
    assert fit_calls == 2
    assert len(policies) == 1
    with pytest.raises(RuntimeError, match="scheduled Ridge policy is poisoned"):
        policies[0].decide_portfolio(None)  # type: ignore[arg-type]
