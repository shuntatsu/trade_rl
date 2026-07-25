from __future__ import annotations

import pytest

from trade_rl.rl.cost_learning import (
    CostFamily,
    CostLearningSchema,
    CostValueSpec,
    canonical_cost_learning_schema,
)
from trade_rl.rl.environment_constraints import CONSTRAINT_COST_NAMES


def test_canonical_cost_schema_preserves_environment_order_and_families() -> None:
    schema = canonical_cost_learning_schema(event_gae_lambda=0.97)

    assert schema.names == CONSTRAINT_COST_NAMES
    assert schema.continuous_names == (
        "drawdown_excess",
        "margin_deficit_fraction",
        "gross_exposure_request_excess",
        "daily_turnover",
        "execution_cost_fraction",
    )
    assert schema.event_names == (
        "drawdown_stop_event",
        "forced_liquidation_event",
    )
    assert all(schema[name].gamma == 1.0 for name in schema.event_names)
    assert all(schema[name].gae_lambda == 0.97 for name in schema.event_names)


def test_funding_credit_cannot_become_a_cost_head() -> None:
    with pytest.raises(ValueError, match="unknown constraint cost"):
        CostLearningSchema(
            (
                CostValueSpec(
                    name="funding_credit_fraction",
                    family=CostFamily.CONTINUOUS,
                    gamma=1.0,
                    gae_lambda=0.95,
                ),
            )
        )


def test_event_cost_rejects_discounting_without_explicit_objective_change() -> None:
    with pytest.raises(ValueError, match="objective-altering"):
        CostValueSpec(
            name="drawdown_stop_event",
            family=CostFamily.EVENT,
            gamma=0.99,
            gae_lambda=0.95,
        )

    explicit = CostValueSpec(
        name="drawdown_stop_event",
        family=CostFamily.EVENT,
        gamma=0.99,
        gae_lambda=0.95,
        objective_altering_discount=True,
    )
    assert explicit.gamma == pytest.approx(0.99)


def test_cost_schema_fails_closed_on_duplicates_or_reordering() -> None:
    canonical = canonical_cost_learning_schema()

    with pytest.raises(ValueError, match="duplicate"):
        CostLearningSchema((canonical.specs[0], canonical.specs[0]))

    with pytest.raises(ValueError, match="canonical order"):
        CostLearningSchema(tuple(reversed(canonical.specs)))


def test_cost_schema_digest_tracks_optimization_identity() -> None:
    baseline = canonical_cost_learning_schema(event_gae_lambda=0.95)
    changed_lambda = canonical_cost_learning_schema(event_gae_lambda=1.0)
    changed_loss = CostLearningSchema(
        tuple(
            CostValueSpec(
                name=spec.name,
                family=spec.family,
                gamma=spec.gamma,
                gae_lambda=spec.gae_lambda,
                value_loss_coefficient=(
                    0.5 if spec.name == "daily_turnover" else spec.value_loss_coefficient
                ),
                auxiliary_event_loss_coefficient=(
                    spec.auxiliary_event_loss_coefficient
                ),
                objective_altering_discount=spec.objective_altering_discount,
            )
            for spec in baseline.specs
        )
    )

    assert baseline.digest != changed_lambda.digest
    assert baseline.digest != changed_loss.digest
    assert baseline.digest_payload()["names"] == list(CONSTRAINT_COST_NAMES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma", 0.0),
        ("gamma", 1.01),
        ("gae_lambda", -0.1),
        ("gae_lambda", 1.01),
        ("value_loss_coefficient", -1.0),
        ("auxiliary_event_loss_coefficient", -1.0),
    ],
)
def test_cost_value_spec_rejects_invalid_optimization_values(
    field: str,
    value: float,
) -> None:
    values: dict[str, object] = {
        "name": "daily_turnover",
        "family": CostFamily.CONTINUOUS,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "value_loss_coefficient": 1.0,
        "auxiliary_event_loss_coefficient": 0.0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        CostValueSpec(**values)  # type: ignore[arg-type]
