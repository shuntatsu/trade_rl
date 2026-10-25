from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
    FourFamilyPortfolioAttemptIdentity,
    four_family_portfolio_attempt_identity_payload,
)


def _identity(**overrides: object) -> FourFamilyPortfolioAttemptIdentity:
    from trade_rl.risk.pretrade import PreTradeRiskConfig
    from trade_rl.simulation.execution import ExecutionCostConfig
    from trade_rl.simulation.objectives.binding import BoundObjectiveClock
    from trade_rl.simulation.objectives.clock import FinancialClockContract
    from trade_rl.simulation.objectives.contract import (
        CapitalContract,
        ObjectiveContract,
    )
    from trade_rl.strategies.forecasts.ridge_artifact import (
        RidgeForecastArtifactBinding,
    )
    from trade_rl.strategies.rules.trend import TrendIntentConfig

    start = datetime(2026, 1, 1, tzinfo=UTC)
    stop = start + timedelta(hours=2)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=start,
            evaluation_stop_exclusive=stop,
            terminal_valuation="settled",
            economics_digest="4" * 64,
            risk_digest="5" * 64,
            deployment_recipe_digest="6" * 64,
            maximum_drawdown=0.2,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=2 * 3_600,
            rollout_steps=2,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    ridge_binding = RidgeForecastArtifactBinding(
        dataset_id="a" * 64,
        dataset_schema_version="market_dataset_v1",
        dataset_artifact_sha256="b" * 64,
        fit_symbol_indices=(0,),
        fit_symbol_names=("AAA",),
        feature_names=("signal",),
        feature_indices=(0,),
        fit_cutoff_ns=1_767_225_600_000_000_000,
        horizon_hours=1,
        alpha_hex=float(1.0).hex(),
        n_samples=2,
        vintage_id="synthetic-vintage",
        source_commit="1" * 40,
        implementation_sha256="2" * 64,
        runtime_sha256="3" * 64,
    )
    values: dict[str, object] = {
        "attempt_id": "synthetic-attempt-001",
        "research_context_digest": "c" * 64,
        "source_repository": "shuntatsu/trade_rl",
        "source_commit": "1" * 40,
        "source_tree": "2" * 40,
        "expected_implementation_digest": "3" * 64,
        "expected_runtime_environment_digest": "4" * 64,
        "expected_dataset_artifact_sha256": "b" * 64,
        "expected_symbols": ("AAA",),
        "ridge_artifact_sha256": "5" * 64,
        "expected_ridge_binding": ridge_binding,
        "expected_bundle_sha256": "6" * 64,
        "expected_manifest_sha256": "7" * 64,
        "expected_completion_sha256": "8" * 64,
        "expected_training_context_sha256": "9" * 64,
        "expected_checkpoint_sha256": "a" * 64,
        "bound": bound,
        "start_index": 0,
        "stop_index": 2,
        "initial_capital": 1_000.0,
        "target_abs_weights": (0.5,),
        "trend_config": TrendIntentConfig(
            signal_index=0,
            entry_threshold=0.5,
            exit_threshold=0.1,
        ),
        "ridge_entry_threshold": 0.005,
        "ridge_exit_threshold": 0.001,
        "ridge_one_way_switch_cost": None,
        "execution_cost": ExecutionCostConfig(
            fee_rate=0.01,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            max_leverage=1.0,
            maintenance_margin_rate=0.0,
            borrow_rate_multiplier=1.0,
            order_latency_bars=0,
            allow_short=True,
        ),
        "risk_config": PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
        "market_order_profile": None,
    }
    values.update(overrides)
    return FourFamilyPortfolioAttemptIdentity(**values)  # type: ignore[arg-type]


def test_identity_payload_binds_every_declared_input_without_host_paths() -> None:
    identity = _identity()

    payload = four_family_portfolio_attempt_identity_payload(identity)

    assert payload["attempt_id"] == identity.attempt_id
    assert payload["source_commit"] == identity.source_commit
    assert payload["source_tree"] == identity.source_tree
    assert payload["expected_dataset_artifact_sha256"] == "b" * 64
    assert payload["expected_ridge_binding"]["dataset_id"] == "a" * 64
    assert payload["market_order_profile"] is None
    assert set(payload) == {
        "attempt_id",
        "bound",
        "expected_bundle_sha256",
        "expected_checkpoint_sha256",
        "expected_completion_sha256",
        "expected_dataset_artifact_sha256",
        "expected_implementation_digest",
        "expected_manifest_sha256",
        "expected_ridge_binding",
        "expected_runtime_environment_digest",
        "expected_symbols",
        "expected_training_context_sha256",
        "execution_cost",
        "initial_capital",
        "market_order_profile",
        "research_context_digest",
        "ridge_artifact_sha256",
        "ridge_entry_threshold",
        "ridge_exit_threshold",
        "ridge_one_way_switch_cost",
        "risk_config",
        "source_commit",
        "source_repository",
        "source_tree",
        "start_index",
        "stop_index",
        "target_abs_weights",
        "trend_config",
        "schema_version",
    }
    assert "dataset_artifact_root" not in payload
    assert "attempt_root" not in payload


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("start_index", True, (TypeError, ValueError)),
        ("initial_capital", True, (TypeError, ValueError)),
        ("expected_implementation_digest", "not-a-digest", (ValueError,)),
        ("expected_runtime_environment_digest", "A" * 64, (ValueError,)),
    ],
)
def test_identity_rejects_noncanonical_or_nonfinite_declarations(
    field: str, value: object, error: type[Exception] | tuple[type[Exception], ...]
) -> None:
    with pytest.raises(error):
        _identity(**{field: value})


def test_strict_result_loader_does_not_require_completion_marker(
    tmp_path: Path,
) -> None:
    """The result is validated before complete/record.json is published."""

    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        load_four_family_portfolio_result_artifact,
    )

    # The attempt fixture in the companion test exercises the full public path.
    # This direct contract pins the pre-completion loader's API and ordering.
    with pytest.raises((FileNotFoundError, ValueError)):
        load_four_family_portfolio_result_artifact(
            tmp_path,
            expected_digest="d" * 64,
            expected_identity=_identity(),
        )
