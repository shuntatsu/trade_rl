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


def _synthetic_precompletion_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, object, FourFamilyPortfolioAttemptIdentity, dict[str, object]]:
    """Build synthetic replay rows without fitting or loading a trained PPO bundle."""
    from dataclasses import replace

    from tests.evaluation.test_portfolio_family_comparison import (
        _FixedPPOPolicy,
        _prepared_inputs,
    )
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.runs import (
        portfolio_family_comparison as comparison_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        _family_payload,
        four_family_portfolio_attempt_identity_payload,
    )
    from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance

    inputs, _ = _prepared_inputs(tmp_path)
    monkeypatch.setattr(
        comparison_module,
        "load_portfolio_inference_bundle",
        lambda *args, **kwargs: _FixedPPOPolicy(),
    )
    result = run_four_family_shared_portfolio_replay(**inputs)
    provenance = build_candidate_run_provenance(research_context_digest="c" * 64)
    identity = replace(
        _identity(),
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_environment_digest=provenance["runtime_environment_digest"],
        expected_dataset_artifact_sha256=inputs["expected_dataset_artifact_sha256"],
        expected_symbols=inputs["expected_symbols"],
        ridge_artifact_sha256=inputs["expected_ridge_artifact_sha256"],
        expected_ridge_binding=inputs["expected_ridge_binding"],
        expected_bundle_sha256=inputs["expected_bundle_sha256"],
        expected_manifest_sha256=inputs["expected_manifest_sha256"],
        expected_completion_sha256=inputs["expected_completion_sha256"],
        expected_training_context_sha256=inputs["expected_training_context_sha256"],
        expected_checkpoint_sha256=inputs["expected_checkpoint_sha256"],
        bound=inputs["bound"],
        start_index=inputs["start_index"],
        stop_index=inputs["stop_index"],
        initial_capital=inputs["initial_capital"],
        target_abs_weights=inputs["target_abs_weights"],
        trend_config=inputs["trend_config"],
        ridge_entry_threshold=inputs["ridge_entry_threshold"],
        ridge_exit_threshold=inputs["ridge_exit_threshold"],
        ridge_one_way_switch_cost=inputs["ridge_one_way_switch_cost"],
        execution_cost=inputs["execution_cost"],
        risk_config=inputs["risk_config"],
        market_order_profile=inputs["market_order_profile"],
    )
    root = tmp_path / "attempt"
    root.mkdir()
    started = {
        "declared_roots": {
            "bundle_root": str(inputs["bundle_root"]),
            "dataset_artifact_root": str(inputs["dataset_artifact_root"]),
            "ridge_artifact_root": str(inputs["ridge_artifact_root"]),
        },
        "identity": four_family_portfolio_attempt_identity_payload(identity),
        "schema_version": "four_family_portfolio_attempt_started_v1",
    }
    started_path = root / "started" / "record.json"
    started_path.parent.mkdir(parents=True)
    started_path.write_bytes(canonical_json_bytes(started))
    for position in ("before", "after"):
        provenance_path = root / "provenance" / position / "record.json"
        provenance_path.parent.mkdir(parents=True)
        provenance_path.write_bytes(
            canonical_json_bytes(
                {
                    "provenance": provenance,
                    "schema_version": "four_family_portfolio_provenance_v1",
                }
            )
        )
    for outcome in result.families:
        family_path = root / "families" / outcome.family / "record.json"
        family_path.parent.mkdir(parents=True)
        family_path.write_bytes(canonical_json_bytes(_family_payload(outcome)))
    return root, result, identity, provenance


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


def test_identity_normalizes_optional_ridge_switch_cost_before_digest_binding() -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        four_family_portfolio_attempt_identity_digest,
    )

    integer_zero = _identity(ridge_one_way_switch_cost=0)
    float_zero = _identity(ridge_one_way_switch_cost=0.0)

    assert type(integer_zero.ridge_one_way_switch_cost) is float
    assert four_family_portfolio_attempt_identity_payload(
        integer_zero
    ) == four_family_portfolio_attempt_identity_payload(float_zero)
    assert four_family_portfolio_attempt_identity_digest(
        integer_zero
    ) == four_family_portfolio_attempt_identity_digest(float_zero)
    assert _identity(ridge_one_way_switch_cost=None).ridge_one_way_switch_cost is None
    unchanged_float = _identity(ridge_one_way_switch_cost=0.25)
    assert unchanged_float.ridge_one_way_switch_cost == 0.25
    assert (
        four_family_portfolio_attempt_identity_payload(unchanged_float)["attempt_id"]
        == integer_zero.attempt_id
    )


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


def test_strict_result_loader_rejects_missing_attempt_records(
    tmp_path: Path,
) -> None:
    """A missing attempt inventory is rejected independently of completion."""

    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        load_four_family_portfolio_result_artifact,
    )

    with pytest.raises(ValueError, match="missing or unexpected result files"):
        load_four_family_portfolio_result_artifact(
            tmp_path,
            expected_digest="d" * 64,
            expected_identity=_identity(),
        )


def test_strict_result_loader_does_not_require_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        four_family_portfolio_attempt_identity_digest,
        load_four_family_portfolio_result_artifact,
        publish_four_family_portfolio_result_artifact,
    )

    root, result, identity, provenance = _synthetic_precompletion_case(
        tmp_path, monkeypatch
    )

    digest = publish_four_family_portfolio_result_artifact(
        root,
        result,
        identity,
        before_provenance=provenance,
        after_provenance=provenance,
    )
    assert not (root / "complete" / "record.json").exists()

    loaded = load_four_family_portfolio_result_artifact(
        root,
        expected_digest=digest,
        expected_identity=identity,
    )

    assert loaded.artifact_digest == digest
    assert loaded.payload["identity_digest"] == (
        four_family_portfolio_attempt_identity_digest(identity)
    )
    assert len(loaded.payload["families"]) == 4
    with pytest.raises(ValueError, match="digest mismatch"):
        load_four_family_portfolio_result_artifact(
            root,
            expected_digest="e" * 64,
            expected_identity=identity,
        )


def test_publisher_rejects_provenance_arguments_that_differ_from_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        publish_four_family_portfolio_result_artifact,
    )

    root, result, identity, provenance = _synthetic_precompletion_case(
        tmp_path, monkeypatch
    )
    mismatched_after = dict(provenance)
    mismatched_after["research_context_digest"] = "f" * 64

    with pytest.raises(ValueError):
        publish_four_family_portfolio_result_artifact(
            root,
            result,
            identity,
            before_provenance=provenance,
            after_provenance=mismatched_after,
        )

    assert not (root / "result" / "manifest.json").exists()
    assert all(
        (root / "families" / family / "record.json").is_file()
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    )


def test_publisher_rejects_on_disk_provenance_outside_identity_pins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        publish_four_family_portfolio_result_artifact,
    )

    root, result, identity, provenance = _synthetic_precompletion_case(
        tmp_path, monkeypatch
    )
    mismatched_after = dict(provenance)
    mismatched_after["research_context_digest"] = "f" * 64
    after_path = root / "provenance" / "after" / "record.json"
    after_path.write_bytes(
        canonical_json_bytes(
            {
                "provenance": mismatched_after,
                "schema_version": "four_family_portfolio_provenance_v1",
            }
        )
    )

    with pytest.raises(ValueError, match="differs from expected pins"):
        publish_four_family_portfolio_result_artifact(
            root,
            result,
            identity,
            before_provenance=provenance,
            after_provenance=mismatched_after,
        )

    assert not (root / "result" / "manifest.json").exists()


def test_public_result_loader_rejects_a_tampered_family_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        load_four_family_portfolio_result_artifact,
        publish_four_family_portfolio_result_artifact,
    )

    root, result, identity, provenance = _synthetic_precompletion_case(
        tmp_path, monkeypatch
    )
    digest = publish_four_family_portfolio_result_artifact(
        root,
        result,
        identity,
        before_provenance=provenance,
        after_provenance=provenance,
    )
    row_path = root / "families" / "cash" / "record.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["actual_interval_count"] = 0
    row_path.write_bytes(canonical_json_bytes(row))

    with pytest.raises(ValueError, match="digest"):
        load_four_family_portfolio_result_artifact(
            root,
            expected_digest=digest,
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b'{"field":1,"field":2}',
        b'{"field": 1}',
    ],
    ids=("duplicate-key", "noncanonical-spacing"),
)
def test_strict_json_rejects_duplicate_keys_and_noncanonical_bytes(raw: bytes) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import _strict_json

    with pytest.raises(ValueError, match="duplicate JSON key|not canonical JSON"):
        _strict_json(raw, field="synthetic record")
