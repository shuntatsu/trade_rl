from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("gymnasium")

from trade_rl.evaluation.experiments.four_family_input_manifest import (
    ComparisonAttemptSpec,
    DatasetInputPin,
    FourFamilyPostFreezeInputManifest,
    PPOTrainingDeclaration,
    WindowSpec,
    verify_bytes,
)
from trade_rl.evaluation.runs import FourFamilyPortfolioAttemptIdentity
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.forecasts.ridge_artifact import RidgeForecastArtifactBinding
from trade_rl.strategies.rl.portfolio_training import SharedPortfolioPPOConfig
from trade_rl.strategies.rules.trend import TrendIntentConfig


def _datetime_ns(value: datetime) -> int:
    delta = value.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        delta.days * 86_400 + delta.seconds
    ) * 1_000_000_000 + delta.microseconds * 1_000


def _bound(
    start: datetime,
    hours: int,
    *,
    economics_digest: str = "4" * 64,
    gamma: float = 1.0,
) -> BoundObjectiveClock:
    return BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=start,
            evaluation_stop_exclusive=start + timedelta(hours=hours),
            terminal_valuation="settled",
            economics_digest=economics_digest,
            risk_digest="5" * 64,
            deployment_recipe_digest="6" * 64,
            maximum_drawdown=0.2,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=hours * 3_600,
            rollout_steps=2,
            gamma=gamma,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )


def _identity(
    *,
    attempt_id: str,
    bound: BoundObjectiveClock,
    start: int,
    stop: int,
    ridge_binding: RidgeForecastArtifactBinding,
    cost: ExecutionCostConfig | None = None,
) -> FourFamilyPortfolioAttemptIdentity:
    return FourFamilyPortfolioAttemptIdentity(
        attempt_id=attempt_id,
        research_context_digest="c" * 64,
        source_repository="shuntatsu/trade_rl",
        source_commit="1" * 40,
        source_tree="2" * 40,
        expected_implementation_digest="3" * 64,
        expected_runtime_environment_digest="4" * 64,
        expected_dataset_artifact_sha256="b" * 64,
        expected_symbols=("AAA",),
        ridge_artifact_sha256="5" * 64,
        expected_ridge_binding=ridge_binding,
        expected_bundle_sha256="6" * 64,
        expected_manifest_sha256="7" * 64,
        expected_completion_sha256="8" * 64,
        expected_training_context_sha256="9" * 64,
        expected_checkpoint_sha256="a" * 64,
        bound=bound,
        start_index=start,
        stop_index=stop,
        initial_capital=1_000.0,
        target_abs_weights=(0.5,),
        trend_config=TrendIntentConfig(0, 0.5, 0.1),
        ridge_entry_threshold=0.005,
        ridge_exit_threshold=0.001,
        ridge_one_way_switch_cost=None,
        execution_cost=cost
        or ExecutionCostConfig(
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
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ),
        market_order_profile=None,
    )


def _market_order_profile(*, reduce_only_exits: bool = True):
    import hashlib

    import numpy as np

    from trade_rl.data.market import MarketDataset
    from trade_rl.integrations.binance.market_order_profile import (
        build_usdm_market_order_profile,
    )
    from trade_rl.integrations.binance.metadata import BinanceExchangeInfoSnapshot

    symbols = ("AAAUSDT",)
    shape = (6, len(symbols))
    prices = np.full(shape, 100.0)
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((*shape, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=prices,
        high=prices + 1.0,
        low=prices - 1.0,
        close=prices,
        volume=np.full(shape, 1_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((*shape, 1), dtype=bool),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        minimum_notional=np.full(shape, 50.0),
        lot_size=np.full(shape, 0.001),
    ).with_content_identity({"fixture": "four-family-manifest-profile"})
    payload = {
        "symbols": [
            {
                "symbol": "AAAUSDT",
                "status": "TRADING",
                "baseAsset": "AAA",
                "quoteAsset": "USDT",
                "orderTypes": ["MARKET"],
                "contractType": "PERPETUAL",
                "marginAsset": "USDT",
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "minPrice": "0.01",
                        "maxPrice": "1000000",
                        "tickSize": "0.01",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.001",
                        "maxQty": "1000",
                        "stepSize": "0.001",
                    },
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "minQty": "0",
                        "maxQty": "100",
                        "stepSize": "0",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "50"},
                ],
            }
        ]
    }
    raw_payload = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    snapshot = BinanceExchangeInfoSnapshot(
        payload=payload,
        raw_payload=raw_payload,
        source_uri="https://fapi.binance.com/fapi/v1/exchangeInfo",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
    )
    return build_usdm_market_order_profile(
        dataset,
        snapshot,
        selected_symbols=symbols,
        account_mode="one_way",
        reduce_only_exits=reduce_only_exits,
    )


def _manifest_with_market_order_profile(manifest, profile):
    dataset = replace(
        manifest.dataset,
        dataset_id=profile.dataset_id,
        symbols=profile.dataset_symbols,
    )
    ridge_binding = replace(
        manifest.attempts[0].identity.expected_ridge_binding,
        dataset_id=profile.dataset_id,
        fit_symbol_names=profile.dataset_symbols,
    )
    attempts = tuple(
        replace(
            attempt,
            identity=replace(
                attempt.identity,
                expected_symbols=profile.dataset_symbols,
                expected_ridge_binding=ridge_binding,
                market_order_profile=profile,
            ),
        )
        for attempt in manifest.attempts
    )
    return replace(
        manifest,
        dataset=dataset,
        attempts=attempts,
        ppo_training=replace(manifest.ppo_training, market_order_profile=profile),
    )


@pytest.fixture
def input_manifest() -> FourFamilyPostFreezeInputManifest:
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    dataset = DatasetInputPin(
        artifact_sha256="b" * 64,
        dataset_id="a" * 64,
        schema_version="market_dataset_v1",
        symbols=("AAA",),
        feature_names=("signal",),
        global_feature_names=("regime",),
        n_bars=9,
    )
    training = WindowSpec("training", 0, 2, _bound(origin, 2))
    selection = WindowSpec("selection", 2, 4, _bound(origin + timedelta(hours=2), 2))
    evaluation = WindowSpec("evaluation", 4, 7, _bound(origin + timedelta(hours=4), 3))
    ridge_binding = RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=dataset.schema_version,
        dataset_artifact_sha256=dataset.artifact_sha256,
        fit_symbol_indices=(0,),
        fit_symbol_names=("AAA",),
        feature_names=dataset.feature_names,
        feature_indices=(0,),
        fit_cutoff_ns=_datetime_ns(selection.bound.objective.evaluation_start),
        horizon_hours=1,
        alpha_hex=float(1.0).hex(),
        n_samples=2,
        vintage_id="synthetic-vintage",
        source_commit="1" * 40,
        implementation_sha256="2" * 64,
        runtime_sha256="3" * 64,
    )
    attempts = (
        ComparisonAttemptSpec(
            "selection",
            _identity(
                attempt_id="selection-attempt",
                bound=selection.bound,
                start=selection.start_index,
                stop=selection.stop_index,
                ridge_binding=ridge_binding,
            ),
        ),
        ComparisonAttemptSpec(
            "evaluation",
            _identity(
                attempt_id="evaluation-attempt",
                bound=evaluation.bound,
                start=evaluation.start_index,
                stop=evaluation.stop_index,
                ridge_binding=ridge_binding,
            ),
        ),
    )
    identity = attempts[0].identity
    return FourFamilyPostFreezeInputManifest(
        dataset=dataset,
        training_window=training,
        selection_window=selection,
        evaluation_window=evaluation,
        ppo_training=PPOTrainingDeclaration(
            training_context_sha256="9" * 64,
            transform_digest="d" * 64,
            config=SharedPortfolioPPOConfig(
                total_timesteps=8,
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                reset_seed=11,
                ppo_seed=12,
                gae_lambda=0.95,
            ),
            target_abs_weights=identity.target_abs_weights,
            execution_cost=identity.execution_cost,
            risk_config=identity.risk_config,
            market_order_profile=identity.market_order_profile,
        ),
        attempts=attempts,
    )


def test_fixed_roster_and_caps_count_four_family_attempts(input_manifest) -> None:
    assert input_manifest.family_roster == (
        "cash",
        "fixed_trend",
        "ridge_forecast",
        "joint_ppo",
    )
    assert input_manifest.scenario_roster == ("base",)
    assert input_manifest.fit_cap == 0
    assert input_manifest.child_attempt_cap == 2
    assert input_manifest.family_result_row_cap == 8
    assert tuple(item.stage for item in input_manifest.attempts) == (
        "selection",
        "evaluation",
    )


def test_verify_bytes_requires_exact_canonical_expected_declaration(
    input_manifest,
) -> None:
    raw = input_manifest.to_bytes()
    assert verify_bytes(raw, expected=input_manifest) is input_manifest
    with pytest.raises(ValueError, match="canonical"):
        verify_bytes(raw + b" ", expected=input_manifest)
    with pytest.raises(ValueError, match="expected declaration"):
        verify_bytes(
            raw,
            expected=replace(
                input_manifest,
                ppo_training=replace(
                    input_manifest.ppo_training, transform_digest="e" * 64
                ),
            ),
        )


def test_non_null_market_profile_is_bound_in_ppo_and_both_attempts(
    input_manifest,
) -> None:
    profile = _market_order_profile()
    declared = _manifest_with_market_order_profile(input_manifest, profile)
    raw = declared.to_bytes()
    profile_payload = {
        "declaration": profile.canonical_payload(),
        "digest": profile.digest,
    }

    payload = json.loads(raw)
    assert payload["ppo_training"]["market_order_profile"] == profile_payload
    assert all(
        attempt["identity_declaration"]["market_order_profile"] == profile_payload
        for attempt in payload["attempts"]
    )
    assert verify_bytes(raw, expected=declared) is declared

    changed_profile = _market_order_profile(reduce_only_exits=False)
    assert changed_profile.digest != profile.digest
    with pytest.raises(ValueError, match="financial declaration"):
        replace(
            declared,
            ppo_training=replace(
                declared.ppo_training, market_order_profile=changed_profile
            ),
        )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [("n_steps", 4), ("gae_lambda", 0.8)],
)
def test_valid_ppo_clock_mismatches_are_rejected(input_manifest, field, changed_value):
    changed_config = replace(
        input_manifest.ppo_training.config, **{field: changed_value}
    )

    with pytest.raises(ValueError, match="PPO config differs from the training clock"):
        replace(
            input_manifest,
            ppo_training=replace(input_manifest.ppo_training, config=changed_config),
        )


def test_manifest_digest_matches_fixed_current_v1_compatibility_vector(
    input_manifest,
) -> None:
    # Fixed domain-separated vector for this fixture; keep the expected value literal.
    assert input_manifest.digest == (
        "cd0fad5d97f957210429ddf8c9cb99e0c1084bad0a975084f28825c8eefb154b"
    )

    changed = replace(
        input_manifest,
        ppo_training=replace(input_manifest.ppo_training, transform_digest="e" * 64),
    )
    assert changed.digest != input_manifest.digest


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"\xff", "UTF-8"),
        (
            b'{"schema":"four_family_post_freeze_input_manifest_v1",'
            b'"schema":"four_family_post_freeze_input_manifest_v1"}',
            "duplicate",
        ),
        (
            b'{"schema":"four_family_post_freeze_input_manifest_v1","extra":NaN}',
            "non-finite",
        ),
    ],
)
def test_verify_bytes_rejects_invalid_encoding_duplicate_and_nonfinite(
    raw, message, input_manifest
) -> None:
    with pytest.raises(ValueError, match=message):
        verify_bytes(raw, expected=input_manifest)


def test_verify_bytes_rejects_wrong_schema_and_unexpected_fields(
    input_manifest,
) -> None:
    raw = input_manifest.to_bytes()
    wrong_schema = raw.replace(
        b"four_family_post_freeze_input_manifest_v1",
        b"another_manifest_schema_v1",
        1,
    )
    with pytest.raises(ValueError, match="schema"):
        verify_bytes(wrong_schema, expected=input_manifest)

    payload = json.loads(raw)
    payload["undeclared"] = True
    altered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="expected declaration"):
        verify_bytes(altered, expected=input_manifest)


def test_common_cost_must_match_between_comparison_attempts(input_manifest) -> None:
    selection, evaluation = input_manifest.attempts
    changed_cost = replace(evaluation.identity.execution_cost, spread_rate=0.001)
    changed_identity = replace(evaluation.identity, execution_cost=changed_cost)
    with pytest.raises(ValueError, match="share all non-window inputs"):
        replace(
            input_manifest,
            attempts=(selection, replace(evaluation, identity=changed_identity)),
        )


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("execution_cost", "different_cost"),
        ("risk_config", "different_risk"),
        ("target_abs_weights", (0.4,)),
    ],
)
def test_attempt_financials_must_match_declared_ppo_training_context(
    input_manifest, field, changed_value
) -> None:
    def changed_identity(identity):
        if field == "execution_cost":
            value = replace(identity.execution_cost, spread_rate=0.001)
        elif field == "risk_config":
            value = replace(identity.risk_config, drawdown_start=0.05)
        else:
            value = changed_value
        return replace(identity, **{field: value})

    changed_attempts = tuple(
        replace(attempt, identity=changed_identity(attempt.identity))
        for attempt in input_manifest.attempts
    )

    with pytest.raises(ValueError, match="financial declaration"):
        replace(input_manifest, attempts=changed_attempts)


def test_attempt_ids_must_be_unique(input_manifest) -> None:
    selection, evaluation = input_manifest.attempts
    duplicate_id = replace(
        evaluation,
        identity=replace(evaluation.identity, attempt_id=selection.identity.attempt_id),
    )

    with pytest.raises(ValueError, match="attempt IDs must be unique"):
        replace(input_manifest, attempts=(selection, duplicate_id))


@pytest.mark.parametrize("change", ["objective", "clock"])
def test_full_objective_and_clock_contracts_match_across_windows(
    input_manifest, change
) -> None:
    selection, evaluation = input_manifest.attempts
    old_bound = input_manifest.evaluation_window.bound
    if change == "objective":
        bad_bound = _bound(
            old_bound.objective.evaluation_start,
            3,
            economics_digest="f" * 64,
        )
    else:
        bad_bound = _bound(
            old_bound.objective.evaluation_start,
            3,
            gamma=0.9,
        )
    bad_window = replace(input_manifest.evaluation_window, bound=bad_bound)
    bad_identity = replace(evaluation.identity, bound=bad_bound)
    with pytest.raises(ValueError, match="may differ only"):
        replace(
            input_manifest,
            evaluation_window=bad_window,
            attempts=(selection, replace(evaluation, identity=bad_identity)),
        )


def test_windows_and_attempts_are_bound_and_chronological(input_manifest) -> None:
    selection, evaluation = input_manifest.attempts
    with pytest.raises(ValueError, match="exactly bind"):
        replace(
            input_manifest,
            attempts=(
                selection,
                replace(
                    evaluation,
                    identity=replace(evaluation.identity, start_index=3, stop_index=6),
                ),
            ),
        )

    overlapping = replace(input_manifest.evaluation_window, start_index=3, stop_index=6)
    with pytest.raises(ValueError, match="chronological"):
        replace(input_manifest, evaluation_window=overlapping)


def test_ridge_cutoff_after_selection_is_rejected(input_manifest) -> None:
    selection, evaluation = input_manifest.attempts
    binding = replace(
        selection.identity.expected_ridge_binding,
        fit_cutoff_ns=selection.identity.expected_ridge_binding.fit_cutoff_ns + 1,
    )
    changed_selection = replace(
        selection,
        identity=replace(selection.identity, expected_ridge_binding=binding),
    )
    changed_evaluation = replace(
        evaluation,
        identity=replace(evaluation.identity, expected_ridge_binding=binding),
    )
    with pytest.raises(ValueError, match="Ridge fit cutoff"):
        replace(input_manifest, attempts=(changed_selection, changed_evaluation))
