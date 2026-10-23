"""Closed, result-blind specification for the artificial H component path."""

from __future__ import annotations

from tools.market_replay_oracle import canonical

_SPEC_SCHEMA = "issue810_artificial_h_component_spec_v1"
_PULSES = (2905, 3995, 5082, 5089, 6191, 7290, 7297, 8362, 9425, 9429)
_FEATURES = (
    "fixture_pulse",
    "fixture_log_return",
    "fixture_body_return",
    "fixture_volume_change",
    "fixture_close_location",
    "fixture_lower_wick",
)


def is_artificial_h_component_spec(value: object) -> bool:
    """Accept only the exact fixed synthetic input recipe, before results exist."""
    if type(value) is not dict:
        return False
    expected = {
        "schema": _SPEC_SCHEMA,
        "scope": "ARTIFICIAL_COMPARISON_ONLY",
        "dataset_label": "synthetic-issue810-component-v1",
        "dataset_identity_binding": "computed_by_pinned_H_MarketDatasetBuilder",
        "protocol_id": "synthetic-issue810-component-protocol-v1",
        "configuration_id": "H24_RIDGE_A1",
        "symbol": "BTCUSDT",
        "scenario": "BASEL0",
        "actor_start": 2903,
        "actor_stop_exclusive": 9430,
        "actor_advances": 6527,
        "reserve_advances": 72,
        "role_pair": ["NONRL", "CASH"],
        "row_count": 9504,
        "pulse_indices": list(_PULSES),
        "feature_names": list(_FEATURES),
        "dataset_recipe": {
            "timestamp_origin_utc": "2024-01-01T00:00:00Z",
            "interval_seconds": 3600,
            "base_timeframe": "1h",
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "volume_unit": "QUOTE_NOTIONAL",
            "availability": "exact_row_timestamp",
            "tradable": True,
            "ohlc": {
                "open": 100.0,
                "close": 100.0,
                "ordinary_high": 101.0,
                "ordinary_low": 99.0,
                "selected_pulse_high": 151.0,
                "selected_pulse_low": 49.0,
                "pulse_symbol": "BTCUSDT",
            },
            "quote_volume_per_bar": 10000000.0,
            "funding_rate": 0.0,
            "funding_price_rate": 0.0,
            "funding_event_stride": 8,
            "instrument_rules": {
                "BTCUSDT": {
                    "tick_size": 0.1,
                    "lot_size": 0.001,
                    "minimum_notional": 50.0,
                },
                "ETHUSDT": {
                    "tick_size": 0.01,
                    "lot_size": 0.001,
                    "minimum_notional": 20.0,
                },
            },
            "feature_kinds": [
                "HIGH_LOW_RANGE",
                "LOG_RETURN",
                "BODY_RETURN",
                "VOLUME_LOG_CHANGE",
                "CLOSE_LOCATION_VALUE",
                "LOWER_WICK_RATIO",
            ],
        },
        "account_recipe": {
            "initial_cash": 100000.0,
            "initial_quantities": [0.0, 0.0],
            "selected_symbol": "BTCUSDT",
            "other_symbol_always_zero": True,
        },
        "cost_recipe": {
            "fee_rate": 0.0005,
            "spread_rate": 0.0002,
            "max_participation_rate": 0.05,
            "borrow_available": True,
            "borrow_rate": 0.0,
            "market_fee_overlay": 0.0,
            "market_spread_overlay": 0.0,
            "impact_coefficient": 0.0001,
            "cash_rate": 0.0,
            "funding_rate": 0.0,
            "stochastic_slippage": 0.0,
            "random_seed": 0,
            "reference_notional": 100000.0,
            "funding_lookback_hours": 24,
            "funding_minimum_event_count": 2,
            "funding_maximum_counted_close_gap_hours": 9.0,
            "funding_maximum_counted_close_age_hours": 8.0,
        },
        "risk_recipe": {
            "max_gross": 1.0,
            "max_abs_weight": 1.0,
            "drawdown_start": 0.1,
            "drawdown_stop": 0.2,
            "emergency_turnover_override": True,
            "fail_closed_tolerance": 1e-10,
        },
        "allocator_recipe": {
            "lower_weight": -1.0,
            "upper_weight": 1.0,
            "max_turnover": None,
            "risk_aversion": 0.0,
        },
        "forecast_recipe": {
            "method": "fixed_projection_from_stored_float32_inputs_no_fit",
            "fit_called": False,
            "horizon_hours": 24,
            "ridge_alpha": 1.0,
            "vintage_starts": [2903, 5087, 7295],
            "feature_indices": [0, 1, 2, 3, 4, 5],
            "fit_symbol_indices": [0],
            "prediction_symbols": ["BTCUSDT", "ETHUSDT"],
            "training_trace_only": True,
            "mean": "first_feature_at_actor_start_then_zero",
            "scale": "ones",
            "intercept": -0.02,
            "pulse_expected_simple_return": 0.02,
            "nonpulse_expected_simple_return": -0.02,
        },
        "execution_recipe": {
            "order_type": "MARKET",
            "time_in_force": "GTC",
            "latency_bars": 0,
            "capacity_source": "prior_bar_quote_notional_volume",
            "market_fill_price_source": "next_processing_row_open",
            "decision_mark_source": "decision_row_close",
            "actor_stop_exclusive": 9430,
            "reserve_stop_inclusive": 9502,
            "row_9503_used": False,
        },
        "full_b4_authorized": False,
        "outer_attempt_authenticated": False,
    }
    try:
        return canonical(value) == canonical(expected)
    except (TypeError, ValueError):
        return False
