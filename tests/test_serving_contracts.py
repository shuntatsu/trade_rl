import pytest

from mars_lite.serving.contracts import InferenceState, PendingOrderInput


def _state(**overrides) -> InferenceState:
    values = {
        "current_weights": {"BTCUSDT": 0.0},
        "portfolio_value": 100.0,
        "day_start_value": 100.0,
        "peak_value": 100.0,
        "consecutive_losses": 0,
        "turnover_mean": 0.0,
        "turnover_std": 1.0,
    }
    values.update(overrides)
    return InferenceState(**values)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("disagreement", -0.1, "disagreement"),
        ("disagreement", 1.1, "disagreement"),
        ("turnover_mean", -0.1, "turnover_mean"),
        ("turnover_std", -0.1, "turnover_std"),
        ("consecutive_losses", 1.5, "consecutive_losses"),
        ("vol_scale", 1.1, "vol_scale"),
        ("dd_scale", -0.1, "dd_scale"),
        ("disagreement_scale", 1.1, "disagreement_scale"),
        ("est_port_vol", -0.1, "est_port_vol"),
    ],
)
def test_impossible_state_values_are_rejected(field, value, message) -> None:
    with pytest.raises(ValueError, match=message):
        _state(**{field: value}).validate(("BTCUSDT",))


def test_peak_must_include_day_start_value() -> None:
    with pytest.raises(ValueError, match="peak_value"):
        _state(portfolio_value=90.0, day_start_value=100.0, peak_value=95.0).validate(
            ("BTCUSDT",)
        )


def test_pending_order_reduce_only_must_be_boolean() -> None:
    order = PendingOrderInput(
        symbol="BTCUSDT",
        side="buy",
        notional=10.0,
        reduce_only="false",
    )
    with pytest.raises(ValueError, match="reduce_only"):
        _state(pending_orders=(order,)).validate(("BTCUSDT",))
