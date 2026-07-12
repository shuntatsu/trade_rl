from mars_lite.pipeline.residual_pipeline import select_residual_configuration


def _result(excess: float, hybrid_dd: float = 0.10, shadow_dd: float = 0.10):
    return {
        "paired": {"excess_log_return": excess},
        "hybrid": {"max_drawdown": hybrid_dd},
        "shadow": {"max_drawdown": shadow_dd},
    }


def _matrix(a: float, b: float, c: float = 0.0, d: float | None = None):
    values = {"A": _result(a), "B": _result(b), "C": _result(c)}
    if d is not None:
        values["D"] = _result(d)
    return values


def test_selects_identity_when_rl_does_not_improve_base() -> None:
    selected = select_residual_configuration(
        _matrix(0.0, -0.01, 0.02, 0.01),
        cost2x_results=_matrix(0.0, -0.02, 0.01, -0.01),
    )
    assert selected["selected"] == "A"
    assert selected["policy_mode"] == "baseline_only"


def test_selects_trend_mix_when_b_beats_a_and_d_does_not_beat_c() -> None:
    selected = select_residual_configuration(
        _matrix(0.0, 0.03, 0.04, 0.035),
        cost2x_results=_matrix(0.0, 0.01, 0.02, 0.005),
    )
    assert selected["selected"] == "B"
    assert selected["policy_mode"] == "ppo_residual_ensemble"


def test_selects_combined_rl_only_when_d_beats_b_and_c() -> None:
    selected = select_residual_configuration(
        _matrix(0.0, 0.02, 0.03, 0.05),
        cost2x_results=_matrix(0.0, 0.005, 0.01, 0.02),
    )
    assert selected["selected"] == "D"


def test_d_cannot_enable_alpha_when_fixed_alpha_c_does_not_beat_a() -> None:
    selected = select_residual_configuration(
        _matrix(0.0, 0.02, -0.01, 0.05),
        cost2x_results=_matrix(0.0, 0.005, -0.01, 0.02),
    )

    assert selected["selected"] == "B"
    assert selected["eligible"]["C"] is False
    assert selected["eligible"]["D"] is True


def test_rejects_high_drawdown_candidate() -> None:
    selected = select_residual_configuration(
        {"A": _result(0.0), "B": _result(0.04, 0.20, 0.10), "C": _result(0.0)},
        cost2x_results=_matrix(0.0, 0.01, 0.0),
    )
    assert selected["selected"] == "A"


def test_rejects_candidate_that_fails_cost2x_even_when_cost1x_is_positive() -> None:
    selected = select_residual_configuration(
        _matrix(0.0, 0.05, 0.0),
        cost2x_results=_matrix(0.0, -0.01, 0.0),
    )

    assert selected["selected"] == "A"
    assert selected["cost2x_eligible"]["B"] is False
