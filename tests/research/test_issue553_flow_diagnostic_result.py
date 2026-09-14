from __future__ import annotations

from dataclasses import replace

from tools.issue553_flow_diagnostic import (
    FlowReturnSample,
    build_flow_diagnostic_result,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def _ns(text: str) -> int:
    import numpy as np

    return int(np.datetime64(text, "ns").astype(np.int64))


def _accepted_days() -> tuple[str, ...]:
    return tuple(
        [f"2021-{month:02d}-01" for month in range(1, 11)]
        + [f"2022-{month:02d}-01" for month in range(1, 11)]
    )


def _samples(beta_2021: float, beta_2022: float) -> tuple[FlowReturnSample, ...]:
    result: list[FlowReturnSample] = []
    for year, beta in ((2021, beta_2021), (2022, beta_2022)):
        for month in range(1, 11):
            for hour in range(1, 25):
                x = (hour - 12.5) / 12.5
                y = 0.0001 + beta * x * 0.001
                result.append(
                    FlowReturnSample(
                        decision_timestamp_ns=_ns(
                            f"{year}-{month:02d}-01T{hour % 24:02d}:00:00"
                        ),
                        flow_imbalance=x,
                        execution_open=100.0,
                        label_end_open=100.0,
                        log_return=y,
                    )
                )
    return tuple(result)


def _valid_inputs() -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, tuple[FlowReturnSample, ...]],
]:
    accepted = {symbol: _accepted_days() for symbol in _SYMBOLS}
    betas = {
        "BTCUSDT": (1.0, 1.0),
        "ETHUSDT": (0.8, 0.8),
        "BNBUSDT": (0.6, 0.6),
        "XRPUSDT": (0.4, 0.4),
        "ADAUSDT": (-0.2, -0.2),
    }
    samples = {symbol: _samples(*betas[symbol]) for symbol in _SYMBOLS}
    return accepted, samples


def test_valid_stable_signs_pass_flow_screen() -> None:
    accepted, samples = _valid_inputs()

    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )

    assert result.status == "PASS_FLOW_SCREEN"
    assert result.coverage_valid is True
    assert result.positive_full_sample_symbols == 4
    assert result.positive_2021_symbols == 4
    assert result.positive_2022_symbols == 4
    assert len(result.symbols) == 5
    assert all(item.accepted_days_total == 20 for item in result.symbols)
    assert all(item.full.count == 480 for item in result.symbols)
    assert all(item.year_2021.count == 240 for item in result.symbols)
    assert all(item.year_2022.count == 240 for item in result.symbols)
    assert "pnl" not in result.to_payload()
    assert "strategy_return" not in result.to_payload()


def test_valid_but_unstable_signs_are_no_stable_flow_signal() -> None:
    accepted, samples = _valid_inputs()
    samples["XRPUSDT"] = _samples(-0.4, -0.4)

    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )

    assert result.status == "NO_STABLE_FLOW_SIGNAL"
    assert result.coverage_valid is True
    assert result.positive_full_sample_symbols == 3


def test_insufficient_days_or_observations_fail_coverage() -> None:
    accepted, samples = _valid_inputs()
    accepted["ADAUSDT"] = accepted["ADAUSDT"][:-1]

    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )
    assert result.status == "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
    assert result.coverage_valid is False
    assert any("accepted_days" in reason for reason in result.invalid_reasons)

    accepted, samples = _valid_inputs()
    samples["ADAUSDT"] = samples["ADAUSDT"][:-1]
    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )
    assert result.status == "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
    assert any("observations" in reason for reason in result.invalid_reasons)


def test_replacement_day_is_invalid_coverage_not_a_new_roster() -> None:
    accepted, samples = _valid_inputs()
    accepted["BTCUSDT"] = ("2021-01-02",) + accepted["BTCUSDT"][1:]

    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )

    assert result.status == "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
    assert result.coverage_valid is False
    assert any("frozen first-of-month roster" in reason for reason in result.invalid_reasons)


def test_zero_variance_symbol_is_invalid_coverage() -> None:
    accepted, samples = _valid_inputs()
    flat = tuple(replace(sample, flow_imbalance=0.25) for sample in samples["ADAUSDT"])
    samples["ADAUSDT"] = flat

    result = build_flow_diagnostic_result(
        accepted_days_by_symbol=accepted,
        samples_by_symbol=samples,
    )

    assert result.status == "INVALID_FLOW_DIAGNOSTIC_COVERAGE"
    assert result.coverage_valid is False
    assert any("variance" in reason for reason in result.invalid_reasons)
