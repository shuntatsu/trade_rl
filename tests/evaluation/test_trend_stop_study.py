"""Trend-stop study: protocol freeze, refusals, metrics oracles and cell writing.

Everything here runs on in-process synthetic bars or hand-built ledgers. No market
data, network or venue access is involved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation.trend_stop_support import AUTHORIZATION_URL, synthetic_dataset
from tests.strategies.test_trend_stop import (
    BREAKOUT,
    FLAT_BARS,
    GAP_DOWN,
    RUN_UP,
    _config,
    _dataset,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.trend_stop_study import cli, metrics, runner
from trade_rl.evaluation.trend_stop_study.data import (
    build_dataset_from_transport,
    build_vision_dataset,
    plan_vision_cache,
    validate_study_dataset,
)
from trade_rl.evaluation.trend_stop_study.protocol import (
    DATA_END_EXCLUSIVE,
    LABEL_NO_WINNER,
    LABEL_PROMISING,
    PROTOCOL,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    VARIANTS,
    check_protocol_window,
    hard_risk,
    protocol_digest,
    scenario_cost,
    trend_config,
    validate_authorization_url,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.rules.trend_stop import TrendStopPortfolioPolicy

SMALL = replace(
    PROTOCOL,
    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
    min_symbols=3,
    evaluation_start="2021-03-01T00:00:00",
    evaluation_end="2021-08-01T00:00:00",
    data_end_exclusive="2021-08-01T00:00:00",
    causality_cut="2021-06-01T00:00:00",
    sub_years=("2021",),
    min_positive_sub_years=1,
    min_closed_trades=5,
    bootstrap_resamples=200,
)


def _small_datasets(last_close: str = "2021-08-01T00:00:00") -> dict[str, Any]:
    return {
        "4h": synthetic_dataset(
            "4h",
            SMALL.symbols,
            first_close="2021-01-01T04:00:00",
            last_close=last_close,
        ),
        "1d": synthetic_dataset(
            "1d",
            SMALL.symbols,
            first_close="2021-01-02T00:00:00",
            last_close=last_close,
        ),
    }


# -- protocol -----------------------------------------------------------------------


def test_protocol_is_frozen_before_any_data_and_has_at_most_two_variants() -> None:
    assert len(VARIANTS) == 2
    assert [(v.entry_bars, v.exit_bars) for v in VARIANTS] == [(20, 10), (55, 20)]
    assert PROTOCOL.timeframes == ("4h", "1d")
    assert PROTOCOL.n_configurations == len(VARIANTS) * len(PROTOCOL.timeframes)
    assert PROTOCOL.evaluation_end == PROTOCOL.data_end_exclusive == DATA_END_EXCLUSIVE
    assert DATA_END_EXCLUSIVE == "2026-08-01T00:00:00"
    assert PROTOCOL.benchmark_symbol in PROTOCOL.symbols
    assert len(PROTOCOL.symbols) == 11
    assert (
        PROTOCOL.atr_period,
        PROTOCOL.initial_stop_atr,
        PROTOCOL.risk_per_trade,
    ) == (
        14,
        2.0,
        0.005,
    )
    # Deterministic and content-addressed.
    assert protocol_digest(PROTOCOL) == protocol_digest(replace(PROTOCOL))
    assert protocol_digest(PROTOCOL) != protocol_digest(
        replace(PROTOCOL, initial_stop_atr=2.5)
    )
    json.dumps(PROTOCOL.canonical_payload(), allow_nan=False)


def test_scenario_costs_double_every_trading_cost_but_not_the_base() -> None:
    base = scenario_cost(PROTOCOL, SCENARIO_BASE)
    stressed = scenario_cost(PROTOCOL, SCENARIO_COST_X2)
    default = ExecutionCostConfig()
    assert (base.fee_rate, base.spread_rate, base.impact_rate) == (
        default.fee_rate,
        default.spread_rate,
        default.impact_rate,
    )
    assert stressed.fee_rate == 2 * base.fee_rate
    assert stressed.spread_rate == 2 * base.spread_rate
    assert stressed.impact_rate == 2 * base.impact_rate
    with pytest.raises(ValueError):
        scenario_cost(PROTOCOL, "latency_1bar")


def test_derived_rule_and_risk_objects_match_the_protocol() -> None:
    config = trend_config(PROTOCOL, VARIANTS[1])
    assert (config.entry_bars, config.exit_bars, config.risk_per_trade) == (
        55,
        20,
        0.005,
    )
    risk = hard_risk(PROTOCOL).config
    assert risk.max_gross >= PROTOCOL.gross_cap
    assert risk.max_abs_weight >= PROTOCOL.max_symbol_weight
    assert risk.drawdown_start == PROTOCOL.max_drawdown_limit


def test_windows_past_the_exploratory_boundary_are_refused() -> None:
    check_protocol_window(PROTOCOL)
    with pytest.raises(ValueError, match="exploratory data boundary"):
        check_protocol_window(replace(PROTOCOL, evaluation_end="2026-08-02T00:00:00"))
    with pytest.raises(ValueError, match="frozen boundary"):
        check_protocol_window(
            replace(
                PROTOCOL,
                data_end_exclusive="2026-08-02T00:00:00",
                evaluation_end="2026-08-02T00:00:00",
            )
        )


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/someone/else/issues/1#issuecomment-1",
        "http://github.com/shuntatsu/trade_rl/issues/1#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/1#issuecomment-x",
        "https://github.com/shuntatsu/trade_rl/issues/1#issuecomment-1 ",
    ],
)
def test_runner_requires_a_wellformed_g4_authorization_url(url: Any) -> None:
    with pytest.raises(ValueError, match="G4 authorization"):
        validate_authorization_url(url)
    with pytest.raises(ValueError, match="G4 authorization"):
        runner.run_study(_small_datasets(), SMALL, g4_authorization_url=url)


def test_authorization_url_shape_accepts_issue_and_pull_comments() -> None:
    for url in (
        AUTHORIZATION_URL,
        "https://github.com/shuntatsu/trade_rl/pull/12#issuecomment-345",
    ):
        assert validate_authorization_url(url) == url


# -- refusals before any replay -------------------------------------------------------


def test_refusals_happen_before_any_cell_is_replayed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a replay ran before a refusal")

    monkeypatch.setattr(runner, "run_rule_cell", forbidden)
    monkeypatch.setattr(runner, "run_benchmark_cell", forbidden)
    monkeypatch.setattr(runner, "causality_probe", forbidden)

    with pytest.raises(ValueError, match="G4 authorization"):
        runner.run_study(_small_datasets(), SMALL, g4_authorization_url=None)
    with pytest.raises(ValueError, match="exploratory data boundary"):
        runner.run_study(
            _small_datasets(),
            replace(SMALL, evaluation_end="2021-08-02T00:00:00"),
            g4_authorization_url=AUTHORIZATION_URL,
        )
    with pytest.raises(ValueError, match="holds bars after"):
        runner.run_study(
            _small_datasets("2021-08-02T00:00:00"),
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
        )
    partial = _small_datasets()
    del partial["1d"]
    with pytest.raises(ValueError, match="exactly the preregistered timeframes"):
        runner.run_study(partial, SMALL, g4_authorization_url=AUTHORIZATION_URL)


def test_cli_refuses_without_authorization_and_never_loads_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("data was loaded before the authorization check")

    monkeypatch.setattr(cli, "load_exported_dataset", forbidden)
    monkeypatch.setattr(cli, "build_vision_dataset", forbidden)
    out = tmp_path / "out"

    code = cli.main(["--output-dir", str(out), "--g4-authorization-url", "nope"])

    assert code == 2
    assert "G4 authorization" in capsys.readouterr().err
    assert not out.exists()


def test_cli_refuses_a_non_empty_output_and_missing_datasets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("existing", encoding="utf-8")
    assert (
        cli.main(
            [
                "--output-dir",
                str(occupied),
                "--g4-authorization-url",
                AUTHORIZATION_URL,
            ]
        )
        == 2
    )
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "existing"

    fresh = tmp_path / "fresh"
    assert (
        cli.main(
            ["--output-dir", str(fresh), "--g4-authorization-url", AUTHORIZATION_URL]
        )
        == 2
    )
    assert "no 4h Dataset" in capsys.readouterr().err


# -- data path -------------------------------------------------------------------------


def _check(
    timeframe: str = "1d", dataset_timeframe: str = "1d", **overrides: Any
) -> None:
    dataset = overrides.pop("dataset", None) or synthetic_dataset(
        dataset_timeframe,
        SMALL.symbols,
        first_close="2021-01-02T00:00:00",
        last_close="2021-08-01T00:00:00",
    )
    validate_study_dataset(dataset, replace(SMALL, **overrides), timeframe)


def test_dataset_validation_accepts_the_regular_roster_grid() -> None:
    _check()
    _check(
        "4h",
        dataset=synthetic_dataset(
            "4h",
            SMALL.symbols,
            first_close="2021-01-01T04:00:00",
            last_close="2021-08-01T00:00:00",
        ),
    )


def test_dataset_validation_rejects_roster_cadence_warmup_and_boundary() -> None:
    with pytest.raises(ValueError, match="symbols outside the roster"):
        _check(symbols=("BTCUSDT", "ETHUSDT", "XRPUSDT"))
    with pytest.raises(ValueError, match="at least"):
        _check(min_symbols=4)
    with pytest.raises(ValueError, match="benchmark symbol"):
        _check(benchmark_symbol="BNBUSDT")
    with pytest.raises(ValueError, match="cadence"):
        _check("4h", dataset_timeframe="1d")  # a 1d grid is not a 4h grid
    with pytest.raises(ValueError, match="no warm-up"):
        _check(evaluation_start="2020-12-01T00:00:00")
    with pytest.raises(ValueError, match="holds bars after"):
        _check(data_end_exclusive="2021-07-31T00:00:00")


class _FakeTransport:
    """Vision-shaped transport serving seeded synthetic klines (no archives)."""

    def __init__(self, protocol: Any) -> None:
        self._protocol = protocol

    def load_klines(
        self, *, symbol: str, interval: str, start_ms: int, end_ms: int, **_: object
    ) -> tuple[list[list[object]], str]:
        step = {"4h": 14_400_000, "1d": 86_400_000}[interval]
        rows: list[list[object]] = []
        seed = sum(map(ord, symbol))
        rng = np.random.default_rng(seed)
        price = 100.0
        for open_ms in range(start_ms, end_ms, step):
            price *= float(np.exp(rng.normal(0.0, 0.01)))
            rows.append(
                [
                    open_ms,
                    repr(price),
                    repr(price * 1.01),
                    repr(price * 0.99),
                    repr(price),
                    "1000.0",
                    open_ms + step - 1,
                    "100000.0",
                    10,
                    "500.0",
                    "50000.0",
                    "0",
                ]
            )
        return rows, "fixture"

    def load_funding_rates(
        self, *, start_ms: int, end_ms: int, **_: object
    ) -> tuple[list[tuple[int, float]], str]:
        return [
            (ms, 0.0001) for ms in range(start_ms + 28_800_000, end_ms, 28_800_000)
        ], "fixture"

    def load_exchange_information(self, **_: object) -> tuple[dict[str, object], str]:
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "onboardDate": 1_569_398_400_000,
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"},
                    ],
                }
                for symbol in self._protocol.symbols
            ]
        }, "fixture"


@pytest.mark.parametrize("timeframe", ["4h", "1d"])
def test_transport_build_path_makes_a_study_dataset_for_each_timeframe(
    timeframe: str,
) -> None:
    protocol = replace(SMALL, data_start="2021-01-01T00:00:00")
    dataset = build_dataset_from_transport(
        protocol, timeframe, _FakeTransport(protocol), evidence={"fixture": True}
    )

    assert dataset.symbols == protocol.symbols
    assert int(dataset.timestamps[-1].astype("datetime64[ns]").astype(np.int64)) == int(
        np.datetime64(protocol.data_end_exclusive, "ns").astype(np.int64)
    )
    assert float(np.abs(dataset.resolved_array("funding_rate")).sum()) > 0.0


def test_vision_fallback_plans_only_the_requested_timeframe_and_needs_a_cache(
    tmp_path: Path,
) -> None:
    plan = plan_vision_cache(PROTOCOL, "4h")
    kinds = {url.split("/monthly/")[1].split("/")[0] for url in plan.urls}
    assert kinds == {"klines", "fundingRate"}
    assert all("/4h/" in url for url in plan.urls if "/klines/" in url)
    assert all(
        any(f"/{symbol}/" in url for url in plan.urls) for symbol in PROTOCOL.symbols
    )
    months = {"-".join(url.removesuffix(".zip").split("-")[-2:]) for url in plan.urls}
    assert min(months) == "2021-01" and max(months) == "2026-07"

    with pytest.raises(Exception, match="missing|incomplete|cache"):
        build_vision_dataset(
            PROTOCOL, "4h", cache_root=tmp_path / "empty", exchange_info_dir=tmp_path
        )


# -- metrics: independent oracles --------------------------------------------------------


def _replay_hand_scenario(extra: list[tuple[float, float, float, float]] | None):
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP
    bars = bars + (extra if extra is not None else GAP_DOWN)
    dataset = _dataset({"AAA": bars})
    policy = TrendStopPortfolioPolicy(dataset, _config())
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=9,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
    )
    assert result.ledger_evidence is not None
    return dataset, policy, result.ledger_evidence.intervals.intervals


def test_stop_audit_keeps_a_gap_through_the_stop_and_removes_a_better_fill() -> None:
    dataset, policy, intervals = _replay_hand_scenario(None)
    multipliers = dataset.resolved_array("contract_multipliers")
    gap = metrics.stop_exit_audit(intervals, policy.stop_events, multipliers)

    # Exit at 94 against a stop at 101: 7/101 worse than the level, no credit given back.
    assert gap["n_stop_exits"] == gap["n_resolved"] == 1
    assert gap["haircut_usdt"] == 0.0
    assert gap["n_fill_better_than_stop"] == 0
    assert gap["mean_adverse_gap_pct"] == pytest.approx(100.0 * 7.0 / 101.0)

    wick = [
        (109.0, 109.0, 100.0, 107.0),
        (108.0, 109.0, 106.0, 108.0),
        (108.0, 109.0, 106.0, 108.0),
    ]
    dataset, policy, intervals = _replay_hand_scenario(wick)
    audit = metrics.stop_exit_audit(
        intervals, policy.stop_events, dataset.resolved_array("contract_multipliers")
    )
    # Exit at 108 is 7 above the stop level 101: 50/6 units * 7 = 58.33 of optimism.
    assert audit["haircut_usdt"] == pytest.approx(float(Fraction(50, 6) * 7), abs=1e-6)
    assert audit["n_fill_better_than_stop"] == 1
    assert metrics.haircut_net_return_pct(0.0, audit["haircut_usdt"], 10_000.0) == (
        pytest.approx(-100.0 * float(Fraction(50, 6) * 7) / 10_000.0)
    )


def test_closed_trade_reconstruction_matches_the_hand_ledger() -> None:
    dataset, _, intervals = _replay_hand_scenario(None)
    combined, per_symbol = metrics.closed_trades_from_ledger(
        intervals, dataset.resolved_array("contract_multipliers"), dataset.symbols
    )
    # Bought 50/6 at 103, sold at 94, zero costs: one losing cycle of -75.
    assert combined.closed_trades == 1 and combined.losing_trades == 1
    assert combined.net_realized_pnl == pytest.approx(-75.0)
    assert combined.profit_factor == 0.0 and combined.win_rate == 0.0
    assert per_symbol["AAA"].closed_trades == 1


def _cell(
    *,
    net: float = 20.0,
    drawdown: float = 12.0,
    years: dict[str, float] | None = None,
    profit_factor: float | None = 1.5,
    trades: int = 400,
    calmar: float | None = 1.0,
    haircut_net: float | None = None,
) -> dict[str, Any]:
    return {
        "performance": {
            "terminal_net_return_pct": net,
            "max_drawdown_pct": drawdown,
            "calmar": calmar,
            "per_year_return_pct": years
            or {
                "2021": 5.0,
                "2022": 4.0,
                "2023": 3.0,
                "2024": 2.0,
                "2025": -1.0,
                "2026": -2.0,
            },
        },
        "haircut": {
            "haircut_net_return_pct": net if haircut_net is None else haircut_net
        },
        "closed_trades": {"profit_factor": profit_factor, "closed_trades": trades},
    }


def _evaluate(**overrides: Any) -> dict[str, Any]:
    base = _cell(**overrides.get("base", {}))
    stressed = _cell(**overrides.get("stressed", {}))
    benchmark = _cell(**overrides.get("benchmark", {"calmar": 0.3}))
    return metrics.evaluate_configuration(
        base,
        stressed,
        benchmark,
        overrides.get("bootstrap", {"bonferroni_ci_low": 0.1}),
        overrides.get("validity", True),
        PROTOCOL,
    )


def test_gate_passes_only_when_every_preregistered_criterion_holds() -> None:
    assert _evaluate()["passed"] is True


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"base": {"haircut_net": -0.1}}, "haircut_net_return_positive_base"),
        ({"stressed": {"haircut_net": -0.1}}, "haircut_net_return_positive_cost_x2"),
        (
            {"base": {"years": {"2021": 1.0, "2022": 1.0, "2023": 1.0, "2024": -1.0}}},
            "positive_sub_years_base",
        ),
        ({"base": {"profit_factor": 1.19}}, "profit_factor_base"),
        ({"base": {"profit_factor": None}}, "profit_factor_base"),
        ({"stressed": {"drawdown": 20.5}}, "max_drawdown_within_limit_all_scenarios"),
        ({"base": {"trades": 99}}, "closed_trades_base"),
        ({"base": {"calmar": 0.3}}, "calmar_exceeds_btc_buy_and_hold"),
        (
            {"bootstrap": {"bonferroni_ci_low": -0.01}},
            "bonferroni_bootstrap_lower_bound_positive",
        ),
        ({"validity": False}, "validity"),
    ],
)
def test_each_criterion_can_fail_the_configuration_on_its_own(
    overrides: dict[str, Any], failed: str
) -> None:
    outcome = _evaluate(**overrides)
    assert outcome["passed"] is False
    assert [
        name for name, item in outcome["criteria"].items() if not item["passed"]
    ] == [failed]


def test_label_is_never_a_winner_and_no_pass_triggers_the_stop_rule() -> None:
    passing = metrics.select_label({"a": {"passed": True}, "b": {"passed": False}})
    assert passing == {
        "label": LABEL_PROMISING,
        "passing_configurations": ["a"],
        "stop_rule_triggered": False,
    }
    none = metrics.select_label({"a": {"passed": False}})
    assert none["label"] == LABEL_NO_WINNER and none["stop_rule_triggered"] is True


def test_bonferroni_bootstrap_separates_a_real_drift_from_noise() -> None:
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 0.01, size=600)
    drift = noise + 0.004
    assert metrics.bootstrap_lower_bound(noise, SMALL)["bonferroni_ci_low"] < 0.0
    assert metrics.bootstrap_lower_bound(drift, SMALL)["bonferroni_ci_low"] > 0.0
    assert metrics.bootstrap_lower_bound(noise, SMALL)["adjusted_alpha"] == (
        pytest.approx(0.05 / 4)
    )


# -- end to end on synthetic bars: every cell is written --------------------------------


def test_run_study_writes_every_cell_and_refuses_to_overwrite(tmp_path: Path) -> None:
    result, cells, equity = runner.run_study(
        _small_datasets(), SMALL, g4_authorization_url=AUTHORIZATION_URL
    )

    expected_cells = {
        f"{timeframe}__{name}__{scenario}"
        for timeframe in SMALL.timeframes
        for scenario in (SCENARIO_BASE, SCENARIO_COST_X2)
        for name in (
            "btc_buy_and_hold",
            "equal_weight_buy_and_hold",
            *(variant.name for variant in SMALL.variants),
        )
    }
    assert set(cells) == expected_cells
    assert result["protocol_digest"] == protocol_digest(SMALL)
    assert result["g4_authorization_url"] == AUTHORIZATION_URL
    assert result["status"] == "development_evidence_only"
    assert set(result["configurations"]) == {
        f"{timeframe}__{variant.name}"
        for timeframe in SMALL.timeframes
        for variant in SMALL.variants
    }
    assert all(item["passed"] for item in result["causality_probe"].values())
    for name, cell in cells.items():
        assert cell["reconciliation"]["passed"], name
        assert cell["terminated"] == [], name
        assert cell["reconciliation"]["terminal_flat"], name
    assert result["label"] in (LABEL_PROMISING, LABEL_NO_WINNER)

    out = tmp_path / "study"
    written = cli.write_outputs(out, result, cells, equity)
    assert {path.name for path in written if path.parent.name == "cells"} == {
        f"{name}.json" for name in expected_cells
    }
    assert (out / "equity.npz").is_file() and (out / "result.json").is_file()
    with pytest.raises(FileExistsError):
        cli.write_outputs(out, result, cells, equity)


def test_rule_cells_hold_positions_and_report_stop_diagnostics() -> None:
    dataset = _small_datasets()["1d"]
    start, stop = runner.window_rows(dataset, SMALL)
    cell = runner.run_rule_cell(
        dataset,
        SMALL,
        VARIANTS[0],
        SCENARIO_BASE,
        "1d",
        start_index=start,
        stop_index=stop,
    ).summary

    assert cell["policy"]["entries"] > 0
    assert cell["haircut"]["n_stop_exits"] == cell["policy"]["stop_exits"]
    assert cell["closed_trades"]["closed_trades"] > 0
    assert cell["stop_model"] == "bar_extreme_touch_next_open_v1"
    # Per-symbol cycles add up to the portfolio count.
    assert (
        sum(item["closed_trades"] for item in cell["per_symbol"].values())
        == (cell["closed_trades"]["closed_trades"])
    )
    # Funding comes from the Dataset through the canonical ledger.
    assert "funding_pnl_usdt" in cell["costs"]
