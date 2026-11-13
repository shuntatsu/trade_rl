"""Trend-stop study: protocol freeze, refusals, metrics oracles and cell writing.

Everything here runs on in-process synthetic bars or hand-built ledgers. No market
data, network or venue access is involved.
"""

from __future__ import annotations

import json
import subprocess
import types
import zipfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation.trend_stop_support import (
    AUTHORIZATION_URL,
    CLEAN_PROVENANCE,
    DIRTY_PROVENANCE,
    synthetic_dataset,
)
from tests.strategies.test_trend_stop import (
    BREAKOUT,
    FLAT_BARS,
    GAP_DOWN,
    RUN_UP,
    _config,
    _dataset,
    partial_exit_dataset,
    replay_partial_exit,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.trend_stop_study import cli, metrics, protocol, runner
from trade_rl.evaluation.trend_stop_study import data as study_data
from trade_rl.evaluation.trend_stop_study.data import (
    IncompleteArchiveCacheError,
    IncompleteRosterError,
    OperationalDataError,
    build_dataset_from_transport,
    build_vision_dataset,
    classify_vision_cache,
    classify_vision_caches,
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
    SUPERSEDES_PROTOCOL_DIGEST,
    VARIANTS,
    check_protocol_window,
    hard_risk,
    protocol_digest,
    scenario_cost,
    trend_config,
    validate_authorization_url,
)
from trade_rl.integrations.binance import BinanceTransportError, vision_cache_path
from trade_rl.integrations.binance.cache import (
    missing_archive_evidence_path,
    write_missing_archive_evidence,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.portfolio import PortfolioTargetProposal
from trade_rl.strategies.rules.trend_stop import TrendStopPortfolioPolicy

SMALL = replace(
    PROTOCOL,
    symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
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


def test_revision_2_has_no_exclusion_or_reduced_roster_surface() -> None:
    fields = set(PROTOCOL.canonical_payload())
    assert not {name for name in fields if "exclusion" in name or "min_symbols" in name}
    assert PROTOCOL.roster_policy == "complete_roster_no_exclusions"
    assert PROTOCOL.roster_decision_url.endswith("#issuecomment-6818495567")
    assert PROTOCOL.causality_probe_scope == "every_timeframe_and_variant"
    assert PROTOCOL.schema_version == "trend_stop_study_protocol_v2"
    assert PROTOCOL.study_id == "trend_stop_tsmom_20261113"
    # Revision 2 is a new frozen protocol, never the digest of revision 1.
    assert PROTOCOL.supersedes_protocol_digest == SUPERSEDES_PROTOCOL_DIGEST
    assert protocol_digest(PROTOCOL) != SUPERSEDES_PROTOCOL_DIGEST
    joined = " ".join(PROTOCOL.deviations)
    assert "no exclusion path" in joined and "INVALID" in joined
    assert not hasattr(study_data, "parse_exclusions")
    assert not hasattr(study_data, "SymbolExclusion")
    assert not hasattr(study_data, "active_symbols")
    assert "docs/research/trend-stop-protocol.md" not in (protocol.__doc__ or "")
    assert (Path(__file__).parents[2] / "docs/specs/trend-following-stops.md").is_file()


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
        runner.run_study(
            _small_datasets(),
            SMALL,
            g4_authorization_url=url,
            code_provenance=CLEAN_PROVENANCE,
        )


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
        runner.run_study(
            _small_datasets(),
            SMALL,
            g4_authorization_url=None,
            code_provenance=CLEAN_PROVENANCE,
        )
    with pytest.raises(ValueError, match="exploratory data boundary"):
        runner.run_study(
            _small_datasets(),
            replace(SMALL, evaluation_end="2021-08-02T00:00:00"),
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=CLEAN_PROVENANCE,
        )
    with pytest.raises(ValueError, match="holds bars after"):
        runner.run_study(
            _small_datasets("2021-08-02T00:00:00"),
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=CLEAN_PROVENANCE,
        )
    partial = _small_datasets()
    del partial["1d"]
    with pytest.raises(ValueError, match="exactly the preregistered timeframes"):
        runner.run_study(
            partial,
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=CLEAN_PROVENANCE,
        )
    with pytest.raises(runner.DirtyWorkingTreeError, match="dirty working tree"):
        runner.run_study(
            _small_datasets(),
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=DIRTY_PROVENANCE,
        )


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


def test_cli_refuses_a_dirty_working_tree_before_loading_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("data was loaded from a dirty working tree")

    monkeypatch.setattr(cli, "capture_code_provenance", lambda: DIRTY_PROVENANCE)
    monkeypatch.setattr(cli, "load_exported_dataset", forbidden)
    monkeypatch.setattr(cli, "build_vision_dataset", forbidden)
    out = tmp_path / "out"

    code = cli.main(
        [
            "--output-dir",
            str(out),
            "--g4-authorization-url",
            AUTHORIZATION_URL,
            "--dataset-4h",
            str(tmp_path / "unused"),
        ]
    )

    assert code == 2
    assert "dirty working tree" in capsys.readouterr().err
    assert not out.exists()


def test_cli_refuses_a_non_empty_output_and_missing_datasets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
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
    with pytest.raises(IncompleteRosterError, match="XRPUSDT"):
        _check(symbols=(*SMALL.symbols, "XRPUSDT"))
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


WICK = [
    (109.0, 109.0, 100.0, 107.0),
    (108.0, 109.0, 106.0, 108.0),
    (108.0, 109.0, 106.0, 108.0),
]


def _replay_hand_scenario(
    extra: list[tuple[float, float, float, float]] | None, step_hours: int = 1
):
    bars = FLAT_BARS + [BREAKOUT] + RUN_UP
    bars = bars + (extra if extra is not None else GAP_DOWN)
    dataset = _dataset({"AAA": bars}, step_hours=step_hours)
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
    gap, gap_by_fill = metrics.stop_exit_audit(
        intervals, policy.stop_events, multipliers
    )

    # Exit at 94 against a stop at 101: 7/101 worse than the level, no credit given back.
    assert gap["n_stop_exits"] == gap["n_resolved"] == 1
    assert gap["haircut_usdt"] == 0.0 and gap_by_fill == {}
    assert gap["n_fill_better_than_stop"] == 0
    assert gap["mean_adverse_gap_pct"] == pytest.approx(100.0 * 7.0 / 101.0)

    dataset, policy, intervals = _replay_hand_scenario(WICK)
    audit, by_fill = metrics.stop_exit_audit(
        intervals, policy.stop_events, dataset.resolved_array("contract_multipliers")
    )
    # Exit at 108 is 7 above the stop level 101: 50/6 units * 7 = 58.33 of optimism.
    assert audit["haircut_usdt"] == pytest.approx(float(Fraction(50, 6) * 7), abs=1e-6)
    assert audit["n_fill_better_than_stop"] == 1
    # The haircut sits on the exit fill (interval 7 -> 8).
    assert [position for position, _ in by_fill] == [7]
    assert sum(by_fill.values()) == pytest.approx(audit["haircut_usdt"])


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
        ({"base": {"net": -0.1}}, "net_return_positive_base"),
        ({"stressed": {"net": -0.1}}, "net_return_positive_cost_x2"),
        (
            {"base": {"years": {"2021": 1.0, "2022": 1.0, "2023": 1.0, "2024": -1.0}}},
            "positive_sub_years_base",
        ),
        ({"base": {"profit_factor": 1.19}}, "profit_factor_base"),
        ({"base": {"profit_factor": None}}, "profit_factor_base"),
        ({"stressed": {"drawdown": 20.5}}, "max_drawdown_within_limit_all_scenarios"),
        ({"base": {"trades": 99}}, "closed_trades_base"),
        ({"base": {"calmar": 0.3}}, "calmar_exceeds_btc_buy_and_hold"),
        ({"benchmark": {"calmar": None}}, "calmar_exceeds_btc_buy_and_hold"),
        ({"base": {"calmar": None}}, "calmar_exceeds_btc_buy_and_hold"),
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
        _small_datasets(),
        SMALL,
        g4_authorization_url=AUTHORIZATION_URL,
        code_provenance=CLEAN_PROVENANCE,
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
    assert result["code"] == CLEAN_PROVENANCE.to_mapping()
    assert result["roster"] == list(SMALL.symbols)
    assert "exclusions" not in result and "git_head" not in result
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


# -- M2: the haircut reaches every gated quantity ----------------------------------------


def _summarized_hand_cell(extra: list[tuple[float, float, float, float]]) -> Any:
    """The hand scenario on a daily grid through the runner's own cell summary."""

    dataset, policy, _ = _replay_hand_scenario(extra, step_hours=24)
    replayed = run_shared_portfolio_replay(
        dataset,
        TrendStopPortfolioPolicy(dataset, _config()),
        start_index=0,
        stop_index=9,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
    )
    return runner.summarize_cell(
        arm="rule",
        timeframe="1d",
        variant="hand",
        scenario=SCENARIO_BASE,
        result=replayed,
        dataset=dataset,
        protocol=SMALL,
        stop_events=policy.stop_events,
        policy_summary=policy.summary(),
    )


def test_haircut_is_charged_to_the_pnl_stream_and_compounds_through_equity() -> None:
    quantity = Fraction(50, 6)
    outcome = _summarized_hand_cell(WICK)
    cell = outcome.summary

    # Engine: bought at 103, the wick exit fills at the open of 108, +41.67. A resting
    # stop at 101 would have lost 2 per unit: 10000 - 50/6 * 2 = 9983.33.
    engine_end = float(Fraction(10000) + quantity * (108 - 103))
    adjusted_end = float(Fraction(10000) + quantity * (101 - 103))
    assert outcome.engine_equity[-1] == pytest.approx(engine_end, abs=1e-6)
    assert outcome.equity[-1] == pytest.approx(adjusted_end, abs=1e-6)
    assert cell["engine_performance"]["terminal_net_return_pct"] > 0.0
    assert cell["performance"]["terminal_net_return_pct"] < 0.0
    # Before the exit the two paths are identical.
    assert outcome.equity[:7] == pytest.approx(outcome.engine_equity[:7])
    # Closed cycles carry the haircut as a cost on the exit fill: the engine cycle wins,
    # the adjusted cycle loses, so the profit factor differs (None vs 0).
    assert cell["engine_closed_trades"]["winning_trades"] == 1
    assert cell["engine_closed_trades"]["profit_factor"] is None
    assert cell["closed_trades"]["losing_trades"] == 1
    assert cell["closed_trades"]["profit_factor"] == 0.0
    assert cell["closed_trades"]["net_realized_pnl"] == pytest.approx(
        float(quantity * (101 - 103)), abs=1e-6
    )
    # The bootstrap input compounds back to the adjusted terminal value.
    assert float(np.prod(1.0 + outcome.daily)) * 10_000.0 == pytest.approx(
        adjusted_end, abs=1e-6
    )


def test_a_criterion_passes_on_engine_pnl_but_fails_on_the_haircut_adjusted_run() -> (
    None
):
    cell = _summarized_hand_cell(WICK).summary
    benchmark = _cell(calmar=0.3)
    stressed = _cell()

    def verdict(performance: dict[str, Any]) -> bool:
        base = {**cell, "performance": performance}
        criteria = metrics.evaluate_configuration(
            base, stressed, benchmark, {"bonferroni_ci_low": 0.1}, True, PROTOCOL
        )["criteria"]
        return bool(criteria["net_return_positive_base"]["passed"])

    assert verdict(cell["engine_performance"]) is True  # un-haircut run passes
    assert verdict(cell["performance"]) is False  # haircut run fails

    gap = _summarized_hand_cell(GAP_DOWN).summary  # adverse gap: nothing credited back
    assert gap["performance"]["terminal_net_return_pct"] == pytest.approx(
        gap["engine_performance"]["terminal_net_return_pct"]
    )
    assert gap["haircut"]["haircut_usdt"] == 0.0


def test_every_gated_metric_is_read_from_the_adjusted_cell_fields() -> None:
    """The gate may not reach for the un-haircut ``engine_*`` fields."""

    source = Path(metrics.__file__).read_text(encoding="utf-8")
    gate = source[source.index("def evaluate_configuration") :]
    assert "engine_" not in gate.split("def select_label")[0].split('"""')[2]


# -- M1: the roster is complete or the run is INVALID ------------------------------------


def _ten_of_eleven(missing: str = "DOTUSDT") -> Any:
    symbols = tuple(s for s in PROTOCOL.symbols if s != missing)
    return synthetic_dataset(
        "1d",
        symbols,
        first_close="2021-01-02T00:00:00",
        last_close="2021-08-01T00:00:00",
    )


def test_a_ten_of_eleven_dataset_is_not_a_valid_roster() -> None:
    with pytest.raises(IncompleteRosterError, match="DOTUSDT") as caught:
        validate_study_dataset(_ten_of_eleven(), PROTOCOL, "1d")
    assert caught.value.missing == ("DOTUSDT",)
    # The benchmark is part of the roster: a missing BTC archive is INVALID, too.
    with pytest.raises(IncompleteRosterError, match="BTCUSDT"):
        validate_study_dataset(_ten_of_eleven("BTCUSDT"), PROTOCOL, "1d")
    # A complete roster passes.
    complete = synthetic_dataset(
        "1d",
        PROTOCOL.symbols,
        first_close="2021-01-02T00:00:00",
        last_close="2021-08-01T00:00:00",
    )
    assert validate_study_dataset(complete, PROTOCOL, "1d") == PROTOCOL.symbols


def test_no_exclusion_or_reduced_roster_parameter_exists_anywhere() -> None:
    import inspect

    for function in (
        runner.run_study,
        validate_study_dataset,
        plan_vision_cache,
        build_vision_dataset,
        build_dataset_from_transport,
        cli._load_datasets,
    ):
        assert not [
            name
            for name in inspect.signature(function).parameters
            if "exclu" in name or "min_symbols" in name
        ], function.__name__
    with pytest.raises(TypeError):
        validate_study_dataset(  # type: ignore[call-arg]
            _ten_of_eleven(), PROTOCOL, "1d", exclusions=()
        )
    with pytest.raises(TypeError):
        runner.run_study(  # type: ignore[call-arg]
            _small_datasets(),
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=CLEAN_PROVENANCE,
            exclusions=(),
        )


@pytest.mark.parametrize("flag", ["--exclusions", "--exclude", "--min-symbols"])
def test_the_cli_accepts_no_exclusion_option(
    tmp_path: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    help_text = cli.build_parser().format_help()
    assert "exclusion" not in help_text and "min-symbols" not in help_text
    records = tmp_path / "exclusions.json"
    records.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--g4-authorization-url",
                AUTHORIZATION_URL,
                flag,
                str(records),
            ]
        )
    assert stopped.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_an_incomplete_roster_writes_an_invalid_result_and_replays_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a replay ran for an INVALID roster")

    monkeypatch.setattr(runner, "run_rule_cell", forbidden)
    monkeypatch.setattr(runner, "run_benchmark_cell", forbidden)
    monkeypatch.setattr(runner, "causality_probe", forbidden)
    # The 3-symbol datasets lack the fourth roster symbol: no reduced roster is run.
    protocol = replace(SMALL, symbols=(*SMALL.symbols, "XRPUSDT"))

    result, cells, equity = runner.run_study(
        _small_datasets(),
        protocol,
        g4_authorization_url=AUTHORIZATION_URL,
        code_provenance=CLEAN_PROVENANCE,
    )

    assert result["status"] == result["label"] == "INVALID"
    assert result["missing_symbols"] == ["XRPUSDT"]
    assert "no exclusions exist" in result["reason"]
    assert result["stop_rule_triggered"] is False  # INVALID is not NO_WINNER
    assert cells == {} and equity == {}
    assert result["protocol_digest"] == protocol_digest(protocol)
    assert result["code"] == CLEAN_PROVENANCE.to_mapping()
    assert "exclusions" not in result
    written = cli.write_outputs(tmp_path / "invalid", result, cells, equity)
    assert [path.name for path in written] == ["result.json"]
    stored = json.loads(written[0].read_text(encoding="utf-8"))
    assert stored["label"] == "INVALID" and stored["prior_exposure"]


def test_the_cli_writes_an_invalid_result_for_a_ten_of_eleven_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(path: object) -> Any:
        timeframe = "4h" if str(path).endswith("4h") else "1d"
        symbols = tuple(s for s in PROTOCOL.symbols if s != "DOTUSDT")
        first = "2021-01-01T04:00:00" if timeframe == "4h" else "2021-01-02T00:00:00"
        return synthetic_dataset(
            timeframe, symbols, first_close=first, last_close="2021-08-01T00:00:00"
        )

    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    monkeypatch.setattr(cli, "load_exported_dataset", fake_load)
    out = tmp_path / "out"

    code = cli.main(
        [
            "--output-dir",
            str(out),
            "--g4-authorization-url",
            AUTHORIZATION_URL,
            "--dataset-4h",
            "x/4h",
            "--dataset-1d",
            "x/1d",
        ]
    )

    assert code == 3
    assert "label=INVALID" in capsys.readouterr().out
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["label"] == "INVALID"
    assert stored["missing_symbols"] == ["DOTUSDT"]
    assert not (out / "cells").exists() or not any((out / "cells").iterdir())


def test_the_result_carries_the_roster_and_the_prior_exposure_disclosure() -> None:
    result, _, _ = runner.run_study(
        _small_datasets(),
        SMALL,
        g4_authorization_url=AUTHORIZATION_URL,
        code_provenance=CLEAN_PROVENANCE,
    )
    assert result["roster"] == list(SMALL.symbols)
    joined = " ".join(result["prior_exposure"])
    for needle in ("realdata_20260713_signal_metadata", "Phase 2", "M34"):
        assert needle in joined
    assert any("haircut-adjusted" in note for note in result["notes"])


def test_the_vision_plan_always_covers_the_whole_roster() -> None:
    plan = plan_vision_cache(PROTOCOL, "1d")
    for symbol in PROTOCOL.symbols:
        assert any(f"/{symbol}/" in url for url in plan.urls), symbol


# -- B1: one causality probe per variant and timeframe -----------------------------------


def _leaky_factory(leaky_entry_bars: int) -> Any:
    """A policy factory whose ``leaky_entry_bars`` variant peeks 20 bars ahead."""

    class Leaky(TrendStopPortfolioPolicy):
        def decide_portfolio(self, snapshot: Any) -> PortfolioTargetProposal:
            proposal = super().decide_portfolio(snapshot)
            ahead = min(snapshot.index + 20, self._close.shape[0] - 1)
            peek = self._close[ahead, 0] > self._close[snapshot.index, 0]
            weights = list(proposal.target_weights)
            weights[0] = 0.10 if peek else 0.05
            holds = tuple(i for i in proposal.hold_current_quantity_symbols if i != 0)
            return PortfolioTargetProposal(
                target_weights=tuple(weights), hold_current_quantity_symbols=holds
            )

    def factory(dataset: Any, config: Any) -> TrendStopPortfolioPolicy:
        cls = (
            Leaky if config.entry_bars == leaky_entry_bars else TrendStopPortfolioPolicy
        )
        return cls(dataset, config)

    return factory


def test_every_variant_gets_its_own_probe_and_its_own_criterion_8() -> None:
    result, _, _ = runner.run_study(
        _small_datasets(),
        SMALL,
        g4_authorization_url=AUTHORIZATION_URL,
        code_provenance=CLEAN_PROVENANCE,
    )

    expected = {
        f"{timeframe}__{variant.name}"
        for timeframe in SMALL.timeframes
        for variant in SMALL.variants
    }
    assert set(result["causality_probe"]) == expected
    for name, probe in result["causality_probe"].items():
        assert name.endswith(probe["variant"]), name
        assert probe["passed"] and probe["pre_cut_has_positions"], name
    assert set(result["configurations"]) == expected
    assert all(
        item["criteria"]["validity"]["passed"]
        for item in result["configurations"].values()
    )


@pytest.mark.parametrize("leaky_bars", [20, 55])
def test_a_leaky_variant_fails_only_its_own_criterion_8(
    leaky_bars: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "TrendStopPortfolioPolicy", _leaky_factory(leaky_bars))

    result, _, _ = runner.run_study(
        _small_datasets(),
        SMALL,
        g4_authorization_url=AUTHORIZATION_URL,
        code_provenance=CLEAN_PROVENANCE,
    )

    for timeframe in SMALL.timeframes:
        for variant in SMALL.variants:
            name = f"{timeframe}__{variant.name}"
            leaks = variant.entry_bars == leaky_bars
            probe = result["causality_probe"][name]
            assert probe["variant"] == variant.name
            assert probe["passed"] is not leaks, name
            assert probe["identical_pre_cut_trace"] is not leaks, name
            validity = result["configurations"][name]["criteria"]["validity"]
            assert validity["passed"] is not leaks, name
            if leaks:
                assert result["configurations"][name]["passed"] is False
                assert name not in result["passing_configurations"]


# -- code provenance and the dirty-tree refusal ---------------------------------------


def _git_repo(root: Path) -> None:
    root.mkdir()
    (root / "trade_rl").mkdir()
    (root / "trade_rl" / "a.py").write_text("X = 1\n", encoding="utf-8")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "x",
        ],
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_git_state_names_the_commit_the_code_tree_and_the_dirty_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    head, tree, dirty = runner.git_state(root)
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    expected_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:trade_rl"], cwd=root, capture_output=True, text=True
    ).stdout.strip()
    assert (head, tree, dirty) == (expected_head, expected_tree, False)
    assert len(head) == len(tree) == 40

    (root / "trade_rl" / "a.py").write_text("X = 2\n", encoding="utf-8")
    assert runner.git_state(root)[2] is True
    subprocess.run(["git", "checkout", "-q", "."], cwd=root, check=True)
    assert runner.git_state(root)[2] is False
    (root / "trade_rl" / "new.py").write_text("Y = 1\n", encoding="utf-8")
    assert runner.git_state(root)[2] is True  # an untracked file is dirty, too

    with pytest.raises(runner.CodeProvenanceError, match="Git"):
        runner.git_state(tmp_path)  # not a repository


def test_the_loaded_module_digest_binds_bytes_and_refuses_foreign_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    (root / "trade_rl").mkdir(parents=True)
    first = root / "trade_rl" / "a.py"
    first.write_text("X = 1\n", encoding="utf-8")

    def module(path: Path) -> types.ModuleType:
        item = types.ModuleType("m")
        item.__file__ = str(path)
        return item

    modules = {"trade_rl.a": module(first), "numpy": module(tmp_path / "ignored.py")}
    digest, count = runner.loaded_module_digest(root, modules)
    assert count == 1 and len(digest) == 64
    assert runner.loaded_module_digest(root, modules) == (digest, 1)
    first.write_text("X = 2\n", encoding="utf-8")
    assert runner.loaded_module_digest(root, modules)[0] != digest

    foreign = tmp_path / "site-packages" / "b.py"
    foreign.parent.mkdir()
    foreign.write_text("Z = 1\n", encoding="utf-8")
    with pytest.raises(runner.CodeProvenanceError, match="outside the checkout"):
        runner.loaded_module_digest(root, {**modules, "trade_rl.b": module(foreign)})
    with pytest.raises(runner.CodeProvenanceError, match="no trade_rl module"):
        runner.loaded_module_digest(root, {"numpy": module(first)})


def test_capture_describes_the_running_checkout() -> None:
    provenance = runner.capture_code_provenance()
    mapping = provenance.to_mapping()
    assert set(mapping) == {
        "git_head",
        "trade_rl_tree",
        "dirty",
        "module_digest",
        "module_count",
    }
    assert len(provenance.git_head) == len(provenance.trade_rl_tree) == 40
    assert len(provenance.module_digest) == 64 and provenance.module_count > 10
    assert isinstance(provenance.dirty, bool)
    json.dumps(mapping, allow_nan=False)


def test_a_dirty_provenance_is_refused_before_any_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a replay ran from a dirty working tree")

    monkeypatch.setattr(runner, "run_rule_cell", forbidden)
    monkeypatch.setattr(runner, "run_benchmark_cell", forbidden)
    monkeypatch.setattr(runner, "causality_probe", forbidden)
    with pytest.raises(runner.DirtyWorkingTreeError):
        runner.run_study(
            _small_datasets(),
            SMALL,
            g4_authorization_url=AUTHORIZATION_URL,
            code_provenance=DIRTY_PROVENANCE,
        )


# -- a stop exit that fills in several parts --------------------------------------------


def _stop_audit_of(bars: list[tuple[float, float, float, float]]) -> Any:
    dataset = partial_exit_dataset(bars)
    result, policy = replay_partial_exit(bars)
    assert result.ledger_evidence is not None
    return metrics.stop_exit_audit(
        result.ledger_evidence.intervals.intervals,
        policy.stop_events,
        dataset.resolved_array("contract_multipliers"),
    )


def test_the_haircut_audit_counts_a_partly_filled_stop_exit_once() -> None:
    gap = FLAT_BARS + [BREAKOUT] + RUN_UP + GAP_DOWN
    audit, by_fill = _stop_audit_of(gap + [(97.0, 98.0, 96.0, 97.0)] * 4)
    # Every fill (94, 97, 97) is worse than the stop level 101: no haircut.
    assert audit["n_stop_exits"] == audit["n_resolved"] == 1
    assert audit["haircut_usdt"] == 0.0 and by_fill == {}
    average = (4 * 94 + 4 * 97 + (50 / 6 - 8) * 97) / (50 / 6)
    assert audit["mean_adverse_gap_pct"] == pytest.approx(
        100.0 * (101.0 - average) / 101.0
    )

    wick = [
        (109.0, 109.0, 100.0, 107.0),
        (108.0, 109.0, 106.0, 108.0),
        (109.0, 110.0, 107.0, 109.0),
        (109.0, 110.0, 107.0, 109.0),
        (109.0, 110.0, 107.0, 109.0),
    ]
    audit, by_fill = _stop_audit_of(FLAT_BARS + [BREAKOUT] + RUN_UP + wick)
    # Fills at 108, 109 and 109 are 7, 8 and 8 above the stop level 101: the haircut is
    # the summed improvement over every closing fill of the one stop event.
    expected = 4 * 7 + 4 * 8 + (50 / 6 - 8) * 8
    assert audit["n_stop_exits"] == audit["n_resolved"] == 1
    assert audit["n_fill_better_than_stop"] == 1
    assert audit["haircut_usdt"] == pytest.approx(expected, abs=1e-6)
    assert sum(by_fill.values()) == pytest.approx(expected, abs=1e-6)
    assert [position for position, _ in by_fill] == [7, 8, 9]


# -- the Vision fallback: a recorded 404 is INVALID, anything else is a halt ---------------


def _vision_urls(timeframe: str = "4h") -> list[str]:
    return list(plan_vision_cache(PROTOCOL, timeframe).urls)


def _record_404(cache: Path, url: str) -> None:
    write_missing_archive_evidence(
        vision_cache_path(cache, url), url=url, downloader="test"
    )


def _cli_vision(cache: Path, out: Path) -> int:
    return cli.main(
        [
            "--output-dir",
            str(out),
            "--g4-authorization-url",
            AUTHORIZATION_URL,
            "--vision-cache",
            str(cache),
            "--exchange-info-dir",
            str(cache),
        ]
    )


def test_a_cache_archive_with_a_recorded_404_is_invalid_and_names_the_evidence(
    tmp_path: Path,
) -> None:
    url = next(u for u in _vision_urls() if "/DOTUSDT/" in u and "/klines/" in u)
    _record_404(tmp_path, url)

    with pytest.raises(IncompleteRosterError) as caught:
        classify_vision_cache(PROTOCOL, "4h", cache_root=tmp_path)

    error = caught.value
    assert error.missing == ("DOTUSDT",)
    (archive,) = error.archives
    assert (archive.symbol, archive.url, archive.http_status) == ("DOTUSDT", url, 404)
    assert archive.observed_at.endswith("+00:00")
    assert "INVALID" in str(error) and url in str(error)


def test_a_funding_archive_404_is_invalid_too(tmp_path: Path) -> None:
    url = next(u for u in _vision_urls("1d") if "fundingRate" in u and "/SOLUSDT/" in u)
    _record_404(tmp_path, url)
    with pytest.raises(IncompleteRosterError) as caught:
        classify_vision_cache(PROTOCOL, "1d", cache_root=tmp_path)
    assert caught.value.missing == ("SOLUSDT",)


def test_an_archive_missing_without_404_evidence_is_a_resumable_halt(
    tmp_path: Path,
) -> None:
    with pytest.raises(IncompleteArchiveCacheError) as caught:
        classify_vision_cache(PROTOCOL, "4h", cache_root=tmp_path)

    error = caught.value
    assert not isinstance(error, IncompleteRosterError)
    assert isinstance(error, OperationalDataError)
    assert len(error.absent) == len(_vision_urls()) and not error.empty
    assert "NOT evidence of a missing archive" in str(error)
    assert "complete or repair the download" in str(error)


def test_unreadable_404_evidence_is_a_halt_not_a_404(tmp_path: Path) -> None:
    url = _vision_urls()[0]
    path = missing_archive_evidence_path(vision_cache_path(tmp_path, url))
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(IncompleteArchiveCacheError):
        classify_vision_cache(PROTOCOL, "4h", cache_root=tmp_path)


def test_a_checksum_mismatch_is_a_halt(tmp_path: Path) -> None:
    url = _vision_urls()[0]
    binary = vision_cache_path(tmp_path, url)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"archive bytes")
    binary.with_suffix(".json").write_text(
        json.dumps(
            {
                "acquired_at": "2026-11-13T00:00:00+00:00",
                "downloader": "test",
                "etag": None,
                "last_modified": None,
                "schema_version": "binance_vision_raw_cache_v1",
                "sha256": "0" * 64,
                "size_bytes": len(b"archive bytes"),
                "url": url,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IncompleteArchiveCacheError) as caught:
        classify_vision_cache(PROTOCOL, "4h", cache_root=tmp_path)
    assert caught.value.invalid == (url,)


def test_the_cli_writes_an_invalid_result_for_a_recorded_404(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    cache, out = tmp_path / "cache", tmp_path / "out"
    url = next(u for u in _vision_urls() if "/DOTUSDT/" in u and "/klines/" in u)
    _record_404(cache, url)  # every other archive is merely absent: the 404 decides

    code = _cli_vision(cache, out)

    assert code == 3
    assert "label=INVALID" in capsys.readouterr().out
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["label"] == stored["status"] == "INVALID"
    assert stored["missing_symbols"] == ["DOTUSDT"]
    (archive,) = stored["missing_archives"]
    assert archive["url"] == url and archive["http_status"] == 404
    assert archive["symbol"] == "DOTUSDT" and archive["observed_at"]
    assert stored["stop_rule_triggered"] is False and stored["code"]
    assert not any((out / "cells").iterdir()) and not (out / "equity.npz").exists()


def test_the_cli_halts_without_a_result_for_an_unevidenced_missing_archive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()

    code = _cli_vision(cache, out)

    captured = capsys.readouterr()
    assert code == 2
    assert "halted, no result written" in captured.err
    assert "complete or repair the download" in captured.err
    assert "Traceback" not in captured.err and "label=" not in captured.out
    assert not out.exists()  # the single run is not consumed


def test_the_cli_halts_on_a_checksum_mismatch_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    cache, out = tmp_path / "cache", tmp_path / "out"
    url = _vision_urls()[0]
    binary = vision_cache_path(cache, url)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"corrupt")
    binary.with_suffix(".json").write_text("{}", encoding="utf-8")

    code = _cli_vision(cache, out)

    captured = capsys.readouterr()
    assert code == 2 and "failed content check=1" in captured.err
    assert "Traceback" not in captured.err and not out.exists()


def test_the_cli_maps_an_unreadable_dataset_to_a_halt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    out = tmp_path / "out"
    code = cli.main(
        [
            "--output-dir",
            str(out),
            "--g4-authorization-url",
            AUTHORIZATION_URL,
            "--dataset-4h",
            str(tmp_path / "missing-dataset"),
            "--dataset-1d",
            str(tmp_path / "missing-dataset"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2 and "halted, no result written" in captured.err
    assert "Traceback" not in captured.err and not out.exists()


# -- one evidence pass: the outcome never depends on the timeframe order -----------------


def _first_url(timeframe: str, symbol: str, kind: str) -> str:
    return next(
        u for u in _vision_urls(timeframe) if f"/{symbol}/" in u and f"/{kind}/" in u
    )


def _vision_cli_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record: list[str]
) -> tuple[int, Path]:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    cache, out = tmp_path / "cache", tmp_path / "out"
    cache.mkdir()
    for url in record:
        _record_404(cache, url)
    return _cli_vision(cache, out), out


@pytest.mark.parametrize("evidenced_timeframe", ["4h", "1d"])
def test_a_404_in_either_timeframe_is_invalid_whatever_the_build_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    evidenced_timeframe: str,
) -> None:
    # The other timeframe's archives are merely absent (not downloaded).
    url = _first_url(evidenced_timeframe, "ADAUSDT", "klines")
    code, out = _vision_cli_case(tmp_path, monkeypatch, [url])

    assert code == 3, capsys.readouterr().err
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["label"] == "INVALID" and stored["missing_symbols"] == ["ADAUSDT"]
    assert [item["url"] for item in stored["missing_archives"]] == [url]


def test_404_evidence_in_both_timeframes_and_funding_names_every_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls = [
        _first_url("1d", "LTCUSDT", "klines"),
        _first_url("4h", "DOGEUSDT", "klines"),
        _first_url("4h", "SOLUSDT", "fundingRate"),
    ]
    code, out = _vision_cli_case(tmp_path, monkeypatch, urls)

    assert code == 3
    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["missing_symbols"] == ["DOGEUSDT", "LTCUSDT", "SOLUSDT"]
    assert sorted(item["url"] for item in stored["missing_archives"]) == sorted(urls)


def test_a_404_wins_over_a_checksum_failure_and_absent_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    cache, out = tmp_path / "cache", tmp_path / "out"
    corrupt = _first_url("4h", "BTCUSDT", "klines")
    binary = vision_cache_path(cache, corrupt)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"corrupt")
    binary.with_suffix(".json").write_text("{}", encoding="utf-8")
    evidenced = _first_url("1d", "XRPUSDT", "klines")
    _record_404(cache, evidenced)

    assert _cli_vision(cache, out) == 3

    stored = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert stored["missing_symbols"] == ["XRPUSDT"]


def test_without_404_evidence_gaps_in_both_timeframes_halt_with_one_union(
    tmp_path: Path,
) -> None:
    union = set(_vision_urls("4h")) | set(_vision_urls("1d"))
    with pytest.raises(IncompleteArchiveCacheError) as caught:
        classify_vision_caches(PROTOCOL, ("4h", "1d"), cache_root=tmp_path)
    assert set(caught.value.absent) == union
    assert len(caught.value.absent) == len(union)  # the shared funding URLs once


def test_a_404_in_the_second_timeframe_is_found_before_the_first_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a Dataset was built before the evidence pass finished")

    monkeypatch.setattr(cli, "build_vision_dataset", forbidden)
    code, out = _vision_cli_case(
        tmp_path, monkeypatch, [_first_url("1d", "AVAXUSDT", "klines")]
    )
    assert code == 3 and (out / "result.json").is_file()


# -- no data or metadata failure ends in a traceback -----------------------------------


def test_a_missing_exchange_information_snapshot_is_a_halt_not_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    monkeypatch.setattr(cli, "classify_vision_caches", lambda *a, **k: None)
    cache, out = tmp_path / "cache", tmp_path / "out"
    info = tmp_path / "info"
    cache.mkdir()
    info.mkdir()

    code = cli.main(
        [
            "--output-dir",
            str(out),
            "--g4-authorization-url",
            AUTHORIZATION_URL,
            "--vision-cache",
            str(cache),
            "--exchange-info-dir",
            str(info),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "exchange-information snapshot" in captured.err
    assert "RuntimeError" in captured.err and "Traceback" not in captured.err
    assert not out.exists()
    with pytest.raises(OperationalDataError, match="exchange-information"):
        build_vision_dataset(
            PROTOCOL,
            "4h",
            cache_root=cache,
            exchange_info_dir=info,
            cache_verified=True,
        )


_INJECTED = [
    RuntimeError("frozen metadata cache is incomplete"),
    ValueError("bad value"),
    OSError("disk"),
    FileNotFoundError("gone"),
    PermissionError("denied"),
    KeyError("field"),
    IndexError("row"),
    EOFError("short"),
    zipfile.BadZipFile("not a zip"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad"),
    json.JSONDecodeError("bad json", "{", 1),
    OverflowError("big"),
    ZeroDivisionError("zero"),
    AssertionError("invariant"),
    MemoryError("memory"),
    BinanceTransportError("transport"),
]


@pytest.mark.parametrize(
    "error", _INJECTED, ids=[type(item).__name__ for item in _INJECTED]
)
@pytest.mark.parametrize("stage", ["dataset", "cache_pass", "build", "run"])
def test_no_injected_failure_ever_ends_in_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error: Exception,
) -> None:
    def boom(*args: object, **kwargs: object) -> Any:
        raise error

    monkeypatch.setattr(cli, "capture_code_provenance", lambda: CLEAN_PROVENANCE)
    out = tmp_path / "out"
    argv = ["--output-dir", str(out), "--g4-authorization-url", AUTHORIZATION_URL]
    if stage == "dataset":
        monkeypatch.setattr(cli, "load_exported_dataset", boom)
        argv += ["--dataset-4h", "x", "--dataset-1d", "y"]
    else:
        argv += ["--vision-cache", str(tmp_path), "--exchange-info-dir", str(tmp_path)]
        if stage == "cache_pass":
            monkeypatch.setattr(cli, "classify_vision_caches", boom)
        else:
            monkeypatch.setattr(cli, "classify_vision_caches", lambda *a, **k: None)
            monkeypatch.setattr(cli, "build_vision_dataset", boom)
    if stage == "run":
        monkeypatch.setattr(cli, "load_exported_dataset", lambda path: object())
        monkeypatch.setattr(cli, "run_study", boom)
        argv = argv[:4] + ["--dataset-4h", "x", "--dataset-1d", "y"]

    code = cli.main(argv)

    captured = capsys.readouterr()
    assert code == 2, captured
    assert type(error).__name__ in captured.err
    assert "halted, no result written" in captured.err
    assert "Traceback" not in captured.err and "label=" not in captured.out
    assert not out.exists()  # no result: the single run is not consumed
