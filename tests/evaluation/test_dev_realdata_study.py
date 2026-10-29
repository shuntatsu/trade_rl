from __future__ import annotations

import json
import math
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import cli as study_cli
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.metrics import (
    bootstrap_excess,
    daily_equity,
    max_drawdown,
    moving_block_indices,
    performance_summary,
    select_development_label,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_ALLOCATOR_24H,
    ARM_ALLOCATOR_72H,
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    ARMS,
    CANDIDATE_ARMS,
    LABEL_NO_WINNER,
    LABEL_PROMISING,
    PROTOCOL,
    SCENARIO_BASE,
    SCENARIO_COST_X2,
    SCENARIO_LATENCY,
    SCENARIOS,
    DevStudyProtocol,
    protocol_digest,
)
from trade_rl.strategies.portfolio import PortfolioDecisionSnapshot

HOUR = np.timedelta64(1, "h")
DAY = np.timedelta64(1, "D")
START = np.datetime64("2022-01-01T00:00:00", "ns")
N_DAYS = 110
N_BARS = N_DAYS * 24 + 1
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


def synthetic_market(*, seed: int = 11) -> MarketDataset:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, size=(N_BARS, len(SYMBOLS)))
    steps[:, 0] += 0.003
    steps[:, 1] -= 0.0015
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    features = np.zeros((N_BARS, len(SYMBOLS), 4), dtype=np.float32)
    available = np.zeros((N_BARS, len(SYMBOLS), 4), dtype=np.bool_)
    trailing = np.log(close[24:] / close[:-24])
    features[24:, :, 0] = trailing
    features[24:, :, 1] = trailing**2
    features[24:, :, 2] = rng.normal(size=trailing.shape)  # funding-named: excluded
    features[24:, :, 3] = trailing  # 4h-named: excluded
    available[24:] = True
    funding = np.zeros((N_BARS, len(SYMBOLS)))
    due = np.zeros((N_BARS, len(SYMBOLS)), dtype=np.bool_)
    funding[8::8] = 0.0001
    due[8::8] = True
    zero = np.zeros((N_BARS, len(SYMBOLS)))
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=SYMBOLS,
        timestamps=START + np.arange(N_BARS) * HOUR,
        features=features,
        global_features=np.zeros((N_BARS, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((N_BARS, len(SYMBOLS)), 1e12),
        funding_rate=funding,
        funding_due=due,
        tradable=np.ones((N_BARS, len(SYMBOLS)), dtype=np.bool_),
        feature_available=available,
        feature_names=("1h__trail", "1h__trail_sq", "1h__funding_bps", "4h__trail"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zero.copy(),
        maker_fee_rate=zero.copy(),
        taker_fee_rate=zero.copy(),
        spread_rate=zero.copy(),
        contract_multipliers=np.ones(len(SYMBOLS)),
    )


def small_protocol(**overrides: object) -> DevStudyProtocol:
    base = replace(
        PROTOCOL,
        dataset_id="5" * 64,
        covariance_warmup_start="2022-01-03T00:00:00",
        evaluation_start="2022-03-01T00:00:00",
        evaluation_end="2022-04-10T00:00:00",
        causality_cut="2022-03-25T00:00:00",
        refit_interval_days=15,
        min_history_periods=10,
        ewma_halflife_days=15.0,
        bootstrap_resamples=64,
        bootstrap_block_days=5,
        bonferroni_comparisons=7,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def market() -> MarketDataset:
    return synthetic_market()


@pytest.fixture(scope="module")
def study(market: MarketDataset) -> tuple[dict, dict]:
    return runner.run_study(market, small_protocol())


# --- protocol -------------------------------------------------------------


def test_arm_scenario_roster_and_selection_constants_are_fixed() -> None:
    assert ARMS == (
        "cash",
        "equal_weight_buy_and_hold",
        "allocator_24h_long_short",
        "allocator_72h_long_short",
        "allocator_24h_long_only",
    )
    assert CANDIDATE_ARMS == ARMS[2:]
    assert SCENARIOS == ("base", "cost_x2", "latency_1bar")
    assert PROTOCOL.arms == ARMS
    assert (PROTOCOL.gamma, PROTOCOL.w_max, PROTOCOL.gross_max) == (5.0, 0.5, 0.95)
    assert (PROTOCOL.bootstrap_block_days, PROTOCOL.bootstrap_resamples) == (20, 2000)
    assert PROTOCOL.bootstrap_seed == 0
    assert PROTOCOL.max_drawdown_limit == 0.20
    assert PROTOCOL.horizons_hours == (24, 72)
    assert PROTOCOL.initial_capital == 10_000.0


def test_protocol_digest_is_stable_and_sensitive() -> None:
    assert protocol_digest() == protocol_digest(DevStudyProtocol())
    assert protocol_digest(replace(PROTOCOL, gamma=6.0)) != protocol_digest()


def test_runner_rejects_a_changed_arm_roster(market: MarketDataset) -> None:
    changed = small_protocol(arms=ARMS[:-1])
    with pytest.raises(ValueError, match="arm roster"):
        runner.run_study(market, changed)


def test_runner_rejects_a_dataset_identity_mismatch(market: MarketDataset) -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        runner.run_study(market, small_protocol(dataset_id="6" * 64))


# --- metrics --------------------------------------------------------------


def test_max_drawdown_and_summary_on_a_known_path() -> None:
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 110.0])) == pytest.approx(0.25)
    hours = np.arange(0, 24 * 40 + 1)
    stamps = hours.astype(np.int64) * 3_600 * 1_000_000_000
    equity = 10_000.0 * (1.01 ** (hours / 24.0))
    summary = performance_summary(stamps, equity, small_protocol())
    assert summary["terminal_net_profit_pct"] == pytest.approx(100.0 * (1.01**40 - 1.0))
    assert summary["max_drawdown_pct"] == pytest.approx(0.0)
    assert summary["sharpe_daily"] is None or summary["sharpe_daily"] > 100.0
    assert list(summary["per_year_return_pct"]) == ["1970"]
    ts, daily = daily_equity(stamps, equity)
    assert len(ts) == 41 and daily[0] == pytest.approx(10_000.0)


def test_block_bootstrap_is_seeded_blockwise_and_paired() -> None:
    first = moving_block_indices(60, block=20, resamples=8, seed=0)
    again = moving_block_indices(60, block=20, resamples=8, seed=0)
    other = moving_block_indices(60, block=20, resamples=8, seed=1)
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert first.shape == (8, 60)
    steps = np.diff(first.reshape(8, 3, 20), axis=2)
    assert np.all(steps == 1)  # each block is a contiguous run
    protocol = small_protocol()
    constant = bootstrap_excess(np.full(60, 0.001), first, protocol)
    assert constant.ci_low == pytest.approx(constant.ci_high)
    assert constant.annualized_mean == pytest.approx(0.001 * 365)
    noisy = np.random.default_rng(0).normal(0.0005, 0.01, size=60)
    interval = bootstrap_excess(
        noisy, moving_block_indices(60, block=20, resamples=400, seed=0), protocol
    )
    assert (
        interval.bonferroni_ci_low
        <= interval.ci_low
        <= interval.ci_high
        <= interval.bonferroni_ci_high
    )


def table(**per_arm: dict[str, tuple[float, float]]) -> dict:
    return {
        arm: {
            scenario: {"terminal_net_profit_pct": p, "max_drawdown_pct": d}
            for scenario, (p, d) in rows.items()
        }
        for arm, rows in per_arm.items()
    }


def rows(base: tuple[float, float], x2: tuple[float, float], lat: tuple[float, float]):
    return {SCENARIO_BASE: base, SCENARIO_COST_X2: x2, SCENARIO_LATENCY: lat}


def test_selection_rule_cases() -> None:
    protocol = PROTOCOL
    good = rows((30.0, 15.0), (10.0, 18.0), (5.0, 19.0))
    results = table(
        allocator_24h_long_short=good,
        allocator_72h_long_short=rows((50.0, 15.0), (20.0, 21.0), (5.0, 10.0)),
        allocator_24h_long_only=rows((20.0, 10.0), (4.0, 12.0), (1.0, 13.0)),
    )
    valid = {arm: True for arm in CANDIDATE_ARMS}
    outcome = select_development_label(results, valid, protocol)
    # 72h has the highest profit but breaches the drawdown gate under cost x2.
    assert outcome.label == LABEL_PROMISING
    assert outcome.selected_arm == "allocator_24h_long_short"
    assert outcome.rows["allocator_72h_long_short"]["risk_gate_passed"] is False

    # Invalid evidence removes an arm even when it is the best performer.
    outcome = select_development_label(
        results, {**valid, "allocator_24h_long_short": False}, protocol
    )
    assert outcome.selected_arm == "allocator_24h_long_only"

    # Failing to beat CASH under cost x2 (or base) disqualifies.
    negative = table(
        allocator_24h_long_short=rows((30.0, 5.0), (-0.1, 5.0), (5.0, 5.0)),
        allocator_72h_long_short=rows((-1.0, 5.0), (-3.0, 5.0), (-2.0, 5.0)),
        allocator_24h_long_only=rows((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    outcome = select_development_label(negative, valid, protocol)
    assert outcome.label == LABEL_NO_WINNER and outcome.selected_arm is None

    # Latency profit does not gate selection, but its drawdown does.
    latency_dd = table(
        allocator_24h_long_short=rows((30.0, 5.0), (10.0, 5.0), (-9.0, 20.5)),
        allocator_72h_long_short=rows((-1.0, 5.0), (-3.0, 5.0), (-2.0, 5.0)),
        allocator_24h_long_only=rows((1.0, 5.0), (1.0, 5.0), (-5.0, 5.0)),
    )
    outcome = select_development_label(latency_dd, valid, protocol)
    assert outcome.selected_arm == "allocator_24h_long_only"
    assert outcome.label in {LABEL_PROMISING, LABEL_NO_WINNER}


def test_the_rule_can_never_return_a_winner_label() -> None:
    results = table(
        allocator_24h_long_short=rows((900.0, 1.0), (800.0, 1.0), (700.0, 1.0)),
        allocator_72h_long_short=rows((900.0, 1.0), (800.0, 1.0), (700.0, 1.0)),
        allocator_24h_long_only=rows((900.0, 1.0), (800.0, 1.0), (700.0, 1.0)),
    )
    outcome = select_development_label(
        results, {arm: True for arm in CANDIDATE_ARMS}, PROTOCOL
    )
    assert outcome.label == "DEVELOPMENT_PROMISING"
    assert "WINNER" not in {outcome.label}


# --- forecast adapter -----------------------------------------------------


class FakeSource:
    def __init__(self, horizon_hours: int, packets: dict[int, object]) -> None:
        self.horizon_hours = horizon_hours
        self._packets = packets

    def packet(self, symbol_index: int, decision_time: np.datetime64) -> object:
        return self._packets.get(symbol_index)


def snapshot_stub(n: int = 3, timestamp: str = "2022-03-01T00:00:00") -> object:
    return SimpleNamespace(
        symbols=tuple(f"S{i}" for i in range(n)),
        timestamp_ns=int(np.datetime64(timestamp, "ns").astype(np.int64)),
    )


@pytest.mark.parametrize("horizon,divisor", [(24, 1.0), (72, 3.0)])
def test_log_forecast_becomes_a_daily_simple_return(
    horizon: int, divisor: float
) -> None:
    packet = SimpleNamespace(point_forecast=0.03, residual_variance=0.01)
    source = FakeSource(horizon, {0: packet, 2: packet})
    adapter = runner.PrequentialSimpleReturnSource(source, decision_interval_hours=24)  # type: ignore[arg-type]
    values = adapter.expected_simple_returns(snapshot_stub())  # type: ignore[arg-type]
    expected = math.expm1(0.03 + 0.5 * 0.01) / divisor
    assert values == (pytest.approx(expected), 0.0, pytest.approx(expected))
    assert adapter.null_packets == 1 and adapter.calls == 1


def test_adapter_rejects_a_horizon_that_is_not_a_multiple_of_the_interval() -> None:
    with pytest.raises(ValueError, match="multiple"):
        runner.PrequentialSimpleReturnSource(
            FakeSource(36, {}),
            decision_interval_hours=24,  # type: ignore[arg-type]
        )


def test_feature_roster_rule_excludes_funding_and_other_timeframes(
    market: MarketDataset,
) -> None:
    assert runner.feature_indices(market, PROTOCOL) == (0, 1)


# --- replay, reconciliation, causality ------------------------------------


def run_one(market: MarketDataset, arm: str, scenario: str):
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    return runner.run_replay(
        market,
        protocol,
        arm,
        scenario,
        sources,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
    )


def test_cash_arm_is_flat_and_reconciles(market: MarketDataset) -> None:
    outcome = run_one(market, ARM_CASH, SCENARIO_BASE)
    assert outcome.summary["terminal_net_profit_pct"] == 0.0
    assert outcome.summary["n_trades"] == 0
    assert outcome.summary["reconciliation"]["passed"] is True  # type: ignore[index]


def test_buy_and_hold_trades_once_in_and_is_settled_flat(market: MarketDataset) -> None:
    outcome = run_one(market, ARM_BUY_AND_HOLD, SCENARIO_BASE)
    reconciliation = outcome.summary["reconciliation"]
    assert reconciliation["passed"] is True  # type: ignore[index]
    assert reconciliation["terminal_flat"] is True  # type: ignore[index]
    # Entry in 3 symbols + terminal exit in 3 symbols; nothing in between.
    assert reconciliation["n_fills"] >= 6  # type: ignore[index]
    assert outcome.summary["turnover_total"] == pytest.approx(2 * 0.99, rel=0.05)
    assert outcome.summary["execution_costs_usdt"] > 0.0  # type: ignore[operator]


def test_allocator_reconciles_in_every_scenario_and_costs_ordering(
    market: MarketDataset,
) -> None:
    outcomes = {
        scenario: run_one(market, ARM_ALLOCATOR_24H, scenario) for scenario in SCENARIOS
    }
    for outcome in outcomes.values():
        assert outcome.summary["reconciliation"]["passed"] is True  # type: ignore[index]
        assert outcome.summary["n_trades"] > 0  # type: ignore[operator]
    base = outcomes[SCENARIO_BASE].summary
    stressed = outcomes[SCENARIO_COST_X2].summary
    assert stressed["execution_costs_usdt"] > base["execution_costs_usdt"]  # type: ignore[operator]
    # The latency scenario really changes execution timing.
    assert not np.array_equal(
        outcomes[SCENARIO_LATENCY].equity, outcomes[SCENARIO_BASE].equity
    )


def test_reconciliation_detects_tampered_ledger_cash(market: MarketDataset) -> None:
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    built = runner.build_arm(ARM_BUY_AND_HOLD, market, protocol, sources)
    result = runner.run_shared_portfolio_replay(
        market,
        built.policy,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        initial_capital=protocol.initial_capital,
        execution_cost=runner.scenario_cost(protocol, SCENARIO_BASE),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert runner.reconcile_ledger(result, market, protocol)["passed"] is True
    ledger = result.ledger_evidence
    tampered_intervals = replace(
        ledger.intervals, final_cash=ledger.intervals.final_cash + 1.0
    )
    tampered = replace(
        result, ledger_evidence=replace(ledger, intervals=tampered_intervals)
    )
    report = runner.reconcile_ledger(tampered, market, protocol)
    assert report["passed"] is False
    assert report["cash_residual"] == pytest.approx(1.0)


def test_perturb_future_only_changes_rows_after_the_cut(market: MarketDataset) -> None:
    cut = 500
    other = runner.perturb_future(market, after_index=cut)
    assert np.array_equal(other.close[: cut + 1], market.close[: cut + 1])
    assert np.array_equal(other.features[: cut + 1], market.features[: cut + 1])
    assert not np.array_equal(other.close[cut + 1 :], market.close[cut + 1 :])
    assert not np.array_equal(other.features[cut + 1 :], market.features[cut + 1 :])


def test_causality_probe_passes_for_the_real_pipeline(market: MarketDataset) -> None:
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    report = runner.causality_probe(
        market,
        protocol,
        sources,
        start_index=runner._row_of(market, protocol.evaluation_start),
    )
    assert report["passed"] is True
    assert report["pre_cut_trace_has_trades"] is True


def test_causality_probe_catches_a_forecast_that_peeks_at_the_future(
    market: MarketDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Leaky:
        horizon_hours = 24

        def __init__(self, dataset: MarketDataset) -> None:
            self._close = dataset.close
            self._stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)

        def packet(self, symbol_index: int, decision_time: np.datetime64) -> object:
            row = int(
                np.searchsorted(self._stamps, int(decision_time.astype(np.int64)))
            )
            future = float(
                np.log(
                    self._close[row + 72, symbol_index] / self._close[row, symbol_index]
                )
            )
            return SimpleNamespace(point_forecast=future, residual_variance=0.0)

    monkeypatch.setattr(
        runner,
        "build_forecast_sources",
        lambda dataset, protocol: {24: Leaky(dataset), 72: Leaky(dataset)},
    )
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    report = runner.causality_probe(
        market,
        protocol,
        sources,
        start_index=runner._row_of(market, protocol.evaluation_start),
    )
    assert report["passed"] is False
    assert report["identical_pre_cut_trace"] is False


def test_causality_probe_matrix_covers_every_arm_and_scenario(
    market: MarketDataset,
) -> None:
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    matrix = runner.causality_probe_matrix(
        market,
        protocol,
        sources,
        start_index=runner._row_of(market, protocol.evaluation_start),
        cut="2022-03-15T00:00:00",
    )
    assert set(matrix) == set(protocol.arms)
    for arm, per_scenario in matrix.items():
        assert set(per_scenario) == set(protocol.scenarios)
        for report in per_scenario.values():
            assert report["passed"] is True
            assert report["identical_pre_cut_trace"] is True
            assert report["cut"] == "2022-03-15T00:00:00"
            if arm == ARM_CASH:
                assert report["pre_cut_trace_has_trades"] is False
            else:
                assert report["pre_cut_trace_has_trades"] is True


def test_causality_probe_matrix_catches_a_leak_in_a_non_default_arm(
    market: MarketDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Leaky:
        horizon_hours = 72

        def __init__(self, dataset: MarketDataset) -> None:
            self._close = dataset.close
            self._stamps = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)

        def packet(self, symbol_index: int, decision_time: np.datetime64) -> object:
            row = int(
                np.searchsorted(self._stamps, int(decision_time.astype(np.int64)))
            )
            future = float(
                np.log(
                    self._close[row + 72, symbol_index] / self._close[row, symbol_index]
                )
            )
            return SimpleNamespace(point_forecast=future, residual_variance=0.0)

    protocol = small_protocol()
    clean = runner.build_forecast_sources(market, protocol)
    monkeypatch.setattr(
        runner,
        "build_forecast_sources",
        lambda dataset, protocol: {24: clean[24], 72: Leaky(dataset)},
    )
    matrix = runner.causality_probe_matrix(
        market,
        protocol,
        runner.build_forecast_sources(market, protocol),
        start_index=runner._row_of(market, protocol.evaluation_start),
        cut="2022-03-15T00:00:00",
    )
    assert matrix[ARM_ALLOCATOR_24H][SCENARIO_BASE]["passed"] is True
    assert matrix[ARM_ALLOCATOR_72H][SCENARIO_BASE]["passed"] is False
    assert matrix[ARM_ALLOCATOR_72H][SCENARIO_LATENCY]["passed"] is False


def test_policies_are_fresh_per_replay(market: MarketDataset) -> None:
    protocol = small_protocol()
    sources = runner.build_forecast_sources(market, protocol)
    first = runner.build_arm(ARM_ALLOCATOR_24H, market, protocol, sources)
    second = runner.build_arm(ARM_ALLOCATOR_24H, market, protocol, sources)
    assert first.policy is not second.policy
    assert first.adapter is not second.adapter


# --- full study -----------------------------------------------------------


def test_study_covers_every_arm_and_scenario_with_valid_evidence(
    study: tuple[dict, dict],
) -> None:
    results, equity = study
    assert tuple(results["arms"]) == ARMS
    for arm in ARMS:
        assert tuple(results["arms"][arm]) == SCENARIOS
        assert results["validity"][arm] is True
    assert results["causality_probe"]["passed"] is True
    assert results["label"] in {LABEL_PROMISING, LABEL_NO_WINNER}
    assert results["status"] == "development_evidence_only"
    assert results["protocol_digest"] == protocol_digest(small_protocol())
    assert len(equity) == 2 * len(ARMS) * len(SCENARIOS)
    cash = results["arms"][ARM_CASH][SCENARIO_BASE]
    assert cash["terminal_net_profit_pct"] == 0.0
    # Bootstrap: versus-cash for the four non-cash arms, versus-B&H for candidates.
    base = results["bootstrap"][SCENARIO_BASE]
    assert set(base) == set(ARMS) - {ARM_CASH}
    assert set(base[ARM_ALLOCATOR_24H]) == {"vs_cash", "vs_equal_weight_buy_and_hold"}
    assert set(base[ARM_BUY_AND_HOLD]) == {"vs_cash"}
    # Net P&L decomposes exactly into gross price P&L, costs and funding.
    for arm in ARMS:
        row = results["arms"][arm][SCENARIO_BASE]
        assert row["net_pnl_usdt"] == pytest.approx(
            row["gross_price_pnl_usdt"]
            - row["execution_costs_usdt"]
            + row["funding_pnl_usdt"]
        )


def test_study_is_deterministic(
    market: MarketDataset, study: tuple[dict, dict]
) -> None:
    again, _ = runner.run_study(market, small_protocol())
    first, _ = study
    drop = ("git_head",)
    assert json.dumps(
        {k: v for k, v in first.items() if k not in drop}, sort_keys=True
    ) == json.dumps({k: v for k, v in again.items() if k not in drop}, sort_keys=True)


# --- CLI ------------------------------------------------------------------


def test_cli_smoke_writes_once_and_refuses_overwrite(
    market: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = small_protocol()
    monkeypatch.setattr(study_cli, "PROTOCOL", protocol)
    monkeypatch.setattr(
        study_cli,
        "load_published_market_dataset_artifact",
        lambda root: (None, market),
    )
    out = tmp_path / "out"
    summary = tmp_path / "summary" / "result.json"
    argv = [
        "--dataset",
        str(tmp_path / "unused"),
        "--output-dir",
        str(out),
        "--summary-path",
        str(summary),
        "--sequential",
    ]
    assert study_cli.main(argv) == 0
    assert (out / "equity.npz").is_file() and (out / "full_results.json").is_file()
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert written["label"] in {LABEL_PROMISING, LABEL_NO_WINNER}
    assert "protocol" not in written and "protocol_deviations" in written
    assert summary.stat().st_size < 200_000
    assert study_cli.main(argv) == 2


def test_snapshot_type_is_the_real_replay_snapshot() -> None:
    # The adapter reads only symbols and timestamp_ns from the replay snapshot.
    fields = PortfolioDecisionSnapshot.__dataclass_fields__
    assert "symbols" in fields and "timestamp_ns" in fields


def test_parallel_scenarios_match_the_sequential_study(tmp_path: Path) -> None:
    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact

    identified = synthetic_market().with_content_identity()
    root = tmp_path / "dataset"
    publish_market_dataset_artifact(root, identified)
    protocol = small_protocol(dataset_id=identified.dataset_id)
    sequential, _ = runner.run_study(identified, protocol)
    parallel, _ = runner.run_study(
        identified,
        protocol,
        parallel_loader=partial(runner.load_dataset, str(root)),
    )
    drop = ("git_head",)
    assert json.dumps(
        {k: v for k, v in sequential.items() if k not in drop}, sort_keys=True
    ) == json.dumps(
        {k: v for k, v in parallel.items() if k not in drop}, sort_keys=True
    )
