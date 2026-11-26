from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study import trial_sources as sources
from trade_rl.evaluation.dev_realdata_study.metrics import select_development_label
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_CASH,
    LABEL_NO_WINNER,
    LABEL_PROMISING,
    PROTOCOL,
    SCENARIOS,
    DevStudyProtocol,
)
from trade_rl.evaluation.dev_realdata_study.trials import (
    ALL_ARMS,
    CANDIDATE_POOL,
    CONSTANTS,
    PR_C_ARMS,
    TRIAL_ARMS,
    TRIALS,
    WEEKLY_MONDAY_ANCHOR_HOUR,
    registry_digest,
    registry_json_bytes,
    registry_payload,
)
from trade_rl.simulation import ExecutionCostConfig

HOUR = np.timedelta64(1, "h")
START = np.datetime64("2021-01-01T00:00:00", "ns")
N_DAYS = 330
N_BARS = N_DAYS * 24 + 1
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
COST = 0.0007
LIGHTGBM_INSTALLED = importlib.util.find_spec("lightgbm") is not None
REPO_ROOT = Path(__file__).resolve().parents[2]


def synthetic_market(*, seed: int = 21) -> MarketDataset:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, size=(N_BARS, len(SYMBOLS)))
    steps[:, 0] += 0.0008
    steps[:, 1] -= 0.0004
    close = 100.0 * np.exp(np.cumsum(steps, axis=0))
    names = ("1h__trail", "1h__trail_sq", "1h__funding_bps", "4h__trail", "1d__trail")
    features = np.zeros((N_BARS, len(SYMBOLS), len(names)), dtype=np.float32)
    available = np.zeros((N_BARS, len(SYMBOLS), len(names)), dtype=np.bool_)
    trailing = np.log(close[24:] / close[:-24])
    features[24:, :, 0] = trailing
    features[24:, :, 1] = trailing**2
    features[24:, :, 2] = rng.normal(size=trailing.shape)
    features[24:, :, 3] = trailing
    features[24:, :, 4] = 0.5 * trailing
    available[24:] = True
    funding = np.zeros((N_BARS, len(SYMBOLS)))
    due = np.zeros((N_BARS, len(SYMBOLS)), dtype=np.bool_)
    funding[8::8] = rng.normal(0.0001, 0.00005, size=funding[8::8].shape)
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
        feature_names=names,
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
        covariance_warmup_start="2021-01-04T00:00:00",
        evaluation_start="2021-10-04T00:00:00",
        evaluation_end="2021-11-08T00:00:00",
        causality_cut="2021-10-25T00:00:00",
        refit_interval_days=15,
        min_history_periods=10,
        ewma_halflife_days=15.0,
        bootstrap_resamples=64,
        bootstrap_block_days=5,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def market() -> MarketDataset:
    return synthetic_market()


def stamp_ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


def snapshot_stub(timestamp: str) -> Any:
    return SimpleNamespace(symbols=SYMBOLS, timestamp_ns=stamp_ns(timestamp))


# --- registry -------------------------------------------------------------


def test_the_roster_is_exactly_eight_registered_trials() -> None:
    assert len(TRIALS) == 8
    assert [trial.trial_id for trial in TRIALS] == [f"T{i}" for i in range(1, 9)]
    assert len(set(TRIAL_ARMS)) == 8
    assert ALL_ARMS == (*PR_C_ARMS, *TRIAL_ARMS) and len(ALL_ARMS) == 13
    assert CANDIDATE_POOL == (*PROTOCOL.candidate_arms, *TRIAL_ARMS)
    assert CONSTANTS.bonferroni_arms == 13
    for trial in TRIALS:
        assert trial.source_kind in sources.SOURCE_REGISTRY
    by_id = {trial.trial_id: trial for trial in TRIALS}
    assert by_id["T7"].overlay == "drawdown_latch"
    assert by_id["T8"].decision_interval_hours == 168
    assert all(t.decision_interval_hours == 24 for t in TRIALS if t.trial_id != "T8")
    assert not any(trial.long_only for trial in TRIALS)


def test_registry_json_is_the_committed_export() -> None:
    path = REPO_ROOT / "report" / "issue810-phase2-trial-registry-20261029.json"
    assert path.read_bytes() == registry_json_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["registry_digest"] == registry_digest()
    assert payload["status"] == "registered"
    assert [row["trial_id"] for row in payload["trials"]] == [
        f"T{i}" for i in range(1, 9)
    ]
    assert registry_payload()["base_protocol_digest"] == payload["base_protocol_digest"]


def test_weekly_anchor_is_monday_midnight_utc() -> None:
    monday = stamp_ns("2023-01-02T00:00:00")
    interval = 168 * 3_600 * 1_000_000_000
    assert monday % interval == WEEKLY_MONDAY_ANCHOR_HOUR * 3_600 * 1_000_000_000
    assert np.datetime64("2023-01-02", "D").astype(int) % 7 == 4  # Monday


# --- signal math ----------------------------------------------------------


def test_tsmom_signal_math_on_a_steady_uptrend_and_a_mixed_trend() -> None:
    days = 130
    up = 100.0 * 1.01 ** np.arange(days)
    # Rising for 100 days then falling for 30: r30 < 0 but r90 > 0 at the end.
    mixed = np.concatenate([up[:100], up[99] * 0.99 ** np.arange(1, 31)])
    marks = np.column_stack([up, mixed])
    series = sources.tsmom_series(marks, one_way_cost=COST, constants=CONSTANTS)
    vol_up = 0.01 * np.sqrt(365)
    assert series[days - 1, 0] == pytest.approx(2.0 * (0.20 / vol_up) * COST)
    assert np.isnan(series[:89]).all()  # needs 90 days of history
    ann_vol = sources.ewma_annual_vol(
        marks, halflife_days=60.0, min_periods=30, days_per_year=365
    )
    assert np.isnan(ann_vol[:30]).all() and np.isfinite(ann_vol[30:]).all()
    assert ann_vol[days - 1, 0] == pytest.approx(vol_up)
    # mixed: r30 negative, r90 positive -> s = 0.5*(-1) + 0.5*(+1) = 0.
    assert sources.lagged_return(marks, 30)[days - 1, 1] < 0
    assert sources.lagged_return(marks, 90)[days - 1, 1] > 0
    assert series[days - 1, 1] == pytest.approx(0.0)


def test_xsmom_is_demeaned_scaled_and_skips_the_last_day() -> None:
    days = 100
    growth = np.array([1.004, 1.001, 0.998])
    marks = 100.0 * growth[None, :] ** np.arange(days)[:, None]
    marks[-1] *= np.array([0.5, 2.0, 1.0])  # the skipped day must not matter
    row = sources.xsmom_row(marks, days - 1, one_way_cost=COST, constants=CONSTANTS)
    assert row.sum() == pytest.approx(0.0, abs=1e-15)
    assert np.abs(row).mean() == pytest.approx(2.0 * COST)
    assert row[0] > 0 > row[2]
    # Dropping the final day changes nothing about the day-(days-1) score.
    assert np.all(
        np.isnan(sources.xsmom_row(marks, 50, one_way_cost=COST, constants=CONSTANTS))
    )
    flat = np.full((100, 3), 50.0)
    assert np.array_equal(
        sources.xsmom_row(flat, 99, one_way_cost=COST, constants=CONSTANTS),
        np.zeros(3),
    )


def test_funding_carry_is_minus_the_trailing_seven_day_mean_per_24h(
    market: MarketDataset,
) -> None:
    constant = replace(
        market,
        funding_rate=np.where(market.funding_due, 0.0002, 0.0),
    )
    panel = sources.DailyPanel(constant)
    series = panel.funding_carry_series(7)
    day = panel.position(stamp_ns("2021-06-15T00:00:00"))
    # 21 funding events of 0.0002 in seven days -> 0.0006 per 24h, sign flipped.
    assert series[day] == pytest.approx(np.full(3, -0.0006))
    assert np.isnan(series[:6]).all()


def test_daily_panel_rejects_a_missing_daily_bar(market: MarketDataset) -> None:
    panel = sources.DailyPanel(market)
    with pytest.raises(ValueError, match="00:00 UTC"):
        panel.position(stamp_ns("2021-06-15T05:00:00"))


# --- causality: future perturbation leaves every signal unchanged ----------

CUT = "2021-10-20T00:00:00"
MATRIX_CUT = "2021-10-18T00:00:00"


@pytest.fixture(scope="module")
def perturbed(market: MarketDataset) -> MarketDataset:
    return runner.perturb_future(market, after_index=runner._row_of(market, CUT))


def decision_days() -> list[str]:
    days = np.arange(
        np.datetime64("2021-10-04"), np.datetime64("2021-10-21"), np.timedelta64(1, "D")
    )
    return [f"{day}T00:00:00" for day in days]


def values_equal(first: Any, second: Any) -> bool:
    return bool(np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True))


def test_pure_series_signals_ignore_the_future(
    market: MarketDataset, perturbed: MarketDataset
) -> None:
    real, other = sources.DailyPanel(market), sources.DailyPanel(perturbed)
    last = real.position(stamp_ns(CUT))
    assert not values_equal(real.marks[last + 1 :], other.marks[last + 1 :])
    assert values_equal(real.marks[: last + 1], other.marks[: last + 1])
    first = sources.tsmom_series(real.marks, one_way_cost=COST, constants=CONSTANTS)
    second = sources.tsmom_series(other.marks, one_way_cost=COST, constants=CONSTANTS)
    assert values_equal(first[: last + 1], second[: last + 1])
    assert not values_equal(first[last + 1 :], second[last + 1 :])
    for day in range(100, last + 1):
        assert values_equal(
            sources.xsmom_row(real.marks, day, one_way_cost=COST, constants=CONSTANTS),
            sources.xsmom_row(other.marks, day, one_way_cost=COST, constants=CONSTANTS),
        )
    carry_real = real.funding_carry_series(7)
    carry_other = other.funding_carry_series(7)
    assert values_equal(carry_real[: last + 1], carry_other[: last + 1])
    assert not values_equal(carry_real[last + 1 :], carry_other[last + 1 :])


def served(arm: str, data: MarketDataset) -> list[tuple[float, ...]]:
    protocol = small_protocol()
    context = sources.TrialContext(data, protocol)
    built = sources.build_phase2_arm(arm, data, protocol, context)
    return [
        built.adapter.expected_simple_returns(snapshot_stub(day))
        for day in decision_days()
    ]


@pytest.mark.parametrize(
    "arm",
    [
        "t1_tsmom",
        "t2_xsmom",
        "t3_funding_carry",
        "t4_ridge24_all_features",
        "t6_blend",
        "t7_blend_dd_overlay",
    ],
)
def test_every_served_signal_ignores_the_future(
    market: MarketDataset, perturbed: MarketDataset, arm: str
) -> None:
    assert served(arm, market) == served(arm, perturbed)


@pytest.mark.skipif(not LIGHTGBM_INSTALLED, reason="forecast-gbm extra not installed")
def test_lightgbm_signal_ignores_the_future(
    market: MarketDataset, perturbed: MarketDataset
) -> None:
    assert served("t5_lightgbm24", market) == served("t5_lightgbm24", perturbed)


def test_served_signals_are_nontrivial(market: MarketDataset) -> None:
    for arm in ("t1_tsmom", "t3_funding_carry", "t4_ridge24_all_features"):
        history = np.asarray(served(arm, market))
        assert history.shape == (len(decision_days()), 3)
        assert np.abs(history).max() > 0.0
    blend = np.asarray(served("t6_blend", market))
    assert np.abs(blend[:9]).max() == 0.0  # fewer than ten Ridge24 dates
    assert np.abs(blend[10:]).max() > 0.0


@pytest.mark.parametrize(
    "arm",
    ["t1_tsmom", "t2_xsmom", "t3_funding_carry", "t6_blend", "t8_blend_weekly"],
)
def test_real_replay_causality_probe_passes_for_registered_trials(
    market: MarketDataset, arm: str
) -> None:
    protocol = small_protocol()
    report = runner.causality_probe(
        market,
        protocol,
        sources.TrialContext(market, protocol),
        start_index=runner._row_of(market, protocol.evaluation_start),
        arm=arm,
        expect_trades=False,
        arm_builder=sources.build_phase2_arm,
        sources_builder=sources.TrialContext,
    )
    assert report["identical_pre_cut_trace"] is True
    assert report["passed"] is True


def test_trial_probe_rule_does_not_require_pre_cut_trades(
    market: MarketDataset,
) -> None:
    """Pins the amended rule: a non-trading arm passes iff its traces are identical.

    CASH never trades, so under the PR-C default (expect_trades=True) it fails while
    the trace is still identical; with expect_trades=False (the rule the Phase 2 study
    applies to every trial) the same replay passes.
    """

    protocol = small_protocol()
    start = runner._row_of(market, protocol.evaluation_start)
    context = sources.TrialContext(market, protocol)
    common: dict[str, Any] = {
        "start_index": start,
        "arm": ARM_CASH,
        "arm_builder": sources.build_phase2_arm,
        "sources_builder": sources.TrialContext,
    }
    strict = runner.causality_probe(market, protocol, context, **common)
    amended = runner.causality_probe(
        market, protocol, context, expect_trades=False, **common
    )
    assert strict["identical_pre_cut_trace"] is True
    assert strict["pre_cut_trace_has_trades"] is False and strict["passed"] is False
    assert amended["pre_cut_trace_has_trades"] is False and amended["passed"] is True


# --- T4/T5 sources ---------------------------------------------------------


def test_t4_uses_every_feature_and_t5_uses_the_pr_c_roster(
    market: MarketDataset,
) -> None:
    context = sources.TrialContext(market, small_protocol())
    all_source = context.forecast_source("ridge24_all")
    assert all_source.horizon_hours == 24
    assert len(all_source._feature_indices) == 5
    assert runner.feature_indices(market, small_protocol()) == (0, 1)


@pytest.mark.skipif(not LIGHTGBM_INSTALLED, reason="forecast-gbm extra not installed")
def test_lightgbm_trial_fits_with_the_fixed_parameters(market: MarketDataset) -> None:
    fitter = sources.ConfiguredLightGBMFitter()
    assert (fitter.n_estimators, fitter.max_depth, fitter.min_child_samples) == (
        200,
        3,
        200,
    )
    assert fitter.learning_rate == 0.05
    context = sources.TrialContext(market, small_protocol())
    source = context.forecast_source("lgbm24_1h")
    packet = source.packet(0, np.datetime64("2021-10-10T00:00:00", "ns"))
    assert packet is not None
    assert "min_leaf=200" in packet.fitter_identity


# --- blend math -------------------------------------------------------------


class FakeRidge:
    def __init__(self, rows: dict[int, np.ndarray]) -> None:
        self._rows = rows

    def value_at(self, day: int) -> np.ndarray | None:
        return self._rows.get(day)


def test_blend_formula_and_warmup(market: MarketDataset) -> None:
    panel = sources.DailyPanel(market)
    rng = np.random.default_rng(0)
    days = panel.n_days
    ridge_rows = {d: rng.normal(0.0, 0.003, size=3) for d in range(200, 260)}
    tsmom = rng.normal(0.0, 0.001, size=(days, 3))
    carry = rng.normal(0.0, 0.0003, size=(days, 3))
    blend = sources.BlendSource(
        panel,
        FakeRidge(ridge_rows),  # type: ignore[arg-type]
        tsmom,
        carry,
        constants=CONSTANTS,
        interval_scale=1.0,
    )
    assert np.isnan(blend.values(205)).all()  # only 6 Ridge dates so far
    day = 259
    window = range(max(0, day - 364), day + 1)
    ridge_hist = np.vstack([ridge_rows[d] for d in window if d in ridge_rows])
    s_r = np.std(ridge_hist)
    s_t = np.std(tsmom[window.start : day + 1])
    s_c = np.std(carry[window.start : day + 1])
    expected = (ridge_rows[day] / s_r + tsmom[day] / s_t + carry[day] / s_c) / 3.0 * s_r
    assert blend.values(day) == pytest.approx(expected)
    weekly = sources.BlendSource(
        panel,
        FakeRidge(ridge_rows),  # type: ignore[arg-type]
        tsmom,
        carry,
        constants=CONSTANTS,
        interval_scale=7.0,
    )
    assert weekly.values(day) == pytest.approx(7.0 * expected)
    # A symbol without a Ridge packet contributes zero for that component only.
    holes = {d: row.copy() for d, row in ridge_rows.items()}
    holes[day][1] = np.nan
    partial_blend = sources.BlendSource(
        panel,
        FakeRidge(holes),  # type: ignore[arg-type]
        tsmom,
        carry,
        constants=CONSTANTS,
        interval_scale=1.0,
    ).values(day)
    sigma = np.std(
        np.vstack([holes[d] for d in window if d in holes])[
            np.isfinite(np.vstack([holes[d] for d in window if d in holes]))
        ]
    )
    assert partial_blend[1] == pytest.approx(
        (tsmom[day][1] / s_t + carry[day][1] / s_c) / 3.0 * sigma
    )


# --- T7 overlay ---------------------------------------------------------------


def test_overlay_scale_is_linear_between_ten_and_twenty_percent() -> None:
    kwargs = {"start": 0.10, "limit": 0.20}
    assert sources.overlay_scale(0.0, **kwargs) == 1.0
    assert sources.overlay_scale(0.10, **kwargs) == 1.0
    assert sources.overlay_scale(0.125, **kwargs) == pytest.approx(0.75)
    assert sources.overlay_scale(0.15, **kwargs) == pytest.approx(0.5)
    assert sources.overlay_scale(0.20, **kwargs) == 0.0
    assert sources.overlay_scale(0.35, **kwargs) == 0.0


def test_drawdown_latch_holds_zero_until_equity_recovers_to_95_percent_of_peak() -> (
    None
):
    latch = sources.DrawdownLatch()
    assert latch.update(100.0, 100.0) == 1.0
    assert latch.update(92.0, 100.0) == 1.0  # DD 8%
    assert latch.update(85.0, 100.0) == pytest.approx(0.5)
    assert latch.update(80.0, 100.0) == 0.0 and latch.latched  # DD 20%
    # DD back to 10% would give scale 1, but the latch holds at zero.
    assert latch.update(90.0, 100.0) == 0.0 and latch.latched
    assert latch.update(94.9, 100.0) == 0.0 and latch.latched
    assert latch.update(95.0, 100.0) == 1.0 and not latch.latched
    # A new peak resets the reference: 100 -> 120 peak, 110 is an 8.3% DD.
    assert latch.update(110.0, 120.0) == 1.0
    with pytest.raises(ValueError):
        latch.update(100.0, 0.0)


class InnerStub:
    def __init__(self, proposal: Any) -> None:
        self.proposal = proposal
        self.calls = 0
        self.records: tuple[object, ...] = ()

    def decide_portfolio(self, snapshot: Any) -> Any:
        self.calls += 1
        return self.proposal


def overlay_snapshot(timestamp: str, equity: float, peak: float) -> Any:
    return SimpleNamespace(
        timestamp_ns=stamp_ns(timestamp),
        account=SimpleNamespace(portfolio_value=equity, peak_value=peak),
    )


def test_overlay_policy_caps_gross_at_scale_times_gross_max() -> None:
    from trade_rl.strategies.portfolio import PortfolioTargetProposal

    proposal = PortfolioTargetProposal(
        target_weights=(0.5, -0.4, 0.0), hold_current_quantity_symbols=(2,)
    )
    inner = InnerStub(proposal)
    policy = sources.DrawdownOverlayPolicy(
        inner,
        gross_max=0.95,
        decision_interval_hours=24,
        decision_anchor_hour_utc=0,
    )
    # Off-schedule bar: passthrough and no latch/scale update.
    assert (
        policy.decide_portfolio(overlay_snapshot("2023-01-01T05:00:00", 70.0, 100.0))
        is proposal
    )
    assert policy.scales == []
    # Scale 1: the very same proposal object comes back.
    assert (
        policy.decide_portfolio(overlay_snapshot("2023-01-02T00:00:00", 100.0, 100.0))
        is proposal
    )
    # DD 15% -> scale 0.5 -> budget 0.475 against gross 0.9.
    shrunk = policy.decide_portfolio(
        overlay_snapshot("2023-01-03T00:00:00", 85.0, 100.0)
    )
    factor = 0.475 / 0.9
    assert shrunk.target_weights == pytest.approx((0.5 * factor, -0.4 * factor, 0.0))
    assert shrunk.hold_current_quantity_symbols == ()
    # Budget above the proposal's gross: untouched (scale 0.95 * 0.95 = 0.9025).
    big = sources.DrawdownOverlayPolicy(
        InnerStub(proposal),
        gross_max=0.95,
        decision_interval_hours=24,
        decision_anchor_hour_utc=0,
    )
    assert (
        big.decide_portfolio(overlay_snapshot("2023-01-02T00:00:00", 100.0, 100.0))
        is proposal
    )
    # DD 20% -> latched at zero: flat target.
    flat = policy.decide_portfolio(overlay_snapshot("2023-01-04T00:00:00", 80.0, 100.0))
    assert flat.target_weights == (0.0, 0.0, 0.0)
    summary = policy.overlay_summary()
    assert summary["n_decisions"] == 3
    assert summary["n_decisions_at_zero"] == 1
    assert summary["latched_at_end"] is True
    assert inner.calls == 4  # the inner allocator is always consulted


# --- weekly schedule --------------------------------------------------------


def test_t8_decides_only_on_mondays_at_midnight(market: MarketDataset) -> None:
    protocol = small_protocol()
    context = sources.TrialContext(market, protocol)
    built = sources.build_phase2_arm("t8_blend_weekly", market, protocol, context)
    runner.run_shared_portfolio_replay(
        market,
        built.policy,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        initial_capital=protocol.initial_capital,
        execution_cost=ExecutionCostConfig(),
    )
    stamps = [record.timestamp_ns for record in built.policy.records]
    assert len(stamps) == 5  # five Mondays from 2021-10-04 up to 2021-11-01
    for stamp in stamps:
        day = np.datetime64(stamp, "ns").astype("datetime64[D]")
        assert day.astype(int) % 7 == 4  # Monday
        assert stamp % (24 * 3_600 * 1_000_000_000) == 0


# --- full study ---------------------------------------------------------------


def test_selection_over_the_eleven_candidates_applies_the_pr_c_selection_rule() -> None:
    protocol = replace(
        small_protocol(),
        arms=ALL_ARMS,
        candidate_arms=CANDIDATE_POOL,
        bonferroni_comparisons=CONSTANTS.bonferroni_arms,
    )

    def rows(base: float, x2: float, drawdown: float) -> dict[str, dict[str, float]]:
        return {
            "base": {"terminal_net_profit_pct": base, "max_drawdown_pct": drawdown},
            "cost_x2": {"terminal_net_profit_pct": x2, "max_drawdown_pct": drawdown},
            "latency_1bar": {
                "terminal_net_profit_pct": 0.0,
                "max_drawdown_pct": drawdown,
            },
        }

    losing = {arm: rows(-1.0, -2.0, 30.0) for arm in CANDIDATE_POOL}
    valid = {arm: True for arm in CANDIDATE_POOL}
    assert select_development_label(losing, valid, protocol).label == LABEL_NO_WINNER
    # Profitable but breaching the 20% drawdown gate: still NO_WINNER.
    risky = {**losing, "t6_blend": rows(40.0, 20.0, 35.0)}
    assert select_development_label(risky, valid, protocol).label == LABEL_NO_WINNER
    # Gate passed, profitable at cost x2 and best in base: promising, not a winner.
    good = {
        **risky,
        "t7_blend_dd_overlay": rows(12.0, 3.0, 19.0),
        "t2_xsmom": rows(8.0, 1.0, 18.0),
    }
    outcome = select_development_label(good, valid, protocol)
    assert outcome.label == LABEL_PROMISING
    assert outcome.selected_arm == "t7_blend_dd_overlay"
    # Unprofitable at cost x2 removes it.
    fragile = {**good, "t7_blend_dd_overlay": rows(12.0, -0.1, 19.0)}
    assert select_development_label(fragile, valid, protocol).selected_arm == "t2_xsmom"


def _report(name: str) -> Any:
    return json.loads((REPO_ROOT / "report" / name).read_text("utf-8"))


def test_registry_digest_is_frozen() -> None:
    assert (
        registry_digest()
        == "d3cfc7278205e263f960ce2b32a82c8f43daa95f30366b0d83426a07f782b0ee"
    )
    # Frozen text; corrected by the provenance erratum, not edited.
    assert registry_payload()["registered_before"] == (
        "the PR-C result and every T1-T8 result"
    )


def test_committed_result_and_provenance_are_consistent() -> None:
    result = _report("issue810-phase2-trial-results-20261029.json")
    provenance = _report("issue810-phase2-trial-provenance-20261029.json")
    assert result["registry_digest"] == registry_digest()
    assert provenance["registry_digest"] == registry_digest()
    assert result["label"] in {LABEL_PROMISING, LABEL_NO_WINNER}
    assert result["label"] == result["label_pr_c_probe_rule"]
    assert set(result["arms"]) == set(ALL_ARMS)
    assert result["failures"] == {}
    for arm in TRIAL_ARMS:
        assert set(result["causality_matrix"][arm]) == set(SCENARIOS)
        assert result["causality_probes"][arm]["identical_pre_cut_trace"] is True
        for scenario in SCENARIOS:
            row = result["arms"][arm][scenario]
            assert "terminal_residual_notional_usdt" in row
            assert "fail_closed_hold_count" in row
    full = [e for e in provenance["executions"] if e["kind"] == "full"]
    assert len(full) == provenance["real_data_execution_count"]["full_executions"]
    assert result["run_metadata"]["git_head"]
    erratum = provenance["registry_erratum"]
    assert "21:43:33Z" in erratum["stage_1_headline_specs"]
    assert "AFTER the PR-C result" in erratum["stage_2_implementation_details"]
    ids = {item["id"] for item in provenance["declared_deviations_and_clarifications"]}
    assert ids >= {"L1", "L2", "L3"}


def test_execution_one_replay_metrics_match_the_committed_result() -> None:
    result = _report("issue810-phase2-trial-results-20261029.json")
    execution1 = _report("issue810-phase2-trial-execution1-summary-20261029.json")
    compared = 0
    for arm in ALL_ARMS:
        for scenario in SCENARIOS:
            first = execution1["arms"][arm][scenario]
            final = result["arms"][arm][scenario]
            shared = set(first) & set(final)
            # every field both executions recorded (18 to 20 per row) must be equal
            assert len(shared) >= 18
            for name in shared:
                assert first[name] == final[name], (arm, scenario, name)
                compared += 1
    assert compared >= 18 * len(ALL_ARMS) * len(SCENARIOS)
