"""Contract tests for the Phase 3 mechanisms and the committed evidence.

These check the declared contracts (kappa scales only the optimizer cost, VT scales
the solved target, causality, fail-closed paths, pinned digests, committed-result
consistency). They are NOT a result-blind G0-G2 review or G2 approval of any Phase 3
mechanism: no such review took place before the economics were generated.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import (
    COST,
    MATRIX_CUT,
    SYMBOLS,
    FakeRidge,
    small_protocol,
    stamp_ns,
    synthetic_market,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study import (
    phase3_cli,
    phase3_registry,
    runner,
)
from trade_rl.evaluation.dev_realdata_study import phase3_sources as sources
from trade_rl.evaluation.dev_realdata_study import phase3_study as study
from trade_rl.evaluation.dev_realdata_study import trial_sources as phase2_sources
from trade_rl.evaluation.dev_realdata_study.phase3_registry import (
    ALL_ARMS,
    BASELINE_ARMS,
    CANDIDATE_POOL,
    CONTROL_ARMS,
    KAPPA,
    LABEL_FAILED_TO_RUN,
    LABEL_NO_WINNER,
    LABEL_POST_HOC_PROMISING,
    P3_ARMS,
    P3_TRIALS,
    PASSIVE_COMPARATOR_ARM,
    PUBLIC_RECORD_BODY,
    VT_TARGET_ANNUAL,
    registry_digest,
    registry_json_bytes,
    registry_payload,
    roster_digest,
)
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    ARM_CASH,
    PROTOCOL,
    SCENARIOS,
)
from trade_rl.evaluation.dev_realdata_study.trials import CONSTANTS
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.allocation.cost_aware import (
    annualized_portfolio_vol,
    ewma_covariance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HOUR_NS = 3_600 * 1_000_000_000


@pytest.fixture(scope="module")
def market() -> MarketDataset:
    return synthetic_market()


def served(arm: str, data: MarketDataset) -> list[tuple[float, ...]]:
    protocol = small_protocol()
    context = sources.make_phase3_context(data, protocol)
    built = sources.build_phase3_arm(arm, data, protocol, context)
    days = np.arange(
        np.datetime64("2021-10-04"), np.datetime64("2021-10-21"), np.timedelta64(1, "D")
    )
    from types import SimpleNamespace

    out = []
    for day in days:
        snapshot = SimpleNamespace(
            symbols=SYMBOLS, timestamp_ns=stamp_ns(f"{day}T00:00:00")
        )
        out.append(built.adapter.expected_simple_returns(snapshot))
    return out


# --- registry -----------------------------------------------------------------


def test_the_roster_is_exactly_the_eight_public_record_rows() -> None:
    assert [spec.trial_id for spec in P3_TRIALS] == [f"P3-{i}" for i in range(1, 9)]
    assert len(set(P3_ARMS)) == 8
    assert ALL_ARMS == (ARM_CASH, ARM_BUY_AND_HOLD, *P3_ARMS) and len(ALL_ARMS) == 10
    assert BASELINE_ARMS == (ARM_CASH, ARM_BUY_AND_HOLD)
    assert PASSIVE_COMPARATOR_ARM == "p3_7_ew_bh_vt12"
    assert CONTROL_ARMS == (ARM_CASH, ARM_BUY_AND_HOLD, PASSIVE_COMPARATOR_ARM)
    assert CANDIDATE_POOL == tuple(a for a in P3_ARMS if a != PASSIVE_COMPARATOR_ARM)
    assert len(CANDIDATE_POOL) == 7
    by_id = {spec.trial_id: spec for spec in P3_TRIALS}
    expected = {
        "P3-1": ("T4 + kappa3", KAPPA, None),
        "P3-2": ("T4 + kappa3 + VT12", KAPPA, VT_TARGET_ANNUAL),
        "P3-3": ("T2 + VT12", 1.0, VT_TARGET_ANNUAL),
        "P3-4": ("T6 + kappa3 + VT12", KAPPA, VT_TARGET_ANNUAL),
        "P3-5": ("T4 weekly (168h, mu x7) + VT12", 1.0, VT_TARGET_ANNUAL),
        "P3-6": ("z-blend(T2, T4) + kappa3 + VT12", KAPPA, VT_TARGET_ANNUAL),
        "P3-7": (
            "EW buy-and-hold + VT12 (vol-scaled passive comparator)",
            1.0,
            VT_TARGET_ANNUAL,
        ),
        "P3-8": (
            "z-blend(T2, T4) long-only + kappa3 + VT12",
            KAPPA,
            VT_TARGET_ANNUAL,
        ),
    }
    assert KAPPA == 3.0 and VT_TARGET_ANNUAL == 0.12
    for trial_id, (text, kappa, target) in expected.items():
        spec = by_id[trial_id]
        assert (spec.roster_text, spec.kappa, spec.vol_target) == (text, kappa, target)
    assert by_id["P3-5"].decision_interval_hours == 168
    assert by_id["P3-5"].decision_anchor_hour_utc == 96
    assert [t.long_only for t in P3_TRIALS] == [False] * 7 + [True]
    assert all(
        t.decision_interval_hours == 24 for t in P3_TRIALS if t.trial_id != "P3-5"
    )


def test_roster_text_is_verbatim_from_the_public_record() -> None:
    assert (
        hashlib.sha256(PUBLIC_RECORD_BODY.encode("utf-8")).hexdigest()
        == phase3_registry.PUBLIC_RECORD_SHA256
    )
    for spec in P3_TRIALS:
        assert f"| {spec.trial_id} | {spec.roster_text} |" in PUBLIC_RECORD_BODY
    assert "kappa = 3" in PUBLIC_RECORD_BODY and "12%" in PUBLIC_RECORD_BODY
    assert "21 development arms" in PUBLIC_RECORD_BODY
    assert "POST_HOC_PROMISING or NO_WINNER" in PUBLIC_RECORD_BODY
    assert phase3_registry.PUBLIC_RECORD_URL.endswith("#issuecomment-6493851921")
    assert phase3_registry.PUBLIC_RECORD_CREATED_UTC == "2026-10-29T10:10:07Z"
    assert phase3_registry.PUBLIC_RECORD_EDITED_UTC == "2026-10-29T10:10:19Z"


def test_registry_json_is_the_committed_export_and_the_digests_are_pinned() -> None:
    path = REPO_ROOT / "report" / "issue810-phase3-trial-registry-20261029.json"
    assert path.read_bytes() == registry_json_bytes()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["registry_digest"] == registry_digest()
    assert payload["status"] == "post_hoc_exploratory"
    assert "not preregistered" in payload["classification"]
    assert payload["public_pre_run_record"]["body_verbatim"] == PUBLIC_RECORD_BODY
    assert payload["bonferroni_arms"] == 21
    assert registry_payload()["base_protocol_digest"] == (
        "7adc12e842cb8f574bcd3bd502182d9ffdd19dcc7d1a9b59cbc99ab6f4626558"
    )
    assert roster_digest() == (
        "f81282e7ef697516ebe9fe8c4b493daa52268a9542a61368be3da8b4a244b3c5"
    )
    # The roster digest does not cover the implementation notes.
    assert "implementation_notes" not in phase3_registry.roster_payload()


def test_registry_never_calls_the_roster_preregistered() -> None:
    text = json.dumps(registry_payload())
    assert "preregistered" not in text.replace("not preregistered", "")


# --- sources -------------------------------------------------------------------


def test_scaled_source_multiplies_and_records_the_scaled_history() -> None:
    class Inner:
        calls = 3
        null_packets = 1

        def expected_simple_returns(self, snapshot: Any) -> tuple[float, ...]:
            return (0.001, -0.002, 0.0)

    scaled = sources.ScaledSource(Inner(), 7.0)
    assert scaled.expected_simple_returns(object()) == pytest.approx(
        (0.007, -0.014, 0.0)
    )
    assert scaled.history == [scaled.history[0]] and scaled.calls == 3
    assert scaled.null_packets == 1
    with pytest.raises(ValueError):
        sources.ScaledSource(Inner(), 0.0)


def test_xsmom_series_row_matches_the_registered_row_and_is_causal() -> None:
    rng = np.random.default_rng(4)
    marks = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(160, 3)), axis=0))
    series = sources.xsmom_series(marks, one_way_cost=COST, constants=CONSTANTS)
    assert np.isnan(series[:91]).all()
    for day in (91, 120, 159):
        assert series[day] == pytest.approx(
            phase2_sources.xsmom_row(marks, day, one_way_cost=COST, constants=CONSTANTS)
        )
    future = marks.copy()
    future[130:] *= 3.0
    other = sources.xsmom_series(future, one_way_cost=COST, constants=CONSTANTS)
    assert np.array_equal(series[:131], other[:131], equal_nan=True)


def test_pair_blend_formula_and_warmup(market: MarketDataset) -> None:
    panel = phase2_sources.DailyPanel(market)
    rng = np.random.default_rng(1)
    days = panel.n_days
    ridge_rows = {d: rng.normal(0.0, 0.003, size=3) for d in range(200, 260)}
    xsmom = rng.normal(0.0, 0.0014, size=(days, 3))
    blend = sources.PairBlendSource(
        panel,
        FakeRidge(ridge_rows),  # type: ignore[arg-type]
        xsmom,
        constants=CONSTANTS,
    )
    assert np.isnan(blend.values(205)).all()  # only six Ridge dates so far
    day = 259
    window = range(max(0, day - 364), day + 1)
    ridge_hist = np.vstack([ridge_rows[d] for d in window if d in ridge_rows])
    s_r = np.std(ridge_hist)
    s_x = np.std(xsmom[window.start : day + 1])
    expected = (ridge_rows[day] / s_r + xsmom[day] / s_x) / 2.0 * s_r
    assert blend.values(day) == pytest.approx(expected)
    weekly = sources.PairBlendSource(
        panel,
        FakeRidge(ridge_rows),  # type: ignore[arg-type]
        xsmom,
        constants=CONSTANTS,
        interval_scale=7.0,
    )
    assert weekly.values(day) == pytest.approx(7.0 * expected)
    # Too few finite T2 observations: no signal at all (mu = 0 via NaN).
    sparse = xsmom.copy()
    sparse[: day - 5] = np.nan
    short = sources.PairBlendSource(
        panel,
        FakeRidge(ridge_rows),  # type: ignore[arg-type]
        sparse,
        constants=CONSTANTS,
    )
    assert np.isnan(short.values(day)).all()
    with pytest.raises(ValueError):
        sources.PairBlendSource(
            panel,
            FakeRidge({}),
            xsmom[:5],
            constants=CONSTANTS,  # type: ignore[arg-type]
        )


@pytest.fixture(scope="module")
def perturbed(market: MarketDataset) -> MarketDataset:
    return runner.perturb_future(
        market, after_index=runner._row_of(market, "2021-10-20T00:00:00")
    )


@pytest.mark.parametrize(
    "arm",
    [
        "p3_1_t4_kappa3",
        "p3_3_t2_vt12",
        "p3_4_t6_kappa3_vt12",
        "p3_5_t4_weekly_vt12",
        "p3_6_zblend_t2t4_kappa3_vt12",
    ],
)
def test_served_signals_ignore_the_future(
    market: MarketDataset, perturbed: MarketDataset, arm: str
) -> None:
    first, second = served(arm, market), served(arm, perturbed)
    assert np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True)
    assert np.abs(np.asarray(first)).max() > 0.0


def test_weekly_arm_serves_seven_times_the_daily_t4_signal(
    market: MarketDataset,
) -> None:
    daily = np.asarray(served("p3_1_t4_kappa3", market))
    weekly = np.asarray(served("p3_5_t4_weekly_vt12", market))
    assert weekly == pytest.approx(7.0 * daily)


# --- wiring ----------------------------------------------------------------------


def test_arms_receive_exactly_their_registered_mechanisms(
    market: MarketDataset,
) -> None:
    protocol = small_protocol()
    context = sources.make_phase3_context(market, protocol)
    for spec in P3_TRIALS:
        built = sources.build_phase3_arm(spec.arm, market, protocol, context)
        if spec.arm == PASSIVE_COMPARATOR_ARM:
            assert isinstance(built.policy, sources.VolTargetedBuyAndHoldPolicy)
            continue
        policy = built.policy
        assert policy._cost_penalty_multiple == spec.kappa
        assert policy._vol_target == spec.vol_target
        assert policy._long_only is spec.long_only
        assert policy._interval_ns == spec.decision_interval_hours * HOUR_NS
        assert policy._periods_per_year == pytest.approx(
            8760.0 / spec.decision_interval_hours
        )
        # the optimizer cost is kappa x the unchanged base one-way cost
        assert np.allclose(policy._costs, COST)
    assert sources.build_phase3_arm(ARM_CASH, market, protocol, context).adapter is None
    with pytest.raises(ValueError):
        sources.phase3_spec("t1_tsmom")


# --- the vol-scaled passive comparator -----------------------------------------


class _Spy:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seen: list[tuple[int, tuple[float, ...]]] = []

    @property
    def records(self) -> Any:
        return self.inner.records

    def decide_portfolio(self, snapshot: Any) -> Any:
        self.seen.append((snapshot.timestamp_ns, snapshot.account.weights))
        return self.inner.decide_portfolio(snapshot)


def _replay_passive(market: MarketDataset, **protocol_overrides: object) -> Any:
    protocol = small_protocol(**protocol_overrides)
    context = sources.make_phase3_context(market, protocol)
    built = sources.build_phase3_arm(PASSIVE_COMPARATOR_ARM, market, protocol, context)
    spy = _Spy(built.policy)
    result = runner.run_shared_portfolio_replay(
        market,
        spy,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        initial_capital=protocol.initial_capital,
        execution_cost=ExecutionCostConfig(),
    )
    return protocol, built.policy, spy, result


def test_passive_comparator_first_decision_is_equal_weight_scaled_to_the_cap(
    market: MarketDataset,
) -> None:
    protocol, policy, _, result = _replay_passive(market, min_trade_fraction=0.0)
    first = policy.records[0]
    assert first.outcome == "rebalance"
    stamps = runner._timestamps_ns(market)
    start = runner._row_of(market, protocol.evaluation_start)
    # independent covariance: daily marks from the warmup start through the decision
    daily = np.flatnonzero(stamps % (24 * HOUR_NS) == 0)
    first_row = runner._row_of(market, protocol.covariance_warmup_start)
    rows = daily[(daily >= first_row) & (daily <= start)]
    marks = np.asarray(market.resolved_array("mark_price"))[rows]
    sigma = ewma_covariance(
        marks[1:] / marks[:-1] - 1.0,
        halflife_periods=protocol.ewma_halflife_days,
        shrinkage=protocol.shrinkage,
        variance_floor=1e-8,
    )
    unit = np.full(3, protocol.buy_and_hold_gross_budget / 3.0)
    vol = annualized_portfolio_vol(unit, sigma, periods_per_year=365.0)
    assert vol > VT_TARGET_ANNUAL, "the fixture must bind the cap"
    assert first.ex_ante_vol == pytest.approx(vol)
    assert first.vol_scale == pytest.approx(VT_TARGET_ANNUAL / vol)
    assert first.final_vol == pytest.approx(VT_TARGET_ANNUAL)
    proposal = next(d.proposal_weights for d in result.decisions if d.index == start)
    assert proposal == pytest.approx(tuple(unit * VT_TARGET_ANNUAL / vol))
    assert all(w >= 0.0 for w in proposal)


def test_passive_comparator_keeps_the_drifted_composition_and_never_shorts(
    market: MarketDataset,
) -> None:
    _, policy, spy, result = _replay_passive(market, min_trade_fraction=0.0)
    proposals = {d.index: d.proposal_weights for d in result.decisions}
    stamps = runner._timestamps_ns(market)
    later = [r for r in policy.records[1:] if r.outcome == "rebalance"]
    assert later
    seen = dict(spy.seen)
    checked = 0
    for record in later:
        current = np.asarray(seen[int(stamps[record.index])])
        target = np.asarray(proposals[record.index])
        assert np.all(target >= 0.0)
        if np.abs(current).sum() < 1e-12:
            continue
        # relative composition = the drifted composition, not equal weights
        assert target / target.sum() == pytest.approx(current / current.sum())
        # the whole gross is 0.99 * the volatility scale
        assert target.sum() == pytest.approx(0.99 * record.vol_scale)
        checked += 1
    assert checked > 5
    assert all(r.final_vol <= VT_TARGET_ANNUAL + 1e-9 for r in policy.records)


def test_passive_comparator_min_trade_dead_band_holds_small_adjustments(
    market: MarketDataset,
) -> None:
    _, policy, _, result = _replay_passive(market, min_trade_fraction=0.5)
    # With a 50% dead band the very first entry is also inside the band: no trade.
    assert result.book.fill_count == 0
    assert {r.outcome for r in policy.records} == {"hold_no_trade"}


def test_passive_comparator_fails_closed_on_bad_inputs(market: MarketDataset) -> None:
    protocol = small_protocol()
    stamps, marks = runner.warmup_marks(market, protocol)

    def make(**overrides: Any) -> sources.VolTargetedBuyAndHoldPolicy:
        kwargs: dict[str, Any] = {
            "symbols": SYMBOLS,
            "gross_budget": 0.99,
            "vol_target_annual": 0.12,
            "ewma_halflife_days": 15.0,
            "min_history_periods": 10,
            "warmup_timestamps_ns": stamps,
            "warmup_mark_prices": marks,
        }
        kwargs.update(overrides)
        return sources.VolTargetedBuyAndHoldPolicy(**kwargs)

    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            make(vol_target_annual=bad)
    with pytest.raises(ValueError):
        make(gross_budget=1.5)
    with pytest.raises(ValueError):
        make(warmup_mark_prices=marks[:-1])
    # insufficient history -> HOLD with a recorded reason, no orders
    held = make(min_history_periods=10_000)
    result = _run_policy(market, protocol, held)
    assert result.book.fill_count == 0
    assert {r.reason for r in held.records} == {"insufficient_history"}


def _run_policy(market: MarketDataset, protocol: Any, policy: Any) -> Any:
    return runner.run_shared_portfolio_replay(
        market,
        policy,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        initial_capital=protocol.initial_capital,
        execution_cost=ExecutionCostConfig(),
    )


def test_passive_comparator_non_finite_volatility_fails_closed(
    market: MarketDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object, **__: object) -> float:
        raise ValueError("nan")

    monkeypatch.setattr(sources, "annualized_portfolio_vol", broken)
    _, policy, _, result = _replay_passive(market)
    assert result.book.fill_count == 0
    assert {r.reason for r in policy.records} == {"invalid_volatility"}


# --- real-replay causality probes and schedule --------------------------------------


@pytest.mark.parametrize("arm", list(P3_ARMS))
def test_every_arm_passes_the_real_replay_causality_probe(
    market: MarketDataset, arm: str
) -> None:
    protocol = small_protocol()
    report = runner.causality_probe(
        market,
        protocol,
        sources.make_phase3_context(market, protocol),
        start_index=runner._row_of(market, protocol.evaluation_start),
        arm=arm,
        expect_trades=False,
        arm_builder=sources.build_phase3_arm,
        sources_builder=sources.make_phase3_context,
    )
    assert report["identical_pre_cut_trace"] is True and report["passed"] is True
    # explicit-split diagnostic fields
    assert (
        report["n_interval_rows"] + report["n_decision_rows"]
        == (report["n_trace_rows"])
    )
    assert report["n_interval_rows"] > 0 and report["n_decision_rows"] > 0
    assert isinstance(report["pre_cut_interval_rows_have_trades"], bool)


def test_p3_5_decides_only_on_mondays_at_midnight(market: MarketDataset) -> None:
    protocol = small_protocol()
    context = sources.make_phase3_context(market, protocol)
    built = sources.build_phase3_arm("p3_5_t4_weekly_vt12", market, protocol, context)
    _run_policy(market, protocol, built.policy)
    stamps = [record.timestamp_ns for record in built.policy.records]
    assert len(stamps) == 5
    for stamp in stamps:
        day = np.datetime64(stamp, "ns").astype("datetime64[D]")
        assert day.astype(int) % 7 == 4 and stamp % (24 * HOUR_NS) == 0
    assert any(r.ex_ante_vol is not None for r in built.policy.records)


def test_vol_target_diagnostics_reach_the_replay_summary(market: MarketDataset) -> None:
    protocol = small_protocol()
    context = sources.make_phase3_context(market, protocol)
    outcome = runner.run_replay(
        market,
        protocol,
        "p3_3_t2_vt12",
        "base",
        context,
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        arm_builder=sources.build_phase3_arm,
    )
    diag = outcome.summary["vol_target"]
    assert diag["n_decisions"] > 0
    assert 0.0 < diag["min_scale"] <= diag["mean_scale"] <= 1.0
    assert outcome.summary["reconciliation"]["passed"] is True


# --- full study ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def phase3(market: MarketDataset) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return study.run_phase3(market, small_protocol(), matrix_cut=MATRIX_CUT)


def test_every_arm_and_scenario_is_reported(
    phase3: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    results, equity = phase3
    assert tuple(results["arms"]) == ALL_ARMS
    for arm in ALL_ARMS:
        assert tuple(results["arms"][arm]) == SCENARIOS
        for scenario in SCENARIOS:
            row = results["arms"][arm][scenario]
            assert row["reconciliation"]["passed"] is True
            assert "terminal_residual_quantities" in row["reconciliation"]
            assert row["net_pnl_usdt"] == pytest.approx(
                row["gross_price_pnl_usdt"]
                - row["execution_costs_usdt"]
                + row["funding_pnl_usdt"]
            )
    assert results["label"] in {LABEL_POST_HOC_PROMISING, LABEL_NO_WINNER}
    assert results["label"] != "WINNER" and "WINNER" not in {
        results["label_amended_rule"],
        results["label_pr_c_probe_rule"],
    }
    assert results["label"] == results["label_pr_c_probe_rule"]
    assert results["registry_digest"] == registry_digest()
    assert (
        results["bonferroni_all_arms"] == 21 and results["bonferroni_family_arms"] == 8
    )
    assert set(results["selection_pr_c_probe_rule"]["arms"]) == set(CANDIDATE_POOL)
    assert PASSIVE_COMPARATOR_ARM not in results["selection_pr_c_probe_rule"]["arms"]
    assert results["selected_arm_pr_c_probe_rule"] != PASSIVE_COMPARATOR_ARM
    assert [t["role"] for t in results["trials"]] == ["candidate"] * 6 + [
        "control",
        "candidate",
    ]
    cash = results["arms"][ARM_CASH]["base"]
    assert cash["terminal_net_profit_pct"] == 0.0
    assert len(equity) == 2 * len(SCENARIOS) * len(ALL_ARMS)
    # the causality matrix runs for every arm but CASH in every scenario
    for arm in ALL_ARMS:
        if arm == ARM_CASH:
            continue
        cells = results["causality_matrix"][arm]
        assert set(cells) == set(SCENARIOS)
        assert all(cell["identical_pre_cut_trace"] for cell in cells.values())
        detail = results["validity_detail"][arm]
        assert detail["causality_matrix_identical_all_scenarios"] is True
        assert detail["causality_probe_identical"] is True
    # bootstrap: baselines and arms, three comparison columns, nested intervals
    table = results["bootstrap"]["base"]
    assert set(table) == set(ALL_ARMS) - {ARM_CASH}
    assert set(table[ARM_BUY_AND_HOLD]) == {"vs_cash"}
    assert set(table["p3_2_t4_kappa3_vt12"]) == {
        "vs_cash",
        "vs_equal_weight_buy_and_hold",
        "vs_p3_7_vol_scaled_passive",
    }
    assert set(table[PASSIVE_COMPARATOR_ARM]) == {
        "vs_cash",
        "vs_equal_weight_buy_and_hold",
    }
    interval = table["p3_2_t4_kappa3_vt12"]["vs_cash"]
    assert (
        interval["bonferroni_all21_ci_low_annualized"]
        <= interval["bonferroni_family8_ci_low_annualized"]
        <= interval["ci95_low_annualized"]
        <= interval["ci95_high_annualized"]
        <= interval["bonferroni_family8_ci_high_annualized"]
        <= interval["bonferroni_all21_ci_high_annualized"]
    )
    assert "vol_target" in results["arms"]["p3_2_t4_kappa3_vt12"]["base"]
    assert "vol_target" not in results["arms"]["p3_1_t4_kappa3"]["base"]
    assert "vol_target" in results["arms"][PASSIVE_COMPARATOR_ARM]["base"]


def test_validity_rules_differ_only_through_the_trade_requirement(
    phase3: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    results, _ = phase3
    pr_c, amended = (
        results["validity_pr_c_probe_rule"],
        results["validity_amended_rule"],
    )
    for arm in ALL_ARMS:
        detail = results["validity_detail"][arm]
        if pr_c[arm]:
            assert amended[arm]
        if amended[arm] and not pr_c[arm]:
            assert detail["causality_probe_traded_before_cut"] is False


def test_a_failed_arm_is_recorded_and_never_selected(
    market: MarketDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object, **__: object) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(sources, "xsmom_series", broken)
    protocol = small_protocol()
    outcome = study.run_arm(
        market,
        protocol,
        "p3_3_t2_vt12",
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        say=lambda message: None,
        matrix_cut=MATRIX_CUT,
    )
    # p3_3 uses the T2 source directly, so it still runs; p3_6 needs xsmom_series.
    assert set(outcome.outcomes) == set(SCENARIOS)
    broken_arm = study.run_arm(
        market,
        protocol,
        "p3_6_zblend_t2t4_kappa3_vt12",
        start_index=runner._row_of(market, protocol.evaluation_start),
        stop_index=runner._row_of(market, protocol.evaluation_end),
        say=lambda message: None,
        matrix_cut=MATRIX_CUT,
    )
    assert broken_arm.outcomes == {} and broken_arm.probe is None
    assert broken_arm.failures["base"]["error"] == "boom"
    assert broken_arm.failures["base"]["status"] == LABEL_FAILED_TO_RUN
    assert "causality_probe" in broken_arm.failures


def _fabricated(
    profits: dict[str, float], drawdown: float = 5.0
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        arm: {
            "base": {
                "terminal_net_profit_pct": profits.get(arm, 1.0),
                "max_drawdown_pct": drawdown,
            },
            "cost_x2": {"terminal_net_profit_pct": 1.0, "max_drawdown_pct": drawdown},
            "latency_1bar": {
                "terminal_net_profit_pct": 1.0,
                "max_drawdown_pct": drawdown,
            },
        }
        for arm in ALL_ARMS
    }


def test_selection_applies_the_pr_c_rule_with_the_post_hoc_vocabulary() -> None:
    protocol = study.phase3_protocol(small_protocol())
    valid = dict.fromkeys(CANDIDATE_POOL, True)
    # P3-7 is the best arm but a control: never selected.
    table = _fabricated({PASSIVE_COMPARATOR_ARM: 900.0, "p3_2_t4_kappa3_vt12": 40.0})
    chosen = study._select(table, valid, protocol, study.RULE_PR_C)
    assert chosen["label"] == LABEL_POST_HOC_PROMISING
    assert chosen["selected_arm"] == "p3_2_t4_kappa3_vt12"
    # Risk gate: a 20.5% drawdown in any scenario disqualifies.
    gated = _fabricated({"p3_2_t4_kappa3_vt12": 40.0})
    gated["p3_2_t4_kappa3_vt12"]["latency_1bar"]["max_drawdown_pct"] = 20.5
    result = study._select(gated, valid, protocol, study.RULE_PR_C)
    assert result["selected_arm"] != "p3_2_t4_kappa3_vt12"
    # Invalid arms and arms that do not beat CASH at cost x2 are ineligible.
    none = _fabricated({}, drawdown=30.0)
    nothing = study._select(none, valid, protocol, study.RULE_PR_C)
    assert nothing["label"] == LABEL_NO_WINNER and nothing["selected_arm"] is None
    invalid = study._select(
        _fabricated({"p3_1_t4_kappa3": 99.0}),
        {**valid, "p3_1_t4_kappa3": False},
        protocol,
        study.RULE_PR_C,
    )
    assert invalid["selected_arm"] != "p3_1_t4_kappa3"
    weak = _fabricated({"p3_1_t4_kappa3": 99.0})
    weak["p3_1_t4_kappa3"]["cost_x2"]["terminal_net_profit_pct"] = -0.1
    assert (
        study._select(weak, valid, protocol, study.RULE_PR_C)["selected_arm"]
        != "p3_1_t4_kappa3"
    )


def test_reproduction_check_detects_any_difference() -> None:
    baseline = {
        "terminal_net_profit_pct": 95.9,
        "max_drawdown_pct": 63.4,
        "net_pnl_usdt": 9590.0,
        "execution_costs_usdt": 30.0,
        "funding_pnl_usdt": -5.0,
        "turnover_total": 0.99,
        "n_trades": 5,
    }
    arms = {
        arm: {scenario: dict(baseline) for scenario in SCENARIOS}
        for arm in BASELINE_ARMS
    }
    ours = {"arms": arms}
    same = study.phase2_reproduction_check(ours, {"arms": json.loads(json.dumps(arms))})
    assert same["all_bit_identical"] is True
    changed = json.loads(json.dumps(arms))
    changed[ARM_BUY_AND_HOLD]["cost_x2"]["net_pnl_usdt"] += 1e-9
    different = study.phase2_reproduction_check(ours, {"arms": changed})
    assert different["all_bit_identical"] is False
    assert different["rows"][f"{ARM_BUY_AND_HOLD}/cost_x2"]["bit_identical"] is False


# --- CLI and workers ------------------------------------------------------------------


def test_cli_smoke_writes_once_counts_every_execution_and_refuses_overwrite(
    market: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = small_protocol()
    monkeypatch.setattr(phase3_cli, "PROTOCOL", protocol)
    monkeypatch.setattr(
        phase3_cli,
        "load_published_market_dataset_artifact",
        lambda root: (None, market),
    )
    out = tmp_path / "out"
    summary = tmp_path / "summary" / "result.json"
    ledger = tmp_path / "ledger.jsonl"
    argv = [
        "--dataset",
        str(tmp_path / "unused"),
        "--output-dir",
        str(out),
        "--summary-path",
        str(summary),
        "--execution-ledger",
        str(ledger),
        "--causality-matrix-cut",
        MATRIX_CUT,
        "--sequential",
    ]
    assert phase3_cli.main(argv) == 0
    written = json.loads(summary.read_text(encoding="utf-8"))
    assert written["label"] in {LABEL_POST_HOC_PROMISING, LABEL_NO_WINNER}
    assert written["registry_digest"] == registry_digest()
    assert set(written["arms"]) == set(ALL_ARMS)
    metadata = written["run_metadata"]
    assert metadata["started_utc"] <= metadata["finished_utc"]
    assert metadata["sequential"] is True
    assert json.loads((out / "run_metadata.json").read_text("utf-8")) == metadata
    row = written["arms"]["p3_2_t4_kappa3_vt12"]["base"]
    for name in (
        "terminal_net_profit_pct",
        "max_drawdown_pct",
        "turnover_total",
        "execution_costs_usdt",
        "funding_pnl_usdt",
        "fail_closed_hold_count",
        "terminal_residual_notional_usdt",
        "vol_target",
    ):
        assert name in row
    assert summary.stat().st_size < 500_000
    lines = [json.loads(x) for x in ledger.read_text("utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["started", "finished"]
    # An existing output is refused without being counted as an execution.
    assert phase3_cli.main(argv) == 2
    assert len(ledger.read_text("utf-8").splitlines()) == 2


def test_cli_records_a_failed_execution_in_the_ledger(
    market: MarketDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crash(*_: object, **__: object) -> Any:
        raise RuntimeError("interrupted")

    monkeypatch.setattr(phase3_cli, "run_phase3", crash)
    monkeypatch.setattr(
        phase3_cli,
        "load_published_market_dataset_artifact",
        lambda root: (None, market),
    )
    ledger = tmp_path / "ledger.jsonl"
    argv = [
        "--dataset",
        str(tmp_path / "unused"),
        "--output-dir",
        str(tmp_path / "out"),
        "--summary-path",
        str(tmp_path / "summary.json"),
        "--execution-ledger",
        str(ledger),
        "--sequential",
    ]
    with pytest.raises(RuntimeError, match="interrupted"):
        phase3_cli.main(argv)
    lines = [json.loads(x) for x in ledger.read_text("utf-8").splitlines()]
    assert [x["event"] for x in lines] == ["started", "failed"]
    assert lines[1]["error_type"] == "RuntimeError"
    assert not (tmp_path / "summary.json").exists()


def test_arm_worker_runs_in_a_separate_process(tmp_path: Path) -> None:
    from trade_rl.data.artifacts.publication import publish_market_dataset_artifact

    identified = synthetic_market().with_content_identity()
    root = tmp_path / "dataset"
    publish_market_dataset_artifact(root, identified)
    protocol = replace(small_protocol(), dataset_id=identified.dataset_id)
    start = runner._row_of(identified, protocol.evaluation_start)
    stop = runner._row_of(identified, protocol.evaluation_end)
    loader = partial(runner.load_dataset, str(root))
    arm = "p3_2_t4_kappa3_vt12"
    with ProcessPoolExecutor(max_workers=1) as pool:
        remote = pool.submit(
            study._arm_worker, loader, protocol, arm, start, stop, MATRIX_CUT
        ).result()
    local = study.run_arm(
        identified,
        protocol,
        arm,
        start_index=start,
        stop_index=stop,
        say=lambda message: None,
        matrix_cut=MATRIX_CUT,
    )
    assert remote.failures == {}
    assert remote.probe_matrix == local.probe_matrix and remote.probe == local.probe
    for scenario in SCENARIOS:
        assert np.array_equal(
            remote.outcomes[scenario].equity, local.outcomes[scenario].equity
        )


def test_the_roster_runner_rejects_a_different_roster(market: MarketDataset) -> None:
    with pytest.raises(ValueError, match="roster"):
        study.run_phase3(market, small_protocol(), arms=(ARM_CASH,))
    with pytest.raises(ValueError, match="dataset_id"):
        study.run_phase3(market, replace(small_protocol(), dataset_id="0" * 64))


def test_default_protocol_is_the_unchanged_pr_c_protocol() -> None:
    assert PROTOCOL.dataset_id.startswith("0b95547145a8")
    assert study.phase3_protocol(PROTOCOL).bonferroni_comparisons == 21
    assert study.phase3_protocol(PROTOCOL).candidate_arms == CANDIDATE_POOL


# --- committed evidence -----------------------------------------------------------


def _report(name: str) -> Any:
    return json.loads((REPO_ROOT / "report" / name).read_text("utf-8"))


def test_committed_result_and_provenance_are_consistent() -> None:
    result = _report("issue810-phase3-trial-results-20261029.json")
    provenance = _report("issue810-phase3-trial-provenance-20261029.json")
    assert result["schema_version"] == phase3_registry.RESULT_SCHEMA
    assert result["registry_digest"] == registry_digest()
    assert result["roster_digest"] == roster_digest()
    assert result["base_protocol_digest"] == registry_payload()["base_protocol_digest"]
    assert result["label"] in {LABEL_POST_HOC_PROMISING, LABEL_NO_WINNER}
    assert result["label"] == result["label_pr_c_probe_rule"]
    assert tuple(result["arms"]) == ALL_ARMS
    assert result["failures"] == {}
    for arm in ALL_ARMS:
        for scenario in SCENARIOS:
            row = result["arms"][arm][scenario]
            assert row["ledger_reconciled"] is True and row["terminal_flat"] is True
            assert row["terminal_residual_quantities"] == {}
            assert row["terminal_residual_notional_usdt"] == 0.0
    # every probe and matrix cell is recorded and identical
    for arm in ALL_ARMS:
        if arm == ARM_CASH:
            continue
        assert result["causality_probes"][arm]["identical_pre_cut_trace"] is True
        cells = result["causality_matrix"][arm]
        assert set(cells) == set(SCENARIOS)
        assert all(c["identical_pre_cut_trace"] for c in cells.values())
    # both labels are recomputed from the committed rows and validity
    protocol = study.phase3_protocol(PROTOCOL)
    for key, validity_key, rule in (
        ("selection_pr_c_probe_rule", "validity_pr_c_probe_rule", study.RULE_PR_C),
        ("selection_amended_rule", "validity_amended_rule", study.RULE_AMENDED),
    ):
        recomputed = study._select(result["arms"], result[validity_key], protocol, rule)
        assert recomputed == result[key]
    # the re-run baselines reproduce the committed Phase 2 numbers exactly
    assert result["phase2_reproduction_check"]["all_bit_identical"] is True
    # provenance: two executions, code head and summary digest recorded
    assert provenance["execution_count"] == 2
    ledger = [row["event"] for row in provenance["execution_ledger_verbatim"]]
    assert ledger == ["started", "failed", "started", "finished"]
    first, second = provenance["executions"]
    assert first["status"] == "incomplete_infrastructure_failure"
    assert second["status"] == "completed" and second["reported"] is True
    assert (
        second["result_summary_sha256"]
        == hashlib.sha256(
            (
                REPO_ROOT / "report" / "issue810-phase3-trial-results-20261029.json"
            ).read_bytes()
        ).hexdigest()
    )
    assert (
        result["run_metadata"]["git_head"]
        == provenance["code_head_for_every_execution"]
    )
    assert first["code_head"] == second["code_head"]
    assert provenance["classification"].startswith("POST_HOC_EXPLORATORY")
    overlap = provenance["execution_1_vs_execution_2_overlap_check"]
    complete_in_first = [a for a, v in overlap.items() if isinstance(v, dict)]
    assert complete_in_first == first["arms_with_results"]
    assert all(all(overlap[a].values()) for a in complete_in_first)


def test_committed_p3_7_is_a_control_and_never_the_selected_arm() -> None:
    result = _report("issue810-phase3-trial-results-20261029.json")
    assert result["selected_arm_pr_c_probe_rule"] != PASSIVE_COMPARATOR_ARM
    assert result["selected_arm_amended_rule"] != PASSIVE_COMPARATOR_ARM
    assert PASSIVE_COMPARATOR_ARM not in result["selection_pr_c_probe_rule"]["arms"]


def test_provenance_states_that_g0_g2_were_not_established() -> None:
    provenance = _report("issue810-phase3-trial-provenance-20261029.json")
    assurance = provenance["assurance"]
    assert [assurance[g] for g in ("G0", "G1", "G2")] == ["NOT_ESTABLISHED"] * 3
    text = " ".join(assurance["consequences"])
    assert "development diagnostics only" in text
    assert "NOT a G4 candidate" in text and "must not be frozen" in text
    scope = provenance["scope_choices_disclosed_after_review"]
    assert set(scope) == {
        "selection_pool",
        "causality_matrix_in_validity",
        "p3_7_excluded_from_selection",
        "pre_cut_trace_has_trades",
    }
    # the frozen registry and the result summary are untouched by the corrections
    assert (
        _report("issue810-phase3-trial-registry-20261029.json")["registry_digest"]
        == registry_digest()
    )
