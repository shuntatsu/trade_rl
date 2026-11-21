"""Synthetic tests of the spot cash-account amendment and the review follow-ups.

Review 5988926816 of PR 1293 found that lowturn v1 allowed a spot gross of 2.0 while the
engine finances negative cash at 0%. The amendment (made result-blind, before any BACKCAST
run) caps the spot gross at 1.0 for every arm and makes a borrowing cell INVALID. Nothing
here reads market data, the network or an economic result.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tests.evaluation import test_trend_oos as base
from tests.evaluation.test_lowturn_screen import _constrain, _signals, _snapshot
from tools import trend_oos as driver
from trade_rl.evaluation.lowturn_screen.policies import (
    LowturnPolicy,
    family_targets,
    risk_scale_of,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.trend_oos import data, protocol, runner
from trade_rl.evaluation.trend_oos.cash_policy import CashAccountPolicy
from trade_rl.evaluation.trend_oos.signals import build_oos_signals
from trade_rl.risk.pretrade import PreTradeRisk

N_LEGS = 22
SPOTS = np.arange(11) * 2
TOLERANCE = protocol.CASH_TOLERANCE_FRACTION * protocol.INITIAL_CAPITAL


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


# ------------------------------------------------------------------ the wrapper itself


def _trend_pack(rows: int = 400, vol: float = 0.10) -> Any:
    """Every spot leg trends up with vol60 at the floor: each target would be 0.1818
    after the lowturn caps, a gross of 2.0."""
    shape = (rows, N_LEGS)
    r = np.full(shape, np.nan)
    r[:, SPOTS] = 0.1
    v = np.full(shape, np.nan)
    v[:, SPOTS] = vol
    return _signals(rows, vol60=v, r30=r.copy(), r90=r.copy(), r180=r.copy())


def _gross(weights: Any) -> float:
    return float(np.abs(np.asarray(weights)[SPOTS]).sum())


def _wrapped(start: int = 0) -> CashAccountPolicy:
    pack = _trend_pack()
    return CashAccountPolicy(
        LowturnPolicy(family="TS_TREND_LONG_SPOT", signals=pack, start_index=start),
        start_index=start,
    )


def _executes_to(proposal: Any, current: np.ndarray, scale_dd: float) -> np.ndarray:
    """The weights risk and the exact-quantity holds leave of a proposal."""
    risk = _constrain(proposal, current, scale_dd)
    held = np.zeros(N_LEGS, dtype=bool)
    held[list(proposal.hold_current_quantity_symbols)] = True
    return np.where(held, current, risk)


def test_the_frozen_rule_wants_gross_two_and_the_cap_brings_it_to_exactly_one() -> None:
    pack = _trend_pack()
    uncapped = family_targets("TS_TREND_LONG_SPOT", pack, 0)
    assert _gross(uncapped) == pytest.approx(2.0)
    policy = _wrapped()
    proposal = policy.decide_portfolio(_snapshot(0))
    assert _gross(proposal.target_weights) == pytest.approx(1.0, abs=1e-12)
    assert proposal.hold_current_quantity_symbols == ()
    assert np.allclose(np.asarray(proposal.target_weights)[SPOTS], 1 / 11)
    executed = _executes_to(proposal, np.zeros(N_LEGS), 0.0)
    assert _gross(executed) <= 1.0 + 1e-12
    stats = policy.stats.to_dict()
    assert stats["rebalance_decisions"] == 1 and stats["cap_bound_decisions"] == 1
    assert stats["cap_bound_fraction"] == 1.0
    assert stats["mean_pre_cap_gross"] == pytest.approx(2.0)


def test_a_proposal_within_the_limit_is_untouched_and_counted_as_not_bound() -> None:
    pack = _trend_pack(vol=2.0)  # risk weight 0.25 -> targets 0.0227, gross 0.25
    policy = CashAccountPolicy(
        LowturnPolicy(family="TS_TREND_LONG_SPOT", signals=pack, start_index=0),
        start_index=0,
    )
    proposal = policy.decide_portfolio(_snapshot(0))
    assert _gross(proposal.target_weights) == pytest.approx(0.25)
    assert policy.stats.bound_decisions == 0
    assert policy.stats.to_dict()["cap_bound_fraction"] == 0.0


def test_the_deadband_cannot_let_gross_creep_above_the_limit() -> None:
    """Five legs sit within the deadband of the *uncapped* target (0.1818) and are
    held, six are far from it: the lowturn mixture has gross 1.99. The wrapper drops
    every hold and re-targets the whole vector to gross exactly 1.0."""
    current = np.zeros(N_LEGS)
    current[SPOTS[:5]] = 0.18
    snapshot = _snapshot(0, weights=current)
    inner = LowturnPolicy(
        family="TS_TREND_LONG_SPOT", signals=_trend_pack(), start_index=0
    )
    raw = inner.decide_portfolio(snapshot)
    held_spots = {i for i in raw.hold_current_quantity_symbols if i % 2 == 0}
    assert held_spots == {int(i) for i in SPOTS[:5]}
    creeping = _executes_to(raw, current, 0.0)
    assert _gross(creeping) == pytest.approx(5 * 0.18 + 6 * 0.18181818, abs=1e-6)
    assert _gross(creeping) > 1.9  # without the cap the book would creep to ~2.0

    policy = _wrapped()
    proposal = policy.decide_portfolio(snapshot)
    assert proposal.hold_current_quantity_symbols == ()
    executed = _executes_to(proposal, current, 0.0)
    assert _gross(executed) == pytest.approx(1.0, abs=1e-12)
    assert np.all(executed[SPOTS] > 0.0)  # proportional: every leg keeps a share


def test_a_pure_hold_and_an_unchanged_scale_pre_image_are_never_capped() -> None:
    """A trade-free proposal cannot change cash, so even a current gross a hair above
    1.0 (the book sits at minus its costs) is not churned."""
    current = np.zeros(N_LEGS)
    current[SPOTS] = 1.0006 / 11
    # non-rebalance bar at scale 1: the lowturn policy holds every leg
    policy = _wrapped()
    policy.decide_portfolio(_snapshot(0, weights=np.zeros(N_LEGS)))
    hold = policy.decide_portfolio(_snapshot(1, weights=current))
    assert len(hold.hold_current_quantity_symbols) == N_LEGS
    assert policy.stats.bound_decisions == 1  # only the first rebalance bound
    # unchanged scale 0.8: the pre-image current / 0.8 executes back to current
    policy = _wrapped()
    snap0 = _snapshot(0, weights=np.zeros(N_LEGS), value=88_000.0, peak=100_000.0)
    policy.decide_portfolio(snap0)
    snap1 = _snapshot(1, weights=current, value=88_000.0, peak=100_000.0)
    assert risk_scale_of(snap1) == pytest.approx(0.8)
    pre = policy.decide_portfolio(snap1)
    assert np.allclose(np.asarray(pre.target_weights) * 0.8, current)
    assert policy.stats.bound_decisions == 1


@pytest.mark.parametrize(("max_dd", "scale"), [(0.12, 0.8), (0.15, 0.5), (0.18, 0.2)])
def test_the_cap_holds_after_risk_scaling_too(max_dd: float, scale: float) -> None:
    current = np.zeros(N_LEGS)
    current[SPOTS] = 0.05
    snapshot = _snapshot(
        0,
        weights=current,
        value=100_000.0 * (1 - max_dd),
        peak=100_000.0,
        max_dd=max_dd,
    )
    assert risk_scale_of(snapshot) == pytest.approx(scale)
    policy = _wrapped()
    proposal = policy.decide_portfolio(snapshot)
    executed = _executes_to(proposal, current, max_dd)
    assert _gross(executed) <= 1.0 + 1e-9
    # the pre-cap effective gross was scale * 2.0: the cap binds only above 1.0
    binds = scale * 2.0 > 1.0
    assert (policy.stats.bound_decisions == 1) is binds
    if binds:
        assert _gross(executed) == pytest.approx(1.0, abs=1e-9)


def test_a_gross_limit_above_one_or_a_foreign_policy_is_refused() -> None:
    inner = LowturnPolicy(family="CASH", signals=_trend_pack(), start_index=0)
    for limit in (1.5, 0.0):
        with pytest.raises(ValueError, match="gross limit"):
            CashAccountPolicy(inner, start_index=0, gross_limit=limit)
    with pytest.raises(TypeError):
        CashAccountPolicy(object(), start_index=0)  # type: ignore[arg-type]


# ----------------------------------------------- the replay: free leverage and the fix


def _low_vol_window(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    window = base._small_window(monkeypatch, "BACKCAST")
    spot = {
        symbol: base._series(
            window, first_row=0, seed=i + 1, drift=0.0012, sigma=0.0004
        )
        for i, symbol in enumerate(protocol.ROSTER)
    }
    return window, base._assemble(window, spot=spot)


def _replay(dataset: Any, window: Any, policy: Any, scenario: str = "BASE") -> Any:
    return run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=window.first_decision_index,
        stop_index=window.last_row_index,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost(scenario),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def _gross_path(result: Any) -> list[float]:
    return [
        (iv.portfolio_value_after - iv.cash_after) / iv.portfolio_value_after
        for iv in result.ledger_evidence.intervals.intervals
    ]


def test_the_uncapped_rule_reproduces_zero_cost_financing_and_the_cap_removes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, dataset = _low_vol_window(monkeypatch)
    signals = build_oos_signals(dataset)
    start = window.first_decision_index
    wants = family_targets("TS_TREND_LONG_SPOT", signals, start)
    assert _gross(wants) == pytest.approx(2.0)  # the frozen rule wants gross above 1

    # lowturn v1 as it was: gross about 2.0, financed with negative cash at 0%
    free = _replay(
        dataset,
        window,
        LowturnPolicy(family="TS_TREND_LONG_SPOT", signals=signals, start_index=start),
    )
    cash_free, borrowing_free = runner._cash_extremes(free)
    assert cash_free is not None and cash_free < -0.5 * protocol.INITIAL_CAPITAL
    assert (
        borrowing_free is not None and borrowing_free > 0.5 * protocol.INITIAL_CAPITAL
    )
    assert max(_gross_path(free)) > 1.8
    assert free.diagnostics.total_cost > 0.0
    # nothing was charged for the loan: no borrow cost and no cash interest
    assert free.diagnostics.borrow_cost == 0.0

    # the spot cash account: gross never above 1, cash never negative beyond costs
    wrapped = CashAccountPolicy(
        LowturnPolicy(family="TS_TREND_LONG_SPOT", signals=signals, start_index=start),
        start_index=start,
    )
    capped = _replay(dataset, window, wrapped)
    cash, borrowing = runner._cash_extremes(capped)
    assert borrowing is not None and borrowing <= TOLERANCE
    assert cash is not None and cash > -0.01 * protocol.INITIAL_CAPITAL
    path = _gross_path(capped)
    assert max(path) <= 1.0 + 2e-3  # only the execution costs sit above 1.0
    assert max(path) > 0.9  # and the book really is invested
    stats = wrapped.stats.to_dict()
    assert stats["cap_bound_fraction"] > 0.9
    assert stats["mean_pre_cap_gross"] == pytest.approx(2.0, abs=0.05)
    # the unleveraged book earns less than the free-leverage book in an up-trend
    assert capped.book.portfolio_value < free.book.portfolio_value


def test_every_arm_cell_keeps_the_cash_account_and_the_cost_overdraft_is_not_borrowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, dataset = _low_vol_window(monkeypatch)
    result = runner.run_stage(
        dataset,
        window=window,
        output_dir=tmp_path / "out",
        expected_protocol_digest=protocol.protocol_digest(),
    )
    assert result["label"] != "INVALID", result["invalid_cells"]
    for key, cell in result["cells"].items():
        assert cell["max_borrowing"] <= TOLERANCE, key
        assert "negative_cash_borrowing" not in cell["invalid_reasons"], key
        assert cell["causality_probe"]["shared_across_cost_scenarios"] is True
        assert cell["causality_probe"]["key"].startswith(cell["arm"] + "__lag")
        assert cell["returns_series"] == "dust_adjusted"
    # EW_HOLD_SPOT is fully invested at 1.0: its cash is minus the costs it paid, which
    # is exactly what is not counted as borrowing (the literal cash would be invalid)
    ew = result["cells"]["EW_HOLD_SPOT__BASE"]
    assert ew["min_cash"] < -TOLERANCE
    assert ew["max_borrowing"] <= TOLERANCE
    candidate = result["cells"]["TS_TREND_LONG_SPOT__BASE"]
    assert candidate["cap_statistics"]["cap_bound_fraction"] > 0.9
    assert candidate["cap_statistics"]["mean_pre_cap_gross"] > 1.9
    assert (
        result["cap_statistics"]["TS_TREND_LONG_SPOT__BASE"]
        == (candidate["cap_statistics"])
    )
    # the probe is per (arm, lag): BASE and HIGH_COST share it, DELAY_1BAR has its own
    probes = json.loads((tmp_path / "out" / runner.MANIFEST_NAME).read_text("utf-8"))
    assert set(probes["causality_probes"]) == {
        f"{a}__lag{lag}" for a in protocol.ARMS for lag in (0, 1)
    }
    base_cell = result["cells"]["TS_TREND_LONG_SPOT__HIGH_COST"]["causality_probe"]
    assert base_cell["key"] == "TS_TREND_LONG_SPOT__lag0"


def test_a_cell_that_borrows_is_invalid_and_makes_the_label_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative cash injected: the uncapped lowturn policy borrows about as much as the
    capital, so the cell is INVALID with reason negative_cash_borrowing."""
    window, dataset = _low_vol_window(monkeypatch)
    signals = build_oos_signals(dataset)
    start = window.first_decision_index
    free = _replay(
        dataset,
        window,
        LowturnPolicy(family="TS_TREND_LONG_SPOT", signals=signals, start_index=start),
    )
    cell = runner._build_cell(
        arm="TS_TREND_LONG_SPOT",
        scenario="BASE",
        result=free,
        dataset=dataset,
        causality_passed=True,
        probe_key="TS_TREND_LONG_SPOT__lag0",
        cap_statistics={},
    )
    assert "negative_cash_borrowing" in cell.invalid_reasons
    assert not cell.valid and cell.max_borrowing is not None
    assert cell.max_borrowing > 0.5 * protocol.INITIAL_CAPITAL

    # the same through a whole stage: the wrapper bypassed means INVALID, not a label
    class Passthrough:
        def __init__(self, inner: Any, *, start_index: int) -> None:
            self._inner = inner
            self.stats = SimpleNamespace(to_dict=lambda: {})

        def decide_portfolio(self, snapshot: Any) -> Any:
            return self._inner.decide_portfolio(snapshot)

    monkeypatch.setattr(runner, "CashAccountPolicy", Passthrough)
    result = runner.run_stage(
        dataset,
        window=window,
        output_dir=tmp_path / "bypassed",
        expected_protocol_digest=protocol.protocol_digest(),
    )
    assert result["label"] == "INVALID"
    assert "TS_TREND_LONG_SPOT__BASE" in result["invalid_cells"]
    assert (
        "negative_cash_borrowing"
        in result["cells"]["TS_TREND_LONG_SPOT__BASE"]["invalid_reasons"]
    )
    assert "EW_HOLD_SPOT__BASE" not in result["invalid_cells"]  # gross 1.0 stays valid


def test_the_cap_holds_through_a_drawdown_with_risk_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = base._small_window(monkeypatch, "BACKCAST")
    crash_hour = (window.first_decision_index + 60) * 4
    spot = {
        symbol: base._series(
            window,
            first_row=0,
            seed=i + 1,
            drift=0.0012,
            sigma=0.0004,
            crash=(crash_hour, 0.85),
        )
        for i, symbol in enumerate(protocol.ROSTER)
    }
    dataset = base._assemble(window, spot=spot)
    signals = build_oos_signals(dataset)
    start = window.first_decision_index
    for arm in ("TS_TREND_LONG_SPOT", "EW_HOLD_SPOT"):
        wrapped = CashAccountPolicy(
            LowturnPolicy(family=arm, signals=signals, start_index=start),
            start_index=start,
        )
        result = _replay(dataset, window, wrapped)
        intervals = result.ledger_evidence.intervals.intervals
        assert max(iv.max_drawdown_after for iv in intervals) > 0.10, arm  # scale < 1
        _, borrowing = runner._cash_extremes(result)
        assert borrowing is not None and borrowing <= TOLERANCE, arm
        assert max(_gross_path(result)) <= 1.0 + 3e-3, arm


# ----------------------------------------------------------------- N3, N2, N6, N7


def test_the_return_series_for_reporting_carries_the_terminal_dust_cost() -> None:
    result = SimpleNamespace(book=SimpleNamespace(portfolio_value=1_000.0))
    returns = np.array([0.01, -0.02, 0.03])
    dust = [{"exit_cost_usdt": 10.0}]
    adjusted = runner._dust_adjusted(returns, result, dust)  # type: ignore[arg-type]
    assert adjusted[:2].tolist() == returns[:2].tolist()
    assert adjusted[2] == pytest.approx(1.03 * 0.99 - 1.0)
    # compounding the adjusted series reproduces the cell's total return
    start = 1_000.0 / np.prod(1.0 + returns)
    assert np.prod(1.0 + adjusted) - 1.0 == pytest.approx(
        (1_000.0 - 10.0) / start - 1.0
    )
    # no dust: the series is untouched
    assert runner._dust_adjusted(returns, result, []) is returns  # type: ignore[arg-type]


def test_the_protocol_records_the_coverage_commit_and_the_cash_account() -> None:
    assert protocol.COVERAGE_EVIDENCE_CODE_COMMIT == (
        "e9dc2eeda508cd7f329897ae08573e29bda2eb62"
    )
    assert protocol.COVERAGE_EVIDENCE_PROTOCOL_DIGEST == (
        "535a25162edd843687a845e920d178b0ea7af184eb8fd6968202e5222bc3a1bb"
    )
    payload = protocol.protocol_payload()
    halts = payload["data"]["declared_halts"]  # type: ignore[index]
    assert halts["coverage_code_commit"] == protocol.COVERAGE_EVIDENCE_CODE_COMMIT
    assert (
        halts["coverage_protocol_digest"] == protocol.COVERAGE_EVIDENCE_PROTOCOL_DIGEST
    )
    cash = payload["spot_cash_account"]
    assert cash["gross_limit"] == 1.0  # type: ignore[index]
    assert cash["invalid_on_borrowing"]["reason"] == "negative_cash_borrowing"  # type: ignore[index]
    assert cash["invalid_on_borrowing"]["tolerance_fraction"] == 1e-6  # type: ignore[index]


@pytest.mark.parametrize(
    ("name", "value"),
    [("SPOT_GROSS_LIMIT", 0.9), ("CASH_TOLERANCE_FRACTION", 1e-5)],
)
def test_the_cash_account_constants_are_in_the_digest(
    monkeypatch: pytest.MonkeyPatch, name: str, value: float
) -> None:
    before = protocol.protocol_digest()
    monkeypatch.setattr(protocol, name, value)
    assert protocol.protocol_digest() != before


def test_a_published_zero_volume_bar_stays_tradable_but_no_fill_can_follow_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N7, verified from the execution code: with processing_bar_volume_capacity off the
    capacity of an order is 1% of the *previous completed bar's* volume, so an order
    sent after a zero-volume bar has zero capacity and nothing fills."""
    window = base._small_window(monkeypatch, "BACKCAST")
    start = window.first_decision_index
    spot = {
        symbol: base._series(window, first_row=0, seed=i + 1)
        for i, symbol in enumerate(protocol.ROSTER)
    }
    btc = spot["BTCUSDT"]
    assert btc is not None
    volume = btc.volume.copy()
    volume[start * 4 : start * 4 + 4] = 0.0  # the four hours of the decision bar
    spot["BTCUSDT"] = type(btc)(
        timestamps=btc.timestamps,
        open=btc.open,
        high=btc.high,
        low=btc.low,
        close=btc.close,
        volume=volume,
        funding_rate=btc.funding_rate,
        funding_available=btc.funding_available,
        funding_event_count=btc.funding_event_count,
        tradable=btc.tradable,
    )
    dataset = base._assemble(window, spot=spot)
    assert np.asarray(dataset.tradable)[start, 0]  # published, so tradable
    assert np.asarray(dataset.volume)[start, 0] == 0.0
    signals = build_oos_signals(dataset)
    wrapped = CashAccountPolicy(
        LowturnPolicy(family="EW_HOLD_SPOT", signals=signals, start_index=start),
        start_index=start,
    )
    result = _replay(dataset, window, wrapped)
    first = result.ledger_evidence.intervals.intervals[0]
    assert first.exact_quantities_after[0] == "0"  # BTC: zero capacity, no fill
    assert first.exact_quantities_after[2] != "0"  # ETH: bought the same bar


# ------------------------------------------------------------------ N4 and N5


def test_the_bound_data_authorization_comment_must_not_change_between_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base._small_window(monkeypatch, "FORWARD")
    monkeypatch.setattr(
        driver.runner, "run_stage", lambda *a, **k: pytest.fail("must not run")
    )
    gh = base._Gh()
    gh.data.variants = {2: {"updated_at": "2026-11-21T00:00:00Z"}}
    rows = base._Rows({s: datetime(2018, 1, 1, tzinfo=UTC) for s in protocol.ROSTER})
    code, _, err = base._tool(
        tmp_path,
        base._forward_argv(tmp_path, "--interim"),
        gh=gh,
        now=datetime(2026, 8, 20, 6, 30, tzinfo=UTC),
        rows=rows,
    )
    assert code == driver.EXIT_UNAUTHENTICATED
    assert "data-authorization comment was edited" in err
    assert not (tmp_path / "fwd").exists()


class _ProbeRows(base._Rows):
    """Rows whose daily archives exist for chosen days (the probe of a leading month)."""

    def __init__(self, listed: dict[str, datetime | None], days: set[int]) -> None:
        super().__init__(listed)
        self.days = days
        self.probes: list[tuple[str, int]] = []

    def probe_published_day(self, symbol: str, day_start_ms: int) -> bool:
        self.probes.append((symbol, day_start_ms))
        return day_start_ms in self.days


def _load_listing(rows: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    window = base._small_window(monkeypatch, "BACKCAST")
    return driver.load_stage_dataset(
        rows, rows, window=window, limit_close=window.last_row_close, rules=base.RULES
    )


def test_a_leading_unpublished_month_with_a_daily_archive_refuses_the_listing_move(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = (
        base._listings()
    )  # SOL lists on 2018-02-03: January is leading, unpublished
    jan31 = base._ms(2018, 1, 31)
    quiet = _ProbeRows(listed, set())
    _, coverage = _load_listing(quiet, monkeypatch)
    assert coverage["SOLUSDT"]["unpublished_leading_months"] == 1
    assert ("SOLUSDT", jan31) in quiet.probes  # the last day of the month was probed
    # a daily archive exists for 2018-01-31: the listing would silently move to February
    loud = _ProbeRows(listed, {jan31})
    with pytest.raises(data.StageDataError, match="listing would silently move"):
        _load_listing(loud, monkeypatch)


def test_the_probe_of_the_real_month_source_reads_one_day_and_records_nothing() -> None:
    source, made = base._month_source(
        base.CarryCoverageError("404", {"kind": "archive_not_published"}),
        {"2018-02-10": "404", "2018-02-11": "malformed"},
    )
    day = base._ms(2018, 2, 12)
    assert source.probe_published_day("BTCUSDT", day) is True
    assert source.probe_published_day("BTCUSDT", base._ms(2018, 2, 10)) is False
    assert source.probe_published_day("BTCUSDT", base._ms(2018, 2, 11)) is False
    assert source.fills() == [] and source.unusable_days() == []
    assert made  # daily sources were used
    guard = data.StageRowSource(
        base._Rows({"BTCUSDT": datetime(2018, 1, 1, tzinfo=UTC)}),
        window=data.frozen_window("BACKCAST"),
        limit_close=datetime(2018, 3, 1, tzinfo=UTC),
    )
    assert guard.probe_published_day("BTCUSDT", day) is None  # a source that cannot say
    far = data.StageRowSource(
        source,
        window=data.frozen_window("BACKCAST"),
        limit_close=datetime(2018, 3, 1, tzinfo=UTC),
    )
    assert far.probe_published_day("BTCUSDT", day) is True
    with pytest.raises(data.StageEmbargoError, match="outside the stage read range"):
        far.probe_published_day("BTCUSDT", base._ms(2018, 3, 1))


def test_coverage_reports_leading_months_that_have_daily_archives_and_the_helper_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base._small_window(monkeypatch, "BACKCAST")
    rows = _ProbeRows(base._listings(), {base._ms(2018, 1, 31)})
    code, _, err = base._cov_tool(tmp_path, base._cov_argv(tmp_path), rows=rows)
    assert code == driver.EXIT_OK, err  # coverage reports, it never refuses
    path = tmp_path / "cov" / "coverage.json"
    document = json.loads(path.read_text("utf-8"))
    sol = document["symbols"]["SOLUSDT"]
    assert sol["leading_unpublished_months_with_daily_archives"] == [
        "2018-01-01T00:00:00+00:00"
    ]
    assert (
        document["symbols"]["BTCUSDT"]["leading_unpublished_months_with_daily_archives"]
        == []
    )
    out, err_io = io.StringIO(), io.StringIO()
    code = driver.main(
        ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err_io
    )
    assert code == driver.EXIT_REFUSED
    assert "listing would move" in err_io.getvalue()


def test_the_run_receipt_records_the_coverage_commit_and_the_cash_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base._small_window(monkeypatch, "BACKCAST")

    def fake(dataset: Any, **kwargs: Any) -> dict[str, Any]:
        (kwargs["output_dir"] / "decision.json").write_text("{}", encoding="utf-8")
        return {"label": "OOS_NOT_SUPPORTED", "status": None}

    monkeypatch.setattr(driver.runner, "run_stage", fake)
    code, _, err = base._tool(
        tmp_path, base._backcast_argv(tmp_path), gh=base._Gh(), now=base.NOW
    )
    assert code == driver.EXIT_OK, err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["declared_halts"]["coverage_code_commit"] == (
        "e9dc2eeda508cd7f329897ae08573e29bda2eb62"
    )
    assert receipt["declared_halts"]["coverage_protocol_digest"].startswith("535a2516")
    assert receipt["spot_cash_account"] == {
        "gross_limit": 1.0,
        "invalid_on": "negative_cash_borrowing",
    }
