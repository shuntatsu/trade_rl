"""Protocol, decision rule, runner and CLI of the multi-symbol carry screen.

All data is synthetic. The registered window is monkeypatched to a three-day span
only where a full run is exercised; the pure protocol and decision tests use the real
constants.
"""

import gzip
import io
import json
import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from tools import carry_multisymbol_screen as cli
from trade_rl.evaluation.carry_multisymbol import decision, protocol, runner
from trade_rl.evaluation.carry_multisymbol.decision import CellSummary
from trade_rl.integrations.market_store.carry_source import (
    CarryCoverageError,
    CarryScreenData,
)
from trade_rl.integrations.market_store.contracts import (
    FundingRow,
    KlineRow,
    StoreMarket,
)

HOUR = 3_600_000
G4_URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1234567890"
COMMIT = "a" * 40


# --- protocol -------------------------------------------------------------------------


def test_protocol_is_registered_before_data_and_digest_is_stable() -> None:
    first, second = protocol.protocol_digest(), protocol.protocol_digest()
    assert first == second and len(first) == 64
    document = protocol.protocol_document()
    assert document["candidates"] == [
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "AVAXUSDT",
        "LTCUSDT",
    ]
    assert json.loads(json.dumps(document)) == document


def test_every_registered_instant_precedes_the_fs1_warmup_span() -> None:
    from trade_rl.integrations.market_store.embargo import FS1_WARMUP_START

    assert protocol.WINDOW_END <= FS1_WARMUP_START
    assert protocol.LOOKBACK_START < protocol.WINDOW_START < protocol.WINDOW_END
    assert all(
        end <= protocol.WINDOW_END and start >= protocol.WINDOW_START
        for start, end in protocol.SPOT_HALTS
    )
    assert protocol.SUB_PERIODS == ("2022", "2023", "2024", "2025", "2026")


def test_protocol_constants_equal_the_assembled_cost_contract() -> None:
    import numpy as np

    from tests.integrations.test_binance_carry import series
    from trade_rl.integrations.binance.carry import assemble_multisymbol_carry_dataset
    from trade_rl.strategies.carry import CarryConfig

    dataset = assemble_multisymbol_carry_dataset(
        {"SOLUSDT": (series(perpetual=False), series(perpetual=True))},
        universe=protocol.CANDIDATES,
    )
    np.testing.assert_allclose(
        dataset.resolved_array("taker_fee_rate")[0],
        [protocol.SPOT_TAKER_FEE, protocol.PERP_TAKER_FEE],
    )
    assert float(dataset.resolved_array("spread_rate")[0, 0]) == pytest.approx(
        protocol.ADVERSE_EXECUTION
    )
    assert float(dataset.resolved_array("max_participation_rate")[0, 0]) == (
        pytest.approx(protocol.MAX_PARTICIPATION)
    )
    assert float(dataset.resolved_array("lot_size")[0, 0]) == protocol.COMMON_LOT
    assert float(dataset.resolved_array("minimum_notional")[0, 0]) == (
        protocol.MINIMUM_NOTIONAL
    )
    config = CarryConfig()
    assert config.gross_budget == protocol.GROSS_BUDGET
    assert config.common_lot == protocol.COMMON_LOT
    assert config.maximum_drawdown == protocol.MAXIMUM_DRAWDOWN


def test_family_threshold_follows_the_markov_bound_for_any_dependence() -> None:
    p0 = protocol.null_pass_probability()
    assert p0 == 2**-5
    table = {n: protocol.required_symbol_passes(n) for n in range(6, 10)}
    assert table == {6: 4, 7: 5, 8: 5, 9: 6}
    for admitted, required in table.items():
        assert admitted * p0 / required <= 1 / 20
        assert admitted * p0 / (required - 1) > 1 / 20  # and K is the smallest such
    with pytest.raises(ValueError):
        protocol.required_symbol_passes(0)


def test_window_and_authorization_refusals() -> None:
    protocol.validate_window(protocol.WINDOW_START, protocol.WINDOW_END)
    late = datetime(2026, 8, 2, tzinfo=UTC)
    with pytest.raises(ValueError, match="FS1 warm-up or sealed"):
        protocol.validate_window(protocol.WINDOW_START, late)
    with pytest.raises(ValueError, match="registered protocol"):
        protocol.validate_window(
            protocol.WINDOW_START, datetime(2026, 7, 1, tzinfo=UTC)
        )
    with pytest.raises(ValueError, match="registered protocol"):
        protocol.validate_window(datetime(2023, 1, 1, tzinfo=UTC), protocol.WINDOW_END)
    with pytest.raises(ValueError, match="timezone"):
        protocol.validate_window(datetime(2022, 1, 1), protocol.WINDOW_END)
    assert protocol.validate_g4_authorization_url(G4_URL) == G4_URL
    for bad in (
        None,
        "",
        "https://github.com/other/repo/issues/810#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-x",
        G4_URL + " ",
    ):
        with pytest.raises(ValueError, match="G4 authorization"):
            protocol.validate_g4_authorization_url(bad)


def test_lease_text_names_digest_commit_window_and_the_limits() -> None:
    text = protocol.lease_text(code_commit=COMMIT)
    for needle in (
        protocol.protocol_digest(),
        COMMIT,
        "2022-01-01T00:00:00+00:00",
        "2026-08-01T00:00:00+00:00",
        "no edge claim",
        "No orders, no keys",
    ):
        assert needle in text


# --- decision rule ---------------------------------------------------------------------


def summary(arm, treatment, *, total=0.02, years=None, drawdown=0.003, **overrides):
    values = dict(
        arm=arm,
        treatment=treatment,
        dataset_id="d" * 64,
        total_return=total,
        year_returns=years
        if years is not None
        else {period: 0.004 for period in protocol.SUB_PERIODS},
        ledger_max_drawdown=drawdown,
        complete=True,
        terminal_flat=True,
        execution_valid=True,
        stop_reason=None,
        invalid_reasons=(),
        total_cost=1.0,
        funding_pnl=2.0,
        average_realized_gross=0.5,
    )
    values.update(overrides)
    return CellSummary(**values)


def passing_arm(arm):
    return {(arm, item.name): summary(arm, item.name) for item in protocol.TREATMENTS}


def grid(admitted, failing=(), portfolio_ok=True):
    cells = {}
    for arm in (protocol.PORTFOLIO_ARM, *admitted):
        cells.update(passing_arm(arm))
    for arm, treatment, overrides in failing:
        cells[(arm, treatment)] = summary(arm, treatment, **overrides)
    if not portfolio_ok:
        cells[(protocol.PORTFOLIO_ARM, "base")] = summary(
            protocol.PORTFOLIO_ARM, "base", total=-0.01
        )
    return cells


ADMITTED = protocol.CANDIDATES


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"total": 0.0}, False),
        ({"total": -0.001}, False),
        ({"drawdown": 0.10}, False),
        ({"drawdown": 0.0999}, True),
        ({"complete": False}, False),
        ({"terminal_flat": False}, False),
        ({"execution_valid": False}, False),
        ({"stop_reason": "unmatched_hedge"}, False),
        ({"invalid_reasons": ("open_margin_breach",)}, False),
        ({"total": math.nan}, False),
        ({"drawdown": math.nan}, False),
        (
            {"years": {**dict.fromkeys(protocol.SUB_PERIODS, 0.01), "2024": -0.001}},
            False,
        ),
        ({"years": {**dict.fromkeys(protocol.SUB_PERIODS, 0.01), "2024": 0.0}}, False),
        ({"years": {"2022": 0.01}}, False),
        ({"years": {**dict.fromkeys(protocol.SUB_PERIODS, 0.01), "2027": 0.01}}, False),
    ],
)
def test_base_cell_requires_every_sub_period_and_clean_execution(overrides, expected):
    assert decision.cell_passes(summary("a", "base", **overrides)) is expected


def test_stress_cells_need_only_the_full_period_but_the_same_hard_guards() -> None:
    losing_year = {**dict.fromkeys(protocol.SUB_PERIODS, 0.01), "2023": -0.5}
    assert decision.cell_passes(summary("a", "cost2x", years=losing_year))
    assert not decision.cell_passes(summary("a", "cost2x", total=-0.001))
    assert not decision.cell_passes(summary("a", "delay1h", drawdown=0.2))
    assert not decision.cell_passes(summary("a", "capacity0p1", complete=False))


def test_arm_needs_all_four_cells_and_a_missing_cell_is_a_failure() -> None:
    cells = passing_arm("a")
    assert decision.arm_passes(cells, "a")
    for treatment in protocol.TREATMENTS:
        broken = dict(cells)
        broken[("a", treatment.name)] = summary("a", treatment.name, total=-1.0)
        assert not decision.arm_passes(broken, "a")
        del broken[("a", treatment.name)]
        assert not decision.arm_passes(broken, "a")


def test_family_generalizes_only_with_the_portfolio_and_at_least_k_symbols() -> None:
    result = decision.decide(ADMITTED, {}, grid(ADMITTED))
    assert result["outcome"] == decision.GENERALIZES
    assert result["required_symbol_passes"] == 6 and result["symbol_passes"] == 9
    assert result["decision_cell_count"] == 40
    assert result["edge_claim"] is False
    assert result["production_eligible"] is False
    assert result["live_trading_authorized"] is False
    assert result["markov_false_positive_bound"] <= 1 / 20

    five_fail = [(s, "cost2x", {"total": -0.01}) for s in ADMITTED[:3]]
    exactly_six = decision.decide(ADMITTED, {}, grid(ADMITTED, failing=five_fail))
    assert exactly_six["symbol_passes"] == 6
    assert exactly_six["outcome"] == decision.GENERALIZES
    seven_fail = [(s, "base", {"total": -0.01}) for s in ADMITTED[:4]]
    short = decision.decide(ADMITTED, {}, grid(ADMITTED, failing=seven_fail))
    assert short["symbol_passes"] == 5
    assert short["outcome"] == decision.DOES_NOT_GENERALIZE
    assert short["failed_cells"] == sorted(f"{s}__base" for s in ADMITTED[:4])

    portfolio_down = decision.decide(ADMITTED, {}, grid(ADMITTED, portfolio_ok=False))
    assert portfolio_down["symbol_passes"] == 9
    assert portfolio_down["outcome"] == decision.DOES_NOT_GENERALIZE
    assert portfolio_down["portfolio_pass"] is False


def test_excluded_symbols_are_reported_and_never_rescue_or_hide_a_failure() -> None:
    admitted = ADMITTED[:7]
    excluded = {s: "window_coverage" for s in ADMITTED[7:]}
    result = decision.decide(admitted, excluded, grid(admitted))
    assert result["required_symbol_passes"] == 5
    assert result["excluded"] == excluded
    assert result["outcome"] == decision.GENERALIZES


def test_too_few_admitted_symbols_is_inconclusive_not_a_pass_or_a_fail() -> None:
    result = decision.decide(ADMITTED[:5], {s: "x" for s in ADMITTED[5:]}, {})
    assert result["outcome"] == decision.INCONCLUSIVE
    assert "portfolio_pass" not in result


def test_reference_arms_are_descriptive_only() -> None:
    cells = grid(ADMITTED)
    cells.update(passing_arm(protocol.REFERENCE_SHARED_ARM))
    cells[(protocol.REFERENCE_SHARED_ARM, "base")] = summary(
        protocol.REFERENCE_SHARED_ARM, "base", total=-0.5
    )
    result = decision.decide(
        ADMITTED, {}, cells, reference_arms=[protocol.REFERENCE_SHARED_ARM]
    )
    assert result["outcome"] == decision.GENERALIZES
    assert result["reference_arm_pass"] == {protocol.REFERENCE_SHARED_ARM: False}
    assert protocol.REFERENCE_SHARED_ARM + "__base" not in result["failed_cells"]


def test_binomial_diagnostic_is_a_plain_independent_tail() -> None:
    assert decision.binomial_tail(3, 0, 0.5) == pytest.approx(1.0)
    assert decision.binomial_tail(3, 2, 0.5) == pytest.approx(0.5)
    assert decision.binomial_tail(3, 3, 0.5) == pytest.approx(0.125)


def test_cell_summary_round_trips_and_derives_the_pass_flag() -> None:
    cell = summary("a", "base")
    again = CellSummary.from_dict(json.loads(json.dumps(cell.to_dict())))
    assert again == cell and cell.to_dict()["cell_pass"] is True


# --- synthetic row source and a full tiny run -------------------------------------------


class FakeSource:
    """Flat-basis spot/perpetual rows for every symbol, with injectable defects."""

    def __init__(self, *, funding=None, low_volume=(), holes=None, refuse=()):
        self.funding = funding or {}
        self.low_volume = set(low_volume)
        self.holes = holes or {}
        self.refuse = set(refuse)
        self.calls = 0

    def read_klines(self, market, symbol, interval, start_ms, end_ms):
        self.calls += 1
        if (market, symbol) in self.refuse:
            raise CarryCoverageError(f"{market.value} {symbol} is not published")
        quote = 10_000.0 if symbol in self.low_volume else 1_000_000.0
        skipped = self.holes.get((market, symbol), set())
        base = 50.0 + sum(map(ord, symbol)) % 100
        rows = []
        for ms in range(start_ms, end_ms, HOUR):
            if ms in skipped:
                continue
            price = Decimal(str(base))
            rows.append(
                KlineRow(
                    open_time_ms=ms,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=Decimal("10"),
                    close_time_ms=ms + HOUR - 1,
                    quote_volume=Decimal(str(quote)),
                    trades=5,
                    taker_buy_base_volume=Decimal("5"),
                    taker_buy_quote_volume=Decimal("500"),
                )
            )
        return rows

    def read_funding(self, market, symbol, start_ms, end_ms):
        rate = Decimal(str(self.funding.get(symbol, 0.004)))
        first = start_ms - start_ms % (8 * HOUR) + 8 * HOUR
        return [
            FundingRow(funding_time_ms=ms, rate=rate)
            for ms in range(first, end_ms + 1, 8 * HOUR)
        ]


@pytest.fixture
def tiny(monkeypatch):
    """Three-day registered window, three candidates, one sub-period."""
    monkeypatch.setattr(protocol, "LOOKBACK_START", datetime(2022, 1, 1, tzinfo=UTC))
    monkeypatch.setattr(protocol, "WINDOW_START", datetime(2022, 1, 3, tzinfo=UTC))
    monkeypatch.setattr(protocol, "WINDOW_END", datetime(2022, 1, 6, tzinfo=UTC))
    monkeypatch.setattr(protocol, "CANDIDATES", ("SOLUSDT", "BNBUSDT", "XRPUSDT"))
    monkeypatch.setattr(protocol, "SPOT_HALTS", ())
    monkeypatch.setattr(protocol, "SUB_PERIODS", ("2022",))
    monkeypatch.setattr(protocol, "MIN_ADMITTED", 2)
    monkeypatch.setattr(protocol, "required_symbol_passes", lambda admitted: 2)
    return protocol


def screen_data(source):
    return CarryScreenData(
        source,
        lookback_start=protocol.LOOKBACK_START,
        window_start=protocol.WINDOW_START,
        window_end=protocol.WINDOW_END,
        halt_intervals=protocol.SPOT_HALTS,
    )


def request(tmp_path, **overrides):
    values = dict(
        start=protocol.WINDOW_START,
        end=protocol.WINDOW_END,
        g4_authorization_url=G4_URL,
        output_dir=tmp_path / "screen",
        code_commit=COMMIT,
    )
    values.update(overrides)
    return runner.ScreenRequest(**values)


def test_full_run_writes_every_cell_and_a_decision(tiny, tmp_path) -> None:
    source = FakeSource()
    result = runner.run_screen(request(tmp_path), screen_data(source))
    root = tmp_path / "screen"

    assert result["outcome"] == decision.GENERALIZES
    assert result["admitted"] == ["SOLUSDT", "BNBUSDT", "XRPUSDT"]
    arms = ["portfolio", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    arms += ["ref_shared", "ref_BTCUSDT", "ref_ETHUSDT"]
    expected = {f"{arm}__{t.name}" for arm in arms for t in protocol.TREATMENTS}
    assert set(result["cells"]) == expected
    assert {p.name.split(".")[0] for p in (root / "cells").glob("*.summary.json")} == (
        expected
    )
    assert len(list((root / "cells").glob("*.json.gz"))) == len(expected)
    for name in ("protocol.json", "authorization.json", "inclusion.json"):
        assert (root / name).is_file()
    assert (root / "decision.json").is_file()
    saved = json.loads((root / "protocol.json").read_text("utf-8"))
    assert saved["protocol_digest"] == protocol.protocol_digest()
    assert saved["code_commit"] == COMMIT
    auth = json.loads((root / "authorization.json").read_text("utf-8"))
    assert auth["g4_authorization_url"] == G4_URL and auth["verified"] is False

    portfolio = json.loads(
        (root / "cells" / "portfolio__base.summary.json").read_text("utf-8")
    )
    assert portfolio["capital"] == 10_000.0 and portfolio["decision_arm"] is True
    solo = json.loads(
        (root / "cells" / "SOLUSDT__base.summary.json").read_text("utf-8")
    )
    assert solo["capital"] == pytest.approx(10_000.0 / 3)
    shared = json.loads(
        (root / "cells" / "ref_shared__base.summary.json").read_text("utf-8")
    )
    assert shared["decision_arm"] is False and shared["capital"] == 10_000.0
    single = json.loads(
        (root / "cells" / "ref_BTCUSDT__base.summary.json").read_text("utf-8")
    )
    assert single["capital"] == 5_000.0
    detail = json.loads(
        gzip.decompress((root / "cells" / "portfolio__base.json.gz").read_bytes())
    )
    assert detail["schema"] == "funding_carry_replay_v1" and detail["terminal_flat"]
    for name in ("portfolio__base", "portfolio__cost2x"):
        assert result["cells"][name]["cell_pass"] is True
    assert result["cells"]["portfolio__cost2x"]["total_cost"] == pytest.approx(
        2 * result["cells"]["portfolio__base"]["total_cost"]
    )


def test_failing_cells_are_written_and_decide_against_generalization(
    tiny, tmp_path
) -> None:
    source = FakeSource(funding={"SOLUSDT": -0.004, "BNBUSDT": -0.004})
    result = runner.run_screen(request(tmp_path), screen_data(source))
    assert result["outcome"] == decision.DOES_NOT_GENERALIZE
    assert result["symbol_pass"] == {
        "SOLUSDT": False,
        "BNBUSDT": False,
        "XRPUSDT": True,
    }
    root = tmp_path / "screen" / "cells"
    for name in ("SOLUSDT__base", "BNBUSDT__cost2x"):
        assert (root / f"{name}.json.gz").is_file()
        assert (root / f"{name}.summary.json").is_file()
    assert result["cells"]["SOLUSDT__base"]["total_return"] < 0
    assert "SOLUSDT__base" in result["failed_cells"]


def test_inclusion_rule_uses_only_coverage_and_volume(tiny, tmp_path) -> None:
    gap_ms = int(datetime(2022, 1, 4, 5, tzinfo=UTC).timestamp() * 1000)
    source = FakeSource(
        low_volume={"BNBUSDT"},
        holes={(StoreMarket.USDS_M, "XRPUSDT"): {gap_ms}},
    )
    result = runner.run_screen(request(tmp_path), screen_data(source))
    assert result["admitted"] == ["SOLUSDT"]
    assert result["excluded"] == {
        "BNBUSDT": "liquidity_below_threshold",
        "XRPUSDT": "window_coverage",
    }
    assert result["outcome"] == decision.INCONCLUSIVE
    assert (
        not list((tmp_path / "screen" / "cells").glob("*"))
        if (tmp_path / "screen" / "cells").exists()
        else True
    )
    inclusion = json.loads((tmp_path / "screen" / "inclusion.json").read_text("utf-8"))
    assert inclusion["symbols"]["BNBUSDT"]["reason_code"] == "liquidity_below_threshold"
    assert inclusion["symbols"]["SOLUSDT"]["admitted"] is True


def test_missing_lookback_history_excludes_a_young_listing(tiny, tmp_path) -> None:
    source = FakeSource(refuse={(StoreMarket.SPOT, "XRPUSDT")})
    result = runner.run_screen(
        request(tmp_path, coverage_only=True), screen_data(source)
    )
    assert result["coverage_only"] is True
    assert result["excluded"] == {"XRPUSDT": "lookback_incomplete"}
    assert not (tmp_path / "screen" / "decision.json").exists()
    assert not (tmp_path / "screen" / "cells").exists()


def test_run_is_resumable_and_does_not_recompute_finished_cells(
    tiny, tmp_path, monkeypatch
) -> None:
    first = runner.run_screen(request(tmp_path), screen_data(FakeSource()))

    def forbidden(*args, **kwargs):
        raise AssertionError("a finished cell must not be replayed again")

    monkeypatch.setattr(runner, "replay_carry", forbidden)
    again = runner.run_screen(request(tmp_path), screen_data(FakeSource()))
    assert again["cells"] == first["cells"] and again["outcome"] == first["outcome"]


def test_resume_refuses_other_data_other_authorization_or_foreign_roots(
    tiny, tmp_path
) -> None:
    runner.run_screen(request(tmp_path), screen_data(FakeSource()))
    with pytest.raises(FileExistsError, match="other data"):
        runner.run_screen(
            request(tmp_path), screen_data(FakeSource(funding={"SOLUSDT": 0.003}))
        )
    other = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-99"
    with pytest.raises(FileExistsError, match="different comment"):
        runner.run_screen(
            request(tmp_path, g4_authorization_url=other), screen_data(FakeSource())
        )
    with pytest.raises(FileExistsError, match="different protocol or commit"):
        runner.run_screen(
            request(tmp_path, code_commit="b" * 40), screen_data(FakeSource())
        )
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "note.txt").write_text("x")
    with pytest.raises(FileExistsError, match="not an empty or resumable"):
        runner.run_screen(
            request(tmp_path, output_dir=foreign), screen_data(FakeSource())
        )


def test_run_refuses_before_reading_anything_without_authorization_or_in_late_window(
    tiny, tmp_path
) -> None:
    source = FakeSource()
    for bad in (
        request(tmp_path, g4_authorization_url="not a url"),
        request(tmp_path, end=datetime(2022, 1, 7, tzinfo=UTC)),
        request(tmp_path, start=datetime(2022, 1, 2, tzinfo=UTC)),
        request(tmp_path, code_commit="HEAD"),
    ):
        with pytest.raises(ValueError):
            runner.run_screen(bad, screen_data(source))
    assert source.calls == 0 and not (tmp_path / "screen").exists()


def test_the_real_registered_window_cannot_be_moved_past_july_31(tmp_path) -> None:
    with pytest.raises(ValueError, match="FS1 warm-up or sealed"):
        runner.run_screen(
            request(tmp_path, end=datetime(2026, 8, 2, tzinfo=UTC)),
            screen_data(FakeSource()),
        )
    plan = cli.build_plan(datetime(2026, 11, 10, tzinfo=UTC))
    assert max(spec.period_end for spec in plan.specs) <= protocol.WINDOW_END
    symbols = {spec.symbol for spec in plan.specs}
    assert symbols == {*protocol.CANDIDATES, *protocol.REFERENCE}
    funding = [s for s in plan.specs if s.kind.value == "funding_rate"]
    assert funding and min(s.period_start for s in funding) >= protocol.WINDOW_START
    klines = [s for s in plan.specs if s.kind.value == "klines"]
    assert min(s.period_start for s in klines) == protocol.LOOKBACK_START


# --- command line ----------------------------------------------------------------------


def run_cli(argv, *, tmp_path, clean=True, factory=None):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(
        argv,
        stdout=out,
        stderr=err,
        state=lambda: (COMMIT, clean),
        rows_factory=factory,
    )
    return code, out.getvalue(), err.getvalue()


def screen_args(tmp_path, *extra, url=G4_URL, end="2026-08-01"):
    args = [
        "screen",
        "--start",
        "2022-01-01",
        "--end",
        end,
        "--output-dir",
        str(tmp_path / "out"),
        "--archive-dir",
        str(tmp_path / "archives"),
    ]
    if url is not None:
        args += ["--g4-authorization-url", url]
    return [*args, *extra]


def test_cli_plan_is_offline_and_prints_digest_and_lease(tmp_path) -> None:
    code, out, _ = run_cli(["plan", "--code-commit", COMMIT], tmp_path=tmp_path)
    document = json.loads(out)
    assert code == 0
    assert document["protocol_digest"] == protocol.protocol_digest()
    assert document["g4_authorization_text"] == protocol.lease_text(code_commit=COMMIT)
    assert document["required_symbol_passes"] == {"6": 4, "7": 5, "8": 5, "9": 6}
    assert document["archives_planned"]["klines"] > 1000


def test_cli_refuses_without_a_g4_authorization_url(tmp_path) -> None:
    called = []
    code, out, err = run_cli(
        screen_args(tmp_path, url=None),
        tmp_path=tmp_path,
        factory=lambda plan: called.append(plan),
    )
    assert code == 2 and out == "" and "G4 authorization" in err
    assert called == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("end", ["2026-08-02", "2027-01-01", "2026-12-31"])
def test_cli_refuses_a_window_after_2026_07_31(tmp_path, end) -> None:
    code, out, err = run_cli(
        screen_args(tmp_path, end=end),
        tmp_path=tmp_path,
        factory=lambda plan: pytest.fail("no data may be opened"),
    )
    assert code == 2 and "FS1 warm-up or sealed" in err and out == ""


def test_cli_refuses_another_window_a_bad_url_a_dirty_tree_and_no_archive_dir(
    tmp_path,
) -> None:
    def factory(plan):
        pytest.fail("no data may be opened")

    code, _, err = run_cli(
        screen_args(tmp_path, end="2026-07-31"), tmp_path=tmp_path, factory=factory
    )
    assert code == 2 and "registered protocol" in err
    code, _, err = run_cli(
        screen_args(tmp_path, url="https://example.com/x"),
        tmp_path=tmp_path,
        factory=factory,
    )
    assert code == 2 and "G4 authorization" in err
    code, _, err = run_cli(
        screen_args(tmp_path), tmp_path=tmp_path, clean=False, factory=factory
    )
    assert code == 2 and "clean" in err
    args = screen_args(tmp_path)
    del args[args.index("--archive-dir") : args.index("--archive-dir") + 2]
    code, _, err = run_cli(args, tmp_path=tmp_path)
    assert code == 2 and "--archive-dir" in err


def test_cli_screen_runs_with_an_injected_source(tiny, tmp_path) -> None:
    args = [
        "screen",
        "--start",
        "2022-01-03",
        "--end",
        "2022-01-06",
        "--g4-authorization-url",
        G4_URL,
        "--output-dir",
        str(tmp_path / "out"),
        "--source",
        "store",
    ]
    code, out, err = run_cli(args, tmp_path=tmp_path, factory=lambda p: FakeSource())
    assert code == 0, err
    assert json.loads(out)["outcome"] == decision.GENERALIZES
    assert (tmp_path / "out" / "decision.json").is_file()


def test_cli_modules_load_network_and_database_code_lazily() -> None:
    import ast
    from pathlib import Path

    heavy = {
        "trade_rl.integrations.market_store.fetch",
        "trade_rl.integrations.market_store.store",
        "psycopg",
        "urllib.request",
        "socket",
    }
    root = Path(runner.__file__).parent
    for path in (Path(cli.__file__), *sorted(root.glob("*.py"))):
        tree = ast.parse(path.read_text("utf-8"))
        top = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not top & heavy, path.name
    for name in ("protocol.py", "decision.py", "runner.py"):
        text = (root / name).read_text("utf-8")
        assert "trade_rl.integrations.market" not in text
