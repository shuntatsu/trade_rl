"""Synthetic fake-source tests for the Core7 MTF Tier D loader and its driver.

Nothing here reads real market data, the network, a database or an economic result.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.integrations.market_store.support import kline_csv_rows
from tools import core7_mtf_tierd as driver
from trade_rl.data.features.mtf6_signals import (
    build_mtf6_signal_pack,
    enforce_pre_fs1_boundary,
    validate_alternating_spot_perp_dataset,
)
from trade_rl.data.source import RawMarketSeries
from trade_rl.evaluation.core7_mtf_study import protocol, run_causality_probe, tierd
from trade_rl.integrations.binance.carry import CarryCoverageError
from trade_rl.integrations.market_store.contracts import (
    FundingRow,
    KlineRow,
    StoreMarket,
)

HOUR_MS = 3_600_000
START = datetime(2021, 12, 29, tzinfo=UTC)
END = datetime(2022, 1, 4, tzinfo=UTC)  # 144 hours, 36 four-hour bars
HALT = (
    datetime(2022, 1, 1, 1, 30, tzinfo=UTC),
    datetime(2022, 1, 1, 6, 0, tzinfo=UTC),
)
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _kline(row: Sequence[str]) -> KlineRow:
    return KlineRow(
        open_time_ms=int(row[0]),
        open=Decimal(row[1]),
        high=Decimal(row[2]),
        low=Decimal(row[3]),
        close=Decimal(row[4]),
        volume=Decimal(row[5]),
        close_time_ms=int(row[6]),
        quote_volume=Decimal(row[7]),
        trades=int(row[8]),
        taker_buy_base_volume=Decimal(row[9]),
        taker_buy_quote_volume=Decimal(row[10]),
    )


class FakeRows:
    """Deterministic in-memory row source; counts the reads it serves."""

    def __init__(
        self,
        *,
        symbols: Sequence[str] = protocol.ROSTER,
        spot_missing: frozenset[int] = frozenset(),
        perp_missing: frozenset[int] = frozenset(),
        funding: bool = True,
        bump: int = 0,
        ignore_range: bool = False,
        extra_funding_ms: int | None = None,
        extra_kline_ms: int | None = None,
    ) -> None:
        self.symbols = tuple(symbols)
        self.spot_missing = spot_missing
        self.perp_missing = perp_missing
        self.funding = funding
        self.bump = bump
        self.ignore_range = ignore_range
        self.extra_funding_ms = extra_funding_ms
        self.extra_kline_ms = extra_kline_ms
        self.calls = 0

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        self.calls += 1
        if symbol not in self.symbols:
            return []
        lo, hi = (start_ms, end_ms)
        if self.ignore_range:
            hi = _ms(datetime(2026, 8, 1, 4, tzinfo=UTC))
        skip = self.spot_missing if market is StoreMarket.SPOT else self.perp_missing
        rows = kline_csv_rows(symbol, lo, hi, skip=skip, close_bump=self.bump)
        parsed = [_kline(row) for row in rows]
        if self.extra_kline_ms is not None:
            late = kline_csv_rows(
                symbol, self.extra_kline_ms, self.extra_kline_ms + HOUR_MS
            )
            parsed.append(_kline(late[0]))
        return parsed

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        self.calls += 1
        if not self.funding or symbol not in self.symbols:
            return []
        first = start_ms - (start_ms % (8 * HOUR_MS)) + 8 * HOUR_MS
        rows = [
            FundingRow(funding_time_ms=stamp, rate=Decimal("0.0001"))
            for stamp in range(first, end_ms + 1, 8 * HOUR_MS)
        ]
        if self.extra_funding_ms is not None:
            rows.append(FundingRow(self.extra_funding_ms, Decimal("0.0001")))
        return rows


def _rules() -> dict[str, tuple[tierd.LegRules, tierd.LegRules]]:
    return {
        symbol: (
            tierd.LegRules(0.00001 * (index + 1), 5.0),
            tierd.LegRules(0.001 * (index + 1), 10.0 + index),
        )
        for index, symbol in enumerate(protocol.ROSTER)
    }


def _load(source: FakeRows, **kwargs: Any) -> Any:
    kwargs.setdefault("halt_intervals", ())
    return driver.load_tierd_dataset(
        source, rules=_rules(), load_start=START, load_end=END, **kwargs
    )


def _payload(market: str) -> bytes:
    symbols: list[dict[str, object]] = []
    for index, symbol in enumerate(protocol.ROSTER):
        if market == "spot":
            symbols.append(
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": f"0.0{index + 1}000000"},
                        {"filterType": "NOTIONAL", "minNotional": "5.0"},
                    ],
                }
            )
        else:
            symbols.append(
                {
                    "symbol": symbol,
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "100"},
                    ],
                }
            )
    return json.dumps({"symbols": symbols}).encode("utf-8")


# --- shape, roster order, determinism, content hash --------------------------------


def test_dataset_is_the_22_leg_4h_roster_in_protocol_order() -> None:
    dataset = _load(FakeRows())

    assert dataset.symbols == protocol.LEG_ROSTER
    assert dataset.symbols[0] == "BTCUSDT:spot"
    assert dataset.symbols[1] == "BTCUSDT:perp"
    assert list(protocol.ROSTER) != sorted(protocol.ROSTER)  # order is not alphabetical
    assert dataset.n_bars == 36 and dataset.n_symbols == 22
    assert dataset.periods_per_year == 2190
    # timestamps are bar close times on the 4h grid
    first = np.datetime64("2021-12-29T04:00:00", "ns")
    assert dataset.timestamps[0] == first
    assert dataset.timestamps[-1] == np.datetime64("2022-01-04T00:00:00", "ns")
    assert np.all(np.diff(dataset.timestamps) == np.timedelta64(4, "h"))
    assert validate_alternating_spot_perp_dataset(dataset) == protocol.ROSTER
    enforce_pre_fs1_boundary(dataset)


def test_hourly_rows_aggregate_to_4h_bars_exactly() -> None:
    source = FakeRows()
    dataset = _load(source)
    hourly = [
        _kline(row)
        for row in kline_csv_rows("BTCUSDT", _ms(START), _ms(START) + 4 * HOUR_MS)
    ]

    assert dataset.open[0, 0] == float(hourly[0].open)
    assert dataset.close[0, 0] == float(hourly[-1].close)
    assert dataset.high[0, 0] == max(float(row.high) for row in hourly)
    assert dataset.low[0, 0] == min(float(row.low) for row in hourly)
    assert dataset.volume[0, 0] == pytest.approx(
        sum(float(row.quote_volume) for row in hourly)
    )
    assert dataset.resolved_array("mark_price")[0, 0] == dataset.close[0, 0]


def test_content_hash_is_deterministic_and_sensitive_to_content() -> None:
    first = _load(FakeRows())
    again = _load(FakeRows())
    changed = _load(FakeRows(bump=1))

    assert first.dataset_id == again.dataset_id
    assert len(first.dataset_id) == 64
    assert first.dataset_id != changed.dataset_id
    np.testing.assert_array_equal(first.close, again.close)


def test_scenario_costs_are_not_double_counted_and_rules_are_per_leg() -> None:
    dataset = _load(FakeRows())

    for name in ("fee_rate", "maker_fee_rate", "taker_fee_rate", "spread_rate"):
        assert not np.any(dataset.resolved_array(name)), name
    lot = dataset.resolved_array("lot_size")
    minimum = dataset.resolved_array("minimum_notional")
    assert np.all(lot == lot[0])  # static snapshot: constant over time
    for index, symbol in enumerate(protocol.ROSTER):
        spot_rules, perp_rules = _rules()[symbol]
        assert lot[0, 2 * index] == spot_rules.lot_size
        assert lot[0, 2 * index + 1] == perp_rules.lot_size
        assert minimum[0, 2 * index] == spot_rules.min_notional
        assert minimum[0, 2 * index + 1] == perp_rules.min_notional
    funding_due = dataset.resolved_array("funding_due")
    assert not funding_due[:, 0::2].any()  # spot carries no funding
    assert funding_due[:, 1::2].any()


def test_exchange_rule_payloads_map_to_roster_legs_and_refuse_a_missing_symbol() -> (
    None
):
    rules = driver.load_rules(_payload("spot"), _payload("usds_m"))
    assert tuple(rules) == protocol.ROSTER
    assert rules["BTCUSDT"][0].lot_size == 0.01
    assert rules["BTCUSDT"][1].min_notional == 100.0

    payload = json.loads(_payload("spot"))
    payload["symbols"] = payload["symbols"][:-1]
    with pytest.raises(Exception, match="DOTUSDT"):
        driver.load_rules(json.dumps(payload).encode(), _payload("usds_m"))
    bad = json.loads(_payload("spot"))
    bad["symbols"][0]["filters"] = []
    with pytest.raises(ValueError, match="finite and positive"):
        driver.load_rules(json.dumps(bad).encode(), _payload("usds_m"))


# --- SPOT_HALTS stay unavailable ----------------------------------------------------


def _halt_skips() -> frozenset[int]:
    # hours opening 02:00..05:00 lie wholly inside the halt: the source has no row
    base = _ms(datetime(2022, 1, 1, tzinfo=UTC))
    return frozenset(base + hour * HOUR_MS for hour in (2, 3, 4, 5))


def test_spot_halts_are_unavailable_rows_not_filled_bars() -> None:
    dataset = _load(FakeRows(spot_missing=_halt_skips()), halt_intervals=(HALT,))

    closes = dataset.timestamps
    touched = (closes == np.datetime64("2022-01-01T04:00:00", "ns")) | (
        closes == np.datetime64("2022-01-01T08:00:00", "ns")
    )
    assert touched.sum() == 2
    tradable = np.asarray(dataset.tradable)
    assert not tradable[touched][:, 0::2].any()  # every spot leg is unavailable
    assert tradable[~touched].all()  # nothing else is masked
    assert tradable[:, 1::2].all()  # perpetuals have no halt calendar
    unmasked = _load(FakeRows())
    # halted hours carry no volume, so the touched bars trade less than unmasked ones
    assert np.all(dataset.volume[touched][:, 0::2] < unmasked.volume[touched][:, 0::2])
    np.testing.assert_array_equal(dataset.volume[~touched], unmasked.volume[~touched])
    assert np.isfinite(dataset.close).all() and (dataset.close > 0).all()

    pack = build_mtf6_signal_pack(dataset)
    assert not pack.spot_usable[touched].any()
    assert not pack.carry_mask[touched].any()
    assert pack.perp_mask[touched].all()
    assert pack.spot_usable[~touched].all()


def test_an_unmasked_halt_is_refused_by_the_assembler() -> None:
    # the source reports the full hours although the declared halt covers them
    with pytest.raises(CarryCoverageError):
        _load(FakeRows(spot_missing=_halt_skips()), halt_intervals=())

    source = FakeRows()
    dataset = _load(source, halt_intervals=())
    assert np.asarray(dataset.tradable).all()
    with pytest.raises(tierd.TierDDataError, match="tradable bar inside the declared"):
        # a halt declared after the fact, over rows the loader treated as tradable
        driver.tierd.assemble_tierd_dataset(
            _raw_pairs(), rules=_rules(), spot_halt_intervals=(HALT,)
        )


# --- fail closed on missing symbols / rows ------------------------------------------


def _raw_pairs(
    *,
    start: datetime = START,
    hours: int = 144,
    symbols: Sequence[str] = protocol.ROSTER,
) -> dict[str, tuple[RawMarketSeries, RawMarketSeries]]:
    close = (np.datetime64(start.replace(tzinfo=None), "ns")) + (
        np.arange(1, hours + 1) * np.timedelta64(1, "h")
    )
    price = 100.0 + np.arange(hours, dtype=np.float64)
    zeros = np.zeros(hours)
    counts = np.zeros(hours, dtype=np.int32)
    counts[7::8] = 1
    funding = np.where(counts > 0, 0.0001, 0.0)

    def series(perp: bool) -> RawMarketSeries:
        return RawMarketSeries(
            timestamps=close,
            available_at=close,
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=np.full(hours, 1000.0),
            funding_rate=funding if perp else zeros,
            funding_available=(counts > 0) if perp else np.zeros(hours, dtype=bool),
            funding_event_count=counts if perp else np.zeros(hours, dtype=np.int32),
            tradable=np.ones(hours, dtype=bool),
        )

    return {symbol: (series(False), series(True)) for symbol in symbols}


def test_missing_extra_or_reordered_symbols_are_refused() -> None:
    pairs = _raw_pairs()
    assert tierd.assemble_tierd_dataset(pairs, rules=_rules()).symbols == (
        protocol.LEG_ROSTER
    )
    missing = {k: v for k, v in pairs.items() if k != "DOTUSDT"}
    with pytest.raises(tierd.TierDDataError, match="missing=.*DOTUSDT"):
        tierd.assemble_tierd_dataset(missing, rules=_rules())
    extra = {**pairs, "BCHUSDT": pairs["BTCUSDT"]}
    with pytest.raises(tierd.TierDDataError, match="extra=.*BCHUSDT"):
        tierd.assemble_tierd_dataset(extra, rules=_rules())
    # insertion order of the mapping never reorders the legs
    reordered = dict(reversed(list(pairs.items())))
    assert tierd.assemble_tierd_dataset(reordered, rules=_rules()).symbols == (
        protocol.LEG_ROSTER
    )
    thin = {k: v for k, v in _rules().items() if k != "LTCUSDT"}
    with pytest.raises(tierd.TierDDataError, match="rules must cover"):
        tierd.assemble_tierd_dataset(pairs, rules=thin)


def test_missing_rows_and_funding_halt_the_load_instead_of_excluding_a_symbol() -> None:
    hole = frozenset({_ms(START) + 40 * HOUR_MS})
    with pytest.raises(CarryCoverageError, match="usds_m.*missing hourly"):
        _load(FakeRows(perp_missing=hole))
    with pytest.raises(CarryCoverageError, match="spot.*missing hourly"):
        _load(FakeRows(spot_missing=hole))
    with pytest.raises(CarryCoverageError, match="no funding"):
        _load(FakeRows(funding=False))
    with pytest.raises(CarryCoverageError, match="missing hourly"):
        _load(FakeRows(symbols=protocol.ROSTER[:-1]))


def test_a_misaligned_or_ragged_hourly_window_is_refused() -> None:
    pairs = _raw_pairs(start=START + timedelta(hours=1))
    with pytest.raises(tierd.TierDDataError, match="4h grid"):
        tierd.assemble_tierd_dataset(pairs, rules=_rules())
    with pytest.raises(tierd.TierDDataError, match="whole 4h"):
        tierd.assemble_tierd_dataset(_raw_pairs(hours=142), rules=_rules())


# --- the embargo is refused at load time ---------------------------------------------


def test_last_admissible_bar_closes_before_the_embargo_instant() -> None:
    assert tierd.DATA_END_EXCLUSIVE == datetime(2026, 8, 1, tzinfo=UTC)
    assert tierd.LOAD_END == datetime(2026, 7, 31, 20, tzinfo=UTC)
    assert tierd.DATA_END_EXCLUSIVE_MS == 1_785_542_400_000
    tierd.validate_load_window(START)  # defaults to the last admissible close
    with pytest.raises(tierd.TierDEmbargoError, match="refused"):
        tierd.validate_load_window(START, tierd.DATA_END_EXCLUSIVE)
    with pytest.raises(tierd.TierDEmbargoError):
        tierd.validate_load_window(START, tierd.LOAD_END + timedelta(hours=4))
    with pytest.raises(ValueError, match="load_start"):
        tierd.validate_load_window(datetime(2020, 12, 31, 20, tzinfo=UTC))
    with pytest.raises(ValueError, match="load_start"):
        tierd.validate_load_window(datetime(2022, 1, 1, 4, tzinfo=UTC))
    with pytest.raises(ValueError, match="grid"):
        tierd.validate_load_window(START + timedelta(hours=1))


def test_the_guard_refuses_the_request_before_reading_anything() -> None:
    inner = FakeRows()
    guard = tierd.GuardedRowSource(inner)
    for end in (tierd.DATA_END_EXCLUSIVE_MS, tierd.DATA_END_EXCLUSIVE_MS + HOUR_MS):
        with pytest.raises(tierd.TierDEmbargoError):
            guard.read_klines(StoreMarket.SPOT, "BTCUSDT", "1h", _ms(START), end)
        with pytest.raises(tierd.TierDEmbargoError):
            guard.read_funding(StoreMarket.USDS_M, "BTCUSDT", _ms(START), end)
    with pytest.raises(tierd.TierDEmbargoError):
        # even a request that ends before the embargo but after the last bar close
        guard.read_klines(
            StoreMarket.SPOT,
            "BTCUSDT",
            "1h",
            _ms(START),
            tierd.LOAD_END_MS + HOUR_MS,
        )
    with pytest.raises(ValueError, match="only 1h"):
        guard.read_klines(StoreMarket.SPOT, "BTCUSDT", "4h", _ms(START), _ms(END))
    assert inner.calls == 0
    assert guard.rows_read == 0


def test_the_guard_refuses_an_embargoed_row_even_from_a_permissive_source() -> None:
    # the source serves a row opening exactly at the embargo instant
    guard = tierd.GuardedRowSource(FakeRows(extra_kline_ms=tierd.DATA_END_EXCLUSIVE_MS))
    with pytest.raises(tierd.TierDEmbargoError, match="at or after the embargo"):
        guard.read_klines(StoreMarket.SPOT, "BTCUSDT", "1h", _ms(START), _ms(END))
    # a source that ignores the requested range is refused as well
    with pytest.raises(tierd.TierDEmbargoError, match="outside the requested"):
        tierd.GuardedRowSource(FakeRows(ignore_range=True)).read_klines(
            StoreMarket.SPOT, "BTCUSDT", "1h", _ms(START), _ms(END)
        )
    # ... and a funding row at the embargo instant
    leaky = tierd.GuardedRowSource(
        FakeRows(extra_funding_ms=tierd.DATA_END_EXCLUSIVE_MS)
    )
    with pytest.raises(tierd.TierDEmbargoError, match="embargo"):
        leaky.read_funding(StoreMarket.USDS_M, "BTCUSDT", _ms(START), _ms(END))
    # the full loader therefore halts rather than loading the embargoed span
    with pytest.raises(tierd.TierDEmbargoError):
        _load(FakeRows(ignore_range=True))


def test_the_assembler_refuses_a_bar_reaching_the_embargo_instant() -> None:
    late = datetime(2026, 7, 29, tzinfo=UTC)  # 72 hours: last close 2026-08-01T00:00
    with pytest.raises(tierd.TierDEmbargoError, match="embargo"):
        tierd.assemble_tierd_dataset(_raw_pairs(start=late, hours=72), rules=_rules())
    ok = tierd.assemble_tierd_dataset(
        _raw_pairs(start=late - timedelta(hours=4), hours=72), rules=_rules()
    )
    enforce_pre_fs1_boundary(ok)


# --- replay window and summary (no returns) -------------------------------------------


def test_replay_window_starts_at_the_first_tier_d_bar_and_summary_has_no_returns() -> (
    None
):
    dataset = _load(FakeRows())
    start, stop = tierd.replay_indices(dataset)
    assert dataset.timestamps[start] == np.datetime64("2022-01-01T04:00:00", "ns")
    assert stop == dataset.n_bars - 1

    summary = tierd.dataset_summary(dataset)
    assert summary["shape"] == [36, 22]
    assert summary["dataset_id"] == dataset.dataset_id
    per_leg = summary["per_leg"]
    assert isinstance(per_leg, dict) and len(per_leg) == 22
    assert per_leg["BTCUSDT:spot"] == {
        "bars": 36,
        "tradable": 36,
        "unavailable": 0,
        "funding_events": 0,
    }
    assert per_leg["BTCUSDT:perp"]["funding_events"] > 0
    assert "return" not in json.dumps(summary).lower()

    short = tierd.assemble_tierd_dataset(
        _raw_pairs(start=START, hours=48), rules=_rules()
    )
    with pytest.raises(tierd.TierDDataError, match="replay window"):
        tierd.replay_indices(short)


def test_the_identity_bound_dataset_passes_the_causality_probe_unchanged() -> None:
    dataset = _load(FakeRows())
    assert dataset.identity_payload_json is not None
    assert run_causality_probe(
        dataset, cut_index=26, policy_mode="anc_bidir_75_25", start_index=18
    ).passed


# --- the CLI: dry check and the authorization-gated run -------------------------------


def _cli_files(tmp_path: Path) -> list[str]:
    spot, perp = tmp_path / "spot.json", tmp_path / "perp.json"
    spot.write_bytes(_payload("spot"))
    perp.write_bytes(_payload("usds_m"))
    return [
        "--load-start",
        "2021-12-29",
        "--exchange-info-spot",
        str(spot),
        "--exchange-info-perp",
        str(perp),
        "--source",
        "store",
    ]


def _main(
    argv: list[str], *, clean: bool = True, source: FakeRows | None = None
) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rows = FakeRows() if source is None else source
    code = driver.main(
        argv,
        stdout=out,
        stderr=err,
        state=lambda: ("a" * 40, clean),
        rows_factory=lambda plan: rows,
        load_end=END,
    )
    return code, out.getvalue(), err.getvalue()


def test_dry_check_loads_only_and_never_calls_run_study(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-check must not call run_study")

    monkeypatch.setattr(driver, "run_study", forbidden)
    code, out, err = _main(["dry-check", *_cli_files(tmp_path)], clean=False)

    assert code == driver.EXIT_OK, err
    document = json.loads(out)
    assert document["summary"]["shape"] == [36, 22]
    assert document["summary"]["dataset_id"] == document["receipt"]["dataset_id"]
    assert document["receipt"]["working_tree_clean"] is False
    assert len(document["receipt"]["exchange_info_sha256"]["spot"]) == 64
    assert not (tmp_path / "out").exists()
    # identical data => identical hash across invocations
    _, again, _ = _main(["dry-check", *_cli_files(tmp_path)])
    assert (
        json.loads(again)["summary"]["dataset_id"] == document["summary"]["dataset_id"]
    )


def test_run_requires_an_authorization_url_and_refuses_a_malformed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        driver,
        "run_study",
        lambda *a, **k: pytest.fail("must not run without authorization"),
    )
    base = ["run", *_cli_files(tmp_path), "--output-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        driver.main(base)  # no --authorization-url: argparse refuses, no default
    for bad in (
        "",
        "https://example.com/x",
        "https://github.com/a/b/issues/1#issuecomment-1",
    ):
        code, out, err = _main([*base, "--authorization-url", bad])
        assert code == driver.EXIT_REFUSED and not out
        assert "authorization_url" in err
    assert not (tmp_path / "out").exists()


def test_run_needs_a_clean_tree_and_a_new_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        driver, "run_study", lambda *a, **k: pytest.fail("must not run")
    )
    out_dir = tmp_path / "out"
    argv = [
        "run",
        *_cli_files(tmp_path),
        "--output-dir",
        str(out_dir),
        "--authorization-url",
        AUTH,
    ]
    code, _, err = _main(argv, clean=False)
    assert code == driver.EXIT_REFUSED and "clean" in err
    assert not out_dir.exists()
    out_dir.mkdir()
    code, _, err = _main(argv)
    assert code == driver.EXIT_REFUSED and "write-once" in err


def test_run_calls_run_study_once_with_the_frozen_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def spy(dataset: Any, **kwargs: Any) -> dict[str, object]:
        calls.append({"dataset": dataset, **kwargs})
        return {"outcome": "PASS_PARITY_ONLY", "evidence_eligible": True}

    monkeypatch.setattr(driver, "run_study", spy)
    out_dir = tmp_path / "out"
    code, out, err = _main(
        [
            "run",
            *_cli_files(tmp_path),
            "--output-dir",
            str(out_dir),
            "--authorization-url",
            AUTH,
        ]
    )

    assert code == driver.EXIT_OK, err
    assert len(calls) == 1
    call = calls[0]
    assert call["authorization_url"] == AUTH
    assert call["allow_synthetic_unauthenticated"] is False
    assert call["require_full_roster"] is True
    assert call["enforce_pre_fs1"] is True
    assert call["start_index"] == call["fit_stop_index"] == 18
    assert call["output_dir"] == out_dir
    assert call["dataset"].symbols == protocol.LEG_ROSTER
    # the real, identity-verified dataset flows through unchanged
    assert call["dataset"].identity_payload_json is not None
    _, dry, _ = _main(["dry-check", *_cli_files(tmp_path)])
    assert call["dataset"].dataset_id == json.loads(dry)["summary"]["dataset_id"]
    assert call["expected_protocol_digest"] == protocol.protocol_digest()
    receipt = json.loads((out_dir / driver.RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["dataset_id"] == call["dataset"].dataset_id
    assert receipt["expected_protocol_digest"] == protocol.protocol_digest()
    assert receipt["protocol_digest"] == protocol.protocol_digest()
    assert receipt["code_commit"] == "a" * 40


def test_the_loaded_dataset_runs_through_the_real_study_on_synthetic_prices(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "study"
    code, out, err = _main(
        [
            "run",
            *_cli_files(tmp_path),
            "--output-dir",
            str(out_dir),
            "--authorization-url",
            AUTH,
        ]
    )
    assert code == driver.EXIT_OK, err
    decision = json.loads(out)
    assert decision["evidence_eligible"] is True
    run_receipt = json.loads(
        (out_dir / driver.RUN_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert run_receipt["evidence_eligible"] is True
    assert (
        run_receipt["dataset_id"]
        == json.loads((out_dir / driver.RECEIPT_NAME).read_text(encoding="utf-8"))[
            "dataset_id"
        ]
    )
    assert decision["roster_complete"] is True
    assert decision["synthetic_only"] is False
    assert {path.name for path in out_dir.iterdir()} == {
        driver.RECEIPT_NAME,
        driver.RUN_RECEIPT_NAME,
        "decision.json",
        "study-manifest.json",
    }
    # outputs are write-once: a second run into the same directory is refused
    code, _, err = _main(
        [
            "run",
            *_cli_files(tmp_path),
            "--output-dir",
            str(out_dir),
            "--authorization-url",
            AUTH,
        ]
    )
    assert code == driver.EXIT_REFUSED


def test_coverage_reports_gaps_per_leg_without_building_a_dataset(
    tmp_path: Path,
) -> None:
    argv = ["coverage", "--load-start", "2021-12-29", "--source", "store"]
    code, out, err = _main(argv)
    assert code == driver.EXIT_OK, err
    report = json.loads(out)
    assert report["complete"] is True and len(report["legs"]) == 22
    assert report["legs"]["BTCUSDT:perp"]["funding_events"] > 0

    hole = frozenset({_ms(START) + 40 * HOUR_MS, _ms(START) + 41 * HOUR_MS})
    code, out, _ = _main(argv, source=FakeRows(perp_missing=hole))
    assert code == driver.EXIT_REFUSED  # a gap is never filled or excluded
    report = json.loads(out)
    assert report["complete"] is False
    leg = report["legs"]["BTCUSDT:perp"]
    assert leg["unexplained_missing_hours"] == 2
    assert leg["unexplained_gaps"] == [
        {"first_missing_open": "2021-12-30T16:00:00+00:00", "hours": 2}
    ]
    assert report["legs"]["BTCUSDT:spot"]["unexplained_missing_hours"] == 0

    # a declared halt explains only the hours it wholly contains
    spot_halt = FakeRows(spot_missing=_halt_skips())
    gaps = driver.coverage_report(
        spot_halt, load_start=START, load_end=END, halt_intervals=(HALT,)
    )
    assert gaps["complete"] is True
    assert gaps["legs"]["ETHUSDT:spot"]["explained_by_halt_hours"] == 4  # type: ignore[index]
    with pytest.raises(tierd.TierDEmbargoError):
        driver.coverage_report(
            FakeRows(), load_start=START, load_end=tierd.DATA_END_EXCLUSIVE
        )


# --- spot legs are not borrowable -------------------------------------------------------


def test_spot_legs_are_not_borrowable_and_the_engine_enforces_it() -> None:
    from trade_rl.simulation.execution import MarketExecutor

    dataset = _load(FakeRows())
    borrow = dataset.resolved_array("borrow_available")
    assert not borrow[:, 0::2].any()  # every spot leg
    assert borrow[:, 1::2].all()  # perpetuals short natively

    executor = MarketExecutor(dataset, protocol.scenario_cost("BASE"))
    desired = np.full(dataset.n_symbols, -5.0)
    flat = np.zeros(dataset.n_symbols)
    result = executor._constrain_borrow(desired, current=flat, index=3)
    assert np.all(result[0::2] == 0.0)  # a spot short is clipped to flat
    assert np.all(result[1::2] == -5.0)  # a perpetual short is allowed


# --- Vision daily fallback for hours missing from the primary source -------------------


_DAILY_ARCHIVES: dict[str, bytes] = {}  # zip bytes embed a clock: build each url once


def _daily_fallback(
    perp_missing: frozenset[int],
    *,
    unpublished: frozenset[str] = frozenset(),
) -> tuple[Any, Any]:
    from tests.integrations.market_store.support import FakeFetcher, archive_bytes
    from trade_rl.integrations.market_store.carry_source import VisionArchiveSource

    fetcher = FakeFetcher()

    def daily(plan: Any) -> Any:
        for spec in plan.specs:
            if spec.url not in fetcher.files and spec.url not in unpublished:
                payload = _DAILY_ARCHIVES.setdefault(spec.url, archive_bytes(spec))
                fetcher.add(spec.url, payload)
        return VisionArchiveSource(fetcher, plan)

    primary = FakeRows(perp_missing=perp_missing)
    return driver.DailyFallbackSource(primary, daily, halts=(HALT,)), fetcher


def test_daily_fallback_fills_only_missing_hours_with_per_hour_provenance() -> None:
    hole = frozenset(_ms(START) + h * HOUR_MS for h in (40, 41, 70))
    source, fetcher = _daily_fallback(hole)
    dataset = _load(source, halt_intervals=(HALT,))

    fills = source.fills()
    # one entry per filled hour and symbol, ascending, never forward-filled
    assert len(fills) == 3 * len(protocol.ROSTER)
    assert fills == sorted(
        fills, key=lambda i: (i["market"], i["symbol"], i["open_time"])
    )
    assert [item["open_time"] for item in fills[:3]] == [
        "2021-12-30T16:00:00+00:00",
        "2021-12-30T17:00:00+00:00",
        "2021-12-31T22:00:00+00:00",
    ]
    assert {item["source"] for item in fills} == {"vision_daily"}
    assert {item["market"] for item in fills} == {"usds_m"}
    for item in fills:
        assert str(item["archive_url"]).endswith(
            f"{item['symbol']}-1h-{str(item['open_time'])[:10]}.zip"
        )
        assert "/futures/um/daily/klines/" in str(item["archive_url"])
        assert len(str(item["archive_sha256"])) == 64
    # only the daily archives of the holes were requested (plus their checksums)
    assert fetcher.requested and all("/daily/" in url for url in fetcher.requested)
    # the 4h bar [16:00, 20:00) mixes two archive-filled hours with two primary rows
    day = [
        _kline(r)
        for r in kline_csv_rows(
            "BTCUSDT",
            _ms(datetime(2021, 12, 30, tzinfo=UTC)),
            _ms(datetime(2021, 12, 31, tzinfo=UTC)),
        )
    ]
    primary = [_kline(r) for r in kline_csv_rows("BTCUSDT", _ms(START), _ms(END))]
    bar = dataset.timestamps == np.datetime64("2021-12-30T20:00:00", "ns")
    expected_high = max(
        float(day[16].high),
        float(day[17].high),
        float(primary[42].high),
        float(primary[43].high),
    )
    assert dataset.high[bar][0, 1] == expected_high
    assert dataset.open[bar][0, 1] == float(day[16].open)
    assert dataset.n_bars == 36
    assert np.asarray(dataset.tradable)[:, 1::2].all()  # no perp bar is masked
    # deterministic
    again, _ = _daily_fallback(hole)
    assert _load(again, halt_intervals=(HALT,)).dataset_id == dataset.dataset_id
    assert again.fills() == fills


def test_daily_fallback_never_overrides_present_rows_or_halt_explained_hours() -> None:
    source, fetcher = _daily_fallback(frozenset())
    clean = _load(FakeRows(), halt_intervals=())
    same = _load(source, halt_intervals=())
    assert same.dataset_id == clean.dataset_id
    assert source.fills() == [] and fetcher.requested == []

    halted, fetcher = _daily_fallback(frozenset())
    halted._primary.spot_missing = _halt_skips()  # type: ignore[attr-defined]
    _load(halted, halt_intervals=(HALT,))
    assert halted.fills() == [] and fetcher.requested == []  # a halt is not a hole


def test_an_unpublished_daily_archive_leaves_the_hole_and_the_load_refuses() -> None:
    hole = frozenset({_ms(START) + 40 * HOUR_MS})
    day = "BTCUSDT-1h-2021-12-30.zip"
    source, _ = _daily_fallback(
        hole,
        unpublished=frozenset(
            f"https://data.binance.vision/data/futures/um/daily/klines/{s}/1h/{s}-1h-2021-12-30.zip"
            for s in protocol.ROSTER
        ),
    )
    assert day
    with pytest.raises(CarryCoverageError, match="missing hourly"):
        _load(source, halt_intervals=(HALT,))
    assert source.fills() == []


def test_a_corrupt_daily_archive_is_an_integrity_halt() -> None:
    from tests.integrations.market_store.support import FakeFetcher, archive_bytes
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt
    from trade_rl.integrations.market_store.carry_source import VisionArchiveSource

    fetcher = FakeFetcher()

    def daily(plan: Any) -> Any:
        for spec in plan.specs:
            fetcher.add(spec.url, archive_bytes(spec), checksum=b"0" * 64 + b"  x\n")
        return VisionArchiveSource(fetcher, plan)

    hole = frozenset({_ms(START) + 40 * HOUR_MS})
    source = driver.DailyFallbackSource(FakeRows(perp_missing=hole), daily)
    with pytest.raises(CarryIntegrityHalt):
        _load(source, halt_intervals=())


def test_fallback_provenance_reaches_the_receipt_and_the_cli_flag_needs_an_archive_dir(
    tmp_path: Path,
) -> None:
    hole = frozenset({_ms(START) + 40 * HOUR_MS})
    source, _ = _daily_fallback(hole)
    argv = [*_cli_files(tmp_path), "--daily-fallback"]
    code, out, err = _main(["dry-check", *argv], source=source)  # type: ignore[arg-type]
    assert code == driver.EXIT_OK, err
    receipt = json.loads(out)["receipt"]["daily_fallback"]
    assert receipt["enabled"] is True and receipt["filled_hours"] == 11
    assert receipt["forward_fill"] is False and receipt["rest_fallback"] is False
    assert len(receipt["fills_digest"]) == 64

    # without an injected source the flag is refused unless an archive dir is given
    out_s, err_s = io.StringIO(), io.StringIO()
    code = driver.main(
        ["dry-check", *argv],
        stdout=out_s,
        stderr=err_s,
        state=lambda: ("a" * 40, True),
        load_end=END,
    )
    assert code == driver.EXIT_REFUSED and "--archive-dir" in err_s.getvalue()


def test_default_load_start_is_2021_10_01_and_the_protocol_pin_is_v3() -> None:
    assert driver.DEFAULT_LOAD_START == datetime(2021, 10, 1, tzinfo=UTC)
    args = driver.build_parser().parse_args(
        [
            "dry-check",
            "--exchange-info-spot",
            "s.json",
            "--exchange-info-perp",
            "p.json",
        ]
    )
    assert args.load_start == datetime(2021, 10, 1, tzinfo=UTC)
    run_args = driver.build_parser().parse_args(
        [
            "run",
            "--exchange-info-spot",
            "s.json",
            "--exchange-info-perp",
            "p.json",
            "--authorization-url",
            AUTH,
            "--output-dir",
            "o",
        ]
    )
    assert run_args.load_start == datetime(2021, 10, 1, tzinfo=UTC)
    assert protocol.PROTOCOL_ID == "core7_mtf_shadow_anchor_ppo_v3"
    assert protocol.protocol_digest() == (
        "239b37a73ab002ef45c1a3ae07f7b1e1b636f59e519a416dce33202a546d630a"
    )


def test_run_does_not_report_success_for_a_result_that_is_not_evidence_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for verdict in (
        {"outcome": "SYNTHETIC_ONLY_NOT_EVIDENCE", "evidence_eligible": False},
        {"outcome": "PASS_PARITY_ONLY"},  # eligibility missing is not eligible
    ):
        monkeypatch.setattr(driver, "run_study", lambda *a, _v=verdict, **k: dict(_v))
        out_dir = tmp_path / f"out{len(verdict)}"
        code, out, err = _main(
            [
                "run",
                *_cli_files(tmp_path),
                "--output-dir",
                str(out_dir),
                "--authorization-url",
                AUTH,
            ]
        )
        assert code == driver.EXIT_NOT_EVIDENCE != driver.EXIT_OK
        assert "not evidence_eligible" in err
        receipt = json.loads(
            (out_dir / driver.RUN_RECEIPT_NAME).read_text(encoding="utf-8")
        )
        assert receipt["evidence_eligible"] is False


@pytest.mark.parametrize("digest", (None, "a" * 63, "g" * 64))
def test_daily_fallback_refuses_unverified_source_digest_before_filling(
    digest: str | None,
) -> None:
    """Source bytes without valid archive provenance cannot become a 4h leg."""
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    missing_hour = _ms(START) + 40 * HOUR_MS

    class UnverifiedDaily(FakeRows):
        def __init__(self, archive_url: str) -> None:
            super().__init__()
            self.archive_url = archive_url

        def evidence(self) -> list[dict[str, object]]:
            return (
                []
                if digest is None
                else [{"url": self.archive_url, "archive_sha256": digest}]
            )

    source = driver.DailyFallbackSource(
        FakeRows(perp_missing=frozenset({missing_hour})),
        lambda plan: UnverifiedDaily(plan.specs[0].url),
        halts=(),
    )
    with pytest.raises(CarryIntegrityHalt, match="archive digest"):
        _load(source, halt_intervals=())
    assert source.fills() == []


def test_daily_fallback_refuses_partial_fills_across_two_day_source_failure() -> None:
    """A later bad daily digest cannot leave earlier per-call fill provenance."""
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    base = _ms(START)
    holes = frozenset({base + 2 * HOUR_MS, base + 26 * HOUR_MS})

    class OnlyFirstDayVerified(FakeRows):
        def __init__(self, first_url: str) -> None:
            super().__init__()
            self.first_url = first_url

        def evidence(self) -> list[dict[str, object]]:
            return [{"url": self.first_url, "archive_sha256": "a" * 64}]

    source = driver.DailyFallbackSource(
        FakeRows(perp_missing=holes),
        lambda plan: OnlyFirstDayVerified(plan.specs[0].url),
        halts=(),
    )
    with pytest.raises(CarryIntegrityHalt, match="archive digest"):
        source.read_klines(
            StoreMarket.USDS_M,
            protocol.ROSTER[0],
            "1h",
            base,
            base + 48 * HOUR_MS,
        )
    assert source.fills() == []
    assert source.evidence() == []


def test_daily_fallback_failed_retry_preserves_only_previous_committed_fills() -> None:
    """A failed larger read must not add to or erase earlier completed fills."""
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    base = _ms(START)
    holes = frozenset(
        {
            base + 2 * HOUR_MS,
            base + 3 * HOUR_MS,
            base + 26 * HOUR_MS,
        }
    )

    class OnlyFirstDayVerified(FakeRows):
        def __init__(self, first_url: str, digest: str) -> None:
            super().__init__()
            self.first_url = first_url
            self.digest = digest

        def evidence(self) -> list[dict[str, object]]:
            return [{"url": self.first_url, "archive_sha256": self.digest}]

    source_calls = 0

    def daily_factory(plan: Any) -> OnlyFirstDayVerified:
        nonlocal source_calls
        source_calls += 1
        return OnlyFirstDayVerified(
            plan.specs[0].url,
            "a" * 64 if source_calls == 1 else "b" * 64,
        )

    source = driver.DailyFallbackSource(
        FakeRows(perp_missing=holes),
        daily_factory,
        halts=(),
    )
    source.read_klines(
        StoreMarket.USDS_M,
        protocol.ROSTER[0],
        "1h",
        base,
        base + 3 * HOUR_MS,
    )
    prior = source.fills()
    prior_evidence = source.evidence()
    assert len(prior) == 1 and prior[0]["archive_sha256"] == "a" * 64
    assert len(prior_evidence) == 1

    with pytest.raises(CarryIntegrityHalt, match="archive digest"):
        source.read_klines(
            StoreMarket.USDS_M,
            protocol.ROSTER[0],
            "1h",
            base,
            base + 48 * HOUR_MS,
        )
    assert source.fills() == prior
    assert source.evidence() == prior_evidence


def test_daily_fallback_rejects_wrong_archive_day_provenance() -> None:
    """A valid day-1 SHA cannot justify a missing row belonging to day 2."""
    from trade_rl.integrations.binance.carry import CarryIntegrityHalt

    base = _ms(START)
    holes = frozenset({base + 2 * HOUR_MS, base + 26 * HOUR_MS})

    class SourceWithWrongDayRow(FakeRows):
        def __init__(self, archive_url: str) -> None:
            super().__init__()
            self.archive_url = archive_url

        def evidence(self) -> list[dict[str, object]]:
            return [{"url": self.archive_url, "archive_sha256": "a" * 64}]

        def read_klines(
            self,
            market: StoreMarket,
            symbol: str,
            interval: str,
            start_ms: int,
            end_ms: int,
        ) -> list[KlineRow]:
            if start_ms != base:
                return []
            rows = super().read_klines(
                market,
                symbol,
                interval,
                start_ms,
                end_ms,
            )
            forged = kline_csv_rows(
                symbol,
                base + 26 * HOUR_MS,
                base + 27 * HOUR_MS,
            )
            rows.append(_kline(forged[0]))
            return rows

    source = driver.DailyFallbackSource(
        FakeRows(perp_missing=holes),
        lambda plan: SourceWithWrongDayRow(plan.specs[0].url),
        halts=(),
    )
    with pytest.raises(CarryIntegrityHalt, match="daily.*day"):
        source.read_klines(
            StoreMarket.USDS_M,
            protocol.ROSTER[0],
            "1h",
            base,
            base + 48 * HOUR_MS,
        )
    assert source.fills() == []
